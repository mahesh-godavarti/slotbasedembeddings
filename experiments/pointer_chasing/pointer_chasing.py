#!/usr/bin/env python3
"""Pointer chasing dataset and training for TC^0 separation experiment.

Demonstrates that D=1 look-ahead trained with BPTT can solve k-hop pointer
chasing for any k, while N=k transformer fails for k+1 hops.

Dataset format (3-hop example):
    Table:  A=5 B=3 C=8 D=1
    Index1: X=B Y=D Z=A
    Index2: P=X Q=Z R=Y
    Query:  P
    Answer: 3  (P -> X -> B -> 3)

The sequence is encoded as tokens. The model must predict the answer token
at the final position.
"""

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks2 import RoFormerBlock, FeedForward, build_rotation_matrix, apply_rotation
from blocks_datadep import make_datadep_model, DATADEP_VARIANTS, DATADEP2_VARIANTS


class PointerChasingDataset:
    """Generate pointer chasing examples with configurable hop count."""

    def __init__(self, n_keys=8, n_values=16, n_hops=3, n_show=None, permutation=False, seed=None, vocab_hops=None, multi_q=False, shuffle=True):
        self.n_keys = n_keys      # number of keys per level
        self.n_show = n_show if n_show is not None else n_keys  # entries shown per table
        self.permutation = permutation  # use bijective mappings in index tables
        self.n_values = n_values  # number of possible values at base level
        self.n_hops = n_hops
        self.multi_q = multi_q    # multiple Q sections per level (helper signals)
        self.shuffle = shuffle    # shuffle table/Q section entry order
        self._n_keys_active = None  # for key curriculum: block permutations
        self.vocab_hops = vocab_hops if vocab_hops is not None else n_hops

        # Vocabulary:
        # 0: PAD
        # 1: QUERY token
        # 2: EQUALS token (=)
        # 3: LEVEL_SEP token (|)
        # 4 .. 4+n_keys*vocab_hops-1: per-level key tokens (A0,B0,..,A1,B1,..,)
        # 4+n_keys*vocab_hops .. 4+n_keys*vocab_hops+n_values-1: value tokens
        self.PAD = 0
        self.QUERY = 1
        self.EQUALS = 2
        self.LEVEL_SEP = 3
        self.key_offset = 4
        self.value_offset = 4 + n_keys * self.vocab_hops
        self.vocab_size = 4 + n_keys * self.vocab_hops + n_values

        if seed is not None:
            random.seed(seed)

    def key_token(self, i, level=0):
        """Key token for key i at given level."""
        return self.key_offset + level * self.n_keys + i

    def value_token(self, v):
        return self.value_offset + v

    def _resolve(self, key, from_level, levels, base_table):
        """Resolve key from from_level down through base. Returns base value."""
        current = key
        for lev in range(from_level, 0, -1):
            current = levels[lev][current]
        return base_table[current]

    def _chain_all_active(self, key, from_level, levels, n_keys_active):
        """Check if the entire resolution chain from key at from_level stays within active keys."""
        if key >= n_keys_active:
            return False
        current = key
        for lev in range(from_level, 0, -1):
            current = levels[lev][current]
            if current >= n_keys_active:
                return False
        return True

    def _resolve_hops(self, key, from_level, n_hops, levels, base_table):
        """Resolve key by exactly n_hops steps. Returns (token, is_value).
        n_hops=1: one index lookup (returns key at from_level-1)
        n_hops=from_level: full resolution through base (returns value)
        n_hops=from_level+1 if we count base lookup: not needed, _resolve handles it
        """
        current = key
        for i in range(n_hops):
            level = from_level - i
            if level > 0:
                current = levels[level][current]
            else:
                # level 0 = base table lookup, returns value
                return self.value_token(base_table[current])
        # Still a key (didn't reach base)
        result_level = from_level - n_hops
        return self.key_token(current, level=result_level)

    def generate_example(self):
        """Generate one pointer chasing example with dense targets.

        Returns: (input_tokens, target_tokens, sequence_length)
        target_tokens: same length as input_tokens. Value token at key positions, -1 elsewhere.
        """
        keys = list(range(self.n_keys))

        # Level 0: base table. Each key maps to a random value.
        base_table = {}
        for k in keys:
            base_table[k] = random.randint(0, self.n_values - 1)

        # Levels 1..n_hops-1: each key maps to a key from the previous level
        levels = [base_table]
        for level in range(1, self.n_hops):
            if self.permutation:
                shuffled = list(keys)
                random.shuffle(shuffled)
                table = dict(zip(keys, shuffled))
            else:
                table = {}
                for k in keys:
                    table[k] = random.choice(keys)
            levels.append(table)

        query_key = random.choice(keys)
        answer = self._resolve(query_key, self.n_hops - 1, levels, base_table)

        # For n_show: track path keys
        path_keys = [query_key]
        current = query_key
        for level in range(self.n_hops - 1, -1, -1):
            current = levels[level][current]
            path_keys.append(current)

        def pick_shown_keys(required_key):
            if self.n_show >= self.n_keys:
                shown = keys[:]
                if self.shuffle:
                    random.shuffle(shown)
                return shown
            others = [k for k in keys if k != required_key]
            if self.shuffle:
                random.shuffle(others)
            shown = [required_key] + others[:self.n_show - 1]
            if self.shuffle:
                random.shuffle(shown)
            return shown

        tokens = []
        targets = []
        IGNORE = -1

        # Base table (level 0): value=key0, no targets in table
        base_required = path_keys[self.n_hops - 1]
        base_shown = pick_shown_keys(base_required)
        for k in base_shown:
            tokens.append(self.value_token(base_table[k]))
            targets.append(IGNORE)
            tokens.append(self.EQUALS)
            targets.append(IGNORE)
            tokens.append(self.key_token(k, level=0))
            targets.append(IGNORE)
        tokens.append(self.LEVEL_SEP)
        targets.append(IGNORE)

        # Query section for base level: Q key -> target = base value (0 hops)
        n_active = self._n_keys_active
        for k in pick_shown_keys(base_required):
            tokens.append(self.QUERY)
            targets.append(IGNORE)
            tokens.append(self.key_token(k, level=0))
            if n_active is not None and k >= n_active:
                targets.append(IGNORE)
            else:
                targets.append(self.value_token(base_table[k]))
        tokens.append(self.LEVEL_SEP)
        targets.append(IGNORE)

        # Index levels: no targets in table, multiple Q sections per level
        for level in range(1, self.n_hops):
            level_required = path_keys[self.n_hops - 1 - level]
            level_shown = pick_shown_keys(level_required)
            for k in level_shown:
                tokens.append(self.key_token(levels[level][k], level=level - 1))
                targets.append(IGNORE)
                tokens.append(self.EQUALS)
                targets.append(IGNORE)
                tokens.append(self.key_token(k, level=level))
                targets.append(IGNORE)
            tokens.append(self.LEVEL_SEP)
            targets.append(IGNORE)

            if self.multi_q:
                # Multiple Q sections: Q1 (1-hop), Q2 (2-hop), ..., Q_{level+1} (fully resolved)
                for hop in range(1, level + 2):
                    for k in pick_shown_keys(level_required):
                        tokens.append(self.QUERY)
                        targets.append(IGNORE)
                        tokens.append(self.key_token(k, level=level))
                        if n_active is not None and not self._chain_all_active(k, level, levels, n_active):
                            targets.append(IGNORE)
                        else:
                            targets.append(self._resolve_hops(k, level, hop, levels, base_table))
                    tokens.append(self.LEVEL_SEP)
                    targets.append(IGNORE)
            else:
                # Single Q section: fully resolved target
                for k in pick_shown_keys(level_required):
                    tokens.append(self.QUERY)
                    targets.append(IGNORE)
                    tokens.append(self.key_token(k, level=level))
                    if n_active is not None and not self._chain_all_active(k, level, levels, n_active):
                        targets.append(IGNORE)
                    else:
                        resolved = self._resolve(k, level, levels, base_table)
                        targets.append(self.value_token(resolved))
                tokens.append(self.LEVEL_SEP)
                targets.append(IGNORE)

        # Final query: single key at top level
        tokens.append(self.QUERY)
        targets.append(IGNORE)
        tokens.append(self.key_token(query_key, level=self.n_hops - 1))
        if n_active is not None and not self._chain_all_active(query_key, self.n_hops - 1, levels, n_active):
            targets.append(IGNORE)
        else:
            targets.append(self.value_token(answer))

        return tokens, targets, len(tokens)

    def generate_batch(self, batch_size):
        """Generate a batch with dense targets. Padded positions get -1 (ignore)."""
        examples = [self.generate_example() for _ in range(batch_size)]
        max_len = max(e[2] for e in examples)

        input_seqs = []
        target_seqs = []
        for tokens, tgts, length in examples:
            input_seqs.append(tokens + [self.PAD] * (max_len - length))
            target_seqs.append(tgts + [-1] * (max_len - length))

        return (torch.tensor(input_seqs, dtype=torch.long),
                torch.tensor(target_seqs, dtype=torch.long),
                max_len)


class LookAheadD1Sequential(nn.Module):
    """D=1 look-ahead trained with BPTT using KV-cached incremental block.

    Uses the block's own weights for incremental computation.
    The block is run incrementally for corrections (O(T²)),
    then one final full pass for output logits.

    Cost: O(T²) instead of O(T³).
    """

    def __init__(self, vocab_size, n_embed, block_size, n_head=4, dropout=0.0, z_residual=False, no_rope=False, window=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.block = RoFormerBlock(n_embed, block_size, dropout, use_softmax=True, n_head=n_head, no_rope=no_rope, window=window)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.z_residual = z_residual
        self.no_rope = no_rope
        self.window = window

    def _precompute_rope(self, T, device):
        """Precompute standard RoPE matrices (no flip, matches blocks2.py)."""
        D = self.head_dim
        angle1 = torch.arange(T, device=device)
        angle2 = torch.arange(D // 2, device=device)
        angle = torch.outer(angle1, angle2).unsqueeze(0)  # (1, T, D//2)
        return build_rotation_matrix(torch.cos(angle), torch.sin(angle))

    def _incremental_block(self, px_t, t, k_cache, v_cache, rope_matrices):
        """Run block on position t using KV cache. Matches blocks2.py exactly.

        blocks2.py uses q @ k^T with standard RoPE. Standard KV caching.
        """
        B = px_t.shape[0]
        C = self.n_embed
        H = self.n_head
        D = self.head_dim
        attn = self.block.sa_head

        x_ln = self.block.ln1(px_t)

        q = attn.queries(x_ln)
        k = attn.keys(x_ln)
        v = attn.values(x_ln)

        if H > 1:
            q = q.view(B, 1, H, D).transpose(1, 2).reshape(B * H, 1, D)
            k = k.view(B, 1, H, D).transpose(1, 2).reshape(B * H, 1, D)
            v = v.view(B, 1, H, D).transpose(1, 2).reshape(B * H, 1, D)

        if self.no_rope:
            k_rot = k
            q_rot = q
        else:
            matrix_t = rope_matrices[:, t:t+1, :, :, :]
            k_rot = apply_rotation(k, matrix_t)
            q_rot = apply_rotation(q, matrix_t)

        if len(k_cache) > 0:
            all_k = torch.cat(k_cache + [k_rot], dim=1)
            all_v = torch.cat(v_cache + [v], dim=1)
        else:
            all_k = k_rot
            all_v = v

        # Apply window: only attend to last W positions
        if self.window is not None and all_k.shape[1] > self.window:
            all_k = all_k[:, -self.window:, :]
            all_v = all_v[:, -self.window:, :]

        # q @ k^T — matches blocks2.py exactly
        wei = q_rot @ all_k.transpose(-1, -2) * D ** (-0.5)  # (B*H, 1, t+1)
        wei = F.softmax(wei, dim=-1)
        attn_out = wei @ all_v  # (B*H, 1, D)

        if H > 1:
            attn_out = attn_out.view(B, H, 1, D).transpose(1, 2).reshape(B, 1, C)

        attn_out = attn.proj(attn_out)

        h = px_t + attn_out
        z_t = h + self.block.ffn(self.block.ln2(h))

        return z_t, k_rot, v

    def forward(self, idx):
        """Sequential BPTT with KV cache. O(T²) instead of O(T³)."""
        B, T = idx.shape
        device = idx.device
        C = self.n_embed

        tok_emb = self.token_embedding(idx)

        # Precompute RoPE matrices for all positions (flipped, matching blocks.py)
        rope_matrices = self._precompute_rope(T, device)

        px_list = []
        z_list = []
        k_cache = []
        v_cache = []

        # Position 0
        zero = torch.zeros(B, 1, C, device=device)
        corr_0 = self.corr_ffn(self.ln_corr(zero + tok_emb[:, 0:1, :]))
        if self.z_residual:
            px_0 = tok_emb[:, 0:1, :] + zero + corr_0
        else:
            px_0 = tok_emb[:, 0:1, :] + corr_0
        px_list.append(px_0)

        z_0, k_0, v_0 = self._incremental_block(px_0, 0, k_cache, v_cache, rope_matrices)
        z_list.append(z_0)
        k_cache.append(k_0)
        v_cache.append(v_0)

        for t in range(1, T):
            z_prev = z_list[-1]
            corr_t = self.corr_ffn(self.ln_corr(z_prev + tok_emb[:, t:t+1, :]))
            if self.z_residual:
                px_t = tok_emb[:, t:t+1, :] + z_prev + corr_t
            else:
                px_t = tok_emb[:, t:t+1, :] + corr_t
            px_list.append(px_t)

            z_t, k_t, v_t = self._incremental_block(px_t, t, k_cache, v_cache, rope_matrices)
            z_list.append(z_t)
            k_cache.append(k_t)
            v_cache.append(v_t)

        # Final full block pass for output logits
        px_full = torch.cat(px_list, dim=1)
        z_final = self.block(px_full)

        logits = self.head(self.ln_f(z_final))
        return logits


class TransformerBaseline(nn.Module):
    """Standard N-layer transformer (parallel, no look-ahead)."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_head=4, dropout=0.0, no_rope=False, window=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax=True, n_head=n_head, no_rope=no_rope, window=window)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        """Standard parallel forward pass."""
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # Output at all positions
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


def _get_level_target_positions(dataset, block_size):
    """Precompute target positions for each level in the sequence."""
    n_keys = dataset.n_show
    n_hops = dataset.n_hops
    multi_q = dataset.multi_q
    q_section_size = 2 * n_keys + 1
    table_size = 3 * n_keys + 1
    level_positions = {}
    pos = 0
    for lev in range(n_hops):
        pos += table_size
        n_q_sections = (1 if lev == 0 else lev + 1) if multi_q else 1
        # Collect ALL Q section key positions for this level
        all_positions = []
        for q_idx in range(n_q_sections):
            positions = [pos + 1 + 2 * k for k in range(n_keys)]
            all_positions.extend(positions)
            pos += q_section_size
        level_positions[lev] = all_positions
    # Final query position
    level_positions['final'] = [pos + 1]  # after the last QUERY token
    return level_positions


def _mask_targets_by_hop(targets, level_positions, max_hop):
    """Mask targets for levels > max_hop to -1."""
    targets = targets.clone()
    for lev, positions in level_positions.items():
        if lev == 'final':
            continue
        if lev >= max_hop:
            for p in positions:
                if p < targets.shape[1]:
                    targets[:, p] = -1
    # Also mask final query if max_hop < n_hops
    if max_hop < max(k for k in level_positions if k != 'final') + 1:
        for p in level_positions['final']:
            if p < targets.shape[1]:
                targets[:, p] = -1
    return targets


def _mask_targets_by_key(targets, inputs, n_keys_active, n_keys_total, key_offset, value_offset):
    """Mask targets where the resolution chain passes through inactive keys.

    A target is valid only if:
    1. The query key itself is active (key_id < n_keys_active)
    2. The target value corresponds to an active base key

    For condition 2: the target is a value token. We check if the base key
    that produces this value is active. Since we can't easily trace the chain
    from target positions alone, we check both the query key AND whether the
    target value comes from an active base key.

    Simpler approach: mask if the query key_id >= n_keys_active OR if the
    target value doesn't match any active base key's value. But values can
    collide. Instead, just mask by query key identity — the chain-based
    masking happens naturally because inactive keys produce targets that
    the model doesn't learn from.

    Actually, the correct approach: mask if query key_id >= n_keys_active.
    For active query keys whose chain goes through inactive keys, the target
    is still set but the model may struggle — this is acceptable. The key
    curriculum gradually makes more chains valid.

    REVISED: To properly mask chains through inactive keys, we need the
    dataset to mark which targets have all-active chains. We do this by
    checking the target: if target is a value token, check if the chain
    stayed within active keys. This requires regenerating with chain tracking.

    SIMPLEST CORRECT APPROACH: At each target position, the input token is
    the query key. Mask if key_id >= n_keys_active. Additionally, check if
    the TARGET itself is reachable only through active keys. Since we can't
    easily do that post-hoc, we accept that some active-key targets may
    have chains through inactive keys — those targets remain active but
    the model learns from whatever the chain produces.
    """
    targets = targets.clone()
    mask = targets != -1
    if not mask.any():
        return targets
    key_tokens = inputs[mask]
    key_ids = (key_tokens - key_offset) % n_keys_total
    suppress = key_ids >= n_keys_active
    target_vals = targets[mask]
    target_vals[suppress] = -1
    targets[mask] = target_vals
    return targets


def train_and_eval(model, dataset, n_iters=5000, batch_size=64, lr=1e-3, device='cuda',
                   eval_every=500, eval_batches=20, checkpoint_dir=None, checkpoint_every=1000,
                   curriculum=None, curriculum_datasets=None,
                   hop_curriculum=None, key_curriculum=None):
    """Train model with dense targets and evaluate per-level accuracy.

    hop_curriculum: list of (iter, max_hop) — mask targets for levels >= max_hop
    key_curriculum: list of (iter, n_keys_active) — mask targets for keys >= n_keys_active
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_iter = 1
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, 'latest.pt')):
        ckpt = torch.load(os.path.join(checkpoint_dir, 'latest.pt'), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = ckpt['iter'] + 1
        print(f"  Resumed from iter {start_iter - 1}")

    current_dataset = dataset
    current_hops = dataset.n_hops

    # Precompute level positions for hop/key curriculum masking
    level_pos = None
    if hop_curriculum or key_curriculum:
        level_pos = _get_level_target_positions(dataset, 0)

    prev_max_hop = None
    prev_n_keys_active = None

    for it in range(start_iter, n_iters + 1):
        # Curriculum: switch dataset based on iteration
        if curriculum and curriculum_datasets:
            for c_iter, c_hops in reversed(curriculum):
                if it >= c_iter:
                    if c_hops != current_hops:
                        current_hops = c_hops
                        current_dataset = curriculum_datasets[c_hops]
                        print(f"  [Curriculum] Switching to {c_hops}-hop at iter {it}")
                    break

        # Determine current hop/key curriculum
        max_hop = dataset.n_hops
        if hop_curriculum:
            for c_iter, c_val in reversed(hop_curriculum):
                if it >= c_iter:
                    max_hop = c_val
                    break
            if max_hop != prev_max_hop:
                print(f"  [Hop curriculum] max_hop={max_hop} at iter {it}")
                prev_max_hop = max_hop

        n_keys_active = dataset.n_show
        if key_curriculum:
            for c_iter, c_val in reversed(key_curriculum):
                if it >= c_iter:
                    n_keys_active = c_val
                    break
            if n_keys_active != prev_n_keys_active:
                print(f"  [Key curriculum] n_keys_active={n_keys_active} at iter {it}")
                prev_n_keys_active = n_keys_active

        model.train()
        # Set n_keys_active for block permutations
        if key_curriculum:
            current_dataset._n_keys_active = n_keys_active if n_keys_active < current_dataset.n_keys else None
        inputs, targets, _ = current_dataset.generate_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply curriculum masking
        if hop_curriculum and max_hop < dataset.n_hops:
            targets = _mask_targets_by_hop(targets, level_pos, max_hop)
        # Key curriculum masking is done in generate_example via _n_keys_active

        logits = model(inputs)  # (B, T, vocab)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                inputs, targets, _ = current_dataset.generate_batch(batch_size * 4)
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=-1)

                # Per-level accuracy: key positions are every 3rd token in each table
                # Base: positions 2, 5, 8, ... (0-indexed, every 3rd starting at 2)
                # Index levels follow after separator
                mask = targets != -1
                if mask.sum() > 0:
                    overall_acc = (preds[mask] == targets[mask]).float().mean().item()
                else:
                    overall_acc = 0.0

                # Per-level: targets are in Q sections after each table
                # Level 0: table + 1 Q section
                # Level L (L>0): table + (L+1) Q sections (1-hop through fully resolved)
                # We eval on the LAST Q section of each level (fully resolved)
                n_keys = current_dataset.n_show
                n_hops = current_dataset.n_hops
                multi_q = current_dataset.multi_q
                level_accs = []
                q_section_size = 2 * n_keys + 1  # Q key pairs + sep
                table_size = 3 * n_keys + 1  # value=key triplets + sep
                pos = 0
                for lev in range(n_hops):
                    pos += table_size  # skip table
                    n_q_sections = (1 if lev == 0 else lev + 1) if multi_q else 1
                    # Skip to the last Q section (fully resolved)
                    pos += (n_q_sections - 1) * q_section_size
                    # Key positions in last Q section: pos + 1 + 2*k
                    level_positions = [pos + 1 + 2 * k for k in range(n_keys)]
                    pos += q_section_size  # advance past last Q section
                    level_targets = targets[:, level_positions]
                    level_preds = preds[:, level_positions]
                    level_mask = level_targets != -1
                    if level_mask.sum() > 0:
                        acc = (level_preds[level_mask] == level_targets[level_mask]).float().mean().item()
                    else:
                        acc = 0.0
                    level_accs.append(acc)

                # Query position (last)
                query_acc = (preds[:, -1] == targets[:, -1]).float().mean().item()
                level_accs.append(query_acc)

                labels = [f'L{i}' for i in range(len(level_accs) - 1)] + ['LF']
                level_str = ' '.join(f'{labels[i]}:{a:.3f}' for i, a in enumerate(level_accs))
                print(f"  iter {it:5d}: loss={loss.item():.4f}, acc={overall_acc:.4f} | {level_str}")

        if checkpoint_dir and it % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': it,
            }, os.path.join(checkpoint_dir, 'latest.pt'))

    # Final eval
    model.eval()
    with torch.no_grad():
        inputs, targets, _ = dataset.generate_batch(1024)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        mask = targets != -1
        overall_acc = (preds[mask] == targets[mask]).float().mean().item()
    return overall_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hops', type=int, default=3)
    parser.add_argument('--n_keys', type=int, default=8)
    parser.add_argument('--n_values', type=int, default=16)
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_iters', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_show', type=int, default=None, help='Entries shown per table (default: all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bptt_only', action='store_true', help='Skip transformer baselines, run only D=1 BPTT')
    parser.add_argument('--min_layers', type=int, default=1, help='Skip transformer N < min_layers')
    parser.add_argument('--run', type=str, default=None, help='Comma-separated list of models: N1,N2,N3,bptt,bptt_zresid')
    parser.add_argument('--permutation', action='store_true', help='Use bijective (permutation) mappings in index tables')
    parser.add_argument('--z_residual', action='store_true', help='Add z[t-1] residual: px = tok_emb + z_prev + corr')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for saving/resuming checkpoints')
    parser.add_argument('--no_rope', action='store_true', help='Disable RoPE positional encoding')
    parser.add_argument('--vocab_hops', type=int, default=None, help='Vocab sized for this many hops (default: n_hops)')
    parser.add_argument('--curriculum', type=str, default=None, help='Curriculum: "iter:hops,iter:hops" e.g. "0:2,100000:3"')
    parser.add_argument('--window', type=int, default=None, help='Sliding window size for attention')
    parser.add_argument('--multi_q', action='store_true', help='Multiple Q sections per level with helper targets')
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling of table/Q section entries')
    parser.add_argument('--hop_curriculum', type=str, default=None, help='Hop curriculum: "iter:max_hop,..." e.g. "0:2,50000:5,80000:10"')
    parser.add_argument('--key_curriculum', type=str, default=None, help='Key curriculum: "iter:n_keys_active,..." e.g. "0:2,50000:5,80000:10"')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    # Parse curriculum schedules
    curriculum = None
    if args.curriculum:
        curriculum = []
        for entry in args.curriculum.split(','):
            it, hops = entry.split(':')
            curriculum.append((int(it), int(hops)))
        curriculum.sort()

    hop_curriculum = None
    if args.hop_curriculum:
        hop_curriculum = []
        for entry in args.hop_curriculum.split(','):
            it, val = entry.split(':')
            hop_curriculum.append((int(it), int(val)))
        hop_curriculum.sort()

    key_curriculum = None
    if args.key_curriculum:
        key_curriculum = []
        for entry in args.key_curriculum.split(','):
            it, val = entry.split(':')
            key_curriculum.append((int(it), int(val)))
        key_curriculum.sort()

    dataset = PointerChasingDataset(
        n_keys=args.n_keys, n_values=args.n_values, n_hops=args.n_hops,
        n_show=args.n_show, permutation=args.permutation, seed=args.seed,
        vocab_hops=args.vocab_hops, multi_q=args.multi_q,
        shuffle=not args.no_shuffle
    )

    # Sequence length for block_size
    sample_tokens, _, seq_len = dataset.generate_example()
    block_size = seq_len + 10  # some padding
    print(f"Pointer chasing: {args.n_hops} hops, {args.n_keys} keys, {args.n_values} values")
    print(f"Sequence length: ~{seq_len}, vocab size: {dataset.vocab_size}")
    print(f"n_embed={args.n_embed}, n_head={args.n_head}, device={device}")
    print()

    # Determine which models to run
    if args.run:
        run_models = [m.strip() for m in args.run.split(',')]
    elif args.bptt_only:
        run_models = ['bptt_zresid' if args.z_residual else 'bptt']
    else:
        run_models = [f'N{n}' for n in range(args.min_layers, args.n_hops + 2)]
        run_models.append('bptt_zresid' if args.z_residual else 'bptt')

    for model_name in run_models:
        if model_name.startswith('N'):
            n_layers = int(model_name[1:])
            print(f"=== Transformer N={n_layers} ===")
            model = TransformerBaseline(
                dataset.vocab_size, args.n_embed, n_layers, block_size,
                n_head=args.n_head, no_rope=args.no_rope, window=args.window
            )
        elif model_name == 'bptt':
            print(f"=== D=1 Look-Ahead (BPTT, sequential training) ===")
            model = LookAheadD1Sequential(
                dataset.vocab_size, args.n_embed, block_size, n_head=args.n_head,
                z_residual=False, no_rope=args.no_rope, window=args.window
            )
        elif model_name == 'bptt_zresid':
            print(f"=== D=1 Look-Ahead (BPTT, sequential training + z_residual) ===")
            model = LookAheadD1Sequential(
                dataset.vocab_size, args.n_embed, block_size, n_head=args.n_head,
                z_residual=True, no_rope=args.no_rope, window=args.window
            )
        elif '_' in model_name and model_name.rsplit('_', 1)[0] in list(DATADEP_VARIANTS) + list(DATADEP2_VARIANTS):
            # Format: variant_N e.g. datadep_N4, joformer_N3
            parts = model_name.rsplit('_', 1)
            variant = parts[0]
            n_layers = int(parts[1][1:])
            ws = args.window if args.window else block_size
            print(f"=== {variant} N={n_layers} ===")
            model = make_datadep_model(
                variant, dataset.vocab_size, args.n_embed, n_layers, block_size,
                n_heads=args.n_head, dropout=0.0, window_size=ws
            )
        else:
            print(f"Unknown model: {model_name}, skipping")
            continue

        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        ckpt_dir = os.path.join(args.checkpoint_dir, model_name) if args.checkpoint_dir else None

        # Build curriculum datasets if needed
        c_datasets = None
        if curriculum:
            c_datasets = {}
            for _, c_hops in curriculum:
                if c_hops not in c_datasets:
                    c_datasets[c_hops] = PointerChasingDataset(
                        n_keys=args.n_keys, n_values=args.n_values, n_hops=c_hops,
                        n_show=args.n_show, permutation=args.permutation, seed=args.seed,
                        vocab_hops=args.vocab_hops, multi_q=args.multi_q,
                        shuffle=not args.no_shuffle
                    )

        acc = train_and_eval(
            model, dataset, n_iters=args.n_iters, batch_size=args.batch_size,
            lr=args.lr, device=device, checkpoint_dir=ckpt_dir,
            curriculum=curriculum, curriculum_datasets=c_datasets,
            hop_curriculum=hop_curriculum, key_curriculum=key_curriculum
        )
        print(f"  Final accuracy: {acc:.4f}")
        print()


if __name__ == '__main__':
    main()

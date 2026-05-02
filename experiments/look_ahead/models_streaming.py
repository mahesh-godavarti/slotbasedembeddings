"""Streaming-inference model definitions, state_dict-compatible with look_ahead8.

Two models, drop-in for trained checkpoints:
  - StreamingRoFormer            ↔ blocks.RoFormer
  - StreamingLookAhead           ↔ models.BlockHeadCorrFFNAddModel  (D=d_block, K=1 inference)

State_dict compatibility (matches names exactly):
  RoFormer:                token_embedding_table, blocks.<i>.{sa_head.{keys,queries,values,proj},
                           ffn.ffn.<j>, ln1, ln2}, ln_f, lm_head, blocks.<i>.sa_head.tril
  BlockHeadCorrFFNAddModel: token_embedding_table, block (D=1) | blocks.<i> (D>1),
                            corr_ffn.ffn.<j>, ln_corr, ln_f, head, drop (no params)

Math correctness:
  - Attention follows codebase exactly: wei = k @ q.T then softmax(dim=-1) over q-positions,
    out = wei @ v.  We cache Q (post-RoPE) and V (no rotation); fresh K each step.
  - RoPE: integer angles outer(arange, arange) with adjacent-pair grouping (x[0::2], x[1::2]).
    Codebase uses flipped (T-1-pos)*j angles which depend on T; we use static -pos*j angles
    that produce the SAME relative attention pattern (Δα = (q-k)*j) — this lets us cache once
    and never re-rotate as the sequence grows.

Streaming look-ahead at K=1 corresponds to BlockHeadCorrFFNAddModel.forward_sequential(seq_k=1).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (state_dict-compatible with look_ahead8/blocks.py)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Match blocks.FeedForward: self.ffn = Sequential(Linear, GELU, Linear, Dropout)."""

    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


def apply_rotation_pairs(x, cos, sin):
    """Adjacent-pair RoPE matching blocks.apply_rotation_fast.

    x: (..., D); cos, sin: broadcastable to (..., D//2).
    Pairs are (x[..., 0::2], x[..., 1::2]).
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).reshape(x.shape)


class StreamingRoFormerAttention(nn.Module):
    """KV-cached version of blocks.RoFormerAttention.

    Streaming notes:
    - Codebase: `wei = k @ q.T` then `softmax(dim=-1)` (over q-positions), `out = wei @ v`.
      So the K-stream is the "current" stream and the Q-stream is what we cache.
      We cache (Q post-RoPE) and V (unrotated).
    - RoPE: codebase uses flipped angles `(T-1-pos)*j`; we use static `-pos*j` which
      yields the same relative pattern (Δα = (q-k)*j) regardless of T. Static caching OK.
    - tril buffer is registered (matching the trained checkpoint) but unused at inference.
    """

    def __init__(self, n_embed, block_size, dropout=0.0, use_softmax=True, n_head=1):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.use_softmax = use_softmax
        # Param-name parity with RoFormerAttention:
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def init_cache(self, batch_size, max_seq_len, device, dtype):
        return {
            'q': torch.empty(batch_size, self.n_head, max_seq_len, self.head_dim,
                             device=device, dtype=dtype),
            'v': torch.empty(batch_size, self.n_head, max_seq_len, self.head_dim,
                             device=device, dtype=dtype),
            'len': 0,
        }

    def forward_step(self, x, cache):
        """x: (B, 1, C). Returns (B, 1, C)."""
        B, T, C = x.shape
        assert T == 1
        H, D = self.n_head, self.head_dim
        half = D // 2

        k = self.keys(x).view(B, 1, H, D).transpose(1, 2)     # (B, H, 1, D)
        q = self.queries(x).view(B, 1, H, D).transpose(1, 2)
        v = self.values(x).view(B, 1, H, D).transpose(1, 2)

        pos = cache['len']
        # Static RoPE: angle = -pos * arange(half). Yields same relative attn as codebase's
        # flipped (T-1-pos)*j convention. (Both give Δα = (q-k)*j in dot products.)
        j_idx = torch.arange(half, device=x.device, dtype=torch.float32)
        angle = -float(pos) * j_idx
        cos = torch.cos(angle).to(x.dtype).view(1, 1, 1, half)
        sin = torch.sin(angle).to(x.dtype).view(1, 1, 1, half)
        q = apply_rotation_pairs(q, cos, sin)
        k = apply_rotation_pairs(k, cos, sin)

        cache['q'][:, :, pos:pos + 1, :] = q
        cache['v'][:, :, pos:pos + 1, :] = v
        cache['len'] = pos + 1

        L = cache['len']
        q_all = cache['q'][:, :, :L, :]   # (B, H, L, D)
        v_all = cache['v'][:, :, :L, :]
        # Codebase: wei = k @ q.T, softmax over q-positions, then wei @ v.
        wei = torch.matmul(k, q_all.transpose(-1, -2)) * (D ** -0.5)   # (B, H, 1, L)
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            # Codebase fallback when --softmax is off.
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.matmul(wei, v_all)        # (B, H, 1, D)
        out = out.transpose(1, 2).reshape(B, 1, C)
        return self.proj(out)


class StreamingRoFormerBlock(nn.Module):
    """Match blocks.RoFormerBlock: sa_head, ffn, ln1, ln2."""

    def __init__(self, n_embed, block_size, dropout=0.0, use_softmax=True, n_head=1):
        super().__init__()
        self.sa_head = StreamingRoFormerAttention(n_embed, block_size, dropout,
                                                  use_softmax, n_head=n_head)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def init_cache(self, batch_size, max_seq_len, device, dtype):
        return self.sa_head.init_cache(batch_size, max_seq_len, device, dtype)

    def forward_step(self, x, cache):
        x = x + self.sa_head.forward_step(self.ln1(x), cache)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Streaming RoFormer (state_dict-compatible with blocks.RoFormer)
# ---------------------------------------------------------------------------

class StreamingRoFormer(nn.Module):
    """N-layer transformer with KV cache.

    State dict: token_embedding_table, blocks.<i>.{sa_head, ffn.ffn.<j>, ln1, ln2},
                ln_f, lm_head.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.0,
                 use_softmax=True, n_head=1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            StreamingRoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=n_head)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    @torch.no_grad()
    def streaming_forward_logits(self, idx, max_seq_len, dtype):
        """Run prefill on full prompt and return logits at every position."""
        device = idx.device
        B, T = idx.shape
        caches = [b.init_cache(B, max_seq_len, device, dtype) for b in self.blocks]
        autocast = torch.amp.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' \
                   else _NoAutocast()
        all_logits = []
        with autocast:
            for t in range(T):
                x = self.token_embedding_table(idx[:, t:t + 1])
                for blk, cache in zip(self.blocks, caches):
                    x = blk.forward_step(x, cache)
                logits = self.lm_head(self.ln_f(x))   # (B, 1, V)
                all_logits.append(logits)
        return torch.cat(all_logits, dim=1)            # (B, T, V)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, max_seq_len, dtype, on_chunk=None, chunk=100):
        device = prompt.device
        B = prompt.shape[0]
        caches = [b.init_cache(B, max_seq_len, device, dtype) for b in self.blocks]
        idx = prompt
        autocast = torch.amp.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' \
                   else _NoAutocast()
        with autocast:
            for t in range(prompt.shape[1]):
                x = self.token_embedding_table(prompt[:, t:t + 1])
                for blk, cache in zip(self.blocks, caches):
                    x = blk.forward_step(x, cache)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        chunk_start = time.perf_counter()
        run_start = chunk_start
        for step in range(max_new_tokens):
            with autocast:
                last_token = idx[:, -1:]
                x = self.token_embedding_table(last_token)
                for blk, cache in zip(self.blocks, caches):
                    x = blk.forward_step(x, cache)
                logits = self.lm_head(self.ln_f(x))[:, -1, :]
                probs = F.softmax(logits.float(), dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if (step + 1) % chunk == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                now = time.perf_counter()
                chunk_dt = now - chunk_start
                cum_dt = now - run_start
                chunk_start = now
                if on_chunk is not None:
                    on_chunk(step + 1, chunk_dt, cum_dt, idx.shape[1])
        return idx


# ---------------------------------------------------------------------------
# Streaming look-ahead (state_dict-compatible with BlockHeadCorrFFNAddModel)
# ---------------------------------------------------------------------------

class StreamingLookAhead(nn.Module):
    """Streaming K=1 inference for BlockHeadCorrFFNAddModel.

    State dict keys (D=1):  token_embedding_table, block.{sa_head, ffn, ln1, ln2},
                            corr_ffn.ffn.<j>, ln_corr, ln_f, head
    State dict keys (D>1):  token_embedding_table, blocks.<i>.{...}, corr_ffn, ln_corr,
                            ln_f, head

    Per step:
        h_prev := z_{t-1}                          (recurrent state, init zeros)
        corr_t := corr_ffn(ln_corr(h_prev + tok_emb_t))
        px_t   := tok_emb_t + corr_t
        z_t    := block-stack(px_t, KV-cache)
        head sees z_t; h_prev := z_t.

    Matches BlockHeadCorrFFNAddModel.forward_sequential(seq_k=1) for n_iters > 1.
    """

    def __init__(self, vocab_size, n_embed, d_block, block_size, dropout=0.0,
                 use_softmax=True, n_head=1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)
        self.d_block = d_block
        if d_block == 1:
            self.block = StreamingRoFormerBlock(n_embed, block_size, dropout,
                                                use_softmax, n_head=n_head)
        else:
            self.blocks = nn.ModuleList([
                StreamingRoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=n_head)
                for _ in range(d_block)
            ])
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.n_embed = n_embed
        self.block_size = block_size

    def _init_caches(self, batch_size, max_seq_len, device, dtype):
        if self.d_block == 1:
            return [self.block.init_cache(batch_size, max_seq_len, device, dtype)]
        return [b.init_cache(batch_size, max_seq_len, device, dtype) for b in self.blocks]

    def _block_step(self, px, caches):
        if self.d_block == 1:
            return self.block.forward_step(px, caches[0])
        z = px
        for blk, cache in zip(self.blocks, caches):
            z = blk.forward_step(z, cache)
        return z

    @torch.no_grad()
    def streaming_forward_logits(self, idx, max_seq_len, dtype):
        """Run streaming K=1 over the full prompt and return logits at every position."""
        device = idx.device
        B, T = idx.shape
        caches = self._init_caches(B, max_seq_len, device, dtype)
        h_prev = torch.zeros(B, 1, self.n_embed, device=device, dtype=dtype)
        autocast = torch.amp.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' \
                   else _NoAutocast()
        all_logits = []
        with autocast:
            for t in range(T):
                tok = self.drop(self.token_embedding_table(idx[:, t:t + 1]))
                corr = self.corr_ffn(self.ln_corr(h_prev + tok))
                px = tok + corr
                z = self._block_step(px, caches)
                h_prev = z
                logits = self.head(self.ln_f(z))    # (B, 1, V)
                all_logits.append(logits)
        return torch.cat(all_logits, dim=1)         # (B, T, V)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, max_seq_len, dtype, on_chunk=None, chunk=100):
        device = prompt.device
        B = prompt.shape[0]
        caches = self._init_caches(B, max_seq_len, device, dtype)
        h_prev = torch.zeros(B, 1, self.n_embed, device=device, dtype=dtype)
        idx = prompt
        autocast = torch.amp.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' \
                   else _NoAutocast()
        with autocast:
            for t in range(prompt.shape[1]):
                tok = self.drop(self.token_embedding_table(prompt[:, t:t + 1]))
                corr = self.corr_ffn(self.ln_corr(h_prev + tok))
                px = tok + corr
                z = self._block_step(px, caches)
                h_prev = z

        if device.type == 'cuda':
            torch.cuda.synchronize()
        chunk_start = time.perf_counter()
        run_start = chunk_start
        for step in range(max_new_tokens):
            with autocast:
                last_token = idx[:, -1:]
                tok = self.drop(self.token_embedding_table(last_token))
                corr = self.corr_ffn(self.ln_corr(h_prev + tok))
                px = tok + corr
                z = self._block_step(px, caches)
                h_prev = z
                logits = self.head(self.ln_f(z))[:, -1, :]
                probs = F.softmax(logits.float(), dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if (step + 1) % chunk == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                now = time.perf_counter()
                chunk_dt = now - chunk_start
                cum_dt = now - run_start
                chunk_start = now
                if on_chunk is not None:
                    on_chunk(step + 1, chunk_dt, cum_dt, idx.shape[1])
        return idx


class _NoAutocast:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

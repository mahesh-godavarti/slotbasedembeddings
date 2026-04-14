"""
Multi-hop left neighbor task.

Each level is a random permutation of tokens {0..K-1}.
"Left neighbor" defines the mapping at each level.
Multi-hop: query at level L -> get left neighbor -> find it in level L-1 -> get left neighbor -> ... -> level 0.

Example (K=6, 2 levels):
  Level 0: A B F G H D
  Level 1: G D A H B F
  Q B at level 1: B ->(left in L1)-> H ->(left in L0)-> G. Answer: G

Format:
  [level 0 seq] SEP Q key0 Q key1 ... SEP [level 1 seq] SEP Q key0 Q key1 ... SEP ... QUERY final_key

Targets:
  L0 Q section: left neighbor in level 0 (1 hop)
  L1 Q section: fully resolved through level 1 -> level 0 (2 hops)
  Final query: fully resolved (n_hops hops)
"""

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks2 import RoFormerBlock


class LeftNeighborMultiHopDataset:
    def __init__(self, K=8, n_hops=3, seed=None):
        self.K = K
        self.n_hops = n_hops
        # Vocab: 0..K-1 are the elements, K is SEP, K+1 is QUERY
        self.SEP = K
        self.QUERY = K + 1
        self.vocab_size = K + 2
        if seed is not None:
            random.seed(seed)

    def generate_example(self):
        K = self.K
        IGNORE = -1

        # Generate random permutations for each level
        levels = []
        for _ in range(self.n_hops):
            perm = list(range(K))
            random.shuffle(perm)
            levels.append(perm)

        # Build left-neighbor mapping for each level
        # left_map[level][token] = token to the left of 'token' in that level's permutation
        left_map = []
        for perm in levels:
            mapping = {}
            for i in range(1, K):
                mapping[perm[i]] = perm[i - 1]
            left_map.append(mapping)

        def resolve(key, from_level):
            """Resolve key from from_level down to level 0."""
            current = key
            for lev in range(from_level, -1, -1):
                if current not in left_map[lev]:
                    return None  # at position 0, no left neighbor
                current = left_map[lev][current]
            return current

        tokens = []
        targets = []

        for level in range(self.n_hops):
            # Level sequence
            for tok in levels[level]:
                tokens.append(tok)
                targets.append(IGNORE)
            tokens.append(self.SEP)
            targets.append(IGNORE)

            # Q section: query each token
            q_keys = list(range(K))
            random.shuffle(q_keys)
            for key in q_keys:
                tokens.append(self.QUERY)
                targets.append(IGNORE)
                tokens.append(key)
                resolved = resolve(key, level)
                targets.append(resolved if resolved is not None else IGNORE)
            tokens.append(self.SEP)
            targets.append(IGNORE)

        # Final query
        valid_keys = [k for k in range(K) if resolve(k, self.n_hops - 1) is not None]
        if valid_keys:
            query_key = random.choice(valid_keys)
            answer = resolve(query_key, self.n_hops - 1)
        else:
            query_key = 0
            answer = 0

        tokens.append(self.QUERY)
        targets.append(IGNORE)
        tokens.append(query_key)
        targets.append(answer)

        return tokens, targets, len(tokens)

    def generate_batch(self, batch_size):
        batch = [self.generate_example() for _ in range(batch_size)]
        max_len = max(length for _, _, length in batch)
        input_seqs = []
        target_seqs = []
        for toks, tgts, length in batch:
            input_seqs.append(toks + [0] * (max_len - length))
            target_seqs.append(tgts + [-1] * (max_len - length))
        return (torch.tensor(input_seqs, dtype=torch.long),
                torch.tensor(target_seqs, dtype=torch.long),
                max_len)


class TransformerModel(nn.Module):
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
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def train_and_eval(model, dataset, n_iters=5000, batch_size=64, lr=1e-3, device='cuda', eval_every=500):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    K = dataset.K
    n_hops = dataset.n_hops
    level_block = K + 1 + 2 * K + 1  # seq + SEP + Q section + SEP

    for it in range(1, n_iters + 1):
        model.train()
        inputs, targets, _ = dataset.generate_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                inputs, targets, _ = dataset.generate_batch(batch_size * 4)
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=-1)

                mask = targets != -1
                overall_acc = (preds[mask] == targets[mask]).float().mean().item() if mask.sum() > 0 else 0.0

                # Per-level accuracy
                level_accs = []
                for lev in range(n_hops):
                    q_start = lev * level_block + K + 1
                    level_positions = [q_start + 1 + 2 * j for j in range(K)]
                    if max(level_positions) < targets.shape[1]:
                        lev_targets = targets[:, level_positions]
                        lev_preds = preds[:, level_positions]
                        lev_mask = lev_targets != -1
                        acc = (lev_preds[lev_mask] == lev_targets[lev_mask]).float().mean().item() if lev_mask.sum() > 0 else 0.0
                    else:
                        acc = 0.0
                    level_accs.append(acc)

                # Final query
                query_acc = (preds[:, -1] == targets[:, -1]).float().mean().item()
                level_accs.append(query_acc)

                level_str = ' '.join(f'L{i}:{a:.3f}' for i, a in enumerate(level_accs))
                print(f"  iter {it:5d}: loss={loss.item():.4f}, acc={overall_acc:.4f} | {level_str}")

    model.eval()
    with torch.no_grad():
        inputs, targets, _ = dataset.generate_batch(batch_size * 4)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        mask = targets != -1
        return (preds[mask] == targets[mask]).float().mean().item() if mask.sum() > 0 else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=8, help='Tokens per level')
    parser.add_argument('--n_hops', type=int, default=3, help='Number of levels')
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_rope', action='store_true')
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--run', type=str, default=None, help='Comma-separated: N1,N3,N5,...')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    dataset = LeftNeighborMultiHopDataset(K=args.K, n_hops=args.n_hops, seed=args.seed)
    sample, _, seq_len = dataset.generate_example()
    block_size = seq_len + 10

    print(f"Left neighbor multi-hop: K={args.K}, n_hops={args.n_hops}")
    print(f"Sequence length: ~{seq_len}, vocab size: {dataset.vocab_size}")
    print(f"n_embed={args.n_embed}, n_head={args.n_head}, device={device}")
    print(f"RoPE: {'DISABLED' if args.no_rope else 'enabled'}")
    if args.window:
        print(f"Window: {args.window}")
    print()

    # Verify with a sample
    random.seed(args.seed)
    toks, tgts, _ = dataset.generate_example()
    names = [chr(65 + i) for i in range(args.K)] + ['|', 'Q']
    def decode(t): return names[t] if t < len(names) else f'?{t}'
    line = ''
    for t, tgt in zip(toks, tgts):
        s = decode(t)
        if tgt is not None and tgt != -1:
            s += f'->{decode(tgt)}'
        line += s + ' '
        if t == args.K:  # SEP
            print(f'  {line}')
            line = ''
    if line:
        print(f'  {line}')
    print()

    # Reset seed for training
    random.seed(args.seed)
    run_models = args.run.split(',') if args.run else ['N3']

    for model_name in run_models:
        if model_name.startswith('N'):
            n_layers = int(model_name[1:])
            print(f"=== Transformer N={n_layers} ===")
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            model = TransformerModel(
                dataset.vocab_size, args.n_embed, n_layers, block_size,
                n_head=args.n_head, no_rope=args.no_rope, window=args.window
            )
            params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {params:,}")
            acc = train_and_eval(
                model, dataset, n_iters=args.n_iters, batch_size=args.batch_size,
                lr=args.lr, device=device
            )
            print(f"  Final accuracy: {acc:.4f}")
            print()

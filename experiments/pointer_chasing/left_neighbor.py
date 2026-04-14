"""
Left neighbor task: find a token in a sequence, predict the token to its left.

Example: A B F G H D | G -> F
(G is the query, F is to G's left in the sequence)

Requires both content matching (find the query token) and positional awareness (look left).
RoPE should outperform no-RoPE.
"""

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks2 import RoFormerBlock


class LeftNeighborDataset:
    def __init__(self, V=20, N=10, seed=None):
        self.V = V      # vocab size (number of distinct elements)
        self.N = N      # sequence length (before separator)
        self.SEP = V    # separator token
        self.vocab_size = V + 1  # V elements + SEP
        if seed:
            random.seed(seed)

    def generate_batch(self, batch_size):
        inputs, targets = [], []
        for _ in range(batch_size):
            # Random sequence of N distinct tokens
            seq = random.sample(range(self.V), self.N)
            # Pick query: any token except the first (needs a left neighbor)
            query_idx = random.randint(1, self.N - 1)
            query_token = seq[query_idx]
            left_token = seq[query_idx - 1]
            # Input: seq + SEP + query
            inp = seq + [self.SEP, query_token]
            # Target: only at the last position (query)
            tgt = [-1] * (self.N + 1) + [left_token]
            inputs.append(inp)
            targets.append(tgt)
        return torch.tensor(inputs), torch.tensor(targets), self.N + 2


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


def run_experiment(no_rope, dataset, args, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    block_size = dataset.N + 5
    model = TransformerModel(
        dataset.vocab_size, args.n_embed, args.n_layers, block_size,
        n_head=args.n_head, no_rope=no_rope, window=args.window
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    for it in range(1, args.n_iters + 1):
        model.train()
        inp, tgt, _ = dataset.generate_batch(args.batch_size)
        inp, tgt = inp.to(device), tgt.to(device)
        loss = F.cross_entropy(model(inp).reshape(-1, dataset.vocab_size), tgt.reshape(-1), ignore_index=-1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % 1000 == 0:
            model.eval()
            with torch.no_grad():
                inp, tgt, _ = dataset.generate_batch(256)
                inp, tgt = inp.to(device), tgt.to(device)
                preds = model(inp).argmax(-1)
                mask = tgt != -1
                acc = (preds[mask] == tgt[mask]).float().mean().item()
            print(f"  iter {it:5d}: loss={loss.item():.4f} acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        inp, tgt, _ = dataset.generate_batch(1024)
        inp, tgt = inp.to(device), tgt.to(device)
        preds = model(inp).argmax(-1)
        mask = tgt != -1
        return (preds[mask] == tgt[mask]).float().mean().item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--V', type=int, default=20, help='Vocab size (number of distinct elements)')
    parser.add_argument('--N', type=int, default=10, help='Sequence length')
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--window', type=int, default=None)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    dataset = LeftNeighborDataset(V=args.V, N=args.N, seed=args.seed)

    print(f"Left neighbor task: V={args.V}, N={args.N}")
    print(f"n_embed={args.n_embed}, n_head={args.n_head}, n_layers={args.n_layers}, device={device}")
    if args.window:
        print(f"Window: {args.window}")
    print()

    print("=== With RoPE ===")
    acc_rope = run_experiment(False, dataset, args, device)
    print(f"  Final accuracy: {acc_rope:.4f}")
    print()

    print("=== Without RoPE ===")
    acc_norope = run_experiment(True, dataset, args, device)
    print(f"  Final accuracy: {acc_norope:.4f}")
    print()

    print(f"=== Summary: RoPE={acc_rope:.4f} vs no-RoPE={acc_norope:.4f} ===")

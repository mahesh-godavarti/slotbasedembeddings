"""
Missing element task: order-invariant, causal.

Present N-1 elements from {0,...,N-1} in random order.
At each position, target = the missing element.
The answer depends only on the SET of tokens seen, not the order.
RoPE and no-RoPE should behave identically.
"""

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks2 import RoFormerBlock


class MissingElementDataset:
    def __init__(self, N=10, seed=None):
        self.N = N
        # Vocab: 0..N-1 are the elements, N is the target vocab
        # Actually, elements ARE the tokens and targets
        self.vocab_size = N
        if seed is not None:
            random.seed(seed)

    def generate_example(self):
        """Generate one example. Returns (input_tokens, target_tokens, length)."""
        elements = list(range(self.N))
        missing = random.choice(elements)
        remaining = [e for e in elements if e != missing]
        random.shuffle(remaining)

        tokens = remaining  # N-1 tokens
        targets = [missing] * len(remaining)  # same target at every position

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

                # Per-position accuracy (early positions are harder)
                N = dataset.N
                pos_accs = []
                for p in range(N - 1):
                    if p < targets.shape[1]:
                        pos_mask = targets[:, p] != -1
                        if pos_mask.sum() > 0:
                            acc = (preds[:, p][pos_mask] == targets[:, p][pos_mask]).float().mean().item()
                        else:
                            acc = 0.0
                        pos_accs.append(acc)

                # Show first, middle, last position accuracy
                first = pos_accs[0] if pos_accs else 0
                mid = pos_accs[len(pos_accs)//2] if pos_accs else 0
                last = pos_accs[-1] if pos_accs else 0
                print(f"  iter {it:5d}: loss={loss.item():.4f}, acc={overall_acc:.4f} | first={first:.3f} mid={mid:.3f} last={last:.3f}")

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
    parser.add_argument('--N', type=int, default=10, help='Number of elements')
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_iters', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_rope', action='store_true')
    parser.add_argument('--window', type=int, default=None)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    dataset = MissingElementDataset(N=args.N, seed=args.seed)
    block_size = args.N + 5

    print(f"Missing element task: N={args.N}")
    print(f"n_embed={args.n_embed}, n_head={args.n_head}, n_layers={args.n_layers}, device={device}")
    print(f"RoPE: {'DISABLED' if args.no_rope else 'enabled'}")
    if args.window:
        print(f"Window: {args.window}")
    print()

    # Run with RoPE
    if not args.no_rope:
        print("=== With RoPE ===")
        model = TransformerModel(
            dataset.vocab_size, args.n_embed, args.n_layers, block_size,
            n_head=args.n_head, no_rope=False, window=args.window
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        acc = train_and_eval(model, dataset, n_iters=args.n_iters, batch_size=args.batch_size, lr=args.lr, device=device)
        print(f"  Final accuracy: {acc:.4f}")
        print()

    # Run without RoPE
    print("=== Without RoPE ===")
    torch.manual_seed(args.seed)
    model = TransformerModel(
        dataset.vocab_size, args.n_embed, args.n_layers, block_size,
        n_head=args.n_head, no_rope=True, window=args.window
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    acc = train_and_eval(model, dataset, n_iters=args.n_iters, batch_size=args.batch_size, lr=args.lr, device=device)
    print(f"  Final accuracy: {acc:.4f}")

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
from blocks import RoFormerBlock, FeedForward


class PointerChasingDataset:
    """Generate pointer chasing examples with configurable hop count."""

    def __init__(self, n_keys=8, n_values=16, n_hops=3, seed=None):
        self.n_keys = n_keys      # number of keys per level
        self.n_values = n_values  # number of possible values at base level
        self.n_hops = n_hops

        # Vocabulary:
        # 0: PAD
        # 1: QUERY token
        # 2: EQUALS token (=)
        # 3: LEVEL_SEP token (|)
        # 4 .. 4+n_keys-1: key tokens (A, B, C, ...)
        # 4+n_keys .. 4+n_keys+n_values-1: value tokens (0, 1, 2, ...)
        self.PAD = 0
        self.QUERY = 1
        self.EQUALS = 2
        self.LEVEL_SEP = 3
        self.key_offset = 4
        self.value_offset = 4 + n_keys
        self.vocab_size = 4 + n_keys + n_values

        if seed is not None:
            random.seed(seed)

    def key_token(self, i):
        return self.key_offset + i

    def value_token(self, v):
        return self.value_offset + v

    def generate_example(self):
        """Generate one pointer chasing example.

        Returns: (input_tokens, target_value_token, sequence_length)
        """
        keys = list(range(self.n_keys))

        # Level 0: base table. Each key maps to a random value.
        base_table = {}
        for k in keys:
            base_table[k] = random.randint(0, self.n_values - 1)

        # Levels 1..n_hops-1: each key maps to a key from the previous level
        levels = [base_table]
        for level in range(1, self.n_hops):
            table = {}
            for k in keys:
                table[k] = random.choice(keys)
            levels.append(table)

        # Pick a random query key at the top level
        query_key = random.choice(keys)

        # Resolve the chain
        current = query_key
        for level in range(self.n_hops - 1, -1, -1):
            current = levels[level][current]
        answer = current  # this is a value from base_table

        # Encode as token sequence
        tokens = []

        # Base table: A=5 B=3 ...
        for k in keys:
            tokens.append(self.key_token(k))
            tokens.append(self.EQUALS)
            tokens.append(self.value_token(base_table[k]))
        tokens.append(self.LEVEL_SEP)

        # Intermediate levels
        for level in range(1, self.n_hops):
            for k in keys:
                tokens.append(self.key_token(k))
                tokens.append(self.EQUALS)
                tokens.append(self.key_token(levels[level][k]))
            tokens.append(self.LEVEL_SEP)

        # Query
        tokens.append(self.QUERY)
        tokens.append(self.key_token(query_key))

        return tokens, self.value_token(answer), len(tokens)

    def generate_batch(self, batch_size):
        """Generate a batch of examples, padded to same length."""
        examples = [self.generate_example() for _ in range(batch_size)]
        max_len = max(e[2] for e in examples)

        input_seqs = []
        targets = []
        for tokens, answer, length in examples:
            padded = tokens + [self.PAD] * (max_len - length)
            input_seqs.append(padded)
            targets.append(answer)

        return (torch.tensor(input_seqs, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long),
                max_len)


class LookAheadD1Sequential(nn.Module):
    """D=1 look-ahead model trained with BPTT (sequential)."""

    def __init__(self, vocab_size, n_embed, block_size, n_head=4, dropout=0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.block = RoFormerBlock(n_embed, block_size, dropout, use_softmax=True, n_head=n_head)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.n_embed = n_embed

    def forward(self, idx):
        """Sequential processing: one position at a time."""
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding(idx)  # (B, T, C)
        h_prev = torch.zeros(B, 1, self.n_embed, device=device)

        all_h = []
        all_px = []

        for t in range(T):
            x_t = tok_emb[:, t:t+1, :]  # (B, 1, C)

            # Correction from previous hidden state
            corr = self.corr_ffn(self.ln_corr(h_prev + x_t))
            px_t = x_t + corr

            all_px.append(px_t)

            # Block sees all contextualized inputs up to t
            px_seq = torch.cat(all_px, dim=1)  # (B, t+1, C)
            h_seq = self.block(px_seq)
            h_t = h_seq[:, -1:, :]  # (B, 1, C) — last position's output

            all_h.append(h_t)
            h_prev = h_t

        # Output from final position
        h_final = all_h[-1]  # (B, 1, C)
        logits = self.head(self.ln_f(h_final)).squeeze(1)  # (B, vocab_size)
        return logits


class TransformerBaseline(nn.Module):
    """Standard N-layer transformer (parallel, no look-ahead)."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_head=4, dropout=0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax=True, n_head=n_head)
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
        # Output from final position
        logits = self.head(x[:, -1, :])  # (B, vocab_size)
        return logits


def train_and_eval(model, dataset, n_iters=5000, batch_size=64, lr=1e-3, device='cuda',
                   eval_every=500, eval_batches=20):
    """Train model and evaluate accuracy."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for it in range(1, n_iters + 1):
        model.train()
        inputs, targets, _ = dataset.generate_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % eval_every == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(eval_batches):
                    inputs, targets, _ = dataset.generate_batch(batch_size)
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            acc = correct / total
            print(f"  iter {it:5d}: loss={loss.item():.4f}, accuracy={acc:.4f}")

    return acc


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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    dataset = PointerChasingDataset(
        n_keys=args.n_keys, n_values=args.n_values, n_hops=args.n_hops
    )

    # Sequence length for block_size
    sample_tokens, _, seq_len = dataset.generate_example()
    block_size = seq_len + 10  # some padding
    print(f"Pointer chasing: {args.n_hops} hops, {args.n_keys} keys, {args.n_values} values")
    print(f"Sequence length: ~{seq_len}, vocab size: {dataset.vocab_size}")
    print(f"n_embed={args.n_embed}, n_head={args.n_head}, device={device}")
    print()

    # Test N=1 through N=n_hops+1 transformers
    for n_layers in range(1, args.n_hops + 2):
        print(f"=== Transformer N={n_layers} ===")
        model = TransformerBaseline(
            dataset.vocab_size, args.n_embed, n_layers, block_size,
            n_head=args.n_head
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        acc = train_and_eval(
            model, dataset, n_iters=args.n_iters, batch_size=args.batch_size,
            lr=args.lr, device=device
        )
        print(f"  Final accuracy: {acc:.4f}")
        print()

    # Test D=1 look-ahead with BPTT
    print(f"=== D=1 Look-Ahead (BPTT, sequential training) ===")
    model = LookAheadD1Sequential(
        dataset.vocab_size, args.n_embed, block_size, n_head=args.n_head
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    acc = train_and_eval(
        model, dataset, n_iters=args.n_iters, batch_size=args.batch_size,
        lr=args.lr, device=device
    )
    print(f"  Final accuracy: {acc:.4f}")
    print()


if __name__ == '__main__':
    main()

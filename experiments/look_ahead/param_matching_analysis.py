#!/usr/bin/env python3
"""
Parameter matching analysis: look_ahead_mlp vs roformer.

Given a look_ahead_mlp at a fixed n_embed, find all (n_embed, n_layers) combos
for roformer that yield the same total parameter count.

Key insight:
- look_ahead_mlp has 1 shared block + MLP head (Linear(C->4C)->GELU->Linear(4C->vocab))
- roformer has N separate blocks + linear head (Linear(C->vocab))
- Per block: 12*C^2 + 13*C params (attention Q/K/V/proj + FFN + 2 LayerNorms)
- MLP head: 4*C^2 + 4*C + 4*C*vocab + vocab (dominates at small C due to 4*C*vocab term)
- Linear head: C*vocab + vocab

At small n_embed, the MLP head (linear in C via 4*C*vocab) dominates, so
look_ahead_mlp is expensive. At large n_embed, blocks (quadratic in C) dominate,
so roformer with many layers is more expensive.

This means look_ahead_mlp at small n_embed can match roformer at larger n_embed
with fewer layers — the look-ahead model trades block depth for head capacity.

Usage:
    python param_matching_analysis.py --la_embed 200 --vocab_size 16000
    python param_matching_analysis.py --la_embed 768 --vocab_size 16000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'joformer'))
from models import MODEL_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Parameter matching analysis")
    parser.add_argument('--la_embed', type=int, default=200,
                        help='n_embed for look_ahead_mlp')
    parser.add_argument('--la_layers', type=int, default=10,
                        help='n_layers (iterations) for look_ahead_mlp')
    parser.add_argument('--vocab_size', type=int, default=16000)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Param match tolerance (default 5%%)')
    args = parser.parse_args()

    bs, d = args.block_size, args.dropout

    # Target: look_ahead_mlp
    m = MODEL_CLASSES['roformer_look_ahead_mlp'](
        args.vocab_size, args.la_embed, args.la_layers, bs, d)
    target = sum(p.numel() for p in m.parameters())
    print(f"Target: look_ahead_mlp n_embed={args.la_embed} N={args.la_layers}: {target:,} params")
    print()

    # Breakdown
    C = args.la_embed
    V = args.vocab_size
    block = 12 * C * C + 13 * C
    embed = C * V
    mlp_head = 4 * C * C + 4 * C + 4 * C * V + V
    print(f"  1 block:   {block:>12,}  (12*C^2 + 13*C)")
    print(f"  embed:     {embed:>12,}  (C * vocab)")
    print(f"  MLP head:  {mlp_head:>12,}  (4*C^2 + 4*C + 4*C*vocab + vocab)")
    print()

    # Grid
    n_layers_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    embed_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 768]

    print(f"{'':>8}", end='')
    for N in n_layers_list:
        print(f"{'N='+str(N):>10}", end='')
    print()

    matches = []
    for c in embed_list:
        print(f"c={c:<4}", end='')
        for N in n_layers_list:
            m = MODEL_CLASSES['roformer'](args.vocab_size, c, N, bs, d)
            p = sum(p.numel() for p in m.parameters())
            ratio = p / target
            if 1 - args.tolerance <= ratio <= 1 + args.tolerance:
                print(f"  *{p/1e6:>6.1f}M", end='')
                matches.append((c, N, p))
            else:
                print(f"   {p/1e6:>6.1f}M", end='')
        print()

    if matches:
        print(f"\nMatches within {args.tolerance*100:.0f}% of target ({target:,}):")
        for c, N, p in matches:
            diff_pct = (p - target) / target * 100
            print(f"  roformer n_embed={c}, N={N}: {p:,} ({diff_pct:+.1f}%)")


if __name__ == '__main__':
    main()

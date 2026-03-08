#!/usr/bin/env python3
"""
Find roformer (C, N) configurations that match the parameter count
of roformer_look_ahead_nocat at a given embedding size.

roformer_look_ahead_nocat (shared weights, linear head):
  params = C*V + (12*C^2 + 13*C) + (C*V + V)

roformer (separate weights, linear head):
  params = C*V + N*(12*C^2 + 13*C) + (C*V + V)

Usage:
    python param_matching_roformer.py --nocat_embed 768 --vocab_size 16000
"""

import math
import argparse


def main():
    parser = argparse.ArgumentParser(description="Find roformer configs matching nocat params")
    parser.add_argument('--nocat_embed', type=int, default=768)
    parser.add_argument('--vocab_size', type=int, default=16000)
    parser.add_argument('--tolerance', type=float, default=0.10,
                        help='Match tolerance (default 10%%)')
    args = parser.parse_args()

    V = args.vocab_size
    C_nocat = args.nocat_embed

    nocat_total = C_nocat * V + (12 * C_nocat**2 + 13 * C_nocat) + (C_nocat * V + V)
    print(f"roformer_look_ahead_nocat (C={C_nocat}): {nocat_total:,} params")
    print(f"  embed:   {C_nocat * V:>12,}")
    print(f"  1 block: {12 * C_nocat**2 + 13 * C_nocat:>12,}")
    print(f"  head:    {C_nocat * V + V:>12,}")
    print()

    # Same C: solve for N
    per_block = 12 * C_nocat**2 + 13 * C_nocat
    blocks_budget = nocat_total - 2 * C_nocat * V - V
    N_exact = blocks_budget / per_block
    print(f"Same C={C_nocat}: N = {N_exact:.2f}")
    print()

    # Grid search
    print(f"Roformer configs matching {nocat_total:,} params (within {args.tolerance*100:.0f}%):")
    print(f"{'C':>6} {'N':>4} {'params':>12} {'diff%':>8}")
    for C in [100, 150, 200, 256, 300, 384, 400, 500, 512, 600, 768]:
        overhead = 2 * C * V + V  # embed + head
        per_block = 12 * C**2 + 13 * C
        if nocat_total <= overhead:
            continue
        N_exact = (nocat_total - overhead) / per_block
        for N in [math.floor(N_exact), math.ceil(N_exact)]:
            if N < 1:
                continue
            total = overhead + N * per_block
            diff = (total - nocat_total) / nocat_total * 100
            if abs(diff) <= args.tolerance * 100:
                print(f"{C:>6} {N:>4} {total:>12,} {diff:>+7.1f}%")


if __name__ == '__main__':
    main()

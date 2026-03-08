#!/usr/bin/env python3
"""
Compute the embedding size needed for roformer_look_ahead_nocat to match
the parameter count of a standard roformer.

roformer (separate weights):
  params = C*V + N*(12*C^2 + 13*C) + (C*V + V)

roformer_look_ahead_nocat (shared weights, linear head):
  params = C*V + (12*C^2 + 13*C) + (C*V + V)
         = 12*C^2 + (2*V + 13)*C + V

Solve for C given a target param count.
"""

import math
import argparse


def roformer_params(C, V, N):
    embed = C * V
    blocks = N * (12 * C**2 + 13 * C)
    head = C * V + V
    return embed + blocks + head


def nocat_params(C, V):
    embed = C * V
    block = 12 * C**2 + 13 * C
    head = C * V + V
    return embed + block + head


def solve_nocat_embed(target, V):
    """Solve 12*C^2 + (2*V+13)*C + V = target for C."""
    a = 12
    b = 2 * V + 13
    c = V - target
    disc = b**2 - 4 * a * c
    return (-b + math.sqrt(disc)) / (2 * a)


def main():
    parser = argparse.ArgumentParser(description="Parameter matching for nocat model")
    parser.add_argument('--roformer_embed', type=int, default=768)
    parser.add_argument('--roformer_layers', type=int, default=10)
    parser.add_argument('--vocab_size', type=int, default=16000)
    args = parser.parse_args()

    V = args.vocab_size
    C_roformer = args.roformer_embed
    N_roformer = args.roformer_layers

    target = roformer_params(C_roformer, V, N_roformer)
    print(f"Roformer (C={C_roformer}, N={N_roformer}): {target:,} params")
    print(f"  embed:     {C_roformer * V:>12,}")
    print(f"  {N_roformer} blocks: {N_roformer * (12 * C_roformer**2 + 13 * C_roformer):>12,}")
    print(f"  head:      {C_roformer * V + V:>12,}")
    print()

    C_nocat = solve_nocat_embed(target, V)
    C = round(C_nocat)
    actual = nocat_params(C, V)

    print(f"roformer_look_ahead_nocat needs C = {C_nocat:.1f} (rounded: {C})")
    print(f"  Verification: {actual:,} params")
    print(f"  embed:   {C * V:>12,}")
    print(f"  1 block: {12 * C**2 + 13 * C:>12,}")
    print(f"  head:    {C * V + V:>12,}")
    print(f"  diff:    {actual - target:+,} ({(actual - target) / target * 100:+.2f}%)")


if __name__ == '__main__':
    main()

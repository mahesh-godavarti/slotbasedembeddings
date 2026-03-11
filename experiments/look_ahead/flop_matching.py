#!/usr/bin/env python3
"""Compute FLOP-matched C for look-ahead models vs roformer baselines.

FLOPs per component per token (dominant linear projection cost):
  Block (attn + FFN):     12C²  (QKV=3C², out=C², FFN=8C²)
  Extra FFN:               8C²  (up=4C², down=4C²)

Model inference FLOPs (all block_head variants use shared weights across K iterations):
  roformer N:              N * 12C²
  roformer_head_ffn N:     N * 12C² + 8C²  = (12N + 8) * C²
  block_head:              12C²              (single shared block, aka deep block_head D=1)
  block_head_corr_ffn:     12C² + 8C²       = 20C²
  block_head_ffn:          12C² + 8C²       = 20C²  (same as corr_ffn)

For deep block_head (D>1): D distinct blocks per iteration step.
  deep block_head D:       D * 12C²
  deep block_head_corr_ffn D: D * 12C² + 8C²

Usage:
  python flop_matching.py --target roformer --target_c 50 --target_n 3 --model block_head
  python flop_matching.py --target roformer --target_c 50 --target_n 3 --model block_head_corr_ffn
  python flop_matching.py --target roformer_head_ffn --target_c 50 --target_n 3 --model block_head_corr_ffn
"""

import argparse
import math

FLOP_COEFFICIENTS = {
    'roformer': lambda n: 12 * n,
    'roformer_head_ffn': lambda n: 12 * n + 8,
    'block_head': lambda _: 12,
    'block_head_corr_ffn': lambda _: 20,
    'block_head_ffn': lambda _: 20,
}


def main():
    parser = argparse.ArgumentParser(description="FLOP-matched C calculator")
    parser.add_argument('--target', type=str, required=True,
                        choices=list(FLOP_COEFFICIENTS.keys()),
                        help='Target model to match FLOPs against')
    parser.add_argument('--target_c', type=int, required=True, help='Target embedding dim')
    parser.add_argument('--target_n', type=int, default=1, help='Target number of layers')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(FLOP_COEFFICIENTS.keys()),
                        help='Model to compute FLOP-matched C for')
    parser.add_argument('--model_n', type=int, default=1, help='Model number of layers (for roformer variants)')
    args = parser.parse_args()

    target_coeff = FLOP_COEFFICIENTS[args.target](args.target_n)
    model_coeff = FLOP_COEFFICIENTS[args.model](args.model_n)

    target_flops = target_coeff * args.target_c ** 2
    c_exact = args.target_c * math.sqrt(target_coeff / model_coeff)
    c_even = round(c_exact)
    if c_even % 2 != 0:
        c_even += 1
    model_flops = model_coeff * c_even ** 2

    print(f"{args.target} N={args.target_n} C={args.target_c}: {target_coeff}C² = {target_flops:,} multiplies/token")
    print(f"{args.model} N={args.model_n} FLOP-matched C={c_exact:.1f} (exact)")
    print(f"Rounded to even: C={c_even}")
    print(f"{args.model} C={c_even}: {model_coeff}C² = {model_flops:,} multiplies/token")
    print(f"FLOP ratio: {model_flops / target_flops:.4f}")

    # Param comparison (vocab=16000)
    vocab = 16000
    target_params = (2 * vocab * args.target_c
                     + args.target_n * (12 * args.target_c**2 + 13 * args.target_c)
                     + 2 * args.target_c)
    if args.target == 'roformer_head_ffn':
        target_params += 8 * args.target_c**2 + 5 * args.target_c  # head FFN + LN

    model_block_params = 12 * c_even**2 + 13 * c_even
    model_params = 2 * vocab * c_even + args.model_n * model_block_params + 2 * c_even
    if args.model in ('block_head_corr_ffn', 'block_head_ffn'):
        model_params += 8 * c_even**2 + 5 * c_even  # extra FFN + LN

    print(f"\nParam comparison (vocab={vocab}):")
    print(f"  {args.target} N={args.target_n} C={args.target_c}: {target_params:,} params")
    print(f"  {args.model} C={c_even}: {model_params:,} params")
    print(f"  {args.model} has {model_params - target_params:+,} params ({model_params/target_params:.2f}x)")


if __name__ == '__main__':
    main()

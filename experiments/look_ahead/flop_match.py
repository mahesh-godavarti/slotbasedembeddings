#!/usr/bin/env python3
"""Calculate FLOP-matched C values for concat v2 variants vs roformer baselines."""
import math

# FLOPs per token at sequential K=1:
# roformer N=3:          3 × 12C² = 36C²
# roformer_head_ffn N=3: 3 × 12C² + 8C² = 44C²
# D=1 concat v2:         12C² (block) + 12C² (corr_ffn 2C→4C→C) = 24C²
# D=3 concat v2:         3 × 12C² (blocks) + 12C² (corr_ffn) = 48C²
# Stacked N=3 concat v2: 3 × (12C² + 12C²) = 72C²

VARIANTS = {
    'D=1 concat v2': 24,
    'D=3 concat v2': 48,
    'Stacked N=3 concat v2': 72,
}

BASELINES = {
    'roformer N=3 C=50': (36, 50),
    'roformer_head_ffn N=3 C=50': (44, 50),
}

for bname, (b_coeff, b_c) in BASELINES.items():
    target = b_coeff * b_c**2
    print(f'=== FLOP match vs {bname} ({b_coeff}C²={target:,} FLOPs) ===')
    for vname, v_coeff in VARIANTS.items():
        c = math.sqrt(target / v_coeff)
        c_int = round(c)
        flops = v_coeff * c_int**2
        print(f'  {vname}: C={c_int}, FLOPs={flops:,} (target {target:,})')
    print()

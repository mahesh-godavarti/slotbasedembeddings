#!/usr/bin/env python3
"""Calculate param-matched D=1 concat v2 C for roformer baselines."""
import sys
sys.path.insert(0, '/home/ubuntu/look_ahead5')
from models import MODEL_CLASSES

for N in [3, 6]:
    m = MODEL_CLASSES['roformer'](vocab_size=16000, n_embed=100, n_layers=N, block_size=256, dropout=0.0, use_softmax=True)
    p = sum(p.numel() for p in m.parameters())

    best_c, best_diff, best_p = None, float('inf'), 0
    for c in range(50, 400):
        m2 = MODEL_CLASSES['block_head_corr_ffn_concat'](vocab_size=16000, n_embed=c, n_layers=5, block_size=256, dropout=0.0, use_softmax=True)
        p2 = sum(pp.numel() for pp in m2.parameters())
        if abs(p2 - p) < best_diff:
            best_diff = abs(p2 - p)
            best_c = c
            best_p = p2

    print(f"roformer N={N} C=100: {p:,} params, {N*12*100**2:,} FLOPs")
    print(f"D=1 concat v2 C={best_c}: {best_p:,} params, {24*best_c**2:,} FLOPs")
    print(f"  param diff: {best_p - p:+,}, FLOP ratio: {24*best_c**2 / (N*12*100**2):.2f}x")
    print()

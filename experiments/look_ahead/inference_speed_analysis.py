#!/usr/bin/env python3
"""
Inference speed analysis: roformer_look_ahead_nocat (K=1) vs roformer.

Compares param-matched pairs:
  1. nocat C=1786 K=1  vs  roformer C=768 N=10   (~95.5M params each)
  2. nocat C=768  K=1  vs  roformer C=384 N=11   (~31.8M params each)

Theoretical analysis + empirical benchmark on the current machine.

Usage:
    python inference_speed_analysis.py                    # theoretical only
    python inference_speed_analysis.py --benchmark        # theoretical + GPU benchmark
    python inference_speed_analysis.py --benchmark --gen_tokens 100
"""

import argparse
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'joformer'))


COMPARISONS = [
    {
        'name': 'Large (~95.5M params)',
        'nocat_embed': 1786,
        'roformer_embed': 768,
        'roformer_layers': 10,
    },
    {
        'name': 'Small (~31.8M params)',
        'nocat_embed': 768,
        'roformer_embed': 384,
        'roformer_layers': 11,
    },
]


def theoretical_analysis(C_nocat, C_roformer, N_roformer, V, T, label=''):
    """Compute FLOPs and sequential depth for both models."""

    print("=" * 70)
    print(f"THEORETICAL ANALYSIS{': ' + label if label else ''}")
    print("=" * 70)

    # Parameter counts
    nocat_block = 12 * C_nocat**2 + 13 * C_nocat
    nocat_embed = C_nocat * V
    nocat_head = C_nocat * V + V
    nocat_total = nocat_embed + nocat_block + nocat_head

    roformer_block = 12 * C_roformer**2 + 13 * C_roformer
    roformer_embed = C_roformer * V
    roformer_head = C_roformer * V + V
    roformer_total = roformer_embed + N_roformer * roformer_block + roformer_head

    print(f"\nroformer_look_ahead_nocat (C={C_nocat}, K=1): {nocat_total:,} params")
    print(f"  embed:   {nocat_embed:>12,}")
    print(f"  1 block: {nocat_block:>12,}")
    print(f"  head:    {nocat_head:>12,}")

    print(f"\nroformer (C={C_roformer}, N={N_roformer}): {roformer_total:,} params")
    print(f"  embed:       {roformer_embed:>12,}")
    print(f"  {N_roformer} blocks:  {N_roformer * roformer_block:>12,}")
    print(f"  head:        {roformer_head:>12,}")

    print(f"\n  Param ratio: {nocat_total / roformer_total:.4f}")

    # FLOPs per token (autoregressive, at sequence position T)
    # Per block: QKV projections (3*C*C) + attn output proj (C*C) + FFN (2*4*C*C) = 12*C^2
    #            + attention scores (C*T) + attention weighted sum (C*T)
    #            + layernorms, biases, etc. (small)
    # Head: C * V

    print(f"\n--- FLOPs per generated token (at seq position T={T}) ---")

    nocat_block_flops = 12 * C_nocat**2 + 2 * C_nocat * T
    nocat_head_flops = C_nocat * V
    nocat_total_flops = nocat_block_flops + nocat_head_flops

    roformer_block_flops = N_roformer * (12 * C_roformer**2 + 2 * C_roformer * T)
    roformer_head_flops = C_roformer * V
    roformer_total_flops = roformer_block_flops + roformer_head_flops

    print(f"\nroformer_look_ahead_nocat K=1:")
    print(f"  1 block: {nocat_block_flops:>12,}  (12*C^2 + 2*C*T)")
    print(f"  head:    {nocat_head_flops:>12,}  (C*V)")
    print(f"  total:   {nocat_total_flops:>12,}")

    print(f"\nroformer N={N_roformer}:")
    print(f"  {N_roformer} blocks: {roformer_block_flops:>12,}  (N*(12*C^2 + 2*C*T))")
    print(f"  head:    {roformer_head_flops:>12,}  (C*V)")
    print(f"  total:   {roformer_total_flops:>12,}")

    flop_ratio = roformer_total_flops / nocat_total_flops
    print(f"\n  FLOP ratio (roformer / nocat_K1): {flop_ratio:.2f}x")

    # Sequential depth
    ops_per_block = 8

    print(f"\n--- Sequential depth ---")
    print(f"  roformer:   {N_roformer} layers x {ops_per_block} ops = {N_roformer * ops_per_block} serial ops")
    print(f"  nocat K=1:  1 layer  x {ops_per_block} ops = {ops_per_block} serial ops")
    print(f"  Depth ratio: {N_roformer}x fewer serial operations for K=1")

    # Memory bandwidth (weight reads)
    bytes_per_param = 4  # float32
    nocat_bytes = nocat_total * bytes_per_param
    roformer_bytes = roformer_total * bytes_per_param

    print(f"\n--- Memory bandwidth (weight reads per token, float32) ---")
    print(f"  roformer:  {roformer_bytes / 1e6:.1f} MB")
    print(f"  nocat K=1: {nocat_bytes / 1e6:.1f} MB")
    print(f"  Ratio: {roformer_bytes / nocat_bytes:.2f}x")

    # KV cache
    nocat_kv = 2 * T * C_nocat * bytes_per_param  # 1 layer
    roformer_kv = N_roformer * 2 * T * C_roformer * bytes_per_param

    print(f"\n--- KV cache size (at T={T}) ---")
    print(f"  roformer:  {roformer_kv / 1024:.1f} KB  ({N_roformer} layers x 2 x T x C)")
    print(f"  nocat K=1: {nocat_kv / 1024:.1f} KB  (1 layer x 2 x T x C)")
    print(f"  Ratio: {roformer_kv / nocat_kv:.1f}x")

    print(f"\n--- Summary ---")
    print(f"  FLOPs:            {flop_ratio:.2f}x fewer for K=1")
    print(f"  Sequential depth: {N_roformer}x fewer serial ops for K=1")
    print(f"  Weight reads:     {roformer_bytes / nocat_bytes:.2f}x")
    print(f"  KV cache:         {roformer_kv / nocat_kv:.1f}x smaller for K=1")
    print(f"\n  Expected speedup (autoregressive, batch=1):")
    print(f"    - Compute-bound (large batch): ~{flop_ratio:.1f}x")
    print(f"    - Latency-bound (batch=1): up to ~{N_roformer}x")

    return nocat_total, roformer_total


def benchmark_one_pair(C_nocat, C_roformer, N_roformer, V, block_size,
                       gen_tokens, label='', n_warmup=5, n_trials=3):
    """Empirical benchmark for one param-matched pair."""
    import torch
    from models import MODEL_CLASSES

    print(f"\n{'─' * 70}")
    print(f"BENCHMARK{': ' + label if label else ''}")
    print(f"{'─' * 70}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout = 0.0

    # Create models
    model_nocat = MODEL_CLASSES['roformer_look_ahead_nocat'](
        V, C_nocat, 10, block_size, dropout).to(device)
    model_nocat.eval()
    nocat_params = sum(p.numel() for p in model_nocat.parameters())
    print(f"  nocat (C={C_nocat}): {nocat_params:,} params")

    model_roformer = MODEL_CLASSES['roformer'](
        V, C_roformer, N_roformer, block_size, dropout).to(device)
    model_roformer.eval()
    roformer_params = sum(p.numel() for p in model_roformer.parameters())
    print(f"  roformer (C={C_roformer}, N={N_roformer}): {roformer_params:,} params")

    prompt = torch.randint(0, V, (1, 10), device=device)

    def generate_with_depth(model, idx, n_tokens, n_iters):
        """Autoregressive generation with explicit iteration depth."""
        for _ in range(n_tokens):
            idx_crop = idx[:, -block_size:]
            tok_emb = model._get_embeddings(idx_crop)
            processed_x, correction, _ = model._run_iterations(tok_emb, n_iters)
            output = model._build_output(processed_x, correction)
            logits = model._classify(output)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def generate_standard(model, idx, n_tokens):
        """Autoregressive generation for standard (non-look-ahead) models."""
        for _ in range(n_tokens):
            idx_crop = idx[:, -block_size:]
            logits, _ = model(idx_crop)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def bench(model, depth=None):
        is_look_ahead = hasattr(model, '_run_iterations')

        for _ in range(n_warmup):
            with torch.no_grad():
                if is_look_ahead and depth is not None:
                    generate_with_depth(model, prompt, gen_tokens, depth)
                else:
                    generate_standard(model, prompt, gen_tokens)

        if device == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(n_trials):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                if is_look_ahead and depth is not None:
                    generate_with_depth(model, prompt, gen_tokens, depth)
                else:
                    generate_standard(model, prompt, gen_tokens)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        return avg, gen_tokens / avg, avg / gen_tokens * 1000

    print(f"\n  Generating {gen_tokens} tokens ({n_warmup} warmup, {n_trials} trials)...")

    t1, tps1, ms1 = bench(model_nocat, depth=1)
    t10, tps10, ms10 = bench(model_nocat, depth=10)
    tr, tpsr, msr = bench(model_roformer)

    print(f"\n  {'Model':<40} {'Total(s)':>9} {'tok/s':>8} {'ms/tok':>8}")
    print(f"  {'-'*68}")
    print(f"  {'nocat C=%d K=1' % C_nocat:<40} {t1:>9.3f} {tps1:>8.1f} {ms1:>8.2f}")
    print(f"  {'nocat C=%d K=10' % C_nocat:<40} {t10:>9.3f} {tps10:>8.1f} {ms10:>8.2f}")
    print(f"  {'roformer C=%d N=%d' % (C_roformer, N_roformer):<40} {tr:>9.3f} {tpsr:>8.1f} {msr:>8.2f}")
    print(f"\n  Speedup K=1 vs roformer: {tr / t1:.2f}x")
    print(f"  Speedup K=1 vs K=10:     {t10 / t1:.2f}x")

    del model_nocat, model_roformer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {'nocat_K1_ms': ms1, 'nocat_K10_ms': ms10, 'roformer_ms': msr,
            'speedup_vs_roformer': tr / t1, 'speedup_vs_K10': t10 / t1}


def main():
    parser = argparse.ArgumentParser(description="Inference speed analysis")
    parser.add_argument('--vocab_size', type=int, default=16000)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=64,
                        help='Sequence length for FLOP calculation')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run empirical GPU benchmark')
    parser.add_argument('--gen_tokens', type=int, default=50,
                        help='Number of tokens to generate in benchmark')
    args = parser.parse_args()

    V = args.vocab_size
    T = args.seq_len

    all_results = []
    for comp in COMPARISONS:
        print(f"\n\n{'#' * 70}")
        print(f"# {comp['name']}")
        print(f"{'#' * 70}\n")

        theoretical_analysis(
            comp['nocat_embed'], comp['roformer_embed'],
            comp['roformer_layers'], V, T, label=comp['name'])

        if args.benchmark:
            res = benchmark_one_pair(
                comp['nocat_embed'], comp['roformer_embed'],
                comp['roformer_layers'], V, args.block_size,
                args.gen_tokens, label=comp['name'])
            all_results.append((comp['name'], res))

    if all_results:
        import torch
        print(f"\n\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        print(f"Generated {args.gen_tokens} tokens per trial\n")
        print(f"{'Comparison':<25} {'nocat K=1':>10} {'roformer':>10} {'Speedup':>10}")
        print(f"{'':<25} {'(ms/tok)':>10} {'(ms/tok)':>10} {'':>10}")
        print("-" * 58)
        for name, res in all_results:
            print(f"{name:<25} {res['nocat_K1_ms']:>10.2f} {res['roformer_ms']:>10.2f} {res['speedup_vs_roformer']:>9.2f}x")


if __name__ == '__main__':
    main()

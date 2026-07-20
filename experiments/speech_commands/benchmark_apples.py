#!/usr/bin/env python3
"""Apples-to-apples: cumsum+window vs scan+lambda.

Same mel front-end, same projection, same BN, same GLU, same frequencies,
same number of layers, same tied weights. The ONLY difference is the
sequence operation:

  cumsum:  d[t] = sum(rotated[t-W:t])      (hard window W, FIR)
  scan:    d[t] = λ·d[t-1] + rotated[t]    (exponential decay λ, IIR)

cumsum uses torch.cumsum (single CUDA kernel, parallel prefix sum).
scan uses our Triton fused kernel (sequential over time, but fused).
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(__file__))
from speech_commands import (
    SpeechCommandsDataset, load_noise_wavs,
    MelCumsumFixed, MelScanFixed,
    train_one_epoch, evaluate,
    NUM_CLASSES, SAMPLE_RATE,
)


def train_and_time(model, name, train_loader, val_loader, device, epochs, lr=1e-3):
    """Train model, return (model, best_val_acc, wall_seconds)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    best_state = None
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_secs = time.perf_counter() - t_ep
        print(f"  [{name}] Epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:.4f}  val={val_acc:.4f}  {epoch_secs:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wall_secs = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc, wall_secs


def benchmark_inference(model, test_loader, device, n_warmup=50, n_iter=200):
    """Single-sample inference latency."""
    model.eval()
    # Get a test waveform
    waveform, _ = next(iter(test_loader))
    x = waveform[:1].to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(n_iter):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return sorted(times)[len(times) // 2]  # median ms


def benchmark_primitive(device, n_warmup=50, n_iter=200):
    """Benchmark torch.cumsum vs Triton scan at matched sizes."""
    from speech_commands import _triton_scan_complex, _HAS_TRITON
    if not _HAS_TRITON:
        print("  Triton not available, skipping primitive benchmark")
        return

    # T=200 matches mel frame count at hop=80
    configs = [(1, 200, 40), (8, 200, 40), (32, 200, 40), (128, 200, 40)]

    print(f"\n  {'Batch':>5}  {'T':>4}  {'N':>4}  {'cumsum':>10}  {'triton_scan':>12}  {'ratio':>8}")
    print(f"  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*8}")

    for B, T, N in configs:
        x = torch.randn(B, T, N, device=device)
        gates = torch.complex(
            torch.full((B, T, N), 0.9, device=device),
            torch.zeros(B, T, N, device=device))
        values = torch.complex(
            torch.randn(B, T, N, device=device),
            torch.randn(B, T, N, device=device))

        for _ in range(n_warmup):
            torch.cumsum(x, dim=1)
            _triton_scan_complex(gates, values)
        torch.cuda.synchronize()

        times_cs = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cumsum(x, dim=1)
            torch.cuda.synchronize()
            times_cs.append((time.perf_counter() - t0) * 1e6)

        times_ts = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _triton_scan_complex(gates, values)
            torch.cuda.synchronize()
            times_ts.append((time.perf_counter() - t0) * 1e6)

        cs_us = sorted(times_cs)[len(times_cs) // 2]
        ts_us = sorted(times_ts)[len(times_ts) // 2]
        ratio = ts_us / cs_us
        print(f"  {B:>5}  {T:>4}  {N:>4}  {cs_us:>8.1f}us  {ts_us:>10.1f}us  {ratio:>6.1f}x")


def main():
    parser = argparse.ArgumentParser(description='Apples-to-apples: cumsum vs scan')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--n_warmup', type=int, default=50)
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    # Architecture (same for both models)
    parser.add_argument('--n_embed', type=int, default=80)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--hop_length', type=int, default=80)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--tie_layers', action='store_true', default=True)
    parser.add_argument('--no_tie_layers', dest='tie_layers', action='store_false')
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2
        args.n_warmup = 5
        args.n_iter = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ──
    print("\nLoading data...")
    noise_wavs = load_noise_wavs(args.data_dir)
    train_ds = SpeechCommandsDataset(args.data_dir, 'training',
                                      augment=True, noise_wavs=noise_wavs)
    val_ds = SpeechCommandsDataset(args.data_dir, 'validation', noise_wavs=noise_wavs)
    test_ds = SpeechCommandsDataset(args.data_dir, 'testing', noise_wavs=noise_wavs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Models ──
    print(f"\nArchitecture: n_embed={args.n_embed}, n_layers={args.n_layers}, "
          f"hop={args.hop_length}, tie_layers={args.tie_layers}")

    cumsum_model = MelCumsumFixed(
        n_embed=args.n_embed, n_layers=args.n_layers,
        window=args.window, hop_length=args.hop_length,
        tie_layers=args.tie_layers,
    ).to(device)

    scan_model = MelScanFixed(
        n_embed=args.n_embed, n_layers=args.n_layers,
        hop_length=args.hop_length,
        tie_layers=args.tie_layers,
    ).to(device)

    print(f"\n  MelCumsumFixed:  {cumsum_model.param_count():>7,} params  (window={args.window})")
    print(f"  MelScanFixed:    {scan_model.param_count():>7,} params  (learned decay)")
    print(f"  Difference:      {scan_model.param_count() - cumsum_model.param_count():>+7,} params "
          f"({args.n_layers} layers × {args.n_embed // 2} decay scalars)")

    # ── Training ──
    print(f"\n{'=' * 60}")
    print(f"Training ({args.epochs} epochs, bs={args.batch_size}, lr={args.lr})")
    print(f"{'=' * 60}")

    print(f"\nMelCumsumFixed (cumsum + window={args.window}):")
    cumsum_model, cumsum_acc, cumsum_secs = train_and_time(
        cumsum_model, 'Cumsum', train_loader, val_loader, device, args.epochs, args.lr)

    print(f"\nMelScanFixed (scan + learned decay):")
    scan_model, scan_acc, scan_secs = train_and_time(
        scan_model, 'Scan', train_loader, val_loader, device, args.epochs, args.lr)

    # ── Test accuracy ──
    print(f"\n{'=' * 60}")
    print("Test Accuracy")
    print(f"{'=' * 60}")
    cumsum_model.eval()
    scan_model.eval()
    cumsum_test_acc, _, _ = evaluate(cumsum_model, test_loader, device)
    scan_test_acc, _, _ = evaluate(scan_model, test_loader, device)
    print(f"  MelCumsumFixed:  {cumsum_test_acc*100:.2f}%")
    print(f"  MelScanFixed:    {scan_test_acc*100:.2f}%")

    # ── Training summary ──
    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<20} {'Params':>7}  {'Wall Time':>10}  {'Per Epoch':>10}  {'Val Acc':>8}  {'Test Acc':>8}")
    print(f"  {'-'*20} {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    print(f"  {'MelCumsumFixed':<20} {cumsum_model.param_count():>7,}  {cumsum_secs:>8.1f}s  "
          f"{cumsum_secs/args.epochs:>8.1f}s  {cumsum_acc*100:>6.1f}%  {cumsum_test_acc*100:>6.1f}%")
    print(f"  {'MelScanFixed':<20} {scan_model.param_count():>7,}  {scan_secs:>8.1f}s  "
          f"{scan_secs/args.epochs:>8.1f}s  {scan_acc*100:>6.1f}%  {scan_test_acc*100:>6.1f}%")
    print(f"\n  Cumsum speedup: {scan_secs/cumsum_secs:.2f}x")

    # ── Inference ──
    print(f"\n{'=' * 60}")
    print("Inference Latency (single sample)")
    print(f"{'=' * 60}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    lat_cumsum = benchmark_inference(cumsum_model, test_loader, device, args.n_warmup, args.n_iter)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    lat_scan = benchmark_inference(scan_model, test_loader, device, args.n_warmup, args.n_iter)

    print(f"\n  MelCumsumFixed:  {lat_cumsum:.2f} ms  ({1000/lat_cumsum:.0f} seq/s)")
    print(f"  MelScanFixed:    {lat_scan:.2f} ms  ({1000/lat_scan:.0f} seq/s)")
    print(f"  Cumsum speedup:  {lat_scan/lat_cumsum:.2f}x")

    # ── Primitive operation comparison ──
    print(f"\n{'=' * 60}")
    print("Primitive: torch.cumsum vs Triton scan (T=200, N=40 = mel config)")
    print(f"{'=' * 60}")
    benchmark_primitive(device, args.n_warmup, args.n_iter)

    # ── Learned decay values ──
    print(f"\n{'=' * 60}")
    print("Learned Decay (sigmoid(param)) per Layer")
    print(f"{'=' * 60}")
    for i, dp in enumerate(scan_model.decay_params):
        d = torch.sigmoid(dp).detach().cpu()
        eff_window = 1.0 / (1.0 - d + 1e-8)
        print(f"  Layer {i}: decay mean={d.mean():.3f}  "
              f"range=[{d.min():.3f}, {d.max():.3f}]  "
              f"effective window=[{eff_window.min():.0f}, {eff_window.max():.0f}]")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Key Takeaway")
    print(f"{'=' * 60}")
    print(f"  Same mel front-end, same layers, same params (~{cumsum_model.param_count():,}).")
    print(f"  Only difference: cumsum+window vs scan+lambda.")
    print(f"  cumsum = torch.cumsum (parallel prefix sum, single CUDA kernel)")
    print(f"  scan   = Triton fused sequential scan (h[t] = λ·h[t-1] + x[t])")
    print(f"  Training speedup:  {scan_secs/cumsum_secs:.2f}x")
    print(f"  Inference speedup: {lat_scan/lat_cumsum:.2f}x")
    print(f"  Accuracy: cumsum {cumsum_test_acc*100:.1f}% vs scan {scan_test_acc*100:.1f}%")
    print()


if __name__ == '__main__':
    main()

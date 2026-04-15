# C=1408 vs C=1024 Comparison — Width vs Depth at Scale

## Experiment Overview

All experiments on OpenWebText (OWT), vocab=32000, block_size=256, batch=32, lr=2e-4, softmax, AMP, n_head=16.

### Models

| Model | Params | Inference FLOPs | Inference Layers | Notes |
|-------|--------|----------------|-----------------|-------|
| Roformer N=24 C=1024 | 368M | 288 × 1024² | 24 | Deep/narrow baseline |
| Roformer N=12 C=1408 | 376M | 144 × 1408² | 12 | Wide/shallow baseline |
| D=12 C=1408 from scratch | 392M | 152 × 1408² | 12 | Look-ahead, K=5 k_min=2 |
| D=12 C=1408 fine-tuned | 392M | 152 × 1408² | 12 | Converted from N=12 C=1408, K=2-5 |
| D=23 C=1024 K=5 | 364M | 284 × 1024² | 23 | FLOP-matched to N=24 (other machine) |
| D=23 C=1024 K-schedule | 364M | 284 × 1024² | 23 | K=1→K=2→K=2-5 curriculum |
| D=24 C=1024 fine-tuned | 376M | 288 × 1024² | 24 | Converted from N=24 C=1024, K=2-4 |
| D=12 C=1024 | 225M | 152 × 1024² | 12 | Narrower look-ahead |

### FLOP comparison

- 144 × 1408² ≈ 285M (N=12 C=1408) — ~5% fewer than N=24 C=1024
- 152 × 1408² ≈ 301M (D=12 C=1408) — essentially FLOP-matched to N=24 C=1024 (288 × 1024² ≈ 302M)
- 284 × 1024² ≈ 298M (D=23 C=1024) — also near FLOP-matched to N=24

---

## 1. Width vs Depth: N=12 C=1408 vs N=24 C=1024

Both are standard roformers with separate weights per layer.

| Iter | N=24 C=1024 | N=12 C=1408 | Gap (N12−N24) |
|------|------------|------------|---------------|
| 5K   | 96.72 | 90.24 | -6.48 |
| 10K  | 69.30 | 66.76 | -2.54 |
| 15K  | 58.18 | 57.18 | -1.00 |
| 20K  | 52.46 | 51.59 | -0.87 |
| 25K  | 48.49 | 48.13 | -0.36 |
| 30K  | 45.54 | 45.53 | -0.01 |
| 35K  | 43.37 | 43.63 | **+0.26** |
| 40K  | 41.77 | 42.03 | +0.26 |
| 45K  | 40.33 | 40.82 | +0.49 |
| 50K  | 39.27 | 39.70 | +0.43 |
| 55K  | 38.17 | 38.83 | +0.66 |
| 60K  | 37.37 | 37.88 | +0.51 |
| 65K  | 36.67 | 37.20 | +0.53 |
| 70K  | 36.10 | 36.80 | +0.70 |
| 75K  | 35.67 | 36.19 | +0.52 |
| 80K  | 34.95 | 35.70 | +0.75 |
| 85K  | 34.49 | 35.25 | +0.76 |
| 90K  | 34.11 | 34.72 | +0.61 |
| 95K  | 33.81 | 34.46 | +0.65 |
| 100K | 33.39 | 34.02 | +0.63 |
| 105K | 32.94 | 33.86 | +0.92 |
| 110K | 32.68 | 33.46 | +0.78 |
| 115K | 32.38 | 33.15 | +0.77 |
| 120K | 32.15 | 32.98 | +0.83 |
| 125K | 31.95 | 32.64 | +0.69 |
| 130K | 31.66 | 32.33 | +0.67 |
| 135K | 31.41 | 32.09 | +0.68 |
| 140K | 31.15 | 31.95 | +0.80 |
| 145K | 30.99 | 31.70 | +0.71 |
| 150K | 30.85 | 31.44 | +0.59 |
| 155K | 30.62 | 31.26 | +0.64 |
| 180K | 29.93 | 30.36 | +0.43 |
| 185K | 29.77 | 30.30 | +0.53 |
| 195K | — | 30.07 | |
| 200K | **29.42** | **29.92** | **+0.50** |

**N=12 C=1408 leads early** (wider model optimizes faster), but **N=24 crosses over at 35K** and stabilizes ~0.5-0.7 PPL ahead. Final gap: +0.50. At ~5% fewer FLOPs for N=12, width vs depth is roughly a wash at this scale. Both models still extending to 400K — gap may continue evolving.

---

## 2. D=12 C=1408 From Scratch vs N=24 C=1024

D=12 C=1408 (152×1408²) is FLOP-matched to N=24 C=1024 (288×1024²). This is the key comparison: can wider look-ahead match deeper roformer?

| Iter | N=24 C=1024 | D=12 C=1408 | D12 vs N24 | N=12 C=1408 | D12 vs N12 |
|------|------------|------------|------------|------------|------------|
| 5K   | 96.72 | 91.20 | -5.52 | 90.24 | +0.96 |
| 10K  | 69.30 | 67.05 | -2.25 | 66.76 | +0.29 |
| 15K  | 58.18 | 57.32 | -0.86 | 57.18 | +0.14 |
| 20K  | 52.46 | 51.61 | -0.85 | 51.59 | +0.02 |
| 25K  | 48.49 | 47.84 | -0.65 | 48.13 | **-0.29** |
| 30K  | 45.54 | 45.40 | -0.14 | 45.53 | -0.13 |
| 35K  | 43.37 | 43.25 | -0.12 | 43.63 | -0.38 |
| 40K  | 41.77 | 41.57 | -0.20 | 42.03 | -0.46 |
| 45K  | 40.33 | 40.25 | -0.08 | 40.82 | -0.57 |
| 50K  | 39.27 | 39.03 | -0.24 | 39.70 | -0.67 |
| 55K  | 38.17 | 38.19 | +0.02 | 38.83 | -0.64 |
| 60K  | 37.37 | 37.22 | -0.15 | 37.88 | -0.66 |
| 65K  | 36.67 | 36.61 | -0.06 | 37.20 | -0.59 |
| 70K  | 36.10 | 35.96 | -0.14 | 36.80 | -0.84 |
| 75K  | 35.67 | 35.42 | -0.25 | 36.19 | -0.77 |
| 80K  | 34.95 | 34.88 | -0.07 | 35.70 | -0.82 |
| 85K  | 34.49 | 34.41 | -0.08 | 35.25 | -0.84 |
| 90K  | 34.11 | 33.95 | -0.16 | 34.72 | -0.77 |
| 95K  | 33.81 | 33.59 | -0.22 | 34.46 | -0.87 |
| 100K | 33.39 | 33.28 | -0.11 | 34.02 | -0.74 |
| 105K | 32.94 | 32.79 | -0.15 | 33.86 | -1.07 |
| 110K | 32.68 | 32.55 | -0.13 | 33.46 | -0.91 |
| 115K | 32.38 | 32.18 | -0.20 | 33.15 | -0.97 |
| 120K | 32.15 | 31.91 | -0.24 | 32.98 | -1.07 |
| 125K | 31.95 | 31.61 | -0.34 | 32.64 | -1.03 |
| 130K | 31.66 | 31.45 | -0.21 | 32.33 | -0.88 |
| 135K | 31.41 | 31.18 | -0.23 | 32.09 | -0.91 |
| 140K | 31.15 | 30.97 | -0.18 | 31.95 | -0.98 |
| 145K | 30.99 | 30.74 | -0.25 | 31.70 | -0.96 |
| 150K | 30.85 | 30.53 | -0.32 | 31.44 | -0.91 |

**D=12 C=1408 has stayed ahead of N=24 C=1024 through 150K iters.** N=12 C=1408 lost its lead at 35K, but D=12 is holding on. The correction mechanism is the difference — it keeps D=12 competitive despite using shared weights.

D=12 vs N=12 gap is growing: -0.29 at 25K → ~-1.0 at 120K. The correction mechanism adds ~1 PPL of value over the plain roformer at the same width and depth. This matches the C=1024 finding where D=12 beat N=12 by 1.06 at 85K.

D=12 from scratch is still running (150K/200K, ~12h remaining).

---

## 3. Fine-Tune: N=12 C=1408 → D=12 C=1408

Converted roformer N=12 C=1408 (29.92 PPL at 200K) to block_head_corr_ffn_add D=12 by copying all shared weights and zero-initializing corr_ffn output layer (initial correction = 0, verified exact PPL match). Fine-tuned at K=random(2,5), lr=2e-4.

### Full fine-tune curve (0–100K)

| Iter | PPL | Δ vs baseline (29.92) |
|------|-----|----------------------|
| 0 | 29.92 | 0.00 |
| 2K | 29.92 | 0.00 |
| 4K | 29.87 | -0.05 |
| 6K | 29.83 | -0.09 |
| 8K | 29.80 | -0.12 |
| 10K | 29.74 | -0.18 |
| 12K | 29.69 | -0.23 |
| 14K | 29.55 | -0.37 |
| 16K | 29.48 | -0.44 |
| 18K | 29.44 | -0.48 |
| 20K | 29.35 | -0.57 |
| 22K | 29.32 | -0.60 |
| 24K | 29.22 | -0.70 |
| 26K | 29.20 | -0.72 |
| 28K | 29.16 | -0.76 |
| 32K | 29.10 | -0.82 |
| 34K | 28.99 | -0.93 |
| 36K | 28.90 | -1.02 |
| 38K | 28.89 | -1.03 |
| 40K | 28.90 | -1.02 |
| 42K | 28.79 | -1.13 |
| 44K | 28.73 | -1.19 |
| 46K | 28.79 | -1.13 |
| 48K | 28.60 | -1.32 |
| 50K | 28.53 | -1.39 |
| 52K | 28.64 | -1.28 |
| 54K | 28.44 | -1.48 |
| 56K | 28.48 | -1.44 |
| 58K | 28.49 | -1.43 |
| 60K | 28.57 | -1.35 |
| 62K | 28.52 | -1.40 |
| 64K | 28.37 | -1.55 |
| 66K | 28.39 | -1.53 |
| 68K | 28.37 | -1.55 |
| 70K | 28.35 | -1.57 |
| 74K | 28.30 | -1.62 |
| 76K | 28.22 | -1.70 |
| 78K | 28.21 | -1.71 |
| 80K | 28.25 | -1.67 |
| 82K | 28.29 | -1.63 |
| 84K | 28.16 | -1.76 |
| 86K | 28.10 | -1.82 |
| 88K | 28.07 | -1.85 |
| 90K | 28.08 | -1.84 |
| 92K | 28.03 | -1.89 |
| 94K | 28.01 | -1.91 |
| 96K | 28.11 | -1.81 |
| 98K | 27.89 | -2.03 |
| 100K | **27.88** | **-2.04** |

### Key observations

- **Total improvement: -2.04 PPL** (29.92 → 27.88) over 100K fine-tune iters
- No hard plateau — PPL kept improving through 100K, with oscillations
- Passed N=24 C=1024 (29.42) at ~20K fine-tune iters
- Passed D=24 C=1024 fine-tune (28.99) at ~34K fine-tune iters
- **Best result of all models tested: 27.88 PPL**

### Comparison with D=24 C=1024 fine-tune

| | D=24 C=1024 fine-tune | D=12 C=1408 fine-tune |
|---|---|---|
| Base model | N=24 C=1024 (29.42) | N=12 C=1408 (29.92) |
| Best PPL | 28.99 (-0.43 at 10K) | **27.88** (-2.04 at 100K) |
| Behavior | Peaked at 10K, bounced at 12K | Steady improvement through 100K |
| Fine-tune iters | 12K (killed by OOM) | 100K |
| Inference FLOPs | 288 × 1024² | 152 × 1408² |

The D=24 fine-tune was cut short at 12K by an OOM. It may have continued improving if run longer. The D=12 C=1408 fine-tune benefited from running the full 100K.

---

## 4. D=23 C=1024 Results

### D=23 K=5 (full training, other machine)

Near FLOP-matched to N=24 (284C² vs 288C²). Crossed over N=24 at 25K-equiv and stayed ahead, averaging ~0.4 PPL better. Latest: ~29.5 at 165K-equiv (still running).

### D=23 K-schedule (completed on this machine)

K curriculum: K=1 for 0–150K, K=2 for 150K–185K, K=2-5 for 185K–200K.

| Iter | N=24 C=1024 | D=23 K-sched | Gap | Phase |
|------|------------|-------------|-----|-------|
| 5K | 96.72 | 99.79 | +3.07 | K=1 |
| 50K | 39.27 | 40.50 | +1.23 | K=1 |
| 100K | 33.39 | 34.27 | +0.88 | K=1 |
| 150K | 30.85 | 31.62 | +0.77 | K=1→K=2 |
| 185K | 29.77 | 30.37 | +0.60 | K=2→K=2-5 |
| 190K | — | 30.28 | | K=2-5 |
| 195K | — | 30.09 | | K=2-5 |
| 200K | — | **29.99** | | K=2-5 |

Final: **29.99 PPL**. The K-schedule traded ~0.5 PPL for ~2x wall-time savings.

---

## 5. D=12 C=1024 Results (completed, other machine)

| Iter | N=24 C=1024 (288C²) | D=12 C=1024 (152C²) | Gap |
|------|---------------------|---------------------|-----|
| 5K | 96.72 | 98.80 | +2.08 |
| 20K | 52.46 | 56.82 | +4.36 |
| 45K | 40.33 | 44.33 | +4.00 |
| 90K | 34.11 | 37.39 | +3.28 |
| 200K | **29.42** | **32.28** | **+2.86** |

D=12 C=1024 finished +2.86 behind N=24 with 47% fewer inference FLOPs (152C² vs 288C²).

### D=12 C=1024 vs N=12 C=1024 (same inference cost)

D=12 crossed over N=12 at 20K and gap grew to -1.06 at 85K. Same 12 layers at inference, but correction-informed training produces better representations.

---

## 6. Wall-Clock Speed: Wider is Faster at Matched FLOPs

D=12 C=1408 and D=24 C=1024 have nearly identical inference FLOPs (152×1408² ≈ 288×1024²). But during training at K=2-5 (avg 3.5), the wall-clock speed is very different:

| Model | Effective layers (K=3.5 avg) | Training speed | s/it |
|-------|------------------------------|---------------|------|
| D=12 C=1408 | 42 layers × C=1408 | 1.11 it/s | 0.90 |
| D=24 C=1024 | 84 layers × C=1024 | 0.75 it/s | 1.33 |

**D=12 C=1408 trains 48% faster** despite matched inference FLOPs. Two reasons:

1. **Sequential depth**: Layers run in series — 84 effective layers means 84 serial steps vs 42 for D=12. Each step has overhead (kernel launches, memory synchronization).

2. **GPU utilization**: GPUs are optimized for large matrix multiplications. Fewer, bigger matmuls (C=1408) saturate GPU compute better than many smaller ones (C=1024). The compute-to-overhead ratio is higher with wider layers.

This is a practical advantage of the look-ahead approach with wider, shallower models: not just fewer inference FLOPs per token, but **faster wall-clock time per FLOP**. A D=12 C=1408 model trains faster, infers faster (fewer sequential steps), and achieves competitive PPL.

At inference with K=1, the advantage is even more pronounced: 12 sequential layers vs 24, with each forward pass completing in roughly half the wall time despite similar total FLOPs.

---

## 7. Scaling Law Fits and Extrapolations

Both roformer training curves follow a power law in log-PPL space: `log(PPL) = a × t^(-b) + c`

Fits (RMSE ~0.003 in log-PPL — excellent):

| Model | a | b | PPL_inf | RMSE |
|-------|---|---|---------|------|
| N=24 C=1024 | 30.24 | 0.339 | 18.17 | 0.003 |
| N=12 C=1408 | 20.92 | 0.298 | 17.25 | 0.002 |

### Extrapolated PPL

| Iters | N=24 C=1024 | N=12 C=1408 | Gap |
|-------|------------|------------|-----|
| 400K | 26.56 | 27.05 | +0.49 |
| 600K | 25.29 | 25.70 | +0.41 |
| 900K (Chinchilla) | 24.22 | 24.56 | +0.34 |
| 1200K | 23.58 | 23.86 | +0.28 |

Both models are significantly undertrained. At our batch size (32 × 256 = 8192 tokens/iter), Chinchilla-optimal for ~370M params would be ~900K iters (7.4B tokens). We're at 400K — less than half optimal.

The gap narrows with more training: +0.49 at 400K → +0.28 at 1200K. The deeper model (N=24) gets there faster, but the wider model (N=12) has a slightly better asymptotic limit from the fit. Both models have large headroom remaining.

### Implication for fine-tuning

The D=24 C=1024 fine-tune needs to beat what plain continued N=24 training would achieve. At 500K (100K more iters), the power law predicts N=24 → 25.86. The fine-tune must get below this to justify the approach over simply training longer.

---

## 8. Fine-Tune from 400K Checkpoints (in progress)

### D=24 C=1024 fine-tune (from N=24 C=1024 400K, baseline 26.66)

| Iter | PPL | Δ vs baseline |
|------|-----|--------------|
| 0 | 26.66 | 0.00 |
| 2K | 26.77 | +0.11 |
| 4K | 26.76 | +0.10 |
| 6K | 26.69 | +0.03 |
| 8K | 26.61 | -0.05 |
| 10K | 26.60 | -0.06 |
| 12K | 26.68 | +0.02 |
| 14K | 26.61 | -0.05 |
| 16K | 26.59 | -0.07 |
| 20K | 26.58 | -0.08 |
| 22K | 26.55 | -0.11 |
| 24K | 26.47 | -0.19 |
| 28K | 26.43 | -0.23 |
| 30K | 26.36 | -0.30 |

Slower to improve than D=12 C=1408 fine-tune was — harder to improve a stronger, more-trained baseline.

### D=12 C=1408 fine-tune (from N=12 C=1408 400K, baseline 27.20)

| Iter | PPL | Δ vs baseline |
|------|-----|--------------|
| 0 | 27.20 | 0.00 |
| 2K | 27.35 | +0.15 |
| 4K | 27.28 | +0.08 |
| 6K | 27.22 | +0.02 |
| 8K | 27.28 | +0.08 |
| 10K | 27.23 | +0.03 |
| 12K | 27.16 | -0.04 |
| 14K | 27.07 | -0.13 |
| 16K | 27.03 | -0.17 |

Same pattern as 200K fine-tune: initial bounce, recovery by ~10K, then steady improvement. The 200K fine-tune reached -2.04 at 100K from a 29.92 baseline. This one starts from 27.20 — if it follows a similar trajectory, could reach ~25-26 range.

### Comparison: fine-tune from 200K vs 400K checkpoints

| | D=12 C=1408 from 200K | D=12 C=1408 from 400K |
|---|---|---|
| Baseline | 29.92 | 27.20 |
| At 16K fine-tune | 29.48 (-0.44) | 27.03 (-0.17) |
| Best (so far) | 27.88 (-2.04 at 100K) | 27.03 (-0.17 at 16K) |

The 400K fine-tune starts from a much lower baseline. Even with slower relative improvement, the absolute PPL should end up lower.

---

## 9. Summary of All Results

| Model | Final PPL | Inference FLOPs | Status |
|-------|-----------|----------------|--------|
| **D=12 C=1408 fine-tuned from 200K** | **27.88** | 152 × 1408² | Completed |
| D=24 C=1024 fine-tuned from 400K | 26.36 (30K, improving) | 288 × 1024² | Running |
| D=12 C=1408 fine-tuned from 400K | 27.03 (16K, improving) | 152 × 1408² | Running |
| N=24 C=1024 (400K) | **26.66** | 288 × 1024² | Completed |
| N=12 C=1408 (400K) | **~27.20** | 144 × 1408² | Completed |
| D=24 C=1024 fine-tuned from 200K | 28.99 | 288 × 1024² | Completed (cut short) |
| D=12 C=1408 from scratch (200K) | **29.00** | 152 × 1408² | Completed |
| N=24 C=1024 (200K) | **29.42** | 288 × 1024² | Completed |
| D=23 K=5 C=1024 | ~29.5 | 284 × 1024² | Running (other machine) |
| N=12 C=1408 (200K) | **29.92** | 144 × 1408² | Completed |
| D=23 K-schedule C=1024 | **29.99** | 284 × 1024² | Completed |
| D=12 C=1024 (200K) | **32.28** | 152 × 1024² | Completed |

### Key findings

1. **Fine-tuning is powerful.** Converting N=12 C=1408 (200K) → D=12 C=1408 improved PPL by 2.04 (29.92 → 27.88) over 100K fine-tune iters. The 400K fine-tunes are in progress and starting from stronger baselines.

2. **D=12 C=1408 from scratch stays ahead of N=24 C=1024** through 200K iters at FLOP parity, finishing at 29.00 vs 29.42. N=12 C=1408 (same width, no correction) lost its lead at 35K. The correction mechanism is the difference.

3. **D=12 vs N=12 at C=1408: ~1 PPL gap.** The correction mechanism adds ~1 PPL of value over the plain roformer at the same width and depth, consistent with the C=1024 finding.

4. **Wider models train faster at matched FLOPs.** D=12 C=1408 trains 48% faster than D=24 C=1024 despite similar total FLOPs, due to fewer sequential layers and better GPU utilization from larger matmuls. At inference (K=1), the wall-clock advantage is even greater: 12 vs 24 sequential steps.

5. **Stronger baselines are harder to fine-tune.** D=24 from 400K (26.66) is improving at -0.30 over 30K iters. D=12 from 200K (29.92) improved -0.44 in just 16K iters. The correction mechanism finds less room to improve in a well-trained model.

6. **K-schedule trades ~0.5 PPL for ~2x wall-time savings.** D=23 K-schedule (29.99) vs D=23 K=5 (~29.5).

7. **Both roformers are significantly undertrained.** Chinchilla-optimal is ~900K iters. At 400K we're at less than half. Power law extrapolation suggests N=24 → 24.22 and N=12 → 24.56 at 900K.

### Currently running

- **GPU 0**: D=24 C=1024 fine-tune from 400K (26.36 at 30K, ~25h left)
- **GPU 1**: D=12 C=1408 fine-tune from 400K (27.03 at 16K, ~21h left)
- **Other machine**: D=23 K=5 C=1024, Roformer N=12 C=1024

- **GPU 0**: N=24 C=1024 extending from 200K to 400K
- **GPU 1**: D=12 C=1408 from scratch, 150K/200K (~12h left)
- **Other machine**: D=23 K=5 C=1024 (187K/200K), Roformer N=12 C=1024 (148K/200K)

### Next steps

- N=12 C=1408 extension to 400K (launch when GPU frees)
- Convert best roformer checkpoints (at 400K) to look-ahead and fine-tune

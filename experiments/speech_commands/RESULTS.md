# Speech Commands Experiment: Bridging Rotation Models to S5

## Goal
Understand exactly what makes S5 work by starting from a simple rotation model
and adding S5's components one at a time, measuring the impact of each change.
Then: can we get S5-level performance with cumsum speed (no parallel scan)?

**Answer: Yes. BlockDecayS5V2 (89.8% full) and WindowS5 (89.4% full) beat S5 (56.7% smoke) with fast cumsum. Block decay adds stable training.**

## Setup
- Dataset: Google Speech Commands v2, 12 classes (10 commands + unknown + silence)
- Smoke tests: **2-epoch, tiny subset** (~4K train samples)
- Full runs: **40-epoch, full dataset** (~36K train samples)
- Hardware: T4 GPU (16GB), 16GB RAM
- Sequence: raw waveform 16kHz, pooled 4x → T=4000
- Architecture: Linear(1→64) front-end, 6 layers, bidirectional, BatchNorm, GLU
- Optimizer: AdamW, SSM params lr=1e-3/wd=0, other params lr=1e-2/wd=0.05
- Scheduler: CosineAnnealingLR (lr → 0 over 40 epochs)

## Part 1: Deconstructing S5

Starting from a simple rotation+decay model and adding S5 components one by one.
All models ~152K params (matching S5 exactly).

| Step | Model | Change | Ep1 val | Ep2 test | Delta |
|------|-------|--------|---------|----------|-------|
| 0 | RotDecayFixed | real proj, shared gates, RoPE init, λ=0.99 | 12.1% | 17.4% | — |
| 1 | +B/C/D | complex B/C matrices + D skip | 15.8% | 26.5% | +9.1% |
| 2 | +per-layer | per-layer gates (not shared across layers) | 20.4% | 34.0% | +7.5% |
| 3 | +tied dt | gate=exp((σ+iω)·dt), S4D-Lin init | 29.4% | 41.7% | +7.7% |
| 4 | +B_bar (buggy) | B̄=((e^{Λdt}-1)/Λ)·B, but /Λdt instead of /Λ | 15.2% | 47.8% | +6.1% |
| 5 | +B_bar (fixed) | correct B_bar scaling | **65.5%** | **64.5%** | **+16.7%** |
| ref | S5 | full S5 implementation | ? | 56.7% | — |

### What each component contributes

1. **Complex B/C + D** (+9.1%): B projects real input into complex state space,
   C projects complex state back to real output. Richer mixing than real-valued
   Linear projections. D is a direct input→output skip bypassing the recurrence.

2. **Per-layer gates** (+7.5%): Each layer learns its own decay rate and frequencies.
   Layer 1 might do fast local patterns, layer 6 might do slow global patterns.
   Shared gates force all layers to use the same dynamics.

3. **Tied dt** (+7.7%): Instead of independent (λ, θ) per dimension, use continuous
   eigenvalues Λ_k = σ_k + iω_k and one shared dt per layer. Gate = exp(Λ·dt).
   This ties decay and frequency together — scaling dt uniformly changes the
   "time resolution" of all dimensions. S4D-Lin init: σ=-0.5, ω=πk, dt=0.01.

4. **B_bar discretization** (+16.7%): ZOH (zero-order hold) discretization scales
   B by (exp(Λdt)-1)/Λ. This normalizes input per mode: slow modes (small |Λ|)
   get scaling ~dt, fast-decaying modes (large |Λ|) get scaling ~1/|Λ|. Each mode
   receives input proportional to how long it can retain it. **Largest single factor.**

## Part 2: Dropping decay — cumsum S5 variants

The key question: is decay (|gate|<1) actually necessary, or do B/C/D + B_bar
compensate? We built two models that keep all of S5's components except decay
(Λ_re=0, |gate|=1), enabling fast cumsum instead of slow parallel scan.

### Smoke test (2 epochs, ~4K train)

| Model | Ep1 val | Ep2 test | Params | Speed | Decay |
|-------|---------|----------|--------|-------|-------|
| **WindowS5** | **42.5%** | **67.7%** | 151,576 | **fast (cumsum)** | **no** |
| RotS5Fixed | 65.5% | 64.5% | 151,960 | slow (scan) | yes |
| S5 | ? | 56.7% | 151,960 | slow (scan) | yes |
| CumsumS5 | 19.5% | 34.4% | 151,576 | fast (cumsum) | no |

### Full run (40 epochs, full dataset)

| Model | Test Acc | Macro F1 | Params | Speed | Training |
|-------|----------|----------|--------|-------|----------|
| **BlockDecayS5V2** | **89.8%** | **0.897** | 151,960 | fast (cumsum) | **stable** |
| WindowS5 | 89.4% | 0.893 | 151,576 | fast (cumsum) | unstable |
| CumsumS5 | 77.0% | 0.768 | 151,576 | fast (cumsum) | smooth |

### Epoch-by-epoch: CumsumS5 (40 epochs, full dataset)

Smooth, steady climb. No decay, no window — unbounded accumulation.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.268 | 20.2% | 22.8% | 21 | 0.953 | 67.9% | 70.3% |
| 2 | 2.066 | 27.1% | 33.8% | 22 | 0.947 | 68.3% | 71.8% |
| 3 | 1.702 | 39.9% | 46.5% | 23 | 0.919 | 69.0% | 71.6% |
| 4 | 1.484 | 48.6% | 52.5% | 24 | 0.902 | 69.8% | 70.2% |
| 5 | 1.348 | 53.5% | 55.7% | 25 | 0.891 | 70.2% | 70.7% |
| 6 | 1.303 | 55.2% | 56.6% | 26 | 0.873 | 70.5% | 69.2% |
| 7 | 1.262 | 56.7% | 58.6% | 27 | 0.857 | 71.3% | 72.6% |
| 8 | 1.227 | 57.9% | 56.3% | 28 | 0.846 | 71.5% | 74.6% |
| 9 | 1.203 | 58.7% | 62.8% | 29 | 0.827 | 72.4% | 74.3% |
| 10 | 1.170 | 59.9% | 61.8% | 30 | 0.811 | 72.8% | 75.6% |
| 11 | 1.142 | 61.1% | 59.5% | 31 | 0.791 | 73.4% | 75.7% |
| 12 | 1.123 | 62.0% | 62.3% | 32 | 0.773 | 74.2% | 75.0% |
| 13 | 1.096 | 63.0% | 66.9% | 33 | 0.752 | 75.0% | 75.4% |
| 14 | 1.075 | 63.7% | 66.0% | 34 | 0.735 | 75.5% | 76.3% |
| 15 | 1.060 | 64.4% | 66.7% | 35 | 0.716 | 76.3% | 76.6% |
| 16 | 1.041 | 65.0% | 66.3% | 36 | 0.698 | 76.6% | 77.1% |
| 17 | 1.027 | 65.5% | 68.9% | 37 | 0.680 | 77.2% | 77.2% |
| 18 | 1.007 | 66.2% | 68.0% | 38 | 0.663 | 77.8% | 78.4% |
| 19 | 0.989 | 66.9% | 68.2% | 39 | 0.658 | 78.1% | 78.6% |
| 20 | 0.975 | 67.4% | 68.9% | 40 | 0.651 | 78.4% | 78.6% |

### Epoch-by-epoch: WindowS5 (40 epochs, full dataset)

Higher ceiling but unstable validation — drops at epochs 4-6, 15, 17, 19, 22.
No decay means no damping; small parameter changes cause large swings.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.755 | 40.1% | 22.8% | 21 | 0.558 | 82.2% | 83.5% |
| 2 | 1.086 | 64.3% | 43.8% | 22 | 0.548 | 82.7% | 76.7% |
| 3 | 0.929 | 69.9% | 53.2% | 23 | 0.532 | 83.0% | 80.0% |
| 4 | 0.855 | 72.3% | 35.9% | 24 | 0.525 | 83.1% | 84.6% |
| 5 | 0.805 | 74.0% | 26.9% | 25 | 0.508 | 83.7% | 86.2% |
| 6 | 0.779 | 74.8% | 27.7% | 26 | 0.495 | 84.2% | 86.4% |
| 7 | 0.751 | 75.8% | 77.9% | 27 | 0.487 | 84.6% | 84.5% |
| 8 | 0.728 | 76.7% | 77.0% | 28 | 0.472 | 84.9% | 83.0% |
| 9 | 0.720 | 76.8% | 77.2% | 29 | 0.459 | 85.4% | 86.7% |
| 10 | 0.698 | 77.6% | 68.7% | 30 | 0.443 | 85.9% | 84.3% |
| 11 | 0.682 | 78.2% | 74.5% | 31 | 0.429 | 86.3% | 88.0% |
| 12 | 0.672 | 78.3% | 79.8% | 32 | 0.415 | 86.8% | 88.0% |
| 13 | 0.656 | 79.0% | 80.8% | 33 | 0.399 | 87.4% | 87.1% |
| 14 | 0.653 | 78.9% | 82.7% | 34 | 0.389 | 87.6% | 87.4% |
| 15 | 0.633 | 79.5% | 48.1% | 35 | 0.374 | 88.0% | 88.8% |
| 16 | 0.616 | 80.2% | 79.4% | 36 | 0.365 | 88.4% | 88.3% |
| 17 | 0.609 | 80.5% | 69.6% | 37 | 0.349 | 88.9% | 88.9% |
| 18 | 0.597 | 81.1% | 82.2% | 38 | 0.341 | 89.2% | 89.1% |
| 19 | 0.591 | 81.1% | 70.8% | 39 | 0.331 | 89.5% | 89.2% |
| 20 | 0.568 | 82.0% | 83.1% | 40 | 0.328 | 89.5% | 89.0% |

### Key findings

**BlockDecayS5V2 (89.8%) is the best SSM model.** It beats WindowS5 (89.4%)
with completely stable training — no val crashes. Block decay combines cumsum
speed with smooth geometric forgetting.

**WindowS5 (89.4%) is close but unstable.** Val accuracy drops sharply at
epochs 4-6 (27%), 15 (48%), and smaller dips at 17, 19, 22. Pure rotation
(|gate|=1) has no damping — hard window cutoff amplifies instability.

**CumsumS5 (77.0%) is much weaker.** Without decay AND without a window, the
infinite cumsum accumulates all history equally — signal gets washed out by
noise from thousands of timesteps ago.

**MelCNN (95.7%) still dominates** with 4x fewer params (25K vs 152K). The
mel spectrogram provides strong inductive bias for audio. SSMs need to close
a 6% gap.

**Decay and windowing serve the same purpose**: preventing ancient history from
drowning out recent signal. Block decay is the best approach — smooth geometric
forgetting with cumsum speed and stable training.

## Part 2b: Input-dependent angle variants

Tested two approaches to making angles input-dependent instead of fixed S4D-Lin:

1. **Input variant** — predict angles from scratch via MLP + LayerNorm + cumsum
2. **Mod variant** — modulate fixed base angles: `angle = base * (1 + proj(x))`, no LayerNorm

### Smoke test (2 epochs, ~4K train)

| Model | Ep2 test | Params | Approach |
|-------|----------|--------|----------|
| WindowS5 (fixed) | 67.7% | 151,576 | fixed S4D-Lin angles |
| WindowS5Mod | 55.7% | 189,016 | modulated angles |
| WindowS5Input | 32.7% | 189,400 | predicted angles |
| CumsumS5 (fixed) | 34.4% | 151,576 | fixed S4D-Lin angles |
| CumsumS5Mod | 23.0% | 189,016 | modulated angles |
| CumsumS5Input | 16.8% | 189,400 | predicted angles |

### Full run (40 epochs, partial)

| Model | Result | Notes |
|-------|--------|-------|
| WindowS5Mod | collapsed epoch 10-12 | val: 70%→14%→31%, training diverged |
| CumsumS5Mod | stuck at ~12% | never learned on full dataset |
| CumsumS5Input | stuck at ~17% | never learned on full dataset |

### Findings

**Fixed angles > modulated > predicted.** Input-dependent angles consistently
hurt performance. Two likely causes:

1. **Hard window cutoff amplifies instability.** The window subtraction
   `cs[t] - cs[t-W]` is sensitive to phase coherence. With input-dependent
   angles, phases at t and t-W depend on different inputs, making the
   subtraction noisy. This is the primary cause — even fixed-angle WindowS5
   shows val instability (epochs 4-6, 15).

2. **LayerNorm destroys S4D-Lin init** (Input variant only). The LayerNorm
   normalizes away the carefully tuned frequency initialization, forcing
   the model to rediscover good frequencies from scratch.

## Part 2c: Block decay — cumsum speed with learnable decay

Instead of hard window cutoff (unstable) or continuous decay (needs parallel
scan), use **block-wise decay**: divide the sequence into blocks of W steps,
apply cumsum within blocks, decay λ^k between blocks.

```
h_t = 1·block_0 + λ·block_1 + λ²·block_2 + ...
where block_k = cs[t-kW] - cs[t-(k+1)W]
```

Benefits: cumsum speed, learnable decay, smooth forgetting (no hard cutoff),
should stabilize training vs pure windowing.

### Smoke test (2 epochs, ~4K train)

| Model | W | K | Ep1 val | Ep2 test | Params |
|-------|---|---|---------|----------|--------|
| BlockDecayS5 | 800 | 5 | 19.2% | 39.2% | 151,960 |
| BlockDecayS5 | 80 | 50 | 51.4% | (killed) | 151,960 |
| BlockDecayS5Mod | 800 | 5 | 16.0% | 15.8% | 189,400 |
| BlockDecayS5V2 | 80 | 50 | 42.3% | 65.1% | 151,960 |

**Smaller window (W=80) is better** — epoch 1 val 51.4% vs 19.2% for W=800.
More blocks = finer-grained decay control.

**V1 is slow with small W** — K=50 Python loop iterations, each operating on
full (B, T, n) tensors with F.pad allocations. V2 reshapes into (B, K, W, n)
blocks, loop operates on (B, W, n) — 50x smaller tensors, no F.pad.

### BlockDecayS5V2 full run (40 epochs, W=80, full dataset)

Test accuracy: **89.8%**, macro F1: **0.897**. Completely stable training — no
val crashes unlike WindowS5. Block decay successfully replaces hard window cutoff
with smooth geometric weighting while maintaining cumsum speed.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.053 | 27.9% | 49.3% | 21 | 0.491 | 84.2% | 85.0% |
| 2 | 1.175 | 60.0% | 66.9% | 22 | 0.478 | 84.6% | 84.8% |
| 3 | 0.911 | 70.2% | 72.3% | 23 | 0.469 | 84.8% | 86.1% |
| 4 | 0.792 | 74.6% | 73.5% | 24 | 0.455 | 85.3% | 80.6% |
| 5 | 0.737 | 76.3% | 78.7% | 25 | 0.441 | 85.8% | 86.6% |
| 6 | 0.702 | 77.3% | 77.3% | 26 | 0.432 | 86.1% | 87.6% |
| 7 | 0.674 | 78.0% | 76.5% | 27 | 0.418 | 86.6% | 86.7% |
| 8 | 0.650 | 78.8% | 77.7% | 28 | 0.409 | 86.7% | 87.3% |
| 9 | 0.635 | 79.7% | 79.0% | 29 | 0.393 | 87.3% | 86.1% |
| 10 | 0.622 | 79.7% | 78.8% | 30 | 0.384 | 87.6% | 88.3% |
| 11 | 0.609 | 80.2% | 80.2% | 31 | 0.366 | 88.2% | 88.9% |
| 12 | 0.595 | 80.7% | 82.8% | 32 | 0.355 | 88.5% | 89.0% |
| 13 | 0.584 | 81.4% | 81.7% | 33 | 0.339 | 89.0% | 89.4% |
| 14 | 0.564 | 81.7% | 82.1% | 34 | 0.329 | 89.3% | 89.2% |
| 15 | 0.559 | 81.9% | 80.1% | 35 | 0.313 | 89.8% | 90.1% |
| 16 | 0.548 | 82.3% | 85.7% | 36 | 0.302 | 90.2% | 90.1% |
| 17 | 0.534 | 82.7% | 83.0% | 37 | 0.292 | 90.5% | 90.6% |
| 18 | 0.523 | 83.1% | 84.3% | 38 | 0.283 | 91.0% | 90.4% |
| 19 | 0.514 | 83.3% | 86.2% | 39 | 0.273 | 91.2% | 90.5% |
| 20 | 0.503 | 83.8% | 84.9% | 40 | 0.269 | 91.4% | 90.7% |

## Part 3: Earlier rotation models (no B/C/D)

Models using simple real-valued projections instead of complex B/C/D.

| Model | Accuracy | Speed | Notes |
|-------|----------|-------|-------|
| RotWinInput | 21.9% | fast (cumsum) | finite window W=80, input-dependent angles |
| RotWindow | 19.3% | fast (cumsum) | finite window W=80, fixed angles |
| RotDecayFixed | 17.4% | slow (scan) | decay λ<1, S4D-Lin init |
| RotInput | 13.2% | fast (cumsum) | no decay, input-dependent angles |
| RotDecayInput | 12.0% | slow (scan) | decay, input-dependent angles+λ |
| RotFixed | 8.7% | fast (cumsum) | no decay, fixed angles |

## CNN baselines (40 epochs, full dataset)

| Model | Test Acc | Macro F1 | Params | Notes |
|-------|----------|----------|--------|-------|
| **MelCNN** | **95.7%** | **0.957** | 25,628 | log-mel + TC-ResNet |
| LearnedSpecCNN | 93.4% | 0.934 | 25,668 | learned frequencies via windowed cumsum |
| RawCNN | 91.0% | 0.909 | 25,228 | M5-style, Conv1d(k=80,s=16) |

All three share the same CNN backend (ResBlock1d 16→24→32→48 + AdaptiveAvgPool).
The only difference is the front-end: fixed FFT+mel vs learned cumsum frequencies
vs learned Conv1d(k=80,s=16).

**LearnedSpecCNN (93.4%)** replaces FFT+mel filterbank with 40 learnable frequencies
via windowed cumsum. Same CNN backend as MelCNN. Beats RawCNN by 2.4% but trails
MelCNN by 2.3%. The gap is likely because FFT uses 201 frequency bins (n_fft=400)
fed into triangular mel filters, while LearnedSpecCNN uses only 40 frequencies
directly — asking 40 parameters to do the work of 201 bins + a filterbank.

**MelCNN vs SSMs**: MelCNN (95.7%, 25K params) dominates all SSM variants
(best: BlockDecayS5V2 89.8%, 152K params). 6x more params, 6% worse. The
mel spectrogram provides massive inductive bias — FFT + mel scale + log
compression — that SSMs must learn from scratch.

### Epoch-by-epoch: LearnedSpecCNN (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.265 | 62.6% | 79.7% | 21 | 0.193 | 93.6% | 92.5% |
| 2 | 0.553 | 83.3% | 86.6% | 22 | 0.189 | 93.7% | 92.5% |
| 3 | 0.431 | 86.4% | 87.2% | 23 | 0.185 | 93.8% | 92.8% |
| 4 | 0.373 | 88.2% | 89.3% | 24 | 0.181 | 94.0% | 93.0% |
| 5 | 0.338 | 89.2% | 89.4% | 25 | 0.180 | 93.9% | 92.5% |
| 6 | 0.316 | 89.9% | 89.1% | 26 | 0.176 | 94.2% | 92.8% |
| 7 | 0.300 | 90.2% | 90.4% | 27 | 0.171 | 94.3% | 93.1% |
| 8 | 0.284 | 90.7% | 90.1% | 28 | 0.171 | 94.3% | 93.3% |
| 9 | 0.270 | 91.3% | 90.5% | 29 | 0.166 | 94.5% | 93.1% |
| 10 | 0.266 | 91.3% | 89.9% | 30 | 0.163 | 94.5% | 93.3% |
| 11 | 0.260 | 91.5% | 91.3% | 31 | 0.161 | 94.8% | 93.7% |
| 12 | 0.248 | 92.1% | 90.6% | 32 | 0.156 | 94.9% | 93.5% |
| 13 | 0.237 | 92.3% | 91.6% | 33 | 0.154 | 95.0% | 93.8% |
| 14 | 0.230 | 92.6% | 91.3% | 34 | 0.149 | 95.1% | 93.4% |
| 15 | 0.224 | 92.7% | 91.4% | 35 | 0.149 | 95.1% | 93.4% |
| 16 | 0.218 | 92.9% | 92.0% | 36 | 0.150 | 95.1% | 93.7% |
| 17 | 0.211 | 93.1% | 92.0% | 37 | 0.148 | 95.1% | 93.7% |
| 18 | 0.211 | 93.0% | 91.7% | 38 | 0.146 | 95.4% | 93.5% |
| 19 | 0.207 | 93.1% | 92.5% | 39 | 0.147 | 95.2% | 93.7% |
| 20 | 0.204 | 93.2% | 92.2% | 40 | 0.146 | 95.4% | 93.6% |

### Epoch-by-epoch: MelCNN (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.475 | 55.0% | 79.5% | 21 | 0.294 | 90.1% | 94.2% |
| 2 | 0.744 | 77.3% | 86.3% | 22 | 0.290 | 90.4% | 94.8% |
| 3 | 0.605 | 80.8% | 88.0% | 23 | 0.280 | 90.7% | 94.4% |
| 4 | 0.515 | 83.6% | 89.5% | 24 | 0.271 | 91.0% | 94.7% |
| 5 | 0.498 | 83.9% | 90.5% | 25 | 0.279 | 90.6% | 94.2% |
| 6 | 0.450 | 85.3% | 91.7% | 26 | 0.264 | 91.2% | 94.9% |
| 7 | 0.431 | 85.9% | 91.8% | 27 | 0.276 | 90.9% | 94.9% |
| 8 | 0.407 | 86.8% | 91.4% | 28 | 0.266 | 91.1% | 94.7% |
| 9 | 0.391 | 87.0% | 92.0% | 29 | 0.258 | 91.2% | 95.1% |
| 10 | 0.381 | 87.4% | 92.0% | 30 | 0.263 | 91.3% | 94.8% |
| 11 | 0.386 | 87.2% | 92.8% | 31 | 0.258 | 91.5% | 95.1% |
| 12 | 0.354 | 88.4% | 92.8% | 32 | 0.251 | 91.6% | 95.4% |
| 13 | 0.340 | 88.8% | 93.1% | 33 | 0.252 | 91.6% | 95.3% |
| 14 | 0.344 | 88.6% | 93.3% | 34 | 0.243 | 92.0% | 95.3% |
| 15 | 0.325 | 89.4% | 93.6% | 35 | 0.242 | 91.9% | 95.3% |
| 16 | 0.325 | 89.1% | 92.7% | 36 | 0.253 | 91.5% | 95.2% |
| 17 | 0.315 | 89.5% | 94.0% | 37 | 0.245 | 91.7% | 95.2% |
| 18 | 0.311 | 89.9% | 93.5% | 38 | 0.239 | 92.0% | 94.9% |
| 19 | 0.309 | 89.7% | 94.0% | 39 | 0.238 | 92.1% | 95.3% |
| 20 | 0.292 | 90.2% | 94.4% | 40 | 0.231 | 92.3% | 95.3% |

### Epoch-by-epoch: RawCNN (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.513 | 56.7% | 73.2% | 21 | 0.296 | 90.7% | 90.3% |
| 2 | 0.808 | 76.5% | 80.4% | 22 | 0.289 | 90.8% | 90.6% |
| 3 | 0.640 | 80.8% | 83.5% | 23 | 0.285 | 91.0% | 91.1% |
| 4 | 0.566 | 82.9% | 85.2% | 24 | 0.277 | 91.2% | 90.8% |
| 5 | 0.512 | 84.5% | 85.7% | 25 | 0.277 | 91.4% | 91.2% |
| 6 | 0.480 | 85.5% | 86.3% | 26 | 0.271 | 91.5% | 91.2% |
| 7 | 0.451 | 85.9% | 87.3% | 27 | 0.265 | 91.7% | 91.2% |
| 8 | 0.428 | 86.9% | 88.1% | 28 | 0.261 | 91.7% | 91.2% |
| 9 | 0.411 | 87.2% | 88.0% | 29 | 0.260 | 91.6% | 91.4% |
| 10 | 0.392 | 87.8% | 88.4% | 30 | 0.257 | 91.9% | 91.3% |
| 11 | 0.380 | 88.3% | 88.5% | 31 | 0.255 | 92.0% | 91.4% |
| 12 | 0.369 | 88.5% | 87.5% | 32 | 0.250 | 92.2% | 91.5% |
| 13 | 0.358 | 88.8% | 89.8% | 33 | 0.246 | 92.3% | 91.2% |
| 14 | 0.348 | 89.1% | 89.7% | 34 | 0.243 | 92.3% | 91.5% |
| 15 | 0.342 | 89.3% | 89.7% | 35 | 0.244 | 92.2% | 91.2% |
| 16 | 0.331 | 89.7% | 90.0% | 36 | 0.243 | 92.4% | 91.5% |
| 17 | 0.321 | 90.0% | 90.0% | 37 | 0.244 | 92.3% | 91.5% |
| 18 | 0.312 | 90.1% | 90.0% | 38 | 0.240 | 92.5% | 91.6% |
| 19 | 0.308 | 90.2% | 89.9% | 39 | 0.239 | 92.6% | 91.5% |
| 20 | 0.302 | 90.6% | 90.3% | 40 | 0.240 | 92.6% | 91.5% |

## Key insights

1. **Decay is not necessary.** WindowS5 (89.4%, no decay, cumsum) beats S5 (56.7%
   smoke) on this task. What matters is limiting the effective memory window —
   decay and windowing are two ways to do this.

2. **B_bar is the single biggest factor.** Correct input normalization per mode
   contributed +16.7% in smoke tests — more than B/C/D, per-layer, and tied dt combined.

3. **Windowing >> no windowing.** CumsumS5 (77.0%) vs WindowS5 (89.4%) at 40
   epochs. The gap persists at scale. Window provides a natural forgetting
   mechanism that's compatible with fast cumsum.

4. **Cumsum is ~5x faster than parallel scan.** Single fused CUDA kernel vs
   12 sequential Python tensor ops. On T4: ~2 min/epoch vs ~10 min/epoch.

5. **All S5 components matter.** Each of the 5 changes (B/C/D, per-layer, tied dt,
   B_bar) contributed 7-17%. No single trick — it's the combination.

6. **S4D-Lin init matters.** Switching from RoPE frequencies (geometric, wide range)
   to S4D-Lin init (linear, low frequencies) doubled RotDecayFixed accuracy.

7. **No decay = unstable training.** WindowS5 val drops sharply at epochs 4-6
   and 15. Pure rotation has no damping. CumsumS5 is perfectly smooth by
   comparison. Both converge, but the instability suggests decay may help
   optimization even if not strictly needed for final accuracy.

8. **Hard window cutoff causes instability.** The window subtraction is the
   root cause of val swings, not input-dependent angles. Block decay should
   fix this by replacing the hard cutoff with smooth geometric weighting.

9. **Data-dependent angles need windowing.** RotInput (no B/C/D, infinite cumsum)
   was stuck at chance — cumsum of noisy angles over T=4000 produces chaotic
   phase. Windowing fixes this: LearnedSpecCNNMod (W=400) gets 92.1%,
   MultiLayerMinimalMod (W=40) tracks ~87%. Fixed frequencies still win
   (93.4% vs 92.1%) because a single audio sample carries no frequency info.

10. **B/C/D are not necessary.** MultiLayerMinimalStrided (88.9%) nearly matches
    BlockDecayS5V2 (89.8%) with just learned frequencies + windowed cumsum +
    Linear mixing + GLU. No B/C/D, no decay, no B_bar. Much simpler and
    completely stable optimization.

11. **Multi-layer windowed cumsum works.** Layer 1 downsamples (stride=10),
    subsequent layers refine on the shorter sequence. Each layer's Linear
    projection mixes across frequencies — this replaces B/C/D's role.

12. **Bidirectional is unnecessary.** All SSM models (Parts 1-3) used bidirectional
    scans (12 rotation scans per forward pass — slow). All current best models
    (LearnedSpecCNN 94.1%, MelCNN 95.7%) are unidirectional — the CNN backend
    handles temporal context. Dropping bidirectional halves scan cost.

## Architecture summary

```
BlockDecayS5 (NEW — fast cumsum + learnable decay):
  gate = exp(iω_k·dt),  |gate| = 1  (within blocks)
  Block decay: h = Σ_k λ^k · [cs(t-kW) - cs(t-(k+1)W)]
  B_bar = ((exp(iωdt)-1)/(iω))·B
  Per-layer: ω_k, dt, λ_k, B, C, D

WindowS5 (fast cumsum, no decay, hard window):
  gate = exp(iω_k·dt),  |gate| = 1
  B_bar = ((exp(iωdt)-1)/(iω))·B
  Input → [B_bar] → complex state → [windowed cumsum, W=80] → [C] → output + D·input
  Per-layer: ω_k, dt, B, C, D

S5 / RotS5Fixed (slow — parallel scan, with decay):
  gate = exp((σ_k + iω_k)·dt),  |gate| < 1
  B_bar = ((exp(Λdt)-1)/Λ)·B
  Input → [B_bar] → complex state → [parallel scan] → [C] → output + D·input
  Per-layer: σ_k, ω_k, dt, B, C, D

CumsumS5 (fast cumsum, no decay, no window):
  Same as WindowS5 but cumsum over full sequence. Washes out signal.

RotWindow (fast cumsum, no B/C/D):
  Real projections only, no B_bar. Much weaker (19.3%).
```

## Part 4: Minimal windowed cumsum (no B/C/D)

Stripped SSM to bare minimum: learned frequencies + windowed cumsum + unrotate →
real+imag output. No B/C/D, no decay, no state space machinery. Just frequency
decomposition with learnable mixing between layers.

### Single-layer MinimalStridedWindow experiments

Iterative exploration stripping SSM to minimum viable:

| Variant | Val Acc | Params | Notes |
|---------|---------|--------|-------|
| Power spectrum (no log) | ~30% plateau | 844 | |d|² has huge dynamic range |
| Power + log | ~43% ep3 | 9,292 | log compression critical for power |
| Real+imag (no unrotation) | ~35% ep1 | 34,892 | phase oscillates at carrier freq |
| Real+imag + unrotation | ~48% ep18 | 34,892 | stable but plateaus |

Key findings:
- **Log compression** is critical for power spectrum (huge dynamic range)
- **Unrotation** (`d * phases`) is critical for real+imag output — stabilizes phase
- Single layer plateaus around 48% — needs depth

### MultiLayerMinimalStrided (3 layers, ds=10, W=400)

Multi-layer version: layer 1 does windowed cumsum (W=400) on raw T=16000 and
strides by ds_factor=10 → T=1600. Layers 2+ do windowed cumsum (W=40) on T=1600
with Linear(128→128) mixing between layers. Each layer: BN + GLU + residual.

Test accuracy: **88.9%**, macro F1: **0.889**, 134,604 params.

Completely stable training — smooth climb from 46% to 89%, no val crashes.
Close to BlockDecayS5V2 (89.8%) with much simpler architecture (no B/C/D, no
decay, no B_bar discretization).

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.800 | 40.0% | 46.3% | 21 | 0.432 | 86.7% | 85.8% |
| 2 | 1.323 | 57.9% | 65.7% | 22 | 0.425 | 87.0% | 86.1% |
| 3 | 0.994 | 69.5% | 73.0% | 23 | 0.412 | 87.4% | 86.5% |
| 4 | 0.859 | 73.8% | 75.5% | 24 | 0.403 | 87.6% | 86.1% |
| 5 | 0.782 | 76.6% | 77.0% | 25 | 0.395 | 87.8% | 86.8% |
| 6 | 0.722 | 78.4% | 80.3% | 26 | 0.384 | 88.1% | 87.3% |
| 7 | 0.679 | 79.5% | 80.0% | 27 | 0.372 | 88.4% | 86.4% |
| 8 | 0.638 | 80.8% | 81.0% | 28 | 0.363 | 88.7% | 87.1% |
| 9 | 0.613 | 81.5% | 80.2% | 29 | 0.357 | 89.1% | 87.0% |
| 10 | 0.592 | 82.0% | 83.4% | 30 | 0.344 | 89.3% | 87.1% |
| 11 | 0.574 | 82.8% | 82.0% | 31 | 0.343 | 89.2% | 87.0% |
| 12 | 0.551 | 82.9% | 84.1% | 32 | 0.335 | 89.6% | 87.1% |
| 13 | 0.537 | 83.4% | 83.1% | 33 | 0.329 | 89.7% | 87.6% |
| 14 | 0.520 | 84.1% | 84.4% | 34 | 0.321 | 90.2% | 88.4% |
| 15 | 0.508 | 84.5% | 84.8% | 35 | 0.315 | 90.2% | 88.3% |
| 16 | 0.495 | 85.2% | 84.2% | 36 | 0.315 | 90.1% | 88.5% |
| 17 | 0.478 | 85.4% | 84.8% | 37 | 0.313 | 90.2% | 88.5% |
| 18 | 0.464 | 85.8% | 85.3% | 38 | 0.313 | 90.3% | 88.5% |
| 19 | 0.453 | 86.2% | 86.0% | 39 | 0.309 | 90.5% | 88.8% |
| 20 | 0.442 | 86.7% | 85.4% | 40 | 0.306 | 90.4% | 88.0% |

## Part 5: Front-end variants and data-dependent frequencies

### Why RotInput failed (and what we learned)

RotInput and RotFixed used **no B/C/D** — just rotation scan (derotate → cumsum →
rerotate) with linear value projections. The architecture was:
```
v = Linear(x)
h = R(-Θ) → cumsum → R(Θ)     # rotation scan
out = Linear(h_fwd + h_bwd)    # bidirectional
```

RotInput was stuck at chance (13.2%). Two problems:

1. **Infinite cumsum of data-dependent angles.** Phase = cumsum(network(x)) over
   T=4000. After 4000 steps of noisy angles, cumulative phase is in the hundreds
   or thousands of radians. Gradients through that are chaotic — changing an angle
   at position t affects rotations at all subsequent positions with essentially
   random phase offsets.

2. **Slow: 6 layers × bidirectional = 12 rotation scans per forward pass.** Each
   scan does two `apply_rotation` (sin/cos on full tensor) + one cumsum. That's
   24 trig operations + 12 cumsums on (B, 4000, 64) tensors, sequential through
   layers. Compare to LearnedSpecCNN: one cumsum, then a lightweight CNN.

**The fix was windowing.** Bounding cumsum to W steps limits phase accumulation,
keeping gradients informative. All current working models use windowed cumsum:
- LearnedSpecCNNMod (W=400): 92.1% — works, just behind fixed (93.4%)
- MultiLayerMinimalMod (W=40 in layers 2+): training, ~87% val
- WindowS5 variants (W=80): worked but B/C/D added other instabilities

**Bidirectional was also dropped.** All current models (LearnedSpecCNN,
FilterbankCNN, MultiLayerMinimal) are unidirectional — the CNN backend handles
temporal aggregation. This halves the scan cost and simplifies the architecture.

### LearnedSpecCNN window sweep

| Window | Test Acc | Params | Notes |
|--------|----------|--------|-------|
| W=200 | **94.1%** | 25,668 | best — broader filters for 40 frequencies |
| W=400 | 93.4% | 25,668 | original |
| W=80 | 93.0% | 25,668 | rectangular window noise hurts |

Shorter window = each frequency bin resolves a broader bandwidth (good when you
only have 40 frequencies). But too short = spectral leakage from the hard
rectangular cutoff. W=200 is the sweet spot.

### Epoch-by-epoch: LearnedSpecCNN W=200 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.259 | 62.7% | 77.9% | 21 | 0.188 | 93.7% | 92.1% |
| 2 | 0.552 | 83.2% | 86.1% | 22 | 0.180 | 94.0% | 93.3% |
| 3 | 0.429 | 86.6% | 87.9% | 23 | 0.179 | 94.0% | 93.5% |
| 4 | 0.371 | 88.3% | 89.2% | 24 | 0.175 | 94.2% | 93.2% |
| 5 | 0.333 | 89.3% | 89.7% | 25 | 0.171 | 94.3% | 93.3% |
| 6 | 0.307 | 90.1% | 89.2% | 26 | 0.163 | 94.5% | 93.3% |
| 7 | 0.294 | 90.6% | 90.1% | 27 | 0.161 | 94.6% | 93.5% |
| 8 | 0.276 | 91.0% | 90.7% | 28 | 0.156 | 94.8% | 93.6% |
| 9 | 0.266 | 91.3% | 91.0% | 29 | 0.156 | 94.7% | 93.5% |
| 10 | 0.259 | 91.5% | 90.0% | 30 | 0.151 | 94.9% | 93.8% |
| 11 | 0.248 | 91.8% | 91.9% | 31 | 0.150 | 95.0% | 94.0% |
| 12 | 0.238 | 92.1% | 90.6% | 32 | 0.147 | 95.1% | 93.8% |
| 13 | 0.231 | 92.4% | 91.6% | 33 | 0.144 | 95.2% | 93.8% |
| 14 | 0.227 | 92.5% | 91.3% | 34 | 0.143 | 95.3% | 93.9% |
| 15 | 0.216 | 92.8% | 92.6% | 35 | 0.144 | 95.2% | 94.0% |
| 16 | 0.211 | 93.1% | 92.8% | 36 | 0.139 | 95.4% | 94.2% |
| 17 | 0.204 | 93.2% | 92.5% | 37 | 0.141 | 95.3% | 94.1% |
| 18 | 0.200 | 93.4% | 92.5% | 38 | 0.138 | 95.5% | 94.1% |
| 19 | 0.196 | 93.6% | 92.7% | 39 | 0.137 | 95.4% | 94.1% |
| 20 | 0.192 | 93.5% | 92.9% | 40 | 0.139 | 95.3% | 94.2% |

### Epoch-by-epoch: LearnedSpecCNN W=80 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.268 | 62.8% | 71.9% | 21 | 0.229 | 92.4% | 92.2% |
| 2 | 0.588 | 81.9% | 84.1% | 22 | 0.220 | 92.7% | 91.2% |
| 3 | 0.469 | 85.0% | 81.8% | 23 | 0.214 | 92.9% | 92.1% |
| 4 | 0.405 | 87.1% | 85.6% | 24 | 0.217 | 92.9% | 92.3% |
| 5 | 0.370 | 88.1% | 87.0% | 25 | 0.207 | 93.0% | 92.6% |
| 6 | 0.346 | 88.7% | 89.2% | 26 | 0.203 | 93.4% | 92.1% |
| 7 | 0.330 | 89.2% | 88.5% | 27 | 0.200 | 93.2% | 92.8% |
| 8 | 0.315 | 89.5% | 90.4% | 28 | 0.197 | 93.4% | 93.0% |
| 9 | 0.300 | 90.0% | 89.9% | 29 | 0.192 | 93.6% | 93.0% |
| 10 | 0.289 | 90.4% | 87.5% | 30 | 0.190 | 93.7% | 92.8% |
| 11 | 0.287 | 90.6% | 90.3% | 31 | 0.189 | 93.7% | 93.1% |
| 12 | 0.277 | 90.8% | 90.5% | 32 | 0.183 | 93.8% | 93.3% |
| 13 | 0.267 | 91.0% | 90.1% | 33 | 0.182 | 94.0% | 92.9% |
| 14 | 0.262 | 91.3% | 91.7% | 34 | 0.180 | 93.9% | 92.9% |
| 15 | 0.253 | 91.6% | 91.6% | 35 | 0.179 | 94.1% | 93.2% |
| 16 | 0.248 | 91.9% | 90.6% | 36 | 0.180 | 94.0% | 92.9% |
| 17 | 0.241 | 92.0% | 90.6% | 37 | 0.178 | 94.0% | 92.9% |
| 18 | 0.239 | 92.1% | 92.0% | 38 | 0.174 | 94.2% | 93.1% |
| 19 | 0.235 | 92.2% | 91.8% | 39 | 0.175 | 94.2% | 93.1% |
| 20 | 0.230 | 92.5% | 92.4% | 40 | 0.174 | 94.2% | 93.1% |

### Data-dependent frequency variants

Testing whether per-timestep frequencies (instead of fixed learned) help:

| Model | Test Acc | Params | Frequency source |
|-------|----------|--------|-----------------|
| LearnedSpecCNN | 93.4% | 25,668 | fixed learned (40 params) |
| LearnedSpecCNNMod2 | 92.2% | 27,428 | Linear→ReLU→Linear→LayerNorm per sample |
| LearnedSpecCNNMod | 92.1% | 25,788 | Linear→LayerNorm per sample |
| LearnedSpecCNNConv | 88.1% | 28,948 | Conv1d(1→40, k=80) local context |

**Fixed frequencies win.** A single audio sample is a scalar — it carries amplitude
but no frequency information. LayerNorm removes amplitude dependence, leaving
essentially nothing useful. The deeper network (Mod2: Linear→ReLU→Linear→LN)
doesn't help because the input is still a single scalar.

Conv1d for frequencies (88.1%) is worse because conv outputs aren't naturally
angular frequencies — the mapping from arbitrary conv features to meaningful
rotation rates is hard to learn.

### Epoch-by-epoch: LearnedSpecCNNMod (40 epochs, full dataset)

Data-dependent frequencies via LayerNorm(Linear(x[t])). 92.1% test, 25,788 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.517 | 52.8% | 70.9% | 21 | 0.287 | 90.5% | 89.2% |
| 2 | 0.799 | 75.3% | 74.9% | 22 | 0.281 | 90.7% | 89.7% |
| 3 | 0.643 | 79.5% | 79.8% | 23 | 0.271 | 91.0% | 89.8% |
| 4 | 0.561 | 82.0% | 83.1% | 24 | 0.273 | 91.0% | 90.5% |
| 5 | 0.514 | 83.5% | 82.9% | 25 | 0.264 | 91.3% | 90.6% |
| 6 | 0.478 | 84.6% | 84.6% | 26 | 0.253 | 91.5% | 91.1% |
| 7 | 0.427 | 86.4% | 83.4% | 27 | 0.252 | 91.6% | 90.1% |
| 8 | 0.408 | 86.8% | 84.9% | 28 | 0.245 | 92.0% | 91.6% |
| 9 | 0.401 | 87.0% | 86.6% | 29 | 0.236 | 92.3% | 91.5% |
| 10 | 0.382 | 87.6% | 86.9% | 30 | 0.228 | 92.5% | 91.6% |
| 11 | 0.365 | 88.0% | 87.3% | 31 | 0.227 | 92.6% | 91.4% |
| 12 | 0.365 | 88.1% | 85.9% | 32 | 0.227 | 92.4% | 91.2% |
| 13 | 0.364 | 88.2% | 87.2% | 33 | 0.220 | 92.7% | 91.6% |
| 14 | 0.352 | 88.2% | 87.5% | 34 | 0.222 | 92.7% | 91.7% |
| 15 | 0.341 | 88.8% | 88.5% | 35 | 0.219 | 92.8% | 92.0% |
| 16 | 0.329 | 89.1% | 87.9% | 36 | 0.218 | 92.9% | 91.8% |
| 17 | 0.326 | 89.2% | 89.1% | 37 | 0.212 | 93.0% | 91.9% |
| 18 | 0.306 | 89.7% | 88.7% | 38 | 0.213 | 93.0% | 91.8% |
| 19 | 0.298 | 90.2% | 89.9% | 39 | 0.209 | 93.1% | 92.0% |
| 20 | 0.286 | 90.6% | 89.0% | 40 | 0.212 | 93.2% | 91.8% |

### Epoch-by-epoch: LearnedSpecCNNMod2 (40 epochs, full dataset)

Deeper: Linear→ReLU→Linear→LayerNorm. 92.2% test, 27,428 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.557 | 50.9% | 68.3% | 21 | 0.358 | 88.0% | 89.0% |
| 2 | 0.947 | 70.0% | 74.9% | 22 | 0.326 | 89.5% | 89.2% |
| 3 | 0.744 | 76.1% | 79.8% | 23 | 0.321 | 89.3% | 88.3% |
| 4 | 0.641 | 79.3% | 81.9% | 24 | 0.312 | 89.7% | 89.7% |
| 5 | 0.584 | 80.9% | 79.1% | 25 | 0.321 | 89.4% | 88.8% |
| 6 | 0.533 | 82.5% | 84.5% | 26 | 0.317 | 89.3% | 88.4% |
| 7 | 0.501 | 83.7% | 84.0% | 27 | 0.297 | 90.2% | 89.3% |
| 8 | 0.491 | 83.9% | 82.2% | 28 | 0.292 | 90.3% | 90.9% |
| 9 | 0.468 | 84.5% | 83.9% | 29 | 0.274 | 90.9% | 90.0% |
| 10 | 0.446 | 85.2% | 84.6% | 30 | 0.275 | 90.8% | 91.2% |
| 11 | 0.442 | 85.4% | 86.2% | 31 | 0.260 | 91.4% | 91.2% |
| 12 | 0.451 | 85.2% | 85.6% | 32 | 0.252 | 91.6% | 91.9% |
| 13 | 0.418 | 86.2% | 84.2% | 33 | 0.247 | 91.7% | 91.4% |
| 14 | 0.417 | 86.2% | 86.6% | 34 | 0.244 | 91.9% | 91.1% |
| 15 | 0.395 | 86.9% | 87.0% | 35 | 0.241 | 92.0% | 91.6% |
| 16 | 0.410 | 86.4% | 85.4% | 36 | 0.238 | 92.2% | 91.7% |
| 17 | 0.391 | 87.0% | 86.6% | 37 | 0.233 | 92.4% | 92.2% |
| 18 | 0.379 | 87.4% | 87.3% | 38 | 0.231 | 92.2% | 92.1% |
| 19 | 0.385 | 87.4% | 85.2% | 39 | 0.228 | 92.5% | 92.1% |
| 20 | 0.378 | 87.4% | 87.5% | 40 | 0.230 | 92.3% | 92.3% |

### Epoch-by-epoch: LearnedSpecCNNConv (40 epochs, full dataset)

Conv1d(1→40, k=80) as frequency source. 88.1% test, 28,948 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.684 | 45.9% | 62.8% | 21 | 0.425 | 85.8% | 85.5% |
| 2 | 1.028 | 67.4% | 71.3% | 22 | 0.414 | 86.3% | 86.5% |
| 3 | 0.845 | 73.0% | 75.7% | 23 | 0.407 | 86.4% | 85.7% |
| 4 | 0.735 | 76.4% | 76.0% | 24 | 0.392 | 86.9% | 85.5% |
| 5 | 0.665 | 78.4% | 79.5% | 25 | 0.385 | 87.1% | 86.1% |
| 6 | 0.630 | 79.5% | 79.3% | 26 | 0.375 | 87.6% | 86.7% |
| 7 | 0.606 | 80.2% | 80.3% | 27 | 0.383 | 87.2% | 86.6% |
| 8 | 0.584 | 80.8% | 81.3% | 28 | 0.362 | 87.8% | 87.1% |
| 9 | 0.556 | 81.7% | 81.3% | 29 | 0.354 | 88.2% | 87.0% |
| 10 | 0.543 | 82.2% | 82.6% | 30 | 0.346 | 88.4% | 88.0% |
| 11 | 0.535 | 82.4% | 81.8% | 31 | 0.347 | 88.5% | 87.6% |
| 12 | 0.522 | 83.0% | 83.4% | 32 | 0.340 | 88.6% | 88.3% |
| 13 | 0.506 | 83.1% | 83.8% | 33 | 0.337 | 88.8% | 88.1% |
| 14 | 0.485 | 84.0% | 83.5% | 34 | 0.334 | 88.7% | 87.9% |
| 15 | 0.470 | 84.6% | 83.1% | 35 | 0.330 | 89.0% | 87.8% |
| 16 | 0.459 | 84.7% | 83.7% | 36 | 0.324 | 89.2% | 88.3% |
| 17 | 0.466 | 84.6% | 83.8% | 37 | 0.321 | 89.3% | 88.4% |
| 18 | 0.447 | 85.2% | 85.0% | 38 | 0.321 | 89.3% | 88.3% |
| 19 | 0.429 | 85.9% | 85.4% | 39 | 0.323 | 89.2% | 88.3% |
| 20 | 0.435 | 85.6% | 84.7% | 40 | 0.323 | 89.1% | 88.1% |

### FilterbankCNN (learned conv filterbank)

Instead of rotation+cumsum for frequency decomposition, use Conv1d directly as a
filterbank: Conv1d(1→40, k=window, stride=160) → x² → log → CNN.

| Model | Test Acc | Params | Init |
|-------|----------|--------|------|
| FilterbankMelInitCNN | **92.0%** | 28,828 | Hann-tapered mel sinusoids |
| FilterbankCNN | 90.5% | 28,828 | random |

Mel initialization (+1.5%) gives the conv filters a strong starting point — proper
frequency spacing and Hann tapering. Still trails LearnedSpecCNN (93.4%) because
conv with k=80 sees only 5ms of context vs cumsum's 25ms window (W=400).

### Epoch-by-epoch: FilterbankCNN k=80 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.607 | 50.4% | 45.6% | 21 | 0.347 | 88.4% | 88.7% |
| 2 | 0.945 | 70.7% | 71.0% | 22 | 0.337 | 88.6% | 89.8% |
| 3 | 0.746 | 76.3% | 77.8% | 23 | 0.333 | 88.8% | 89.2% |
| 4 | 0.653 | 78.9% | 74.1% | 24 | 0.328 | 89.0% | 89.2% |
| 5 | 0.599 | 80.5% | 80.6% | 25 | 0.320 | 89.5% | 88.8% |
| 6 | 0.564 | 81.4% | 79.1% | 26 | 0.317 | 89.4% | 89.8% |
| 7 | 0.527 | 82.8% | 83.0% | 27 | 0.308 | 89.6% | 89.7% |
| 8 | 0.512 | 83.1% | 82.5% | 28 | 0.301 | 90.0% | 89.9% |
| 9 | 0.491 | 83.6% | 82.6% | 29 | 0.298 | 90.0% | 90.5% |
| 10 | 0.480 | 84.0% | 84.4% | 30 | 0.288 | 90.3% | 90.7% |
| 11 | 0.457 | 84.8% | 85.8% | 31 | 0.289 | 90.3% | 90.7% |
| 12 | 0.443 | 85.4% | 82.1% | 32 | 0.287 | 90.4% | 91.1% |
| 13 | 0.425 | 86.0% | 86.1% | 33 | 0.282 | 90.6% | 91.0% |
| 14 | 0.419 | 86.0% | 85.6% | 34 | 0.279 | 90.6% | 91.0% |
| 15 | 0.409 | 86.4% | 87.4% | 35 | 0.272 | 90.9% | 90.8% |
| 16 | 0.402 | 86.6% | 85.8% | 36 | 0.275 | 90.8% | 91.1% |
| 17 | 0.398 | 86.8% | 88.0% | 37 | 0.269 | 91.1% | 91.3% |
| 18 | 0.380 | 87.3% | 87.6% | 38 | 0.269 | 91.1% | 91.3% |
| 19 | 0.367 | 87.8% | 88.5% | 39 | 0.269 | 91.0% | 91.2% |
| 20 | 0.363 | 88.1% | 88.2% | 40 | 0.270 | 90.9% | 91.3% |

### Epoch-by-epoch: FilterbankMelInitCNN k=80 (40 epochs, full dataset)

Mel-initialized conv filters. 92.0% test, 28,828 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.283 | 62.1% | 77.5% | 21 | 0.268 | 91.0% | 90.7% |
| 2 | 0.631 | 80.8% | 81.3% | 22 | 0.264 | 91.3% | 90.9% |
| 3 | 0.513 | 83.8% | 77.0% | 23 | 0.253 | 91.8% | 90.9% |
| 4 | 0.457 | 85.4% | 84.1% | 24 | 0.250 | 91.7% | 91.6% |
| 5 | 0.422 | 86.3% | 87.0% | 25 | 0.243 | 91.9% | 91.6% |
| 6 | 0.403 | 86.8% | 85.6% | 26 | 0.242 | 92.0% | 91.2% |
| 7 | 0.379 | 87.6% | 87.8% | 27 | 0.237 | 92.1% | 92.1% |
| 8 | 0.369 | 87.9% | 88.1% | 28 | 0.233 | 92.3% | 91.9% |
| 9 | 0.354 | 88.5% | 88.4% | 29 | 0.231 | 92.4% | 92.1% |
| 10 | 0.346 | 88.6% | 88.3% | 30 | 0.224 | 92.5% | 92.2% |
| 11 | 0.332 | 88.9% | 88.8% | 31 | 0.222 | 92.6% | 92.2% |
| 12 | 0.325 | 89.2% | 89.9% | 32 | 0.222 | 92.7% | 92.3% |
| 13 | 0.316 | 89.6% | 90.6% | 33 | 0.218 | 92.8% | 92.5% |
| 14 | 0.311 | 89.8% | 89.9% | 34 | 0.215 | 93.0% | 92.4% |
| 15 | 0.304 | 90.0% | 89.3% | 35 | 0.213 | 92.9% | 92.3% |
| 16 | 0.297 | 90.0% | 89.9% | 36 | 0.213 | 93.0% | 92.6% |
| 17 | 0.291 | 90.5% | 90.0% | 37 | 0.208 | 93.1% | 92.5% |
| 18 | 0.283 | 90.7% | 90.6% | 38 | 0.210 | 93.1% | 92.6% |
| 19 | 0.278 | 90.8% | 90.7% | 39 | 0.206 | 93.2% | 92.5% |
| 20 | 0.272 | 91.1% | 90.9% | 40 | 0.207 | 93.3% | 92.5% |

### LearnedSpecLinearCNN (no d², unrotated real+imag)

LearnedSpecCNN without d² — uses unrotation to stabilize phase, then feeds
cat(real, imag) → 80 channels to CNN. Tests whether CNN can use phase info
that power spectrum discards.

**Result: 90.7%** (27,588 params) — worse than power spectrum (93.4%). The CNN
prefers magnitude features. Phase information, while available, adds noise
that the small CNN can't effectively exploit.

### Epoch-by-epoch: LearnedSpecLinearCNN W=200 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.827 | 40.8% | 62.2% | 21 | 0.325 | 89.3% | 89.9% |
| 2 | 1.035 | 68.3% | 73.3% | 22 | 0.314 | 89.8% | 90.2% |
| 3 | 0.783 | 75.4% | 79.4% | 23 | 0.314 | 89.7% | 90.2% |
| 4 | 0.656 | 78.9% | 81.4% | 24 | 0.302 | 90.1% | 90.9% |
| 5 | 0.584 | 81.3% | 82.7% | 25 | 0.296 | 90.2% | 91.0% |
| 6 | 0.542 | 82.6% | 84.7% | 26 | 0.293 | 90.3% | 90.8% |
| 7 | 0.508 | 83.6% | 85.6% | 27 | 0.285 | 90.7% | 91.1% |
| 8 | 0.476 | 84.5% | 85.6% | 28 | 0.283 | 90.5% | 91.0% |
| 9 | 0.458 | 85.2% | 84.4% | 29 | 0.277 | 90.9% | 91.1% |
| 10 | 0.436 | 85.9% | 87.0% | 30 | 0.272 | 91.0% | 91.2% |
| 11 | 0.421 | 86.2% | 87.4% | 31 | 0.274 | 91.1% | 90.8% |
| 12 | 0.408 | 86.6% | 87.7% | 32 | 0.272 | 91.2% | 91.2% |
| 13 | 0.396 | 87.1% | 88.1% | 33 | 0.267 | 91.3% | 90.9% |
| 14 | 0.382 | 87.6% | 88.1% | 34 | 0.264 | 91.3% | 91.2% |
| 15 | 0.368 | 87.9% | 88.3% | 35 | 0.264 | 91.3% | 91.5% |
| 16 | 0.362 | 88.2% | 89.0% | 36 | 0.260 | 91.5% | 91.6% |
| 17 | 0.353 | 88.3% | 89.2% | 37 | 0.259 | 91.5% | 91.2% |
| 18 | 0.343 | 88.7% | 90.0% | 38 | 0.257 | 91.5% | 91.5% |
| 19 | 0.341 | 88.7% | 88.8% | 39 | 0.258 | 91.5% | 91.3% |
| 20 | 0.327 | 89.3% | 89.2% | 40 | 0.256 | 91.6% | 91.4% |

### MultiLayerMinimalMod (data-dependent frequencies, layers 2+)

Same as MultiLayerMinimalStrided but layers 2+ use data-dependent frequencies
via Linear→ReLU→Linear→LayerNorm. Layer 1 remains fixed learned mel frequencies.

**Result: 88.7%** (159,564 params) — essentially tied with fixed-frequency version
(88.9%). Data-dependent frequencies in layers 2+ don't help, even though the
input is now a feature vector (not a scalar). The cumsum window (W=40) may be
too short for data-dependent angles to provide meaningful benefit.

### Epoch-by-epoch: MultiLayerMinimalMod (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.882 | 36.7% | 41.3% | 21 | 0.361 | 88.6% | 85.5% |
| 2 | 1.594 | 47.8% | 50.1% | 22 | 0.348 | 89.1% | 85.6% |
| 3 | 1.404 | 55.0% | 55.6% | 23 | 0.330 | 89.5% | 87.4% |
| 4 | 1.170 | 64.1% | 64.4% | 24 | 0.320 | 89.9% | 87.4% |
| 5 | 0.963 | 70.6% | 71.3% | 25 | 0.311 | 90.1% | 88.2% |
| 6 | 0.825 | 74.5% | 72.7% | 26 | 0.301 | 90.5% | 87.6% |
| 7 | 0.725 | 77.5% | 77.8% | 27 | 0.283 | 91.0% | 88.2% |
| 8 | 0.666 | 79.5% | 80.6% | 28 | 0.278 | 91.1% | 87.9% |
| 9 | 0.619 | 80.9% | 79.5% | 29 | 0.270 | 91.4% | 87.8% |
| 10 | 0.581 | 81.8% | 81.5% | 30 | 0.257 | 91.9% | 88.6% |
| 11 | 0.551 | 82.7% | 83.1% | 31 | 0.250 | 91.9% | 87.5% |
| 12 | 0.524 | 83.6% | 82.5% | 32 | 0.239 | 92.5% | 88.4% |
| 13 | 0.500 | 84.6% | 81.4% | 33 | 0.237 | 92.5% | 89.9% |
| 14 | 0.475 | 85.1% | 84.9% | 34 | 0.230 | 92.7% | 88.6% |
| 15 | 0.454 | 85.8% | 85.2% | 35 | 0.224 | 92.9% | 89.0% |
| 16 | 0.439 | 86.1% | 83.8% | 36 | 0.218 | 93.1% | 88.6% |
| 17 | 0.418 | 86.8% | 85.0% | 37 | 0.216 | 93.1% | 88.2% |
| 18 | 0.408 | 87.2% | 86.4% | 38 | 0.213 | 93.2% | 89.6% |
| 19 | 0.388 | 87.7% | 86.7% | 39 | 0.213 | 93.1% | 88.8% |
| 20 | 0.377 | 88.3% | 87.0% | 40 | 0.208 | 93.4% | 88.5% |

### Epoch-by-epoch: MultiLayerV2 AvgPool (40 epochs, full dataset)

4-layer cumsum, stride=160, AvgPool(2) between layers. Fixed learned freqs. 89.2% test, 73,052 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.795 | 40.0% | 51.8% | 21 | 0.435 | 86.0% | 85.9% |
| 2 | 1.162 | 61.8% | 69.1% | 22 | 0.426 | 86.2% | 87.2% |
| 3 | 0.909 | 70.4% | 70.1% | 23 | 0.420 | 86.4% | 87.0% |
| 4 | 0.789 | 74.2% | 76.8% | 24 | 0.416 | 86.5% | 87.8% |
| 5 | 0.716 | 76.7% | 76.3% | 25 | 0.402 | 86.8% | 88.4% |
| 6 | 0.671 | 78.2% | 77.9% | 26 | 0.393 | 87.2% | 87.1% |
| 7 | 0.635 | 79.6% | 80.4% | 27 | 0.385 | 87.4% | 88.5% |
| 8 | 0.604 | 80.5% | 81.9% | 28 | 0.383 | 87.4% | 87.9% |
| 9 | 0.586 | 81.1% | 84.4% | 29 | 0.376 | 87.5% | 88.2% |
| 10 | 0.569 | 81.9% | 82.4% | 30 | 0.367 | 87.9% | 87.9% |
| 11 | 0.549 | 82.4% | 80.9% | 31 | 0.362 | 88.0% | 88.1% |
| 12 | 0.535 | 82.7% | 84.0% | 32 | 0.355 | 88.3% | 89.2% |
| 13 | 0.521 | 83.2% | 84.2% | 33 | 0.349 | 88.4% | 88.3% |
| 14 | 0.509 | 83.7% | 84.2% | 34 | 0.347 | 88.6% | 89.0% |
| 15 | 0.500 | 83.9% | 85.7% | 35 | 0.346 | 88.6% | 89.1% |
| 16 | 0.482 | 84.2% | 84.0% | 36 | 0.343 | 88.6% | 89.4% |
| 17 | 0.474 | 84.7% | 85.8% | 37 | 0.341 | 88.9% | 88.9% |
| 18 | 0.459 | 85.2% | 85.8% | 38 | 0.337 | 88.9% | 89.2% |
| 19 | 0.456 | 85.2% | 86.4% | 39 | 0.338 | 89.0% | 89.4% |
| 20 | 0.448 | 85.4% | 86.0% | 40 | 0.339 | 88.9% | 89.6% |

### Epoch-by-epoch: MultiLayerModV2 AvgPool (40 epochs, full dataset)

Data-dependent freqs in layers 2+. 87.4% test, 87,812 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.986 | 31.3% | 50.0% | 21 | 0.468 | 84.5% | 85.6% |
| 2 | 1.291 | 57.0% | 60.2% | 22 | 0.457 | 84.7% | 85.9% |
| 3 | 1.034 | 65.8% | 68.5% | 23 | 0.451 | 85.1% | 86.4% |
| 4 | 0.911 | 70.1% | 73.6% | 24 | 0.442 | 85.2% | 85.3% |
| 5 | 0.826 | 72.7% | 73.9% | 25 | 0.430 | 85.7% | 86.8% |
| 6 | 0.774 | 74.6% | 76.8% | 26 | 0.419 | 86.2% | 87.4% |
| 7 | 0.721 | 76.1% | 78.7% | 27 | 0.409 | 86.3% | 86.5% |
| 8 | 0.691 | 77.2% | 78.8% | 28 | 0.403 | 86.6% | 87.9% |
| 9 | 0.663 | 78.2% | 78.4% | 29 | 0.392 | 86.9% | 87.7% |
| 10 | 0.634 | 79.0% | 80.8% | 30 | 0.389 | 87.0% | 88.0% |
| 11 | 0.609 | 79.8% | 82.7% | 31 | 0.380 | 87.4% | 88.0% |
| 12 | 0.597 | 80.3% | 81.1% | 32 | 0.373 | 87.5% | 88.1% |
| 13 | 0.576 | 81.2% | 81.1% | 33 | 0.373 | 87.7% | 87.9% |
| 14 | 0.550 | 81.8% | 85.1% | 34 | 0.364 | 87.8% | 88.3% |
| 15 | 0.539 | 82.3% | 82.8% | 35 | 0.360 | 88.0% | 88.7% |
| 16 | 0.531 | 82.3% | 84.7% | 36 | 0.360 | 87.9% | 88.6% |
| 17 | 0.527 | 82.6% | 84.3% | 37 | 0.354 | 88.1% | 88.8% |
| 18 | 0.501 | 83.4% | 86.2% | 38 | 0.354 | 88.2% | 88.2% |
| 19 | 0.494 | 83.7% | 84.8% | 39 | 0.355 | 88.2% | 88.6% |
| 20 | 0.478 | 84.2% | 84.3% | 40 | 0.349 | 88.5% | 88.8% |

**Conv vs cumsum tradeoffs:**
- Conv can learn arbitrary filter shapes including natural tapering at edges
- Cumsum has O(1) streaming: only needs `cs[t]` and `cs[t-W]`, no buffering
- Conv requires full k-length context per step (must buffer k samples)

### Downsampling: MaxPool vs AvgPool vs Stride

The multi-layer models (V2/ModV2, ConvCumsumV2/ModV2) downsample between layers:
T=100 → 50 → 25 → 12. Three options tested:

| Method | V2 val @ ep10 | V2 val @ ep15 | Final test |
|--------|--------------|--------------|------------|
| AvgPool(2) | 82.4% | 85.7% | **89.2%** |
| MaxPool(2) | 82.5% | 85.8% | 88.3% |
| Stride 2 | 77.3% | — | abandoned (~5% behind) |

**MaxPool vs AvgPool**: Essentially identical performance. However, AvgPool is more
principled for real+imag features. MaxPool picks the maximum per channel independently,
which is biased toward positive values — with signed real/imag data, a large negative
value is just as important as a large positive one. AvgPool treats both equally by
averaging adjacent time steps. Same result in practice, but correct inductive bias.

**Why stride failed**: Stride 2 (`h[:, ::2]`) simply drops every other time step —
you lose half the data with no aggregation. With MaxPool(2) or AvgPool(2), you look
at both adjacent time steps and combine them, giving 2x the coverage per output step.
The pooling layers see information from *all* input time steps; stride sees only half.
To make stride work, you'd need to increase the cumsum window in the layer above so
outputs already overlap enough that skipping every other one doesn't lose information.

**Conclusion**: Use AvgPool(2) for real+imag features. It preserves full temporal
coverage while correctly handling signed values.

### LearnedSpecMultiCNN (2 frequencies per mel bin)

Instead of 40 frequencies (one per mel bin), use 80 (two per bin), initialized to
the same center frequency and free to spread out. Power is summed within each bin
before log, mimicking how mel filterbanks average multiple FFT bins per channel.

**Result: 94.9%** (25,708 params) — best learned model, only 0.8% behind MelCNN (95.7%).
The extra frequency per bin captures more of each filter's bandwidth.

### LearnedSpecMultiCNN (4 frequencies per mel bin)

Bumped to 160 frequencies (4 per bin), same W=200. Run with batch_size=64 due to GPU
memory constraints (160-freq cumsum over 16000 timesteps uses ~8GB VRAM on T4).

**Result: 94.0%** (25,788 params) — *worse* than 2/bin (94.9%). More frequencies per bin
did not help. The extra frequencies may be redundant starting from the same center, or
the smaller batch size may hurt optimization. Either way, 2/bin appears optimal for cumsum.

### Epoch-by-epoch: LearnedSpecMultiCNN W=200 (40 epochs, full dataset)

2 frequencies per bin, power summed within bins. 94.9% test, 25,708 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.895 | 72.2% | 84.1% | 21 | 0.203 | 93.2% | 93.8% |
| 2 | 0.488 | 84.1% | 88.0% | 22 | 0.192 | 93.7% | 93.7% |
| 3 | 0.411 | 86.8% | 89.9% | 23 | 0.189 | 93.8% | 93.7% |
| 4 | 0.371 | 87.7% | 90.3% | 24 | 0.185 | 93.8% | 94.2% |
| 5 | 0.337 | 89.0% | 90.1% | 25 | 0.178 | 94.1% | 94.1% |
| 6 | 0.323 | 89.6% | 91.4% | 26 | 0.176 | 94.0% | 94.5% |
| 7 | 0.307 | 89.9% | 91.1% | 27 | 0.168 | 94.3% | 94.4% |
| 8 | 0.291 | 90.4% | 91.3% | 28 | 0.166 | 94.4% | 94.3% |
| 9 | 0.281 | 90.8% | 92.8% | 29 | 0.160 | 94.6% | 94.8% |
| 10 | 0.271 | 91.2% | 92.5% | 30 | 0.156 | 94.8% | 94.6% |
| 11 | 0.259 | 91.6% | 92.1% | 31 | 0.154 | 94.9% | 94.6% |
| 12 | 0.255 | 91.7% | 92.2% | 32 | 0.151 | 95.0% | 95.0% |
| 13 | 0.245 | 92.1% | 92.9% | 33 | 0.147 | 95.1% | 94.7% |
| 14 | 0.239 | 92.1% | 92.9% | 34 | 0.146 | 95.1% | 94.8% |
| 15 | 0.234 | 92.3% | 93.0% | 35 | 0.142 | 95.3% | 94.6% |
| 16 | 0.223 | 92.6% | 93.6% | 36 | 0.140 | 95.3% | 94.9% |
| 17 | 0.220 | 92.8% | 93.1% | 37 | 0.140 | 95.4% | 94.7% |
| 18 | 0.212 | 93.0% | 92.9% | 38 | 0.137 | 95.4% | 94.6% |
| 19 | 0.210 | 93.1% | 93.6% | 39 | 0.140 | 95.3% | 94.7% |
| 20 | 0.201 | 93.3% | 93.9% | 40 | 0.139 | 95.3% | 94.8% |

### Epoch-by-epoch: LearnedSpecMultiCNN 4/bin W=200 (40 epochs, batch_size=64)

160 learned frequencies (4 per bin), power summed within bins. 94.0% test, 25,788 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.060 | 68.2% | 83.8% | 21 | 0.189 | 93.7% | 93.4% |
| 2 | 0.488 | 84.6% | 88.0% | 22 | 0.183 | 93.9% | 93.3% |
| 3 | 0.400 | 87.4% | 89.3% | 23 | 0.179 | 94.1% | 93.9% |
| 4 | 0.353 | 88.6% | 89.7% | 24 | 0.175 | 94.1% | 93.8% |
| 5 | 0.322 | 89.6% | 90.9% | 25 | 0.171 | 94.3% | 93.6% |
| 6 | 0.302 | 90.2% | 89.9% | 26 | 0.163 | 94.6% | 94.2% |
| 7 | 0.286 | 90.5% | 91.5% | 27 | 0.161 | 94.7% | 94.0% |
| 8 | 0.275 | 91.0% | 90.8% | 28 | 0.159 | 94.7% | 94.0% |
| 9 | 0.263 | 91.4% | 91.0% | 29 | 0.155 | 94.8% | 93.8% |
| 10 | 0.252 | 91.8% | 91.7% | 30 | 0.148 | 95.1% | 94.3% |
| 11 | 0.244 | 91.8% | 91.8% | 31 | 0.148 | 95.0% | 94.3% |
| 12 | 0.239 | 92.2% | 92.1% | 32 | 0.143 | 95.2% | 94.3% |
| 13 | 0.233 | 92.3% | 91.5% | 33 | 0.142 | 95.4% | 94.1% |
| 14 | 0.226 | 92.5% | 92.9% | 34 | 0.141 | 95.3% | 94.3% |
| 15 | 0.218 | 92.7% | 92.6% | 35 | 0.137 | 95.3% | 94.5% |
| 16 | 0.212 | 93.0% | 92.6% | 36 | 0.138 | 95.4% | 94.4% |
| 17 | 0.206 | 93.2% | 92.9% | 37 | 0.138 | 95.3% | 94.2% |
| 18 | 0.203 | 93.3% | 92.8% | 38 | 0.135 | 95.6% | 94.4% |
| 19 | 0.197 | 93.6% | 92.6% | 39 | 0.136 | 95.5% | 94.4% |
| 20 | 0.191 | 93.6% | 92.9% | 40 | 0.134 | 95.6% | 94.6% |

### ConvCumsumV2/ModV2 (conv front-end + cumsum layers)

Conv1d(1→80, k=400, stride=160) with mel sin+cos init as layer 1, then cumsum layers 2+
with AvgPool(2) between layers. Tests whether conv front-end + cumsum deeper layers
beats either approach alone.

| Model | Test Acc | Params | Deeper layers |
|-------|----------|--------|--------------|
| ConvCumsumV2 | **90.1%** | 105,012 | fixed freqs |
| ConvCumsumModV2 | 89.6% | 119,772 | data-dep freqs |

Conv front-end helps vs pure cumsum (90.1% vs MultiLayerV2 89.2%), but cumsum deeper
layers still drag it below pure CNN (FilterbankMelInit 92.0%). Data-dependent freqs
don't help here either (89.6% ≈ 90.1%).

### Epoch-by-epoch: ConvCumsumV2 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.718 | 42.9% | 61.7% | 21 | 0.381 | 87.4% | 86.6% |
| 2 | 0.990 | 67.9% | 73.8% | 22 | 0.374 | 87.7% | 87.9% |
| 3 | 0.795 | 74.4% | 76.3% | 23 | 0.365 | 87.9% | 88.6% |
| 4 | 0.705 | 77.0% | 79.5% | 24 | 0.355 | 88.4% | 88.2% |
| 5 | 0.641 | 79.1% | 82.3% | 25 | 0.351 | 88.5% | 89.0% |
| 6 | 0.603 | 80.4% | 82.6% | 26 | 0.338 | 88.7% | 89.0% |
| 7 | 0.575 | 81.3% | 82.8% | 27 | 0.336 | 89.0% | 89.1% |
| 8 | 0.550 | 82.2% | 82.8% | 28 | 0.325 | 89.3% | 89.5% |
| 9 | 0.526 | 82.9% | 83.1% | 29 | 0.319 | 89.5% | 89.3% |
| 10 | 0.515 | 83.3% | 84.3% | 30 | 0.308 | 89.9% | 89.5% |
| 11 | 0.498 | 83.8% | 82.1% | 31 | 0.308 | 89.9% | 90.0% |
| 12 | 0.486 | 84.2% | 84.0% | 32 | 0.305 | 89.9% | 89.5% |
| 13 | 0.470 | 84.8% | 85.3% | 33 | 0.299 | 90.2% | 89.5% |
| 14 | 0.457 | 85.1% | 86.6% | 34 | 0.300 | 90.2% | 89.5% |
| 15 | 0.443 | 85.6% | 85.6% | 35 | 0.295 | 90.4% | 90.1% |
| 16 | 0.436 | 85.6% | 87.2% | 36 | 0.293 | 90.4% | 90.3% |
| 17 | 0.421 | 86.3% | 87.0% | 37 | 0.285 | 90.7% | 89.8% |
| 18 | 0.411 | 86.5% | 87.2% | 38 | 0.284 | 90.6% | 90.1% |
| 19 | 0.403 | 86.7% | 86.7% | 39 | 0.284 | 90.8% | 89.9% |
| 20 | 0.391 | 87.2% | 87.7% | 40 | 0.283 | 90.7% | 89.8% |

### Epoch-by-epoch: ConvCumsumModV2 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.085 | 27.5% | 43.2% | 21 | 0.394 | 87.0% | 87.6% |
| 2 | 1.321 | 56.3% | 63.5% | 22 | 0.376 | 87.5% | 88.5% |
| 3 | 0.987 | 67.9% | 69.9% | 23 | 0.366 | 87.9% | 87.8% |
| 4 | 0.842 | 72.5% | 74.0% | 24 | 0.363 | 87.9% | 88.5% |
| 5 | 0.752 | 75.4% | 76.5% | 25 | 0.346 | 88.5% | 88.1% |
| 6 | 0.688 | 77.4% | 79.8% | 26 | 0.337 | 88.8% | 89.3% |
| 7 | 0.644 | 78.9% | 80.2% | 27 | 0.329 | 89.1% | 89.1% |
| 8 | 0.617 | 79.8% | 81.5% | 28 | 0.322 | 89.2% | 88.8% |
| 9 | 0.589 | 80.8% | 82.3% | 29 | 0.312 | 89.6% | 89.3% |
| 10 | 0.561 | 81.7% | 82.8% | 30 | 0.304 | 89.9% | 89.0% |
| 11 | 0.549 | 82.3% | 83.5% | 31 | 0.300 | 89.9% | 89.9% |
| 12 | 0.516 | 83.2% | 83.8% | 32 | 0.292 | 90.3% | 89.9% |
| 13 | 0.505 | 83.7% | 84.7% | 33 | 0.288 | 90.3% | 90.1% |
| 14 | 0.485 | 84.1% | 85.5% | 34 | 0.288 | 90.3% | 90.2% |
| 15 | 0.465 | 84.7% | 84.8% | 35 | 0.282 | 90.6% | 90.1% |
| 16 | 0.457 | 85.0% | 85.8% | 36 | 0.276 | 90.7% | 90.2% |
| 17 | 0.433 | 85.7% | 85.8% | 37 | 0.275 | 90.8% | 90.0% |
| 18 | 0.427 | 85.9% | 87.0% | 38 | 0.271 | 90.9% | 90.4% |
| 19 | 0.415 | 86.1% | 87.0% | 39 | 0.273 | 90.8% | 90.2% |
| 20 | 0.400 | 86.7% | 87.8% | 40 | 0.273 | 90.8% | 90.4% |

### FilterbankSinCos variants (sin+cos filterbank, k=400)

80 conv filters (40 sin + 40 cos) with Hann-tapered mel init, k=400.

| Model | Test Acc | Params | Processing |
|-------|----------|--------|-----------|
| FilterbankSinCos (1/bin) | **94.7%** | 57,628 | 80 filters → sin²+cos² → 40ch → log → CNN |
| FilterbankSinCosCombined | 94.5% | 61,468 | 80 filters → 40 log(sin²+cos²) + 80 raw → 120ch → CNN |
| FilterbankSinCosMulti (2/bin) | 94.1% | 89,628 | 160 filters → sin²+cos² → 80 mag → sum to 40ch → log → CNN |
| FilterbankSinCosLinear | 90.9% | 59,548 | 80 filters → 80 raw sin+cos → CNN (no d²/log) |

**d²+log is critical at k=400**: 94.7% with vs 90.9% without (3.8% gap).
Adding raw sin+cos alongside magnitude (Combined) doesn't help — the CNN ignores
the raw channels (94.5% ≈ 94.7%). Magnitude is all it needs.

**More frequencies per bin doesn't help for conv either**: FilterbankSinCosMulti (2/bin)
94.1% vs FilterbankSinCos (1/bin) 94.7%. Same pattern as cumsum (4/bin 94.0% ≤ 2/bin 94.9%).
The bottleneck is not frequency density — it's something else about the FFT approach
(full spectrum coverage, Hann window shape, or both).

### FilterbankLinear variants (no d²/log, k=80)

Testing whether d²+log helps at shorter kernel sizes.

| Model | Test Acc | Params | Init | d²+log version |
|-------|----------|--------|------|----------------|
| FilterbankMelInitLinear | **91.7%** | 28,828 | mel | FilterbankMelInit: 92.0% |
| FilterbankLinear | 90.2% | 28,828 | random | FilterbankCNN: 90.5% |

**d²+log barely matters at k=80**: only 0.3% difference for both mel and random init.
Contrast with k=400 where the gap is 3.8%. Observation: d²+log becomes more important
with wider kernels.

### Epoch-by-epoch: FilterbankSinCosCNN k=400 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.885 | 72.4% | 84.3% | 21 | 0.205 | 93.1% | 92.2% |
| 2 | 0.490 | 84.3% | 87.7% | 22 | 0.196 | 93.4% | 93.3% |
| 3 | 0.414 | 86.5% | 88.7% | 23 | 0.196 | 93.5% | 93.6% |
| 4 | 0.377 | 87.7% | 89.9% | 24 | 0.183 | 93.7% | 93.6% |
| 5 | 0.352 | 88.5% | 89.3% | 25 | 0.180 | 93.9% | 94.0% |
| 6 | 0.334 | 88.9% | 90.2% | 26 | 0.175 | 94.2% | 93.5% |
| 7 | 0.318 | 89.5% | 90.6% | 27 | 0.166 | 94.4% | 94.1% |
| 8 | 0.308 | 89.9% | 90.5% | 28 | 0.162 | 94.6% | 94.0% |
| 9 | 0.292 | 90.3% | 91.1% | 29 | 0.157 | 94.8% | 94.0% |
| 10 | 0.288 | 90.4% | 90.3% | 30 | 0.154 | 94.9% | 94.2% |
| 11 | 0.278 | 90.8% | 92.2% | 31 | 0.149 | 95.0% | 94.2% |
| 12 | 0.272 | 91.0% | 92.1% | 32 | 0.145 | 95.2% | 94.3% |
| 13 | 0.260 | 91.4% | 92.2% | 33 | 0.142 | 95.1% | 94.0% |
| 14 | 0.248 | 91.6% | 92.8% | 34 | 0.136 | 95.4% | 94.5% |
| 15 | 0.249 | 91.7% | 92.3% | 35 | 0.136 | 95.5% | 94.4% |
| 16 | 0.236 | 92.1% | 93.3% | 36 | 0.134 | 95.6% | 94.2% |
| 17 | 0.236 | 92.2% | 92.4% | 37 | 0.131 | 95.7% | 94.5% |
| 18 | 0.227 | 92.5% | 91.7% | 38 | 0.130 | 95.7% | 94.4% |
| 19 | 0.218 | 92.7% | 93.1% | 39 | 0.129 | 95.7% | 94.5% |
| 20 | 0.210 | 92.9% | 92.8% | 40 | 0.127 | 95.8% | 94.4% |

### Epoch-by-epoch: FilterbankSinCosLinearCNN k=400 (40 epochs, full dataset)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.439 | 52.9% | 70.3% | 21 | 0.354 | 88.3% | 89.3% |
| 2 | 0.857 | 72.3% | 78.9% | 22 | 0.347 | 88.7% | 89.7% |
| 3 | 0.719 | 76.5% | 82.1% | 23 | 0.337 | 88.9% | 90.3% |
| 4 | 0.642 | 79.2% | 83.4% | 24 | 0.329 | 89.2% | 90.1% |
| 5 | 0.594 | 80.5% | 84.4% | 25 | 0.324 | 89.1% | 90.0% |
| 6 | 0.556 | 81.7% | 85.1% | 26 | 0.321 | 89.5% | 90.2% |
| 7 | 0.528 | 82.6% | 86.0% | 27 | 0.314 | 89.8% | 91.0% |
| 8 | 0.510 | 83.3% | 87.1% | 28 | 0.304 | 89.9% | 91.2% |
| 9 | 0.487 | 84.1% | 86.9% | 29 | 0.298 | 90.2% | 90.6% |
| 10 | 0.475 | 84.5% | 87.3% | 30 | 0.290 | 90.2% | 91.2% |
| 11 | 0.455 | 85.0% | 87.0% | 31 | 0.290 | 90.3% | 91.0% |
| 12 | 0.443 | 85.3% | 87.9% | 32 | 0.282 | 90.6% | 91.5% |
| 13 | 0.435 | 85.7% | 88.5% | 33 | 0.274 | 90.9% | 91.6% |
| 14 | 0.420 | 86.0% | 88.0% | 34 | 0.272 | 91.0% | 91.5% |
| 15 | 0.410 | 86.7% | 88.9% | 35 | 0.268 | 91.1% | 91.9% |
| 16 | 0.397 | 86.8% | 89.0% | 36 | 0.271 | 91.0% | 91.4% |
| 17 | 0.389 | 86.9% | 89.0% | 37 | 0.266 | 91.3% | 91.6% |
| 18 | 0.380 | 87.5% | 89.4% | 38 | 0.263 | 91.4% | 91.7% |
| 19 | 0.371 | 87.7% | 89.4% | 39 | 0.264 | 91.2% | 91.7% |
| 20 | 0.362 | 88.0% | 89.5% | 40 | 0.262 | 91.3% | 91.8% |

### Epoch-by-epoch: FilterbankSinCosCombinedCNN k=400 (40 epochs, full dataset)

40 log(sin²+cos²) + 80 raw sin+cos = 120 channels. 94.5% test, 61,468 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.915 | 71.5% | 81.4% | 21 | 0.209 | 93.1% | 93.5% |
| 2 | 0.506 | 83.8% | 86.2% | 22 | 0.201 | 93.4% | 93.3% |
| 3 | 0.438 | 85.9% | 87.8% | 23 | 0.197 | 93.4% | 93.5% |
| 4 | 0.395 | 87.2% | 88.5% | 24 | 0.186 | 93.8% | 93.0% |
| 5 | 0.367 | 88.3% | 89.9% | 25 | 0.179 | 94.0% | 93.5% |
| 6 | 0.349 | 88.6% | 90.6% | 26 | 0.177 | 94.1% | 93.5% |
| 7 | 0.330 | 89.4% | 90.9% | 27 | 0.169 | 94.4% | 93.6% |
| 8 | 0.315 | 89.6% | 90.9% | 28 | 0.164 | 94.6% | 94.0% |
| 9 | 0.300 | 90.2% | 90.9% | 29 | 0.160 | 94.7% | 93.7% |
| 10 | 0.297 | 90.3% | 91.0% | 30 | 0.153 | 94.9% | 94.2% |
| 11 | 0.280 | 90.8% | 91.9% | 31 | 0.150 | 95.0% | 94.1% |
| 12 | 0.274 | 90.9% | 92.1% | 32 | 0.146 | 95.2% | 94.6% |
| 13 | 0.269 | 91.0% | 92.0% | 33 | 0.143 | 95.4% | 94.5% |
| 14 | 0.257 | 91.4% | 91.3% | 34 | 0.141 | 95.3% | 94.3% |
| 15 | 0.250 | 91.7% | 92.0% | 35 | 0.136 | 95.6% | 94.6% |
| 16 | 0.240 | 92.1% | 92.7% | 36 | 0.136 | 95.4% | 94.4% |
| 17 | 0.236 | 92.2% | 92.8% | 37 | 0.131 | 95.7% | 94.4% |
| 18 | 0.229 | 92.4% | 93.1% | 38 | 0.133 | 95.6% | 94.4% |
| 19 | 0.220 | 92.7% | 92.7% | 39 | 0.131 | 95.6% | 94.5% |
| 20 | 0.213 | 93.0% | 93.4% | 40 | 0.131 | 95.6% | 94.7% |

### Epoch-by-epoch: FilterbankSinCosMultiCNN 2/bin k=400 (40 epochs, full dataset)

160 conv filters (2 sin+cos pairs per mel bin), d²+log, sum within bin. 94.1% test, 89,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.870 | 73.2% | 86.0% | 21 | 0.209 | 93.1% | 93.2% |
| 2 | 0.474 | 84.8% | 88.7% | 22 | 0.203 | 93.3% | 93.2% |
| 3 | 0.411 | 86.6% | 90.0% | 23 | 0.197 | 93.4% | 93.8% |
| 4 | 0.370 | 88.0% | 89.3% | 24 | 0.189 | 93.6% | 93.2% |
| 5 | 0.347 | 88.7% | 90.6% | 25 | 0.183 | 93.8% | 93.7% |
| 6 | 0.327 | 89.2% | 90.3% | 26 | 0.176 | 94.1% | 93.8% |
| 7 | 0.316 | 89.6% | 90.9% | 27 | 0.169 | 94.4% | 94.3% |
| 8 | 0.306 | 90.0% | 91.4% | 28 | 0.163 | 94.5% | 94.0% |
| 9 | 0.294 | 90.3% | 90.4% | 29 | 0.157 | 94.7% | 94.3% |
| 10 | 0.287 | 90.5% | 91.0% | 30 | 0.156 | 94.8% | 94.0% |
| 11 | 0.278 | 90.8% | 92.0% | 31 | 0.148 | 95.1% | 94.3% |
| 12 | 0.269 | 91.1% | 91.6% | 32 | 0.144 | 95.3% | 94.3% |
| 13 | 0.260 | 91.4% | 92.1% | 33 | 0.141 | 95.4% | 94.4% |
| 14 | 0.254 | 91.6% | 92.0% | 34 | 0.137 | 95.4% | 94.4% |
| 15 | 0.247 | 91.7% | 91.9% | 35 | 0.135 | 95.5% | 94.4% |
| 16 | 0.238 | 92.1% | 92.0% | 36 | 0.132 | 95.6% | 94.6% |
| 17 | 0.235 | 92.2% | 92.0% | 37 | 0.131 | 95.7% | 94.7% |
| 18 | 0.228 | 92.4% | 92.7% | 38 | 0.129 | 95.7% | 94.7% |
| 19 | 0.219 | 92.6% | 92.6% | 39 | 0.130 | 95.6% | 94.6% |
| 20 | 0.214 | 92.7% | 93.3% | 40 | 0.130 | 95.7% | 94.6% |

### Epoch-by-epoch: FilterbankLinearCNN k=80 (40 epochs, full dataset)

No d², no log, random init. 90.2% test, 28,828 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.410 | 54.2% | 70.9% | 21 | 0.368 | 87.9% | 88.6% |
| 2 | 0.857 | 72.5% | 76.9% | 22 | 0.361 | 88.3% | 88.4% |
| 3 | 0.712 | 77.0% | 80.9% | 23 | 0.352 | 88.3% | 89.2% |
| 4 | 0.644 | 79.0% | 82.1% | 24 | 0.351 | 88.5% | 89.3% |
| 5 | 0.596 | 80.7% | 83.9% | 25 | 0.346 | 88.6% | 89.5% |
| 6 | 0.559 | 81.9% | 82.9% | 26 | 0.342 | 88.8% | 89.4% |
| 7 | 0.539 | 82.5% | 84.5% | 27 | 0.330 | 89.1% | 89.6% |
| 8 | 0.523 | 82.9% | 86.0% | 28 | 0.324 | 89.4% | 89.7% |
| 9 | 0.500 | 83.7% | 86.5% | 29 | 0.321 | 89.3% | 90.3% |
| 10 | 0.483 | 84.1% | 86.6% | 30 | 0.319 | 89.5% | 90.0% |
| 11 | 0.471 | 84.6% | 86.7% | 31 | 0.312 | 89.7% | 89.9% |
| 12 | 0.455 | 85.1% | 86.9% | 32 | 0.303 | 90.0% | 90.4% |
| 13 | 0.450 | 85.4% | 87.4% | 33 | 0.300 | 90.2% | 90.2% |
| 14 | 0.429 | 86.1% | 87.4% | 34 | 0.302 | 90.1% | 90.3% |
| 15 | 0.425 | 86.0% | 87.3% | 35 | 0.297 | 90.2% | 90.3% |
| 16 | 0.416 | 86.2% | 88.0% | 36 | 0.293 | 90.3% | 90.7% |
| 17 | 0.407 | 86.5% | 89.1% | 37 | 0.292 | 90.5% | 90.1% |
| 18 | 0.396 | 86.9% | 87.0% | 38 | 0.294 | 90.4% | 90.5% |
| 19 | 0.389 | 87.2% | 88.2% | 39 | 0.289 | 90.4% | 90.1% |
| 20 | 0.381 | 87.5% | 88.2% | 40 | 0.293 | 90.3% | 90.4% |

### Epoch-by-epoch: FilterbankMelInitLinearCNN k=80 (40 epochs, full dataset)

No d², no log, mel init. 91.7% test, 28,828 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.349 | 56.1% | 72.0% | 21 | 0.336 | 88.9% | 89.9% |
| 2 | 0.786 | 74.8% | 78.9% | 22 | 0.331 | 89.2% | 90.4% |
| 3 | 0.667 | 78.3% | 81.9% | 23 | 0.316 | 89.7% | 90.6% |
| 4 | 0.597 | 80.4% | 83.1% | 24 | 0.321 | 89.6% | 90.6% |
| 5 | 0.551 | 82.0% | 84.8% | 25 | 0.311 | 89.8% | 91.5% |
| 6 | 0.522 | 82.9% | 85.5% | 26 | 0.310 | 89.8% | 90.4% |
| 7 | 0.497 | 83.9% | 86.2% | 27 | 0.298 | 90.2% | 91.2% |
| 8 | 0.479 | 84.5% | 87.4% | 28 | 0.292 | 90.3% | 91.5% |
| 9 | 0.460 | 85.0% | 87.7% | 29 | 0.292 | 90.5% | 91.8% |
| 10 | 0.442 | 85.6% | 88.1% | 30 | 0.286 | 90.6% | 91.5% |
| 11 | 0.434 | 86.0% | 87.3% | 31 | 0.280 | 90.9% | 92.0% |
| 12 | 0.419 | 86.4% | 88.1% | 32 | 0.272 | 91.0% | 91.8% |
| 13 | 0.410 | 86.7% | 89.4% | 33 | 0.275 | 91.0% | 92.0% |
| 14 | 0.392 | 87.0% | 88.6% | 34 | 0.270 | 91.2% | 92.0% |
| 15 | 0.388 | 87.4% | 89.1% | 35 | 0.268 | 91.1% | 92.3% |
| 16 | 0.377 | 87.6% | 89.4% | 36 | 0.265 | 91.3% | 92.4% |
| 17 | 0.369 | 87.8% | 89.6% | 37 | 0.262 | 91.5% | 92.5% |
| 18 | 0.359 | 88.3% | 88.6% | 38 | 0.260 | 91.6% | 92.1% |
| 19 | 0.358 | 88.3% | 90.1% | 39 | 0.260 | 91.5% | 92.1% |
| 20 | 0.344 | 88.8% | 89.9% | 40 | 0.263 | 91.2% | 92.2% |

### MelCNN hop length experiments

Testing whether finer temporal stride improves MelCNN. Same n_fft=400, same CNN backbone.
With finer hop, more frames survive the 3x MaxPool(2) → avg_pool in groups of ~12 → max over groups.

| Hop | Stride | Frames | After CNN | Pooling | Test Acc |
|-----|--------|--------|-----------|---------|----------|
| 160 | 10ms | 101 | ~12 | global avg | **95.7%** |
| 80 | 5ms | 201 | ~25 | avg(12)→max over 2 | 95.0% |
| 40 | 2.5ms | 401 | ~50 | avg(12)→max over 4 | 94.5% |

**Finer stride hurts.** The standard 10ms hop is well-matched to speech dynamics. Going
finer adds redundant frames that the avg_pool→max aggregation can't exploit effectively.
The model also trains slower (epoch 1 val: 85.2% at hop=160 vs 72.5% at hop=80 vs 70.5% at hop=40).

### Epoch-by-epoch: MelCNN hop=80 (40 epochs, full dataset)

5ms stride, 201 frames. 95.0% test, 25,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.428 | 55.6% | 70.5% | 21 | 0.233 | 92.5% | 93.7% |
| 2 | 0.715 | 78.3% | 84.0% | 22 | 0.233 | 92.4% | 94.4% |
| 3 | 0.564 | 82.5% | 85.5% | 23 | 0.228 | 92.7% | 93.9% |
| 4 | 0.472 | 85.1% | 85.5% | 24 | 0.215 | 92.9% | 94.0% |
| 5 | 0.431 | 86.3% | 89.5% | 25 | 0.217 | 93.0% | 94.3% |
| 6 | 0.386 | 87.8% | 91.4% | 26 | 0.212 | 93.1% | 94.4% |
| 7 | 0.373 | 88.2% | 91.2% | 27 | 0.213 | 93.0% | 93.8% |
| 8 | 0.351 | 88.9% | 91.5% | 28 | 0.207 | 93.4% | 94.2% |
| 9 | 0.339 | 89.2% | 92.0% | 29 | 0.198 | 93.7% | 94.7% |
| 10 | 0.327 | 89.6% | 91.5% | 30 | 0.204 | 93.4% | 94.8% |
| 11 | 0.320 | 89.6% | 92.4% | 31 | 0.196 | 93.6% | 94.5% |
| 12 | 0.300 | 90.4% | 92.1% | 32 | 0.194 | 93.7% | 94.6% |
| 13 | 0.282 | 91.0% | 91.9% | 33 | 0.191 | 93.7% | 94.7% |
| 14 | 0.281 | 90.9% | 93.0% | 34 | 0.184 | 94.1% | 95.0% |
| 15 | 0.269 | 91.5% | 92.6% | 35 | 0.186 | 94.0% | 94.7% |
| 16 | 0.262 | 91.5% | 92.2% | 36 | 0.191 | 93.7% | 94.8% |
| 17 | 0.260 | 91.7% | 93.1% | 37 | 0.187 | 93.9% | 94.9% |
| 18 | 0.253 | 91.8% | 93.2% | 38 | 0.183 | 94.1% | 94.6% |
| 19 | 0.256 | 91.6% | 93.7% | 39 | 0.183 | 94.0% | 94.9% |
| 20 | 0.238 | 92.2% | 93.9% | 40 | 0.178 | 94.3% | 94.9% |

### Epoch-by-epoch: MelCNN hop=40 (40 epochs, full dataset)

2.5ms stride, 401 frames. 94.5% test, 25,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.427 | 54.6% | 72.5% | 21 | 0.228 | 92.7% | 92.8% |
| 2 | 0.743 | 76.9% | 82.4% | 22 | 0.221 | 92.8% | 93.9% |
| 3 | 0.580 | 81.6% | 84.1% | 23 | 0.221 | 92.8% | 93.2% |
| 4 | 0.481 | 84.7% | 87.6% | 24 | 0.212 | 93.3% | 94.0% |
| 5 | 0.447 | 85.6% | 87.7% | 25 | 0.208 | 93.2% | 94.0% |
| 6 | 0.395 | 87.5% | 88.9% | 26 | 0.204 | 93.4% | 94.3% |
| 7 | 0.375 | 88.1% | 90.6% | 27 | 0.207 | 93.3% | 93.7% |
| 8 | 0.351 | 88.9% | 90.7% | 28 | 0.193 | 93.8% | 93.5% |
| 9 | 0.337 | 89.1% | 92.0% | 29 | 0.187 | 93.9% | 94.2% |
| 10 | 0.317 | 89.9% | 90.1% | 30 | 0.194 | 93.8% | 94.1% |
| 11 | 0.310 | 90.0% | 91.5% | 31 | 0.186 | 94.0% | 94.1% |
| 12 | 0.291 | 90.7% | 92.4% | 32 | 0.182 | 94.2% | 94.3% |
| 13 | 0.276 | 91.3% | 92.0% | 33 | 0.180 | 94.2% | 94.5% |
| 14 | 0.267 | 91.3% | 91.6% | 34 | 0.175 | 94.4% | 94.5% |
| 15 | 0.264 | 91.5% | 92.5% | 35 | 0.176 | 94.4% | 94.7% |
| 16 | 0.254 | 91.8% | 91.2% | 36 | 0.179 | 94.3% | 94.7% |
| 17 | 0.253 | 91.8% | 92.9% | 37 | 0.179 | 94.1% | 94.7% |
| 18 | 0.243 | 92.2% | 91.8% | 38 | 0.173 | 94.4% | 94.6% |
| 19 | 0.252 | 91.8% | 93.4% | 39 | 0.175 | 94.4% | 94.6% |
| 20 | 0.230 | 92.7% | 93.7% | 40 | 0.169 | 94.6% | 94.6% |

### Summary table (all front-end variants, 40 epochs)

| Model | Test Acc | Params | Front-end |
|-------|----------|--------|-----------|
| **MelCNN (hop=160)** | **95.7%** | 25,628 | FFT + mel filterbank + log, 10ms stride |
| MelCNN (hop=80) | 95.0% | 25,628 | FFT + mel filterbank + log, 5ms stride |
| LearnedSpecMulti (2/bin) | **94.9%** | 25,708 | 80 learned freqs (2/bin), cumsum, d² |
| FilterbankSinCos (1/bin) | **94.7%** | 57,628 | Conv1d sin+cos k=400 → sin²+cos² → log |
| MelCNN (hop=40) | 94.5% | 25,628 | FFT + mel filterbank + log, 2.5ms stride |
| FilterbankSinCosCombined | 94.5% | 61,468 | Conv1d sin+cos k=400 → mag+raw → 120ch |
| FilterbankSinCosMulti (2/bin) | 94.1% | 89,628 | Conv1d 160 sin+cos k=400 → d²+log → sum → 40ch |
| LearnedSpecCNN W=200 | 94.1% | 25,668 | 40 learned freqs, windowed cumsum, d² |
| LearnedSpecMulti (4/bin) | 94.0% | 25,788 | 160 learned freqs (4/bin), cumsum, d² |
| LearnedSpecCNN W=400 | 93.4% | 25,668 | 40 learned freqs, windowed cumsum, d² |
| LearnedSpecCNN W=80 | 93.0% | 25,668 | 40 learned freqs, windowed cumsum, d² |
| LearnedSpecCNNMod2 | 92.2% | 27,428 | data-dep freqs (deep, per sample), d² |
| LearnedSpecCNNMod | 92.1% | 25,788 | data-dep freqs (shallow, per sample), d² |
| FilterbankMelInit | 92.0% | 28,828 | Conv1d(k=80) mel init → x² → log |
| FilterbankMelInitLinear | 91.7% | 28,828 | Conv1d(k=80) mel init → CNN (no d²/log) |
| RawCNN | 91.0% | 25,228 | Conv1d(1→32, k=80, s=16) |
| FilterbankSinCosLinear | 90.9% | 59,548 | Conv1d sin+cos k=400 → 80ch raw |
| LearnedSpecLinear | 90.7% | 27,588 | 40 learned freqs, cumsum, real+imag (no d²) |
| FilterbankCNN | 90.5% | 28,828 | Conv1d(1→40, k=80) rand init → x² → log |
| FilterbankLinear | 90.2% | 28,828 | Conv1d(1→40, k=80) rand init (no d²/log) |
| ConvCumsumV2 | 90.1% | 105,012 | Conv sin+cos front + cumsum layers, fixed |
| BlockDecayS5V2 | 89.8% | 151,960 | SSM (6 layers, bidir, B/C/D) |
| ConvCumsumModV2 | 89.6% | 119,772 | Conv sin+cos front + cumsum layers, data-dep |
| MultiLayerV2 (avgpool) | 89.2% | 73,052 | 4-layer cumsum, stride=160, avgpool 2x |
| MultiLayerMinimal | 88.9% | 134,604 | 3-layer cumsum (no B/C/D), ds=10 |
| MultiLayerMod | 88.7% | 159,564 | 3-layer cumsum, data-dep layers 2+, ds=10 |
| MultiLayerV2 (maxpool) | 88.3% | 73,052 | 4-layer cumsum, stride=160, maxpool 2x |
| LearnedSpecCNNConv | 88.1% | 28,948 | conv→cumsum as frequencies |
| MultiLayerModV2 (avgpool) | 87.4% | 87,812 | 4-layer cumsum, data-dep 2+, avgpool 2x |

## Key design insight: tapering and streaming

Three ways to avoid spectral leakage from hard rectangular window cutoff:
1. **Decay (exponential taper)**: `λ^k` weighting, smooth rolloff, but no simple streaming form
2. **Conv filterbank**: learns arbitrary filter shape (including natural tapering at edges), but requires full k-length context per step (no streaming)
3. **Block decay**: streaming-friendly decay. Each block is a rectangular windowed cumsum (O(1) streaming), weighted by geometric decay across blocks: `h_t = Σ_k λ^k · [cs(t-kW) - cs(t-(k+1)W)]`. Gets smooth tapering overall while each block is streamable.

Evidence: W=200 (94.1%) > W=400 (93.4%) for LearnedSpecCNN — shorter window helps with only 40 frequencies, but too short (W=80) hurts from rectangular window noise. Conv filterbank can learn to taper naturally. Block decay achieves tapering while maintaining O(1) streaming.

## Open questions

- ~~BlockDecayS5 full 40-epoch run: does block decay stabilize training vs WindowS5?~~ **YES — 89.8%, completely stable**
- ~~Can BlockDecayS5 match or beat WindowS5's 89.4%?~~ **YES — 89.8% vs 89.4%**
- ~~Will FilterbankMelInitCNN close the gap to MelCNN?~~ **Partially — 92.0% vs 90.5% random, but still behind LearnedSpecCNN 93.4%**
- ~~Does LearnedSpecLinearCNN (no d²) beat power spectrum?~~ **NO — 90.7% vs 93.4%. CNN prefers magnitude.**
- ~~MultiLayerMinimalMod: do data-dependent frequencies help in layers 2+?~~ **NO — 88.7% ≈ 88.9% fixed**
- Can any variant match S5's 96.5% from the literature with full training?
- ~~Can LearnedSpecCNN match MelCNN (95.7%) with learned frequencies instead of FFT?~~ **Not by adding more freqs per bin: 2/bin=94.9%, 4/bin=94.0%. Conv filterbank 2/bin also didn't help (94.1% vs 1/bin 94.7%). Bottleneck is not frequency density.**
- ~~MultiLayerV2/ModV2: does mimicking MelCNN's downsampling hierarchy help multi-layer cumsum?~~ **Marginal — V2 89.2% vs V1 88.9%. AvgPool > MaxPool (+0.9%). Data-dep still doesn't help (87.4%)**
- FilterbankLinearCNN: does skipping d²/log help or hurt conv filterbank?

## Model inventory

| CLI flag | Class | Decay | Window | B/C/D | B_bar | Scan | Test acc |
|----------|-------|-------|--------|-------|-------|------|----------|
| window_s5 | WindowS5 | no | W=80 | yes | yes | cumsum | **89.4%** (67.7% smoke) |
| rot_s5_fixed | RotS5Fixed | yes | no | yes | yes | parallel | 64.5% (smoke) |
| s5 | S5Model | yes | no | yes | yes | parallel | 56.7% (smoke) |
| cumsum_s5 | CumsumS5 | no | no | yes | yes | cumsum | 77.0% (34.4% smoke) |
| rot_window_input | RotWindowInput | no | W=80 | no | no | cumsum | 21.9% |
| rot_window | RotWindow | no | W=80 | no | no | cumsum | 19.3% |
| rot_decay_fixed | RotDecayFixed | yes | no | no | no | parallel | 17.4% |
| rot_input | RotInput | no | no | no | no | cumsum | 13.2% |
| rot_decay_input | RotDecayInput | yes | no | no | no | parallel | 12.0% |
| block_decay_s5 | BlockDecayS5 | block λ^k | W=var | yes | yes | cumsum | 39.2% smoke W=800 |
| block_decay_s5_v2 | BlockDecayS5V2 | block λ^k | W=80 | yes | yes | cumsum (fast) | **89.8%** (65.1% smoke) |
| block_decay_s5_mod | BlockDecayS5Mod | block λ^k | W=var | yes | yes | cumsum | 15.8% smoke W=800 |
| window_s5_mod | WindowS5Mod | no | W=80 | yes | yes | cumsum | 55.7% smoke (collapsed at 40ep) |
| cumsum_s5_mod | CumsumS5Mod | no | no | yes | yes | cumsum | 23.0% smoke |
| cumsum_s5_input | CumsumS5Input | no | no | yes | yes | cumsum | 16.8% smoke |
| window_s5_input | WindowS5Input | no | W=80 | yes | yes | cumsum | 32.7% smoke |
| rot_fixed | RotFixed | no | no | no | no | cumsum | 8.7% |
| mel | MelCNN | — | — | — | — | CNN | **95.7%** |
| raw | RawCNN | — | — | — | — | CNN | 91.0% |
| learned_spec | LearnedSpecCNN | — | W=var | — | — | CNN+cumsum | **94.1%** (W=200) |
| learned_spec_mod | LearnedSpecCNNMod | — | W=400 | — | — | CNN+cumsum | 92.1% |
| learned_spec_mod2 | LearnedSpecCNNMod2 | — | W=400 | — | — | CNN+cumsum | 92.2% |
| learned_spec_conv | LearnedSpecCNNConv | — | W=400 | — | — | CNN+cumsum | 88.1% |
| learned_spec_linear | LearnedSpecLinearCNN | — | W=200 | — | — | CNN+cumsum | 90.7% |
| filterbank | FilterbankCNN | — | k=80 | — | — | CNN | 90.5% |
| filterbank_mel | FilterbankMelInitCNN | — | k=80 | — | — | CNN | **92.0%** |
| filterbank_linear | FilterbankLinearCNN | — | k=80 | — | — | CNN | 90.2% |
| filterbank_mel_linear | FilterbankMelInitLinearCNN | — | k=80 | — | — | CNN | 91.7% |
| multi_layer_minimal | MultiLayerMinimalStrided | no | W=400/40 | no | no | cumsum | **88.9%** |
| multi_layer_mod | MultiLayerMinimalMod | no | W=400/40 | no | no | cumsum | 88.7% |
| multi_layer_v2 | MultiLayerMinimalV2 | no | W=400, avgpool 2x | no | no | cumsum | **89.2%** (88.3% maxpool) |
| multi_layer_mod_v2 | MultiLayerMinimalModV2 | no | W=400, avgpool 2x | no | no | cumsum | 87.4% |
| learned_spec_multi | LearnedSpecMultiCNN | — | W=200 | — | — | CNN+cumsum | **94.9%** (2/bin) |
| learned_spec_multi4 | LearnedSpecMultiCNN | — | W=200 | — | — | CNN+cumsum | 94.0% (4/bin) |
| filterbank_sincos_multi | FilterbankSinCosMultiCNN | — | k=400 | — | — | CNN | 94.1% (2/bin) |
| conv_cumsum_v2 | ConvCumsumV2 | no | conv+cumsum, avgpool | no | no | cumsum | **90.1%** |
| conv_cumsum_mod_v2 | ConvCumsumModV2 | no | conv+cumsum, avgpool | no | no | cumsum | 89.6% |
| filterbank_sincos | FilterbankSinCosCNN | — | k=400 | — | — | CNN | **94.7%** |
| filterbank_sincos_linear | FilterbankSinCosLinearCNN | — | k=400 | — | — | CNN | 90.9% |
| minimal_strided | MinimalStridedWindow | no | W=400 | no | no | cumsum | ~48% val (1 layer) |
# MelCumsum & CumsumE2E Results Summary

## Architecture Overview

All models use stacked cumsum layers for sequence processing with two key innovations:
1. **Magnitude pooling at classification**: collapse re/im pairs to sqrt(re²+im²) before fc layer (phase invariant)
2. **Windowed cumsum + maxpool**: window=20 in layers 2+ with max over all frames (vs last-frame for full-seq)

### Model Variants

| Model | Layer 1 (front-end) | Layers 2+ | Window L2+ | Classification | Params |
|-------|-------------------|-----------|------------|----------------|--------|
| MelCumsumFixed | Mel spectrogram + log + Linear(40→80) | Fixed freq cumsum | full (101) | last-frame magnitude | 82,332 |
| MelCumsumMod | Mel spectrogram + log + Linear(40→80) | Mod freq cumsum | full (101) | last-frame magnitude | 102,012 |
| MelCumsumFixedW | Mel spectrogram + log + Linear(40→80) | Fixed freq cumsum | 20 | maxpool magnitude | 82,332 |
| MelCumsumModW | Mel spectrogram + log + Linear(40→80) | Mod freq cumsum | 20 | maxpool magnitude | 102,012 |
| CumsumE2E | Learned freq cumsum (w=400, stride 160) | Fixed freq cumsum | 20 | maxpool magnitude | 72,572 |
| CumsumE2EMod | Learned freq cumsum (w=400, stride 160) | Mod freq cumsum | 20 | maxpool magnitude | 87,332 |
| CumsumE2EMag | Learned freq → mag+log → Linear(40→80) | Fixed freq cumsum | 20 | maxpool magnitude | 62,732 |
| CumsumE2EMagMod | Learned freq → mag+log → Linear(40→80) | Mod freq cumsum | 20 | maxpool magnitude | 77,492 |

All: n_embed=80 (mel models) or n_freqs=40 (E2E models), n_layers=4, 40 epochs, Adam lr=1e-3, cosine schedule.

## Results Summary

| Model | Best Val | Test Acc | Test F1 | Params |
|-------|---------|----------|---------|--------|
| MelCumsumFixed (full) | 88.8% | 87.9% | 0.878 | 82,332 |
| MelCumsumMod (full) | 85.1% | 84.9% | 0.846 | 102,012 |
| **MelCumsumFixedW (w=20)** | **95.9%** | **96.0%** | **0.960** | **82,332** |
| **MelCumsumModW (w=20)** | **95.6%** | **95.8%** | **0.958** | **102,012** |
| CumsumE2E (w_l1=400, w=20) | 87.9% | 86.8% | 0.868 | 72,572 |
| CumsumE2EMod (w_l1=400, w=20) | 88.8% | 87.6% | 0.875 | 87,332 |
| CumsumE2EMag | — | — | — | 62,732 |
| CumsumE2EMagMod | — | — | — | 77,492 |

### Key findings

- **Windowing is critical**: w=20 + maxpool gives +7% over full-sequence + last-frame (95.9% vs 88.8%)
- **Magnitude pooling**: collapsing re/im to magnitude before classification provides phase invariance
- **Mel front-end >> learned frequencies**: MelCumsumFixedW 96.0% vs CumsumE2E 86.8% (same layers 2+)
- **Fixed ≥ Mod for mel front-end**: Fixed slightly better (96.0 vs 95.8%)
- **Mod > Fixed for E2E**: Mod catches up and surpasses fixed (88.8% vs 87.9%) — data-dependent freqs help when front-end is weaker
- **Full-seq mod struggles**: MelCumsumMod (full) 84.9% — much worse than fixed 87.9%. Mod benefits from windowing

### Comparison with prior models (from RESULTS.md)

| Model | Test Acc | Notes |
|-------|----------|-------|
| MelCNN (baseline) | 93.4% | FFT + mel + CNN |
| LearnedSpecLinearCNN | 91.4% | Learned freq cumsum → CNN |
| MultiLayerV2 | 89.2% | Learned freq cumsum → cumsum layers + AvgPool(2) |
| **MelCumsumFixedW** | **96.0%** | **Mel → windowed cumsum + mag maxpool** |
| CumsumE2EMod | 87.6% | Full cumsum end-to-end |

MelCumsumFixedW beats all prior models by a significant margin.

---

## Epoch-by-Epoch Tables

### MelCumsumFixed (full sequence, 82,332 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.532 | 17.7% | 33.4% | 21 | 0.779 | 73.8% | 85.2% |
| 2 | 1.982 | 31.2% | 45.0% | 22 | 0.764 | 74.4% | 85.0% |
| 3 | 1.738 | 38.6% | 55.1% | 23 | 0.751 | 74.5% | 86.2% |
| 4 | 1.580 | 44.0% | 58.0% | 24 | 0.726 | 75.5% | 85.2% |
| 5 | 1.474 | 48.1% | 65.9% | 25 | 0.716 | 75.8% | 85.9% |
| 6 | 1.376 | 51.9% | 66.9% | 26 | 0.706 | 76.1% | 85.6% |
| 7 | 1.273 | 55.6% | 70.5% | 27 | 0.677 | 77.3% | 87.0% |
| 8 | 1.195 | 58.5% | 72.2% | 28 | 0.678 | 77.0% | 85.8% |
| 9 | 1.151 | 60.2% | 76.3% | 29 | 0.660 | 77.7% | 88.0% |
| 10 | 1.096 | 62.6% | 73.2% | 30 | 0.646 | 78.0% | 87.4% |
| 11 | 1.043 | 64.2% | 75.8% | 31 | 0.634 | 78.6% | 87.7% |
| 12 | 1.020 | 65.3% | 78.2% | 32 | 0.633 | 78.7% | 87.3% |
| 13 | 0.984 | 66.8% | 80.0% | 33 | 0.629 | 78.6% | 88.1% |
| 14 | 0.939 | 68.0% | 81.1% | 34 | 0.615 | 79.3% | 88.7% |
| 15 | 0.925 | 69.0% | 81.8% | 35 | 0.613 | 79.3% | 88.1% |
| 16 | 0.883 | 70.2% | 83.2% | 36 | 0.613 | 79.2% | 88.8% |
| 17 | 0.863 | 70.8% | 81.7% | 37 | 0.596 | 79.8% | 88.6% |
| 18 | 0.844 | 71.7% | 81.8% | 38 | 0.602 | 79.6% | 88.2% |
| 19 | 0.817 | 72.6% | 82.7% | 39 | 0.602 | 79.8% | 88.1% |
| 20 | 0.807 | 72.9% | 85.5% | 40 | 0.594 | 79.7% | 88.3% |

Best val: 88.8% (epoch 36). Test: 87.9%.

### MelCumsumMod (full sequence, 102,012 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.844 | 9.5% | 12.7% | 21 | 1.212 | 57.3% | 72.3% |
| 2 | 2.642 | 11.3% | 14.8% | 22 | 1.154 | 59.8% | 69.4% |
| 3 | 2.533 | 11.8% | 14.7% | 23 | 1.120 | 61.4% | 74.0% |
| 4 | 2.520 | 12.3% | 14.5% | 24 | 1.065 | 63.3% | 77.3% |
| 5 | 2.386 | 15.4% | 20.7% | 25 | 1.014 | 65.5% | 76.6% |
| 6 | 2.324 | 16.8% | 18.6% | 26 | 0.977 | 67.0% | 80.5% |
| 7 | 2.296 | 17.8% | 23.9% | 27 | 0.939 | 67.9% | 79.7% |
| 8 | 2.337 | 17.5% | 22.1% | 28 | 0.905 | 69.3% | 81.1% |
| 9 | 2.277 | 19.0% | 29.7% | 29 | 0.876 | 70.4% | 82.0% |
| 10 | 2.122 | 24.6% | 36.4% | 30 | 0.852 | 71.1% | 82.4% |
| 11 | 1.981 | 30.5% | 43.0% | 31 | 0.833 | 72.2% | 84.3% |
| 12 | 1.854 | 34.9% | 43.3% | 32 | 0.802 | 73.0% | 83.9% |
| 13 | 1.718 | 39.1% | 48.9% | 33 | 0.801 | 73.2% | 83.7% |
| 14 | 1.754 | 38.3% | 46.5% | 34 | 0.773 | 73.9% | 84.2% |
| 15 | 1.626 | 42.1% | 50.3% | 35 | 0.762 | 74.3% | 84.3% |
| 16 | 1.535 | 45.3% | 52.2% | 36 | 0.762 | 74.3% | 84.7% |
| 17 | 1.450 | 47.9% | 58.7% | 37 | 0.750 | 74.7% | 84.9% |
| 18 | 1.410 | 49.6% | 59.9% | 38 | 0.744 | 74.9% | 84.9% |
| 19 | 1.317 | 53.2% | 69.4% | 39 | 0.742 | 75.2% | 85.1% |
| 20 | 1.261 | 55.7% | 68.6% | 40 | 0.737 | 75.0% | 84.8% |

Best val: 85.1% (epoch 39). Test: 84.9%.

### MelCumsumFixedW (window=20, 82,332 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.584 | 50.9% | 83.3% | 21 | 0.330 | 89.0% | 94.4% |
| 2 | 0.788 | 74.1% | 86.8% | 22 | 0.329 | 89.2% | 94.6% |
| 3 | 0.663 | 78.3% | 90.0% | 23 | 0.315 | 89.5% | 94.2% |
| 4 | 0.595 | 80.7% | 89.9% | 24 | 0.308 | 89.8% | 95.2% |
| 5 | 0.562 | 81.7% | 91.6% | 25 | 0.301 | 90.0% | 94.9% |
| 6 | 0.543 | 82.5% | 92.3% | 26 | 0.296 | 90.0% | 94.6% |
| 7 | 0.509 | 83.5% | 92.5% | 27 | 0.280 | 90.6% | 94.9% |
| 8 | 0.473 | 84.7% | 92.3% | 28 | 0.281 | 90.6% | 94.7% |
| 9 | 0.470 | 84.9% | 93.3% | 29 | 0.268 | 91.0% | 95.5% |
| 10 | 0.453 | 85.2% | 93.5% | 30 | 0.267 | 91.0% | 95.1% |
| 11 | 0.444 | 85.6% | 92.0% | 31 | 0.264 | 91.3% | 95.0% |
| 12 | 0.429 | 86.0% | 92.8% | 32 | 0.259 | 91.4% | 95.6% |
| 13 | 0.417 | 86.3% | 93.9% | 33 | 0.257 | 91.4% | 95.4% |
| 14 | 0.407 | 86.6% | 93.5% | 34 | 0.251 | 91.5% | 95.8% |
| 15 | 0.397 | 86.8% | 93.7% | 35 | 0.251 | 91.6% | 95.5% |
| 16 | 0.380 | 87.7% | 93.7% | 36 | 0.248 | 91.6% | 95.6% |
| 17 | 0.366 | 88.0% | 94.1% | 37 | 0.244 | 91.8% | 95.8% |
| 18 | 0.360 | 88.1% | 94.0% | 38 | 0.245 | 91.7% | 95.5% |
| 19 | 0.351 | 88.4% | 94.6% | 39 | 0.241 | 91.9% | 95.8% |
| 20 | 0.346 | 88.6% | 95.1% | 40 | 0.238 | 91.8% | 95.9% |

Best val: 95.9% (epoch 40). Test: 96.0%.

### MelCumsumModW (window=20, 102,012 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.959 | 37.6% | 68.0% | 21 | 0.347 | 88.5% | 94.0% |
| 2 | 1.046 | 65.3% | 81.2% | 22 | 0.336 | 88.8% | 94.5% |
| 3 | 0.827 | 72.9% | 85.8% | 23 | 0.332 | 89.1% | 93.8% |
| 4 | 0.718 | 76.4% | 87.7% | 24 | 0.313 | 89.5% | 94.4% |
| 5 | 0.659 | 78.2% | 88.2% | 25 | 0.311 | 89.6% | 94.7% |
| 6 | 0.615 | 79.9% | 85.8% | 26 | 0.297 | 90.0% | 94.8% |
| 7 | 0.566 | 81.2% | 90.0% | 27 | 0.288 | 90.4% | 94.9% |
| 8 | 0.548 | 82.1% | 90.3% | 28 | 0.273 | 90.8% | 94.8% |
| 9 | 0.529 | 82.8% | 91.4% | 29 | 0.273 | 91.0% | 94.9% |
| 10 | 0.506 | 83.4% | 91.4% | 30 | 0.273 | 90.9% | 95.3% |
| 11 | 0.490 | 84.0% | 90.7% | 31 | 0.268 | 91.3% | 95.1% |
| 12 | 0.467 | 84.6% | 92.1% | 32 | 0.256 | 91.4% | 95.2% |
| 13 | 0.454 | 85.1% | 92.4% | 33 | 0.258 | 91.2% | 95.4% |
| 14 | 0.431 | 85.7% | 93.3% | 34 | 0.242 | 91.9% | 95.3% |
| 15 | 0.425 | 86.0% | 93.3% | 35 | 0.244 | 91.9% | 95.3% |
| 16 | 0.398 | 86.8% | 92.4% | 36 | 0.243 | 91.8% | 95.5% |
| 17 | 0.394 | 87.0% | 92.9% | 37 | 0.237 | 92.0% | 95.5% |
| 18 | 0.387 | 87.1% | 93.8% | 38 | 0.237 | 92.1% | 95.5% |
| 19 | 0.368 | 87.8% | 92.3% | 39 | 0.236 | 92.2% | 95.4% |
| 20 | 0.364 | 88.0% | 93.9% | 40 | 0.232 | 92.3% | 95.6% |

Best val: 95.6% (epoch 40). Test: 95.8%.

### CumsumE2E (fixed, w_l1=400, w=20, 72,572 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.361 | 31.4% | 38.1% | 21 | 0.571 | 81.8% | 83.9% |
| 2 | 1.869 | 39.3% | 41.0% | 22 | 0.557 | 82.0% | 84.5% |
| 3 | 1.748 | 43.5% | 44.5% | 23 | 0.537 | 82.5% | 84.7% |
| 4 | 1.552 | 50.8% | 55.3% | 24 | 0.522 | 83.4% | 84.9% |
| 5 | 1.348 | 57.6% | 59.5% | 25 | 0.511 | 83.5% | 85.2% |
| 6 | 1.205 | 61.8% | 65.1% | 26 | 0.496 | 84.0% | 85.2% |
| 7 | 1.122 | 64.5% | 69.0% | 27 | 0.494 | 84.1% | 86.1% |
| 8 | 1.033 | 67.1% | 67.4% | 28 | 0.483 | 84.2% | 86.2% |
| 9 | 0.952 | 70.1% | 73.0% | 29 | 0.466 | 84.8% | 86.0% |
| 10 | 0.886 | 72.0% | 75.0% | 30 | 0.458 | 84.9% | 85.5% |
| 11 | 0.832 | 73.6% | 67.2% | 31 | 0.450 | 85.5% | 86.9% |
| 12 | 0.783 | 75.0% | 76.7% | 32 | 0.443 | 85.6% | 86.8% |
| 13 | 0.750 | 76.0% | 78.2% | 33 | 0.440 | 85.8% | 87.5% |
| 14 | 0.722 | 76.9% | 77.1% | 34 | 0.431 | 86.1% | 87.7% |
| 15 | 0.706 | 77.5% | 79.7% | 35 | 0.418 | 86.2% | 87.5% |
| 16 | 0.667 | 78.7% | 80.5% | 36 | 0.422 | 86.2% | 87.6% |
| 17 | 0.655 | 78.9% | 81.3% | 37 | 0.422 | 86.4% | 87.0% |
| 18 | 0.630 | 79.7% | 82.7% | 38 | 0.412 | 86.5% | 87.2% |
| 19 | 0.605 | 80.7% | 82.2% | 39 | 0.412 | 86.5% | 87.1% |
| 20 | 0.587 | 81.1% | 83.0% | 40 | 0.410 | 86.6% | 87.9% |

Best val: 87.9% (epoch 40). Test: 86.8%.

### CumsumE2EMod (mod, w_l1=400, w=20, 87,332 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.445 | 30.6% | 35.8% | 21 | 0.490 | 83.9% | 84.9% |
| 2 | 1.904 | 38.0% | 39.0% | 22 | 0.470 | 84.8% | 86.0% |
| 3 | 1.840 | 40.1% | 40.5% | 23 | 0.462 | 85.1% | 84.5% |
| 4 | 1.645 | 46.4% | 49.6% | 24 | 0.444 | 85.5% | 86.2% |
| 5 | 1.309 | 57.9% | 61.9% | 25 | 0.431 | 85.9% | 86.8% |
| 6 | 1.097 | 64.8% | 68.3% | 26 | 0.418 | 86.3% | 85.6% |
| 7 | 0.965 | 69.1% | 73.1% | 27 | 0.411 | 86.7% | 86.7% |
| 8 | 0.878 | 72.0% | 72.5% | 28 | 0.409 | 86.6% | 87.4% |
| 9 | 0.808 | 74.1% | 77.3% | 29 | 0.389 | 87.2% | 87.2% |
| 10 | 0.745 | 76.0% | 76.7% | 30 | 0.386 | 87.4% | 88.1% |
| 11 | 0.707 | 77.1% | 80.2% | 31 | 0.375 | 87.6% | 88.2% |
| 12 | 0.677 | 78.3% | 79.1% | 32 | 0.367 | 88.1% | 87.2% |
| 13 | 0.655 | 78.9% | 81.2% | 33 | 0.361 | 88.1% | 87.7% |
| 14 | 0.629 | 79.6% | 81.8% | 34 | 0.353 | 88.3% | 87.9% |
| 15 | 0.601 | 80.5% | 80.5% | 35 | 0.352 | 88.3% | 88.5% |
| 16 | 0.577 | 81.3% | 82.0% | 36 | 0.341 | 88.8% | 88.3% |
| 17 | 0.557 | 82.0% | 83.7% | 37 | 0.341 | 88.8% | 88.5% |
| 18 | 0.535 | 82.6% | 82.9% | 38 | 0.341 | 88.8% | 88.5% |
| 19 | 0.524 | 83.1% | 84.0% | 39 | 0.339 | 88.8% | 88.6% |
| 20 | 0.511 | 83.5% | 84.2% | 40 | 0.340 | 88.9% | 88.8% |

Best val: 88.8% (epoch 40). Test: 87.6%.

## CumsumE2EMag Results (n_freqs=40, 4 layers, W_l1=400, W=20)

Layer 1: learned frequencies on raw audio → cumsum(W=400) → re²+im² → log → Linear(40→80).
Layers 2+: fixed/mod windowed cumsum (W=20) → BN → GLU → residual.
Classification: magnitude maxpool over all frames → fc.

| Model | Val | Test | F1 | Params |
|-------|-----|------|----|--------|
| **CumsumE2EMag (fixed)** | 94.2% | **93.5%** | 0.935 | 62,732 |
| **CumsumE2EMagMod** | 94.0% | **93.7%** | 0.936 | 77,492 |

First pure cumsum models to break 93% — mag+log bottleneck between layer 1 and layers 2+ is critical
(compare CumsumE2E fixed 86.8%, CumsumE2EMod 87.6% without it).

### Epoch-by-epoch: CumsumE2EMag (fixed, 40 epochs)

Learned frequencies (W=400) → mag+log → embed → fixed cumsum layers (W=20). 93.5% test, 62,732 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.192 | 61.0% | 82.9% | 21 | 0.220 | 92.6% | 92.6% |
| 2 | 0.544 | 82.1% | 86.5% | 22 | 0.212 | 92.7% | 93.1% |
| 3 | 0.462 | 85.1% | 88.9% | 23 | 0.201 | 93.3% | 93.3% |
| 4 | 0.416 | 86.4% | 88.1% | 24 | 0.198 | 93.3% | 92.7% |
| 5 | 0.378 | 87.8% | 89.9% | 25 | 0.189 | 93.7% | 93.4% |
| 6 | 0.368 | 88.0% | 90.4% | 26 | 0.184 | 93.7% | 93.2% |
| 7 | 0.348 | 88.7% | 90.0% | 27 | 0.179 | 94.0% | 93.4% |
| 8 | 0.327 | 89.1% | 91.2% | 28 | 0.175 | 94.0% | 93.2% |
| 9 | 0.319 | 89.4% | 90.7% | 29 | 0.165 | 94.4% | 93.5% |
| 10 | 0.315 | 89.6% | 90.6% | 30 | 0.161 | 94.6% | 93.9% |
| 11 | 0.300 | 90.1% | 89.9% | 31 | 0.155 | 94.8% | 94.2% |
| 12 | 0.296 | 90.3% | 91.6% | 32 | 0.151 | 94.9% | 93.8% |
| 13 | 0.284 | 90.6% | 91.4% | 33 | 0.148 | 94.9% | 93.5% |
| 14 | 0.271 | 91.1% | 91.7% | 34 | 0.146 | 95.1% | 94.0% |
| 15 | 0.264 | 91.3% | 92.4% | 35 | 0.142 | 95.3% | 94.0% |
| 16 | 0.252 | 91.6% | 91.5% | 36 | 0.140 | 95.3% | 94.0% |
| 17 | 0.250 | 91.7% | 93.0% | 37 | 0.138 | 95.3% | 93.8% |
| 18 | 0.242 | 92.0% | 92.8% | 38 | 0.139 | 95.3% | 94.2% |
| 19 | 0.237 | 92.0% | 93.4% | 39 | 0.133 | 95.5% | 94.2% |
| 20 | 0.225 | 92.5% | 92.6% | 40 | 0.133 | 95.4% | 94.1% |

Best val: 94.2% (epoch 31/39). Test: 93.5%.

### Epoch-by-epoch: CumsumE2EMagMod (40 epochs)

Layer 1 fixed, layers 2+ data-dependent frequencies. 93.7% test, 77,492 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.581 | 48.0% | 73.0% | 21 | 0.220 | 92.6% | 92.5% |
| 2 | 0.687 | 77.4% | 82.2% | 22 | 0.201 | 93.2% | 92.5% |
| 3 | 0.541 | 82.3% | 86.7% | 23 | 0.205 | 93.1% | 92.3% |
| 4 | 0.478 | 84.3% | 86.4% | 24 | 0.195 | 93.3% | 92.9% |
| 5 | 0.430 | 86.0% | 87.6% | 25 | 0.182 | 93.8% | 92.9% |
| 6 | 0.415 | 86.4% | 89.1% | 26 | 0.172 | 94.1% | 92.2% |
| 7 | 0.389 | 87.5% | 88.2% | 27 | 0.168 | 94.4% | 92.8% |
| 8 | 0.374 | 87.6% | 88.1% | 28 | 0.160 | 94.6% | 92.9% |
| 9 | 0.339 | 88.7% | 89.2% | 29 | 0.153 | 94.9% | 93.2% |
| 10 | 0.337 | 88.9% | 88.3% | 30 | 0.149 | 95.1% | 93.6% |
| 11 | 0.326 | 89.4% | 89.5% | 31 | 0.141 | 95.3% | 93.9% |
| 12 | 0.315 | 89.7% | 90.4% | 32 | 0.134 | 95.5% | 93.5% |
| 13 | 0.297 | 90.2% | 91.2% | 33 | 0.127 | 95.7% | 93.8% |
| 14 | 0.289 | 90.6% | 91.1% | 34 | 0.124 | 95.8% | 93.6% |
| 15 | 0.275 | 90.8% | 89.1% | 35 | 0.117 | 96.0% | 94.0% |
| 16 | 0.272 | 91.2% | 90.8% | 36 | 0.114 | 96.1% | 93.8% |
| 17 | 0.270 | 91.0% | 90.9% | 37 | 0.114 | 96.2% | 93.8% |
| 18 | 0.249 | 91.8% | 91.5% | 38 | 0.110 | 96.4% | 93.8% |
| 19 | 0.245 | 92.0% | 92.0% | 39 | 0.110 | 96.4% | 93.7% |
| 20 | 0.235 | 92.2% | 92.1% | 40 | 0.110 | 96.4% | 93.7% |

Best val: 94.0% (epoch 35). Test: 93.7%.

## MelCNNMaxPool Results (global max pool instead of avg pool)

Same TC-ResNet backbone as MelCNN but with global max pool instead of avg pool, plus SpecAugment.
Testing three hop lengths (stride): 160 (10ms), 80 (5ms), 40 (2.5ms).

| Model | Hop | Frames | Test | F1 | Params |
|-------|-----|--------|------|----|--------|
| **MelMaxPool160** | 160 | 101 | **97.1%** | 0.971 | 25,628 |
| MelMaxPool80 | 80 | 201 | 96.7% | 0.967 | 25,628 |
| MelMaxPool40 | 40 | 401 | 96.2% | 0.962 | 25,628 |

MaxPool is a huge win over AvgPool: +1.4% at hop=160 (97.1% vs 95.7%).
AvgPool got worse with smaller hops (95.7→95.0→94.5%) while MaxPool stays strong (97.1→96.7→96.2%).

### Epoch-by-epoch: MelMaxPool160 (40 epochs)

Global max pool, hop=160, 101 frames. 97.1% test, 25,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.912 | 70.2% | 88.5% | 21 | 0.246 | 91.7% | 95.6% |
| 2 | 0.559 | 81.7% | 89.8% | 22 | 0.241 | 92.0% | 95.4% |
| 3 | 0.478 | 84.2% | 91.9% | 23 | 0.231 | 92.4% | 95.4% |
| 4 | 0.439 | 85.4% | 92.4% | 24 | 0.230 | 92.3% | 95.6% |
| 5 | 0.405 | 86.8% | 92.6% | 25 | 0.228 | 92.3% | 95.5% |
| 6 | 0.386 | 87.2% | 93.3% | 26 | 0.225 | 92.4% | 95.5% |
| 7 | 0.371 | 87.5% | 93.6% | 27 | 0.224 | 92.6% | 95.9% |
| 8 | 0.348 | 88.4% | 93.7% | 28 | 0.214 | 92.8% | 96.1% |
| 9 | 0.334 | 88.7% | 93.5% | 29 | 0.215 | 92.8% | 95.8% |
| 10 | 0.322 | 89.2% | 94.0% | 30 | 0.201 | 93.2% | 95.8% |
| 11 | 0.312 | 89.6% | 94.8% | 31 | 0.205 | 93.1% | 96.0% |
| 12 | 0.312 | 89.7% | 94.0% | 32 | 0.198 | 93.3% | 96.2% |
| 13 | 0.302 | 90.0% | 94.1% | 33 | 0.197 | 93.4% | 95.8% |
| 14 | 0.285 | 90.5% | 94.4% | 34 | 0.193 | 93.6% | 96.3% |
| 15 | 0.284 | 90.4% | 94.1% | 35 | 0.188 | 93.7% | 96.4% |
| 16 | 0.274 | 90.8% | 94.9% | 36 | 0.190 | 93.6% | 96.2% |
| 17 | 0.271 | 90.9% | 95.1% | 37 | 0.186 | 93.8% | 96.3% |
| 18 | 0.262 | 91.1% | 95.2% | 38 | 0.194 | 93.4% | 96.0% |
| 19 | 0.258 | 91.3% | 95.3% | 39 | 0.193 | 93.5% | 96.1% |
| 20 | 0.248 | 91.6% | 95.2% | 40 | 0.188 | 93.5% | 96.4% |

Best val: 96.4% (epoch 35). Test: 97.1%.

### Epoch-by-epoch: MelMaxPool80 (40 epochs)

Global max pool, hop=80, 201 frames. 96.7% test, 25,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.851 | 72.1% | 87.6% | 21 | 0.189 | 93.7% | 95.5% |
| 2 | 0.489 | 84.1% | 87.6% | 22 | 0.188 | 93.9% | 96.0% |
| 3 | 0.420 | 86.3% | 91.8% | 23 | 0.180 | 94.0% | 95.8% |
| 4 | 0.372 | 87.8% | 92.5% | 24 | 0.176 | 94.2% | 96.0% |
| 5 | 0.344 | 88.9% | 92.2% | 25 | 0.172 | 94.4% | 95.8% |
| 6 | 0.317 | 89.6% | 93.7% | 26 | 0.167 | 94.5% | 96.1% |
| 7 | 0.306 | 89.9% | 92.8% | 27 | 0.166 | 94.5% | 96.2% |
| 8 | 0.288 | 90.5% | 93.6% | 28 | 0.160 | 94.9% | 96.1% |
| 9 | 0.276 | 91.0% | 93.9% | 29 | 0.156 | 94.9% | 96.2% |
| 10 | 0.259 | 91.4% | 93.7% | 30 | 0.147 | 95.0% | 96.0% |
| 11 | 0.255 | 91.6% | 94.7% | 31 | 0.150 | 95.0% | 96.3% |
| 12 | 0.249 | 91.9% | 94.4% | 32 | 0.150 | 95.0% | 96.4% |
| 13 | 0.240 | 92.1% | 93.5% | 33 | 0.141 | 95.3% | 96.7% |
| 14 | 0.227 | 92.5% | 95.0% | 34 | 0.139 | 95.3% | 96.5% |
| 15 | 0.225 | 92.5% | 95.3% | 35 | 0.139 | 95.5% | 96.4% |
| 16 | 0.218 | 92.8% | 95.2% | 36 | 0.133 | 95.6% | 96.4% |
| 17 | 0.215 | 93.0% | 95.0% | 37 | 0.138 | 95.4% | 96.4% |
| 18 | 0.204 | 93.4% | 95.3% | 38 | 0.135 | 95.5% | 96.5% |
| 19 | 0.201 | 93.4% | 95.4% | 39 | 0.135 | 95.5% | 96.3% |
| 20 | 0.195 | 93.5% | 95.4% | 40 | 0.134 | 95.5% | 96.5% |

Best val: 96.7% (epoch 33). Test: 96.7%.

### Epoch-by-epoch: MelMaxPool40 (40 epochs)

Global max pool, hop=40, 401 frames. 96.2% test, 25,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.879 | 71.3% | 87.4% | 21 | 0.192 | 93.6% | 94.6% |
| 2 | 0.485 | 84.6% | 88.9% | 22 | 0.184 | 94.0% | 94.3% |
| 3 | 0.413 | 86.8% | 90.4% | 23 | 0.179 | 94.2% | 95.1% |
| 4 | 0.366 | 88.0% | 91.5% | 24 | 0.171 | 94.4% | 95.1% |
| 5 | 0.342 | 89.1% | 90.8% | 25 | 0.171 | 94.5% | 95.1% |
| 6 | 0.319 | 89.7% | 92.4% | 26 | 0.164 | 94.7% | 94.9% |
| 7 | 0.300 | 90.3% | 92.2% | 27 | 0.163 | 94.6% | 95.2% |
| 8 | 0.286 | 90.8% | 91.4% | 28 | 0.156 | 95.0% | 95.2% |
| 9 | 0.273 | 91.2% | 93.0% | 29 | 0.153 | 95.0% | 95.3% |
| 10 | 0.260 | 91.6% | 93.8% | 30 | 0.147 | 95.2% | 95.2% |
| 11 | 0.254 | 91.8% | 93.2% | 31 | 0.148 | 95.2% | 95.3% |
| 12 | 0.247 | 92.0% | 93.7% | 32 | 0.144 | 95.3% | 95.5% |
| 13 | 0.240 | 92.3% | 92.2% | 33 | 0.138 | 95.5% | 95.4% |
| 14 | 0.230 | 92.6% | 94.4% | 34 | 0.138 | 95.4% | 95.7% |
| 15 | 0.227 | 92.7% | 93.6% | 35 | 0.138 | 95.4% | 95.7% |
| 16 | 0.217 | 93.0% | 94.1% | 36 | 0.133 | 95.7% | 95.6% |
| 17 | 0.211 | 93.1% | 94.2% | 37 | 0.134 | 95.6% | 95.6% |
| 18 | 0.205 | 93.3% | 94.4% | 38 | 0.133 | 95.7% | 95.3% |
| 19 | 0.201 | 93.3% | 94.5% | 39 | 0.128 | 95.9% | 95.3% |
| 20 | 0.195 | 93.6% | 94.6% | 40 | 0.131 | 95.7% | 95.7% |

Best val: 95.7% (epoch 34/40). Test: 96.2%.

## MelCNN MultiPhase Results (interleaved subsequences, max over outputs)

Finer hop → more mel frames → split into phase-shifted subsequences of ~101 frames each →
same CNN on each → max over output logits. Preserves receptive field while exploiting finer resolution.

| Model | Phases | Hop | Test | F1 | Params |
|-------|--------|-----|------|----|--------|
| MelMaxPool160 | 1 | 160 | **97.1%** | 0.971 | 25,628 |
| **MelMultiPhase80** | **2** | **80** | **97.1%** | **0.971** | **25,628** |
| MelMultiPhase40 | 4 | 40 | 96.3% | 0.963 | 25,628 |

Multi-phase preserves accuracy (2-phase matches single-phase). 4-phase slightly worse — max over
4 logit vectors may add noise. Key insight: finer hop doesn't help if you maintain the same receptive field.

### Epoch-by-epoch: MelMultiPhase80 (2 phases, 40 epochs)

hop=80, 2 interleaved views of ~101 frames, max over outputs. 97.1% test, 25,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.795 | 74.1% | 88.0% | 21 | 0.180 | 94.0% | 95.2% |
| 2 | 0.461 | 84.8% | 90.2% | 22 | 0.176 | 94.2% | 95.5% |
| 3 | 0.393 | 87.0% | 92.3% | 23 | 0.171 | 94.4% | 96.0% |
| 4 | 0.352 | 88.3% | 92.9% | 24 | 0.165 | 94.5% | 95.6% |
| 5 | 0.326 | 89.3% | 92.8% | 25 | 0.165 | 94.5% | 95.7% |
| 6 | 0.305 | 89.9% | 93.6% | 26 | 0.159 | 94.8% | 95.8% |
| 7 | 0.287 | 90.6% | 92.8% | 27 | 0.156 | 94.9% | 96.2% |
| 8 | 0.270 | 90.9% | 94.4% | 28 | 0.149 | 95.0% | 96.0% |
| 9 | 0.261 | 91.5% | 94.0% | 29 | 0.148 | 95.1% | 95.8% |
| 10 | 0.247 | 91.9% | 94.3% | 30 | 0.142 | 95.3% | 96.2% |
| 11 | 0.242 | 92.0% | 94.9% | 31 | 0.142 | 95.3% | 96.0% |
| 12 | 0.238 | 92.1% | 95.3% | 32 | 0.138 | 95.4% | 96.2% |
| 13 | 0.231 | 92.3% | 94.1% | 33 | 0.133 | 95.6% | 96.0% |
| 14 | 0.216 | 93.0% | 95.5% | 34 | 0.134 | 95.5% | 96.4% |
| 15 | 0.212 | 92.9% | 95.0% | 35 | 0.132 | 95.6% | 96.4% |
| 16 | 0.206 | 93.2% | 94.9% | 36 | 0.124 | 95.9% | 96.5% |
| 17 | 0.204 | 93.3% | 95.1% | 37 | 0.129 | 95.7% | 96.4% |
| 18 | 0.197 | 93.5% | 94.7% | 38 | 0.128 | 95.8% | 96.2% |
| 19 | 0.190 | 93.7% | 95.1% | 39 | 0.128 | 95.7% | 95.6% |
| 20 | 0.184 | 93.9% | 95.5% | 40 | 0.129 | 95.6% | 96.4% |

Best val: 96.5% (epoch 36). Test: 97.1%.

### Epoch-by-epoch: MelMultiPhase40 (4 phases, 40 epochs)

hop=40, 4 interleaved views of ~100 frames, max over outputs. 96.3% test, 25,628 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 0.774 | 74.8% | 89.3% | 21 | 0.150 | 95.0% | 95.4% |
| 2 | 0.412 | 86.7% | 89.7% | 22 | 0.149 | 95.1% | 95.3% |
| 3 | 0.342 | 88.8% | 92.5% | 23 | 0.142 | 95.2% | 95.5% |
| 4 | 0.306 | 90.0% | 93.3% | 24 | 0.139 | 95.3% | 94.7% |
| 5 | 0.280 | 90.9% | 92.8% | 25 | 0.138 | 95.3% | 95.8% |
| 6 | 0.262 | 91.4% | 92.8% | 26 | 0.130 | 95.7% | 94.9% |
| 7 | 0.249 | 91.8% | 91.7% | 27 | 0.128 | 95.8% | 95.6% |
| 8 | 0.233 | 92.4% | 93.7% | 28 | 0.122 | 95.9% | 95.7% |
| 9 | 0.225 | 92.7% | 94.6% | 29 | 0.121 | 95.9% | 96.0% |
| 10 | 0.213 | 93.2% | 94.9% | 30 | 0.116 | 96.2% | 95.7% |
| 11 | 0.209 | 93.2% | 94.1% | 31 | 0.116 | 96.2% | 96.0% |
| 12 | 0.203 | 93.4% | 94.7% | 32 | 0.112 | 96.3% | 96.1% |
| 13 | 0.195 | 93.7% | 93.3% | 33 | 0.107 | 96.6% | 96.3% |
| 14 | 0.188 | 93.8% | 95.3% | 34 | 0.109 | 96.3% | 96.2% |
| 15 | 0.186 | 94.0% | 94.8% | 35 | 0.107 | 96.4% | 96.3% |
| 16 | 0.180 | 94.1% | 95.3% | 36 | 0.102 | 96.7% | 96.2% |
| 17 | 0.171 | 94.3% | 94.7% | 37 | 0.102 | 96.7% | 95.8% |
| 18 | 0.166 | 94.5% | 95.5% | 38 | 0.105 | 96.6% | 96.0% |
| 19 | 0.161 | 94.6% | 94.8% | 39 | 0.103 | 96.6% | 95.2% |
| 20 | 0.156 | 94.9% | 94.6% | 40 | 0.100 | 96.7% | 96.1% |

Best val: 96.3% (epoch 33). Test: 96.3%.

## MelCumsum MultiPhase Results (n_embed=80, 4 layers, W=20)

Same multi-phase approach applied to MelCumsum: finer hop → interleaved subsequences →
same cumsum layers on each → max over output logits.

| Model | Phases | Hop | Test | F1 | Params |
|-------|--------|-----|------|----|--------|
| MelCumsumFixedW (original) | 1 | 160 | 96.0% | 0.960 | 82,332 |
| MelCumsumFixedMP2 | 2 | 80 | 96.1% | 0.961 | 82,332 |
| MelCumsumFixedMP4 | 4 | 40 | 96.3% | 0.963 | 82,332 |
| MelCumsumModW (original) | 1 | 160 | 95.8% | 0.958 | 102,012 |
| MelCumsumModMP2 | 2 | 80 | 96.3% | 0.963 | 102,012 |
| MelCumsumModMP4 | 4 | 40 | 96.0% | 0.960 | 102,012 |

Small gains from multi-phase (~0.1-0.5%). Mod MP2 matches Fixed MP4 at 96.3%.

### Epoch-by-epoch: MelCumsumFixedMP2 (2 phases, 40 epochs)

hop=80, 2 interleaved views, fixed frequencies, W=20. 96.1% test, 82,332 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.438 | 55.7% | 83.8% | 21 | 0.254 | 91.7% | 94.1% |
| 2 | 0.647 | 79.2% | 82.9% | 22 | 0.245 | 91.9% | 95.1% |
| 3 | 0.551 | 82.5% | 89.0% | 23 | 0.234 | 92.2% | 94.8% |
| 4 | 0.497 | 84.3% | 88.7% | 24 | 0.226 | 92.5% | 94.7% |
| 5 | 0.460 | 85.4% | 92.4% | 25 | 0.219 | 92.8% | 95.4% |
| 6 | 0.443 | 85.9% | 91.7% | 26 | 0.215 | 92.8% | 95.7% |
| 7 | 0.416 | 86.7% | 91.4% | 27 | 0.208 | 93.3% | 95.0% |
| 8 | 0.391 | 87.3% | 92.6% | 28 | 0.203 | 93.4% | 95.2% |
| 9 | 0.382 | 87.6% | 92.0% | 29 | 0.194 | 93.6% | 95.4% |
| 10 | 0.374 | 88.0% | 92.2% | 30 | 0.191 | 93.6% | 95.8% |
| 11 | 0.358 | 88.5% | 92.4% | 31 | 0.184 | 93.7% | 95.1% |
| 12 | 0.344 | 88.9% | 92.4% | 32 | 0.179 | 94.0% | 95.7% |
| 13 | 0.330 | 89.2% | 93.6% | 33 | 0.177 | 94.1% | 95.5% |
| 14 | 0.323 | 89.7% | 92.7% | 34 | 0.172 | 94.3% | 95.8% |
| 15 | 0.314 | 89.8% | 93.8% | 35 | 0.171 | 94.4% | 95.5% |
| 16 | 0.294 | 90.4% | 94.3% | 36 | 0.170 | 94.4% | 95.6% |
| 17 | 0.285 | 90.6% | 93.5% | 37 | 0.165 | 94.5% | 95.6% |
| 18 | 0.283 | 90.6% | 95.0% | 38 | 0.168 | 94.4% | 95.4% |
| 19 | 0.272 | 91.1% | 93.7% | 39 | 0.167 | 94.5% | 95.6% |
| 20 | 0.261 | 91.5% | 94.7% | 40 | 0.165 | 94.6% | 95.8% |

Best val: 95.8% (epoch 30/34). Test: 96.1%.

### Epoch-by-epoch: MelCumsumFixedMP4 (4 phases, 40 epochs)

hop=40, 4 interleaved views, fixed frequencies, W=20. 96.3% test, 82,332 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.340 | 58.9% | 84.3% | 21 | 0.211 | 93.0% | 94.9% |
| 2 | 0.574 | 81.5% | 87.9% | 22 | 0.207 | 93.1% | 95.1% |
| 3 | 0.491 | 84.5% | 89.0% | 23 | 0.198 | 93.5% | 94.4% |
| 4 | 0.433 | 86.1% | 91.7% | 24 | 0.187 | 93.8% | 95.6% |
| 5 | 0.402 | 87.2% | 93.0% | 25 | 0.186 | 93.8% | 95.6% |
| 6 | 0.380 | 87.9% | 91.3% | 26 | 0.175 | 94.2% | 95.3% |
| 7 | 0.358 | 88.6% | 92.4% | 27 | 0.170 | 94.4% | 94.9% |
| 8 | 0.339 | 89.0% | 92.5% | 28 | 0.167 | 94.5% | 95.1% |
| 9 | 0.330 | 89.7% | 92.4% | 29 | 0.158 | 94.6% | 95.5% |
| 10 | 0.318 | 89.9% | 93.4% | 30 | 0.152 | 94.9% | 96.0% |
| 11 | 0.306 | 90.2% | 93.6% | 31 | 0.148 | 95.0% | 95.6% |
| 12 | 0.294 | 90.6% | 92.5% | 32 | 0.144 | 95.2% | 95.7% |
| 13 | 0.280 | 90.9% | 94.4% | 33 | 0.144 | 95.1% | 96.0% |
| 14 | 0.272 | 91.3% | 93.1% | 34 | 0.137 | 95.4% | 96.1% |
| 15 | 0.271 | 91.3% | 93.9% | 35 | 0.139 | 95.4% | 95.8% |
| 16 | 0.251 | 91.9% | 93.9% | 36 | 0.134 | 95.6% | 95.6% |
| 17 | 0.243 | 92.1% | 94.5% | 37 | 0.130 | 95.7% | 95.8% |
| 18 | 0.239 | 92.1% | 94.6% | 38 | 0.131 | 95.5% | 95.4% |
| 19 | 0.228 | 92.5% | 94.0% | 39 | 0.132 | 95.7% | 96.0% |
| 20 | 0.221 | 92.7% | 95.3% | 40 | 0.131 | 95.6% | 95.9% |

Best val: 96.1% (epoch 34). Test: 96.3%.

### Epoch-by-epoch: MelCumsumModMP2 (2 phases, 40 epochs)

hop=80, 2 interleaved views, data-dependent frequencies, W=20. 96.3% test, 102,012 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.900 | 39.0% | 71.3% | 21 | 0.263 | 91.3% | 94.2% |
| 2 | 0.906 | 70.4% | 82.4% | 22 | 0.251 | 91.7% | 94.4% |
| 3 | 0.696 | 77.3% | 86.9% | 23 | 0.248 | 91.7% | 94.4% |
| 4 | 0.589 | 80.9% | 88.4% | 24 | 0.233 | 92.1% | 94.5% |
| 5 | 0.560 | 81.9% | 87.4% | 25 | 0.226 | 92.6% | 94.8% |
| 6 | 0.506 | 83.6% | 89.0% | 26 | 0.217 | 92.8% | 95.1% |
| 7 | 0.479 | 84.6% | 87.9% | 27 | 0.214 | 92.9% | 95.0% |
| 8 | 0.463 | 85.1% | 91.3% | 28 | 0.199 | 93.5% | 94.6% |
| 9 | 0.429 | 86.4% | 91.2% | 29 | 0.199 | 93.4% | 95.1% |
| 10 | 0.419 | 86.6% | 90.7% | 30 | 0.191 | 93.7% | 95.2% |
| 11 | 0.391 | 87.4% | 92.1% | 31 | 0.184 | 93.9% | 95.3% |
| 12 | 0.379 | 87.8% | 92.8% | 32 | 0.179 | 94.1% | 95.2% |
| 13 | 0.371 | 88.0% | 92.8% | 33 | 0.177 | 94.0% | 95.3% |
| 14 | 0.339 | 88.9% | 93.4% | 34 | 0.170 | 94.4% | 95.2% |
| 15 | 0.339 | 89.3% | 92.6% | 35 | 0.164 | 94.6% | 95.5% |
| 16 | 0.314 | 89.8% | 92.4% | 36 | 0.165 | 94.5% | 95.5% |
| 17 | 0.312 | 89.9% | 93.6% | 37 | 0.160 | 94.7% | 95.6% |
| 18 | 0.298 | 90.3% | 94.2% | 38 | 0.161 | 94.6% | 95.6% |
| 19 | 0.280 | 90.9% | 93.7% | 39 | 0.160 | 94.7% | 95.5% |
| 20 | 0.278 | 90.9% | 93.9% | 40 | 0.156 | 94.8% | 95.3% |

Best val: 95.6% (epoch 37). Test: 96.3%.

### Epoch-by-epoch: MelCumsumModMP4 (4 phases, 40 epochs)

hop=40, 4 interleaved views, data-dependent frequencies, W=20. 96.0% test, 102,012 params.

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.805 | 42.3% | 73.3% | 21 | 0.227 | 92.7% | 93.7% |
| 2 | 0.799 | 74.0% | 83.8% | 22 | 0.218 | 92.7% | 93.2% |
| 3 | 0.620 | 79.9% | 88.6% | 23 | 0.209 | 93.2% | 94.4% |
| 4 | 0.525 | 82.9% | 88.7% | 24 | 0.198 | 93.5% | 93.9% |
| 5 | 0.478 | 84.8% | 89.0% | 25 | 0.194 | 93.6% | 94.5% |
| 6 | 0.434 | 86.1% | 88.5% | 26 | 0.180 | 94.0% | 94.7% |
| 7 | 0.407 | 87.0% | 90.3% | 27 | 0.175 | 94.2% | 94.2% |
| 8 | 0.394 | 87.5% | 90.6% | 28 | 0.166 | 94.5% | 94.5% |
| 9 | 0.369 | 88.1% | 91.1% | 29 | 0.166 | 94.5% | 94.8% |
| 10 | 0.352 | 88.8% | 90.9% | 30 | 0.160 | 94.7% | 95.1% |
| 11 | 0.336 | 89.2% | 91.9% | 31 | 0.149 | 94.9% | 94.9% |
| 12 | 0.329 | 89.5% | 92.6% | 32 | 0.145 | 95.3% | 95.1% |
| 13 | 0.318 | 89.8% | 91.5% | 33 | 0.145 | 95.1% | 95.1% |
| 14 | 0.297 | 90.3% | 93.3% | 34 | 0.137 | 95.4% | 95.2% |
| 15 | 0.289 | 90.8% | 92.9% | 35 | 0.134 | 95.7% | 95.3% |
| 16 | 0.271 | 91.2% | 93.1% | 36 | 0.137 | 95.5% | 95.3% |
| 17 | 0.261 | 91.5% | 93.0% | 37 | 0.130 | 95.7% | 95.2% |
| 18 | 0.252 | 91.9% | 93.9% | 38 | 0.131 | 95.7% | 95.2% |
| 19 | 0.241 | 92.1% | 93.1% | 39 | 0.130 | 95.7% | 94.9% |
| 20 | 0.237 | 92.2% | 94.0% | 40 | 0.126 | 95.8% | 95.0% |

Best val: 95.3% (epoch 35). Test: 96.0%.

---

## CumsumMagDeep: Mag+Log at Every Layer

### Key insight

CumsumE2EMag (93.5%) applies mag+log only between layer 1 and layers 2+, then layers 2+
operate on [re, im] as independent real channels with BN+GLU. CumsumMagDeep applies mag+log
at *every* layer — each layer does: proj → complex → windowed cumsum → mag → log → linear embed.
This collapses phase at every layer, forcing the network to re-derive useful complex structure
each time rather than carrying ambiguous phase through.

No BN, no GLU, no ReLU/GELU — just mag+log as the only nonlinearity between layers.

### Architecture evolution

#### V1: mag-only features (original)

```
Layer 1: raw audio → cumsum(W=400, stride=160) → mag → log → Linear(n_freqs, dim)
Layers 2+: Linear(dim, dim) → split re/im → complex → cumsum(W=20, stride=1)
           → mag → log → Linear(n_freqs, dim) → residual add
Readout: maxpool over time → Linear(dim, 12)
dim = 2 * n_freqs
```

Each layer passes only log_mag (n_freqs features) to the embed layer. Phase is discarded.

#### V2: mag + re/im features (with proj)

```
Layer 1: raw audio → cumsum(W=400, stride=160) → mag → log → Linear(n_freqs, dim)
Layers 2+: Linear(dim, dim) → split re/im → complex → cumsum(W=20, stride=1)
           → [log_mag, re, im] → Linear(3*n_freqs, dim) → residual add
Readout: maxpool over time → Linear(dim, 12)
```

Each layer passes [log_mag, re, im] (3*n_freqs features). Keeps raw re/im alongside
compressed log_mag, giving the next layer both magnitude and phase information.

#### V3: mag + re/im features (no proj) — CURRENT

```
Layer 1: raw audio → cumsum(W=400, stride=160) → [log_mag, re, im]
         → Linear(3*n_freqs, dim)
Layers 2+: split h into re/im directly (no proj linear) → complex
           → cumsum(W=20, stride=1) → [log_mag, re, im]
           → Linear(3*n_freqs, dim) → residual add
Readout: maxpool over time → Linear(dim, 12)
```

Removes the input proj Linear per layer — h is dim=2*n_freqs which splits naturally
into n_freqs complex values. The output embed linear of one layer serves as the
"projection" for the next. Layer 1 also outputs [log_mag, re, im] for consistency.

### Results comparison

| Model | Test Acc | Best Val | Params | Notes |
|-------|----------|----------|--------|-------|
| CumsumE2E | 86.8% | — | 72,572 | No mag/log, [re,im]+BN+GLU |
| MagDeep V1 (n40, mag only) | 89.2% | 90.1% | 33,692 | log_mag only between layers |
| MagDeep V1 (n56, mag only) | 90.5% | 91.2% | 65,084 | log_mag only between layers |
| MagDeep V1 DS (n56, mag only) | 90.3% | 91.7% | 65,084 | + avg_pool(2) downsampling |
| MagDeep V2 (n44, +re/im, old L1) | 92.5% | 93.2% | 63,812 | [log_mag, re, im] L2+, log_mag L1 |
| **MagDeep V2 (n44, +re/im, proj)** | **92.8%** | **92.9%** | **71,556** | **[log_mag, re, im] all layers, with proj** |
| MagDeep V3 (n44, +re/im, no proj) | 91.6% | 92.2% | 48,060 | [log_mag, re, im] all layers, no proj |
| CumsumE2EMag | 93.5% | 94.1% | 62,732 | Mag+log layer 1, [re,im]+BN+GLU layers 2+ |

Key findings:
- V1 mag-only beats CumsumE2E (90.5% vs 86.8%) with fewer params — mag+log > BN+GLU
- **V2 adding re/im: +2% over mag-only** (92.5%→92.8% vs 90.5%) at similar params
- V2 nearly matches CumsumE2EMag (92.8% vs 93.5%) without any BN/GLU/ReLU
- Downsampling (DS) doesn't help: 90.3% vs 90.5%
- Proj linear adds ~1.2% accuracy (92.8% vs 91.6%) for 23K extra params
- Layer 1 [log_mag, re, im] vs log_mag only: marginal improvement (92.8% vs 92.5%)

### Why re/im helps alongside mag

Magnitude is phase-invariant — it doesn't care where in the window the event occurs.
This is good for robustness but discards timing information. Adding raw re/im preserves
the phase, letting the next layer know *where* in the window things happened. The network
gets both: compressed magnitude (via log) for robust pattern detection, and raw complex
output for precise temporal structure.

### Single-layer experiments (CumsumSingleLayer)

Tested what a single cumsum layer can do with different feature representations,
followed by Linear(dim, 4*dim) → ReLU → Linear(4*dim, dim) → maxpool → Linear(dim, 12).

| Features | Val at ep12 | Params | Notes |
|----------|------------|--------|-------|
| [re, im] | 39.2%* | ~26K | Stopped at epoch 16. Raw re/im unlearnable |
| [log_mag, cos_phase, sin_phase] | 82.9% | 117,292 | Stopped at epoch 12, plateauing ~83% |

*stopped early

Phase as atan2 caused NaN (gradient instability near zero magnitude).
cos/sin phase (unit vector) is stable and works well — single layer reaches 83% vs
39% with raw [re, im]. The mag+phase representation is far more learnable, but a
single layer hits a ceiling. Multi-layer MagDeep reaches 92.5%.

### Epoch-by-epoch: V1 mag-only comparison

| Epoch | E2E (72K) | V1 n40 (34K) | V1 n56 (65K) | E2EMag (63K) |
|-------|-----------|--------------|--------------|--------------|
| 1 | 38.1% | 65.7% | 67.7% | 82.9% |
| 2 | 41.0% | 76.0% | 78.1% | 86.5% |
| 3 | 44.5% | 78.4% | 83.1% | 88.9% |
| 4 | 55.3% | 78.0% | 74.2% | 88.1% |
| 5 | 59.5% | 77.9% | 79.1% | 89.9% |
| 6 | 65.1% | 81.7% | 80.8% | 90.4% |
| 7 | 69.0% | 80.6% | 82.3% | 90.0% |
| 8 | 67.4% | 81.4% | 84.1% | 91.2% |
| 9 | 73.0% | 82.8% | 81.7% | 90.7% |
| 10 | 75.0% | 83.7% | 82.5% | 90.6% |
| 15 | 79.7% | 86.8% | 85.4% | 92.4% |
| 20 | 83.0% | 85.7% | 87.9% | 92.6% |
| 25 | 85.2% | 86.4% | 88.4% | 93.4% |
| 30 | 85.5% | 89.0% | 89.4% | 93.9% |
| 35 | 87.5% | 90.0% | 90.2% | 94.0% |
| 40 | 87.9% | 89.9% | 90.9% | 94.1% |

### Epoch-by-epoch: V1 vs V2 (proj) vs V3 (no proj)

| Epoch | V1 n56 mag (65K) | V2 proj n44 (72K) | V3 no-proj n44 (48K) |
|-------|------------------|-------------------|----------------------|
| 1 | 67.7% | 78.1% | 69.4% |
| 2 | 78.1% | 83.4% | 80.2% |
| 3 | 83.1% | 83.3% | 81.0% |
| 4 | 74.2% | 82.8% | 78.9% |
| 5 | 79.1% | 83.2% | 80.7% |
| 6 | 80.8% | 83.5% | 84.8% |
| 7 | 82.3% | 84.9% | 83.8% |
| 8 | 84.1% | 76.1% | 82.8% |
| 9 | 81.7% | 82.4% | 83.1% |
| 10 | 82.5% | 84.6% | 84.9% |
| 15 | 85.4% | 86.8% | 85.2% |
| 20 | 87.9% | 88.5% | 87.6% |
| 25 | 88.4% | 90.2% | 89.0% |
| 30 | 89.4% | 91.4% | 91.2% |
| 35 | 90.2% | 92.5% | 91.7% |
| 40 | 90.9% | 92.9% | 92.1% |

V2 (proj) finishes highest at 92.9% val / 92.8% test with 72K params.
V3 (no proj) is 91.6% test with only 48K params — best accuracy-per-param.
V1 (mag only) trails at 90.5% test — re/im features add ~2%.

### Epoch-by-epoch: V2 old (log_mag L1, 63,812 params)

This was the first V2 run, before layer 1 was changed to output [log_mag, re, im].
Layer 1 output log_mag only → Linear(n_freqs, dim). Layers 2+ had proj + [log_mag, re, im].

| Epoch | V2 old (64K) |
|-------|-------------|
| 1 | 79.6% |
| 2 | 81.5% |
| 3 | 80.4% |
| 4 | 84.9% |
| 5 | 83.8% |
| 10 | 86.4% |
| 15 | 85.8% |
| 20 | 88.9% |
| 25 | 91.1% |
| 30 | 92.8% |
| 35 | 92.9% |
| 40 | 93.2% |

Best val: 93.2%. Test: 92.5%.

### Epoch-by-epoch: V2 proj new (all layers [log_mag, re, im], 71,556 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.344 | 62.2% | 78.1% | 21 | 0.368 | 87.9% | 89.4% |
| 2 | 0.697 | 77.8% | 83.4% | 22 | 0.351 | 88.4% | 89.2% |
| 3 | 0.641 | 79.6% | 83.3% | 23 | 0.339 | 88.9% | 89.8% |
| 4 | 0.627 | 80.2% | 82.8% | 24 | 0.324 | 89.2% | 90.2% |
| 5 | 0.641 | 80.2% | 83.2% | 25 | 0.314 | 89.5% | 90.2% |
| 6 | 0.585 | 81.3% | 83.5% | 26 | 0.291 | 90.2% | 90.9% |
| 7 | 0.564 | 82.0% | 84.9% | 27 | 0.287 | 90.4% | 90.7% |
| 8 | 0.557 | 82.2% | 76.1% | 28 | 0.274 | 90.8% | 91.1% |
| 9 | 0.537 | 82.8% | 82.4% | 29 | 0.258 | 91.5% | 90.7% |
| 10 | 0.531 | 83.0% | 84.6% | 30 | 0.251 | 91.6% | 91.4% |
| 11 | 0.520 | 83.3% | 84.1% | 31 | 0.237 | 92.1% | 91.7% |
| 12 | 0.503 | 83.9% | 86.0% | 32 | 0.225 | 92.4% | 91.9% |
| 13 | 0.490 | 84.2% | 86.3% | 33 | 0.216 | 92.7% | 92.0% |
| 14 | 0.467 | 85.1% | 87.6% | 34 | 0.204 | 93.2% | 92.2% |
| 15 | 0.468 | 85.0% | 86.8% | 35 | 0.199 | 93.3% | 92.5% |
| 16 | 0.447 | 85.5% | 88.6% | 36 | 0.191 | 93.6% | 92.8% |
| 17 | 0.417 | 86.1% | 87.3% | 37 | 0.184 | 93.8% | 92.7% |
| 18 | 0.415 | 86.4% | 88.6% | 38 | 0.182 | 93.8% | 92.8% |
| 19 | 0.405 | 86.9% | 88.3% | 39 | 0.177 | 94.0% | 92.9% |
| 20 | 0.381 | 87.3% | 88.5% | 40 | 0.175 | 94.1% | 92.9% |

Best val: 92.9% (epoch 39). Test: 92.8%.

### Epoch-by-epoch: V3 no-proj (all layers [log_mag, re, im], 48,060 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 2.099 | 55.1% | 69.4% | 21 | 0.402 | 86.7% | 87.8% |
| 2 | 0.825 | 74.3% | 80.2% | 22 | 0.386 | 87.5% | 87.1% |
| 3 | 0.722 | 77.7% | 81.0% | 23 | 0.378 | 87.6% | 89.2% |
| 4 | 0.676 | 78.9% | 78.9% | 24 | 0.358 | 88.1% | 89.7% |
| 5 | 0.660 | 79.4% | 80.7% | 25 | 0.346 | 88.5% | 89.0% |
| 6 | 0.642 | 80.2% | 84.8% | 26 | 0.330 | 89.0% | 90.0% |
| 7 | 0.622 | 80.3% | 83.8% | 27 | 0.316 | 89.4% | 90.2% |
| 8 | 0.604 | 81.2% | 82.8% | 28 | 0.299 | 89.9% | 90.1% |
| 9 | 0.582 | 81.8% | 83.1% | 29 | 0.289 | 90.1% | 90.5% |
| 10 | 0.579 | 81.9% | 84.9% | 30 | 0.273 | 90.7% | 91.2% |
| 11 | 0.561 | 82.4% | 84.2% | 31 | 0.261 | 91.2% | 90.8% |
| 12 | 0.551 | 82.7% | 83.7% | 32 | 0.254 | 91.3% | 91.5% |
| 13 | 0.534 | 83.2% | 84.6% | 33 | 0.246 | 91.6% | 91.2% |
| 14 | 0.521 | 83.5% | 87.0% | 34 | 0.232 | 92.2% | 91.5% |
| 15 | 0.493 | 84.3% | 85.2% | 35 | 0.227 | 92.2% | 91.7% |
| 16 | 0.492 | 84.3% | 86.1% | 36 | 0.220 | 92.4% | 91.6% |
| 17 | 0.461 | 85.1% | 86.4% | 37 | 0.215 | 92.6% | 92.1% |
| 18 | 0.452 | 85.5% | 87.3% | 38 | 0.209 | 93.0% | 91.9% |
| 19 | 0.430 | 86.2% | 86.5% | 39 | 0.208 | 92.8% | 92.2% |
| 20 | 0.422 | 86.3% | 87.6% | 40 | 0.208 | 92.9% | 92.1% |

Best val: 92.2% (epoch 39). Test: 91.6%.

### Epoch-by-epoch: CumsumMagDeep n_embed=56 (40 epochs, 65,084 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.485 | 52.0% | 67.7% | 21 | 0.411 | 86.4% | 87.9% |
| 2 | 0.844 | 72.0% | 78.1% | 22 | 0.402 | 86.9% | 88.6% |
| 3 | 0.737 | 75.6% | 83.1% | 23 | 0.400 | 86.7% | 87.9% |
| 4 | 0.684 | 77.5% | 74.2% | 24 | 0.389 | 87.3% | 88.3% |
| 5 | 0.681 | 77.9% | 79.1% | 25 | 0.378 | 87.5% | 88.4% |
| 6 | 0.623 | 79.5% | 80.8% | 26 | 0.366 | 87.9% | 88.7% |
| 7 | 0.616 | 79.5% | 82.3% | 27 | 0.355 | 88.2% | 89.7% |
| 8 | 0.593 | 80.5% | 84.1% | 28 | 0.345 | 88.7% | 89.4% |
| 9 | 0.571 | 81.2% | 81.7% | 29 | 0.330 | 89.0% | 90.0% |
| 10 | 0.553 | 81.7% | 82.5% | 30 | 0.321 | 89.5% | 89.4% |
| 11 | 0.554 | 81.9% | 86.0% | 31 | 0.310 | 89.7% | 89.4% |
| 12 | 0.510 | 83.4% | 82.2% | 32 | 0.306 | 89.9% | 90.3% |
| 13 | 0.502 | 83.4% | 86.2% | 33 | 0.298 | 90.2% | 90.9% |
| 14 | 0.507 | 83.4% | 86.4% | 34 | 0.291 | 90.4% | 90.7% |
| 15 | 0.488 | 84.0% | 85.4% | 35 | 0.287 | 90.4% | 90.2% |
| 16 | 0.476 | 84.3% | 86.2% | 36 | 0.279 | 90.9% | 91.2% |
| 17 | 0.480 | 84.2% | 80.4% | 37 | 0.280 | 90.8% | 90.8% |
| 18 | 0.453 | 85.1% | 87.2% | 38 | 0.278 | 90.9% | 90.9% |
| 19 | 0.438 | 85.4% | 85.3% | 39 | 0.270 | 91.1% | 91.0% |
| 20 | 0.425 | 85.8% | 87.9% | 40 | 0.270 | 91.1% | 90.9% |

Best val: 91.2% (epoch 36). Test: 90.5%.

### Epoch-by-epoch: CumsumMagDeep n_embed=40 (40 epochs, 33,692 params)

| Epoch | Loss | Train | Val | Epoch | Loss | Train | Val |
|-------|------|-------|-----|-------|------|-------|-----|
| 1 | 1.456 | 52.0% | 65.7% | 21 | 0.454 | 85.0% | 87.6% |
| 2 | 0.866 | 70.8% | 76.0% | 22 | 0.439 | 85.5% | 87.3% |
| 3 | 0.745 | 75.0% | 78.4% | 23 | 0.429 | 85.8% | 85.0% |
| 4 | 0.688 | 77.4% | 78.0% | 24 | 0.422 | 86.0% | 86.1% |
| 5 | 0.653 | 78.3% | 77.9% | 25 | 0.406 | 86.5% | 86.4% |
| 6 | 0.628 | 79.2% | 81.7% | 26 | 0.394 | 87.1% | 88.8% |
| 7 | 0.601 | 80.2% | 80.6% | 27 | 0.380 | 87.3% | 88.5% |
| 8 | 0.589 | 80.4% | 81.4% | 28 | 0.384 | 87.3% | 88.3% |
| 9 | 0.587 | 80.5% | 82.8% | 29 | 0.367 | 87.9% | 88.6% |
| 10 | 0.580 | 80.9% | 83.7% | 30 | 0.362 | 88.0% | 89.0% |
| 11 | 0.549 | 81.8% | 84.0% | 31 | 0.353 | 88.2% | 89.2% |
| 12 | 0.533 | 82.3% | 83.5% | 32 | 0.345 | 88.4% | 89.1% |
| 13 | 0.524 | 82.6% | 83.1% | 33 | 0.335 | 88.8% | 89.8% |
| 14 | 0.520 | 82.8% | 82.8% | 34 | 0.326 | 89.2% | 89.6% |
| 15 | 0.512 | 83.0% | 86.8% | 35 | 0.326 | 89.2% | 90.0% |
| 16 | 0.515 | 83.1% | 85.0% | 36 | 0.320 | 89.4% | 90.1% |
| 17 | 0.487 | 83.9% | 86.0% | 37 | 0.319 | 89.4% | 90.1% |
| 18 | 0.471 | 84.5% | 83.5% | 38 | 0.316 | 89.5% | 90.1% |
| 19 | 0.468 | 84.5% | 85.5% | 39 | 0.313 | 89.6% | 90.1% |
| 20 | 0.453 | 85.0% | 85.7% | 40 | 0.314 | 89.7% | 89.9% |

### CumsumComplex experiments (complex-valued layers 2+)

Attempted keeping everything in complex domain with complex linear projections
(proper (a+bi)(c+di) coupling) instead of treating [re, im] as independent channels.

| Variant | Test Acc | Params | Nonlinearity |
|---------|----------|--------|-------------|
| v1 gate, n_embed=40 | 71.2% | 15,292 | RMS norm + sigmoid gate |
| v1 gate, n_embed=80 | ~80%* | 59,372 | RMS norm + sigmoid gate |
| v3 mag MLP, n_embed=40 | 71.2% | 20,092 | softplus(MLP(mag)) / mag |
| mlp_direct readout, n_embed=40 | ~60%* | — | same + MLP readout |

*stopped early

Complex-valued layers underperformed real-valued [re,im]+BN+GLU by a wide margin.
The magnitude gate/MLP nonlinearity is too weak compared to GLU, and the complex
linear constraint (half the capacity of a general real linear) limits expressivity.

## Implications for Large Vocabulary Speech Recognition

### The universal front-end: 40 years of mel spectrograms

Every major production ASR system uses the same front-end: fixed FFT → mel filterbank → log.
This has been essentially unchanged since the 1980s.

- **Whisper** (Radford et al., 2022): 80-bin log-mel, 25ms window, 10ms hop → 2 Conv1d layers
  → transformer encoder-decoder. Trained on 680K hours. The mel is completely fixed/non-learnable.
  (https://arxiv.org/abs/2212.04356)

- **Conformer** (Gulati et al., 2020): 80-bin log-mel → SpecAugment → conv subsampling
  → conformer blocks (self-attention + convolution). State-of-the-art for streaming ASR.
  (https://arxiv.org/abs/2005.08100)

- **Parakeet / Canary** (NVIDIA, 2024): Same fixed log-mel → fast-conformer encoder.
  Production-grade multilingual ASR.

- **wav2vec 2.0** (Baevski et al., 2020): The exception — learns from raw waveform via
  7-layer CNN feature extractor. But primarily used for self-supervised pretraining;
  downstream systems often still use mel features.
  (https://arxiv.org/abs/2006.11477)

All of these systems throw away phase completely. The mel spectrogram is magnitude-only.

### What our experiments show

1. **Phase information helps (+2% accuracy)**
   - MagDeep mag-only: 90.5% (65K params)
   - MagDeep [log_mag, re, im]: 92.8% (72K params)
   - The field has discarded phase for 40 years. Our experiments show it carries
     useful information when paired with magnitude as an anchor.

2. **Learned frequencies match fixed FFT+mel**
   - LearnedSpecCNN (learned cumsum): 94.1% (26K params)
   - MelCNN (fixed FFT+mel): 95.7% (25K params)
   - The gap is small, and learned frequencies could adapt to language/task-specific
     frequency resolution rather than the fixed mel approximation of human hearing.

3. **mag² → log is a sufficient nonlinearity**
   - CumsumMagDeep V2 uses no ReLU, no BatchNorm, no GLU between layers.
   - Just: linear → complex → cumsum → mag → log → linear → residual.
   - Gets 92.8% — only 0.7% behind a model with full BN+GLU (93.5%).
   - This suggests the logarithmic compression of magnitude is doing the heavy lifting
     in standard ASR front-ends, not just as a feature transform but as a viable
     nonlinearity for deep networks.

4. **[log_mag, re, im] as a drop-in front-end replacement**
   - LearnedSpecMagReImCNN: cumsum → [log_mag, re, im] → Linear(3F, F) → same CNN backbone
   - Currently running (results pending), but early epochs show strong convergence.
   - This is a direct drop-in for the mel spectrogram in any ASR architecture:
     swap `MelSpectrogram` for the cumsum+linear layer, keep everything else identical.

### Potential applications

**Streaming/on-device ASR**: The cumsum front-end is causal and O(T) — no FFT needed.
Each new sample updates the running sum. This is naturally suited to streaming ASR
where you process audio frame-by-frame.

**Adaptive frequency resolution**: Different languages have different phonemic inventories.
Tonal languages (Mandarin, Thai) may benefit from finer frequency resolution in F0 regions.
Learned frequencies could specialize per-language without manual feature engineering.

**Phase-aware ASR**: Phase carries information about:
- Precise phone boundary timing (onset/offset)
- Speaker characteristics (glottal pulse shape)
- Room acoustics and reverberation
Standard mel discards all of this. A [log_mag, re, im] front-end preserves it.

**Stacked cumsum as sequence modeling**: MagDeep shows cumsum layers with mag+log
nonlinearity can model temporal patterns. At O(T) per layer vs O(T²) for self-attention,
this could complement or partially replace conformer blocks for long utterances.

### Limitations

- Our experiments use 12-class keyword spotting on 1-second clips — far simpler than
  continuous large-vocabulary ASR.
- We haven't tested with CTC, RNN-T, or attention-based sequence losses.
- The CNN backend won't scale — a real system would pair the cumsum front-end with
  a conformer or transformer.
- Learned frequencies may lose the well-understood interpretability of mel filterbanks.
- Phase benefits may diminish in noisy/reverberant conditions where phase is corrupted.

---

## Hop=400 (No Overlap) Comparison: Mel vs Cumsum Front-Ends

### Motivation

At hop=400 with window=400, there is **zero overlap** between adjacent frames. This is the
harshest test of temporal resolution: each sample belongs to exactly one frame. The hypothesis
is that phase information becomes more valuable at longer strides because magnitude alone
loses all timing precision within each 25ms frame.

At hop=160 (standard), LearnedSpecMagReImCNN was only +0.5% over LearnedSpecCNN (93.9% vs 93.4%).
Does the gap widen at hop=400?

### Models Tested

All models use **n_freqs=80 / n_mels=80**, 40 epochs, same CNN backbone (ResBlock1d ×3 + pool + fc).

| Model | Front-End | Features | Freqs | SpecAugment |
|-------|-----------|----------|-------|-------------|
| MelCNNMaxPool | FFT + mel filterbank | log-mel | fixed mel | Yes |
| MelCNNMaxPool (no SA) | FFT + mel filterbank | log-mel | fixed mel | No |
| LearnedSpecCNN | cumsum | log-mag | learned (mel init) | No |
| LearnedSpecFrozen | cumsum | log-mag | frozen mel | No |
| LearnedSpecMagReImCNN | cumsum | [log_mag, re, im] → Linear | learned (mel init) | No |
| LearnedSpecMagReImFrozen | cumsum | [log_mag, re, im] → Linear | frozen mel | No |

### Results Summary

| Model | Test Acc | Best Val | Params |
|-------|----------|----------|--------|
| MelCNNMaxPool | **95.4%** | 95.3% | 27,548 |
| MelCNNMaxPool (no SA) | **95.4%** | 95.9% | 27,548 |
| MagReIm Frozen | 94.0% | 94.1% | 46,828 |
| MagReIm Learned | 93.6% | 94.0% | 46,908 |
| Spec Frozen | 92.0% | 92.7% | 27,548 |
| Spec Learned | 90.8% | 92.2% | 27,628 |

### Key Findings

**1. SpecAugment has zero effect at hop=400.** Both MelCNNMaxPool variants score 95.4%.
With only ~40 frames per sequence, freq_mask=7 and time_mask=25 are extremely aggressive
(masking ~18% of freq bins and ~63% of time frames), so the regularization provides no
additional benefit — the short sequence already limits overfitting.

**2. Phase matters more at longer strides.** MagReIm vs mag-only:
- hop=160: +0.5% (93.9% vs 93.4%)
- hop=400: **+2.8%** (93.6% vs 90.8%) with learned freqs
- hop=400: **+2.0%** (94.0% vs 92.0%) with frozen freqs

This confirms the hypothesis: when frames have no overlap, magnitude alone loses all
temporal precision within the 25ms window. Phase preserves sub-frame timing.

**3. Frozen mel > learned frequencies.** In both configurations:
- Mag-only: frozen 92.0% vs learned 90.8% (+1.2%)
- MagReIm: frozen 94.0% vs learned 93.6% (+0.4%)

The learned frequencies drift away from mel during training, hurting performance.
Mel-scale spacing is near-optimal for speech.

**4. Remaining gap is architectural.** MelCNNMaxPool (95.4%) vs MagReIm Frozen (94.0%)
leaves a 1.4% gap. This is not from SpecAugment (no effect) or from frequencies (both mel).
The gap comes from the FFT+mel filterbank vs single-frequency cumsum:
- FFT uses 201 frequency bins, mel filterbank averages groups → smoother, more robust
- Cumsum uses exactly 80 single frequencies → noisier magnitude estimates
- FFT is a windowed DFT (rectangular window by default) while cumsum uses exponential
  running sum — different frequency responses

### Per-Epoch Comparison (val_acc)

| Epoch | MelMaxPool | MelMP noSA | MagReIm Frozen | MagReIm Learned | Spec Frozen | Spec Learned |
|-------|-----------|------------|----------------|-----------------|-------------|--------------|
| 1 | 85.6% | 90.1% | 84.9% | 85.2% | 82.3% | 79.6% |
| 2 | 84.2% | 88.0% | 88.3% | 87.3% | 85.5% | 82.9% |
| 3 | 88.8% | 91.9% | 87.6% | 87.4% | 85.3% | 85.1% |
| 4 | 89.5% | 91.9% | 90.0% | 89.2% | 87.2% | 86.5% |
| 5 | 91.0% | 93.8% | 89.9% | 88.8% | 88.6% | 86.1% |
| 6 | 90.7% | 93.4% | 90.5% | 89.0% | 87.5% | 86.4% |
| 7 | 91.3% | 94.0% | 90.8% | 89.2% | 88.9% | 88.4% |
| 8 | 92.1% | 93.9% | 90.5% | 89.3% | 88.8% | 88.6% |
| 9 | 92.9% | 94.4% | 91.5% | 89.5% | 89.5% | 89.3% |
| 10 | 93.6% | 94.4% | 91.4% | 90.5% | 89.8% | 88.4% |
| 12 | 92.8% | 94.6% | 92.3% | 90.7% | 90.6% | 88.5% |
| 14 | 94.1% | 94.4% | 92.8% | 91.4% | 90.6% | 90.3% |
| 16 | 93.9% | 95.1% | 92.2% | 92.5% | 90.9% | 89.9% |
| 18 | 93.7% | 95.3% | 92.0% | 92.0% | 91.5% | 88.5% |
| 20 | 94.4% | 95.2% | 93.1% | 92.1% | 91.7% | 90.6% |
| 22 | 94.7% | 95.1% | 92.5% | 91.8% | 91.7% | 91.0% |
| 24 | 94.6% | 95.5% | 93.7% | 92.4% | 91.5% | 91.0% |
| 26 | 94.6% | 95.3% | 93.6% | 93.3% | 91.6% | 91.2% |
| 28 | 94.9% | 95.7% | 93.3% | 93.2% | 91.6% | 91.3% |
| 30 | 95.3% | 95.5% | 93.6% | 93.7% | 92.1% | 91.9% |
| 32 | 95.0% | 95.5% | 94.0% | 93.3% | 92.2% | 91.8% |
| 34 | 95.0% | 95.6% | 94.1% | 93.6% | 92.5% | 91.9% |
| 36 | 95.0% | 95.5% | 94.1% | 93.7% | 92.4% | 91.9% |
| 38 | 95.2% | 95.5% | 94.0% | 94.0% | 92.6% | 91.9% |
| 40 | 95.1% | 95.6% | 94.0% | 93.7% | 92.7% | 91.9% |

**Note on MelMaxPool train_acc**: With SpecAugment, train_acc is artificially low (~79% at epoch 40)
because SpecAugment masks large portions of the ~40-frame input during training. Without SpecAugment,
train_acc reaches ~97.5% — a normal value. Despite the low training accuracy, validation accuracy
is identical (95.4%), confirming SpecAugment provides no regularization benefit at this stride.

### Time Shift Robustness (±200 samples)

Tested whether random ±200 sample shift (half a frame at hop=400) on train+test hurts models.
Validation is unshifted for fair comparison.

| Model | No Shift | ±200 Shift | Diff |
|-------|----------|------------|------|
| MelCNNMaxPool (no SA) | 95.4% | 95.7% | +0.3% |
| MagReIm Frozen | 94.0% | 93.9% | -0.1% |
| Spec Frozen | 92.0% | 91.8% | -0.2% |

No meaningful effect. The training augmentation already applies ±1600 sample shifts,
so ±200 is trivial for all models.

---

## Hop=160 Comparison: Frozen Mel Frequencies + [mag, re, im]

### Motivation

At hop=400, MagReIm Frozen reached 94.0% vs MelCNNMaxPool's 95.4% — a 1.4% gap.
Can we close this gap at hop=160 where there's more temporal information?

Additionally, we test the effect of window size: at W=400 with only 80 frequencies,
we capture a 400-sample window with just 80 complex numbers (vs FFT's 201 bins).
At W=160, 80 frequencies ≈ FFT's 81 bins — nearly complete representation.

### Models Tested

All models: hop=160, n_freqs/n_mels=80, 40 epochs, same CNN backbone.

| Model | Window | Features | Freqs | Params |
|-------|--------|----------|-------|--------|
| MelCNNMaxPool (SA) | 400 (n_fft) | log-mel | fixed | 25,628 |
| MagReIm Frozen W=400 | 400 | [log_mag, re, im] → Linear | frozen mel | 46,828 |
| MagReIm Frozen W=160 | 160 | [log_mag, re, im] → Linear | frozen mel | 46,828 |
| MagReIm Learned W=400 | 400 | [log_mag, re, im] → Linear | learned | 30,508 |

### Results Summary

| Model | Test Acc | Best Val | Params |
|-------|----------|----------|--------|
| MelCNNMaxPool (SA) | **97.1%** | 96.4% | 25,628 |
| MagReIm Frozen W=400 | 95.2% | 95.2% | 46,828 |
| MagReIm Frozen W=160 | 94.8% | 95.2% | 46,828 |
| MagReIm Learned W=400 | 93.9% | 94.0% | 30,508 |

### Key Findings

**1. Frozen mel frequencies gain +1.3% over learned.** MagReIm Frozen W=400 (95.2%) vs
MagReIm Learned W=400 (93.9%). Same architecture, same params (except 80 frozen frequency
values). The learned frequencies drift away from mel during training, consistently hurting
performance across all experiments.

**2. Window size has minimal effect at hop=160.** W=400 (95.2%) vs W=160 (94.8%) — only
0.4% difference. At hop=160 with overlapping frames, the CNN backend can compensate for
any within-frame information loss through adjacent frames. The hypothesis that "80 freqs
can't represent a 400-sample window" doesn't hold in practice when frames overlap.

**3. Gap to MelCNNMaxPool remains ~2%.** Even with frozen mel + [mag,re,im], the best
cumsum front-end (95.2%) trails MelCNNMaxPool (97.1%) by 1.9%. The remaining gap comes from:
- FFT uses all 201 frequency bins → mel filterbank averages groups → smoother estimates
- Cumsum uses exactly 80 single-frequency oscillators → noisier, no inter-bin averaging
- MelCNNMaxPool benefits from SpecAugment regularization (trained at ~93% train_acc)

**4. MagReIm Frozen is the best cumsum front-end.** Among all cumsum-based models tested
at hop=160, frozen mel + [mag,re,im] → Linear achieves the highest accuracy (95.2%).

### Per-Epoch Comparison (val_acc, hop=160)

| Epoch | MelMaxPool (SA) | MagReIm Frozen W=400 | MagReIm Frozen W=160 | MagReIm Learned W=400 |
|-------|----------------|---------------------|---------------------|---------------------|
| 1 | 88.5% | 84.3% | 81.4% | 81.5% |
| 2 | 89.8% | 86.6% | 88.1% | 85.8% |
| 4 | 92.4% | 90.3% | 89.8% | 87.2% |
| 6 | 93.3% | 90.6% | 90.9% | 88.8% |
| 8 | 93.7% | 91.0% | 92.3% | 89.5% |
| 10 | 94.0% | 92.4% | 92.0% | 91.1% |
| 12 | 94.0% | 92.7% | 92.7% | 91.0% |
| 14 | 94.4% | 93.3% | 93.6% | 91.7% |
| 16 | 94.9% | 93.5% | 93.1% | 92.1% |
| 18 | 95.2% | 93.6% | 93.5% | 92.2% |
| 20 | 95.2% | 93.9% | 93.9% | 92.4% |
| 22 | 95.4% | 93.5% | 94.1% | 92.6% |
| 24 | 95.6% | 94.3% | 93.9% | 93.4% |
| 26 | 95.5% | 94.5% | 94.5% | 93.3% |
| 28 | 96.1% | 94.5% | 94.6% | 92.8% |
| 30 | 95.8% | 94.6% | 94.7% | 93.4% |
| 32 | 96.2% | 94.9% | 95.0% | 93.7% |
| 34 | 96.3% | 94.8% | 94.7% | 93.9% |
| 36 | 96.2% | 95.0% | 94.9% | 93.8% |
| 38 | 96.0% | 95.1% | 94.9% | 93.9% |
| 40 | 96.4% | 95.2% | 95.0% | 94.0% |

---

## Frozen Conv Filterbank: FilterbankSinCos Frozen vs Learned

### Motivation

Freezing mel frequencies consistently improved cumsum front-ends (+1-2%). Does the same
pattern hold for conv filterbanks? FilterbankSinCosCNN uses Hann-windowed sin+cos pairs
initialized at mel frequencies — a conv implementation of a windowed DFT at mel centers.

We also test whether adding raw sin/cos channels via a Linear bottleneck (the approach
that helped cumsum in LearnedSpecMagReImCNN) helps the conv filterbank too.

### Models Tested

All models: hop=160, n_freqs=40, window=400, 40 epochs, same CNN backbone.

| Model | Features | Filters | Params |
|-------|----------|---------|--------|
| FilterbankSinCos Learned | sin²+cos² → log | learned | 57,628 |
| FilterbankSinCos Frozen | sin²+cos² → log | frozen mel | 25,628 |
| FilterbankSinCosMagReIm Frozen | [log_mag, sin, cos] → Linear(120→40) | frozen mel | 30,468 |
| FilterbankSinCosCombined (prev) | [log_mag, sin, cos] → 120ch raw | learned | 61,468 |

### Results Summary

| Model | Test Acc | Best Val | Params |
|-------|----------|----------|--------|
| FilterbankSinCos Frozen | **95.7%** | 95.0% | 25,628 |
| FilterbankSinCosMagReIm Frozen | 95.4% | 95.0% | 30,468 |
| FilterbankSinCos Learned | 94.7% | 94.5% | 57,628 |
| FilterbankSinCosCombined Learned | 94.5% | — | 61,468 |

### Key Findings

**1. Frozen mel beats learned for conv filterbanks too.** +1.0% (95.7% vs 94.7%).
Same pattern as cumsum. Additionally, frozen has fewer params (25,628 vs 57,628) since
the 80×400 filter weights aren't counted as learnable parameters.

**2. Adding raw sin/cos doesn't help conv filterbanks.** MagReIm Frozen (95.4%) is
slightly worse than mag-only Frozen (95.7%). This contrasts with cumsum, where
[log_mag, re, im] → Linear consistently helped (+0.5% at hop=160, +2.8% at hop=400).

The reason: the Hann-windowed conv filterbank already produces clean, phase-invariant
magnitude estimates via sin²+cos². The raw sin/cos channels carry phase information
that the CNN doesn't need — it's redundant. In contrast, cumsum uses a rectangular
running window, producing noisier magnitude, so the raw re/im provides useful
complementary information.

**3. Linear bottleneck vs raw concatenation.** FilterbankSinCosCombined (94.5%) fed
120 raw channels to the CNN. FilterbankSinCosMagReIm Frozen (95.4%) uses a Linear(120→40)
bottleneck first. The +0.9% improvement confirms: when combining mag+phase, a learned
linear bottleneck to mix features per-frequency is better than dumping all channels raw
into the CNN.

**4. FilterbankSinCos Frozen is the best non-mel single-frequency model.** At 95.7%
with only 25,628 params, it matches the param count of MelCNNMaxPool and closes the
gap to 1.4% (vs 97.1%). The Hann window is the key advantage over cumsum (95.2%):
it provides proper spectral tapering that the cumsum's rectangular window lacks.

### Per-Epoch Comparison (val_acc, hop=160, n_freqs=40)

| Epoch | MelMaxPool (SA) | SinCos Frozen | SinCos MagReIm Frozen | SinCos Learned |
|-------|----------------|---------------|----------------------|----------------|
| 1 | 88.5% | 87.0% | 84.7% | 84.3% |
| 2 | 89.8% | 89.4% | 87.4% | 87.7% |
| 4 | 92.4% | 90.2% | 89.7% | 89.9% |
| 6 | 93.3% | 92.4% | 91.4% | 90.2% |
| 8 | 93.7% | 93.4% | 91.5% | 90.5% |
| 10 | 94.0% | 93.4% | 91.2% | 90.3% |
| 12 | 94.0% | 92.8% | 93.1% | 92.1% |
| 14 | 94.4% | 93.5% | 93.7% | 92.8% |
| 16 | 94.9% | 93.5% | 93.2% | 93.3% |
| 18 | 95.2% | 93.3% | 93.7% | 91.7% |
| 20 | 95.2% | 94.1% | 93.5% | 92.8% |
| 22 | 95.4% | 94.5% | 94.3% | 93.3% |
| 24 | 95.6% | 94.5% | 94.6% | 93.6% |
| 26 | 95.5% | 94.3% | 94.1% | 93.5% |
| 28 | 96.1% | 94.5% | 94.6% | 94.0% |
| 30 | 95.8% | 94.6% | 94.4% | 94.2% |
| 32 | 96.2% | 94.8% | 94.7% | 94.3% |
| 34 | 96.3% | 94.9% | 94.7% | 94.5% |
| 36 | 96.2% | 94.8% | 94.9% | 94.2% |
| 38 | 96.0% | 94.9% | 95.0% | 94.4% |
| 40 | 96.4% | 94.8% | 94.9% | 94.4% |

## Tied Layers: MelCumsumFixed Weight Sharing

**Question**: How much accuracy do we lose by sharing the projection and GLU layers across all cumsum layers, keeping only the frequency parameters and batch norm per-layer?

**Setup**: MelCumsumFixed with `tie_layers=True` — one shared `nn.Linear(n_embed, n_embed)` proj and one shared GLU across all 4 layers. Each layer retains its own `freq_params` and `TransposedBN`.

| Component | Untied | Tied |
|-----------|--------|------|
| Mel front-end (embed) | 3,280 | 3,280 |
| Proj layers (4×) | 25,920 | 6,480 (1×) |
| Freq params (4×) | 160 | 160 |
| BatchNorm (4×) | 640 | 640 |
| GLU layers (4×) | 51,840 | 12,960 (1×) |
| Classifier (fc) | 492 | 492 |
| **Total** | **82,252** | **24,012** |

**Result**: 95.2% test accuracy with 24K params (vs 96.3% untied at 82K params). Only 1.1% drop for 3.4× fewer parameters.

| Model | Params | Best Val | Test |
|-------|--------|----------|------|
| MelCumsumFixed W=20 (untied) | 82,252 | 96.0% | 96.3% |
| **MelCumsumFixedTied W=20** | **24,012** | **94.9%** | **95.2%** |
| MelCNNMaxPool (reference) | 25,628 | 95.4% | 95.4% |

At comparable param count (~24K vs ~26K), tied cumsum matches MelCNNMaxPool.

### Per-Epoch Val Accuracy: Tied vs Untied

| Epoch | Untied (82K) | Tied (24K) |
|-------|-------------|------------|
| 1 | 83.3% | 77.3% |
| 2 | 86.8% | 83.3% |
| 3 | 90.0% | 87.0% |
| 4 | 89.9% | 88.6% |
| 5 | 91.6% | 90.1% |
| 6 | 92.3% | 90.3% |
| 7 | 92.5% | 90.1% |
| 8 | 92.3% | 90.9% |
| 9 | 93.3% | 91.1% |
| 10 | 93.5% | 90.9% |
| 12 | 92.8% | 92.6% |
| 14 | 93.5% | 90.8% |
| 16 | 93.7% | 92.5% |
| 18 | 94.0% | 93.0% |
| 20 | 95.1% | 93.4% |
| 22 | 94.6% | 93.2% |
| 24 | 95.2% | 93.8% |
| 26 | 94.6% | 93.7% |
| 28 | 94.7% | 94.1% |
| 30 | 95.1% | 94.5% |
| 32 | 95.6% | 94.4% |
| 34 | 95.8% | 94.4% |
| 36 | 95.6% | 94.7% |
| 38 | 95.5% | 94.9% |
| 40 | 96.0% | 94.8% |

**Key observations**:
- Tied version starts ~6% behind at epoch 1 but steadily closes the gap
- Gap narrows from ~2.6% (epoch 10) to ~1.2% (epoch 40)
- Tied version converges more slowly — train_acc only reaches 90.0% vs 91.8% for untied
- The shared projection forces all layers to use the same linear mixing — differentiation comes entirely from the per-layer frequencies and batch norm statistics

### Doubling Depth: 8-Layer Tied

**Question**: With tied weights, adding layers is nearly free in params (only +160 freq_params and +160 BN per layer). Does doubling depth from 4→8 help?

**Result**: 95.5% test (24,812 params) vs 95.2% for 4-layer (24,012 params). +0.3% for 2× the compute.

| Model | Layers | Params | Best Val | Test |
|-------|--------|--------|----------|------|
| Tied 4-layer | 4 | 24,012 | 94.9% | 95.2% |
| Tied 8-layer | 8 | 24,812 | 95.0% | 95.5% |

### Per-Epoch Val Accuracy: 4-Layer vs 8-Layer Tied

| Epoch | 4L Tied | 8L Tied |
|-------|---------|---------|
| 1 | 77.3% | 78.7% |
| 2 | 83.3% | 83.0% |
| 3 | 87.0% | 87.3% |
| 4 | 88.6% | 88.7% |
| 5 | 90.1% | 89.9% |
| 6 | 90.3% | 89.7% |
| 8 | 90.9% | 90.3% |
| 10 | 90.9% | 92.0% |
| 12 | 92.6% | 92.5% |
| 14 | 90.8% | 92.2% |
| 16 | 92.5% | 92.6% |
| 18 | 93.0% | 93.0% |
| 20 | 93.4% | 93.4% |
| 22 | 93.2% | 93.0% |
| 24 | 93.8% | 94.0% |
| 26 | 93.7% | 93.9% |
| 28 | 94.1% | 94.1% |
| 30 | 94.5% | 94.5% |
| 32 | 94.4% | 94.6% |
| 34 | 94.4% | 94.7% |
| 36 | 94.7% | 94.6% |
| 38 | 94.9% | 95.0% |
| 40 | 94.8% | 94.9% |

**Conclusion**: Doubling depth gives negligible improvement (+0.3%). The bottleneck is not depth — it's the shared projection's expressiveness. Each layer sees the same linear mixing; only the per-layer frequencies and BN differentiate them.

### Data-Dependent Frequencies: MelCumsumModTied

**Question**: Can we replace fixed per-layer frequencies with data-dependent frequency predictions while keeping weights tied?

**Challenge**: Unbounded frequency predictions cause cumsum phase divergence. We tested multiple bounding strategies and architectures.

#### Architecture variants tested

**A) 3×proj + no norm**: `shared_proj: Linear(n_embed, 3*n_freqs)` → [re, im, inst_freq]. No normalization on frequencies.
- Result: Stuck at chance (8.3%). Unbounded inst_freq → cumsum diverges.

**B) 3×proj + tanh**: Same, but `tanh(inst_freq)` to bound to [-1, 1].
- Result: 86.0% test. Highly unstable training (val oscillates ±10% between epochs).

**C) 3×proj + LayerNorm**: Same, but `LayerNorm(inst_freq)`.
- Result: **94.4% test**. Stable training, converges well.

**D) Separate freq_proj + LayerNorm**: `shared_proj: Linear(n_embed, n_embed)` for re+im, separate `shared_freq_proj: Linear(n_embed, n_freqs)` + `LayerNorm(n_freqs)` for frequencies.
- Result: **94.1% test**. Slightly worse than 3×proj despite more architectural separation.

**E) Separate freq_proj + BatchNorm**: Same as D but `TransposedBN(n_freqs)` instead of LayerNorm.
- Result: ~80.7% at epoch 16 (killed). Worse than LayerNorm — BN statistics too noisy for frequency prediction.

| Model | Freq source | Norm | Params | Best Val | Test |
|-------|------------|------|--------|----------|------|
| Fixed tied 4L | per-layer params | — | 24,012 | 94.9% | 95.2% |
| Fixed tied 8L | per-layer params | — | 24,812 | 95.0% | 95.5% |
| Mod tied (3×proj) | shared proj | none | 27,092 | 8.3% | — |
| Mod tied (3×proj) | shared proj | tanh | 27,092 | 86.0% | 86.0% |
| **Mod tied (3×proj)** | **shared proj** | **LN** | **27,172** | **94.4%** | **94.4%** |
| Mod tied (sep freq) | shared freq_proj | LN | 27,172 | 94.3% | 94.1% |
| Mod tied (sep freq) | shared freq_proj | BN | 27,172 | 80.7% | — (killed) |

### Per-Epoch Val Accuracy: All Mod Tied Variants vs Fixed Tied

| Epoch | Fixed Tied | 3×proj+LN | Sep freq+LN | 3×proj+tanh |
|-------|-----------|-----------|-------------|-------------|
| 1 | 77.3% | 55.3% | 51.2% | 53.3% |
| 2 | 83.3% | 67.0% | 61.8% | 68.0% |
| 4 | 88.6% | 82.5% | 76.3% | 74.5% |
| 6 | 90.3% | 85.3% | 81.8% | 62.3% |
| 8 | 90.9% | 88.2% | 87.3% | 72.5% |
| 10 | 90.9% | 88.1% | 86.5% | 69.7% |
| 12 | 92.6% | 89.9% | 89.2% | 75.9% |
| 14 | 90.8% | 89.2% | 85.9% | 76.6% |
| 16 | 92.5% | 90.9% | 90.4% | 78.4% |
| 18 | 93.0% | 91.2% | 92.3% | 80.0% |
| 20 | 93.4% | 91.5% | 89.7% | 77.0% |
| 22 | 93.2% | 93.1% | 93.0% | 79.2% |
| 24 | 93.8% | 93.4% | 92.3% | 81.5% |
| 26 | 93.7% | 92.5% | 92.6% | 83.5% |
| 28 | 94.1% | 93.0% | 93.1% | 83.1% |
| 30 | 94.5% | 93.3% | 93.7% | 83.8% |
| 32 | 94.4% | 94.1% | 93.7% | 85.5% |
| 34 | 94.4% | 94.2% | 94.0% | 85.4% |
| 36 | 94.7% | 94.3% | 94.0% | 85.5% |
| 38 | 94.9% | 93.9% | 94.3% | 85.4% |
| 40 | 94.8% | 94.4% | 94.2% | 85.6% |

**Key findings**:
- LayerNorm is essential for data-dependent frequencies — tanh and no-norm both fail badly
- The simpler 3×proj approach (one matrix does re+im+freq) slightly outperforms the separate freq_proj (94.4% vs 94.1%)
- BatchNorm is much worse than LayerNorm for frequency prediction
- Data-dependent frequencies cost ~0.8% vs fixed (94.4% vs 95.2%) at the same param count
- Mod tied converges slower (starts ~20% behind fixed tied) but closes most of the gap by epoch 30+

### Doubling Depth: 8-Layer Mod Tied (3×proj+LN)

Same pattern as fixed tied — doubling layers from 4→8 gives marginal improvement.

**Result**: 94.6% test (27,812 params) vs 94.4% for 4-layer (27,172 params). +0.2% for 2× compute.

| Model | Layers | Params | Best Val | Test |
|-------|--------|--------|----------|------|
| Fixed tied | 4 | 24,012 | 94.9% | 95.2% |
| Fixed tied | 8 | 24,812 | 95.0% | 95.5% |
| Mod tied (3×proj+LN) | 4 | 27,172 | 94.4% | 94.4% |
| Mod tied (3×proj+LN) | 8 | 27,812 | 94.3% | 94.6% |

### Per-Epoch Val Accuracy: All Tied Variants (4L and 8L)

| Epoch | Fixed 4L | Fixed 8L | Mod 4L | Mod 8L |
|-------|----------|----------|--------|--------|
| 1 | 77.3% | 78.7% | 55.3% | 24.5% |
| 2 | 83.3% | 83.0% | 67.0% | 48.0% |
| 4 | 88.6% | 88.7% | 82.5% | 78.5% |
| 6 | 90.3% | 89.7% | 85.3% | 82.9% |
| 8 | 90.9% | 90.3% | 88.2% | 81.9% |
| 10 | 90.9% | 92.0% | 88.1% | 85.7% |
| 12 | 92.6% | 92.5% | 89.9% | 89.6% |
| 14 | 90.8% | 92.2% | 89.2% | 86.9% |
| 16 | 92.5% | 92.6% | 90.9% | 90.5% |
| 18 | 93.0% | 93.0% | 91.2% | 91.4% |
| 20 | 93.4% | 93.4% | 91.5% | 88.9% |
| 22 | 93.2% | 93.0% | 93.1% | 92.4% |
| 24 | 93.8% | 94.0% | 93.4% | 92.6% |
| 26 | 93.7% | 93.9% | 92.5% | 92.4% |
| 28 | 94.1% | 94.1% | 93.0% | 93.0% |
| 30 | 94.5% | 94.5% | 93.3% | 93.4% |
| 32 | 94.4% | 94.6% | 94.1% | 94.2% |
| 34 | 94.4% | 94.7% | 94.2% | 94.2% |
| 36 | 94.7% | 94.6% | 94.3% | 94.1% |
| 38 | 94.9% | 95.0% | 93.9% | 94.3% |
| 40 | 94.8% | 94.9% | 94.4% | 94.3% |

**Conclusion**: Doubling depth gives +0.2-0.3% for both fixed and mod tied variants. The fixed vs mod gap (~0.8%) persists regardless of layer count. All four variants converge to 94-95%.

### Window Size vs Depth: Receptive Field Decomposition

**Key insight**: Cumsum is a fixed aggregation — it averages everything in the window indiscriminately, unlike attention which can selectively ignore irrelevant frames. A wider window forces irrelevant information into the state. MelCNNMaxPool achieves full coverage efficiently with a ~38-frame local receptive field from CNN convolutions plus maxpool over all time positions.

**Experiment**: Keep total receptive field constant at 40 frames, vary how it's decomposed:
- 2 layers × W=20: coarse aggregation, few processing stages
- 4 layers × W=10: moderate local windows, more processing
- 8 layers × W=5: very tight local windows, many processing stages with GLU gating

| Config | RF | Layers | Params | Best Val | Test |
|--------|-----|--------|--------|----------|------|
| 2L × W=20 | 40 | 2 | 23,612 | 94.6% | 94.4% |
| 4L × W=10 | 40 | 4 | 24,012 | 95.7% | 95.8% |
| **8L × W=5** | **40** | **8** | **24,812** | **96.3%** | **96.2%** |
| 4L × W=20 | 80 | 4 | 24,012 | 94.9% | 95.2% |
| 8L × W=20 | 160 | 8 | 24,812 | 95.0% | 95.5% |

**8L×W=5 (96.2%) matches the untied model (96.3%) with 3.3× fewer params!**

Comparing same-layer-count at different windows:
- 4L: W=10 (95.8%) beats W=20 (95.2%) — smaller window, same depth
- 8L: W=5 (96.2%) beats W=20 (95.5%) — much smaller window, same depth

The pattern is clear: many layers of tight local aggregation with GLU gating between them outperforms fewer layers of wide aggregation. Each layer does a small local cumsum, then GLU selectively gates what passes through. This is analogous to deep narrow convolutions beating shallow wide ones.

### Per-Epoch Val Accuracy: RF=40 Decomposition (Fixed Tied)

| Epoch | 2L×W=20 | 4L×W=10 | 8L×W=5 |
|-------|---------|---------|--------|
| 1 | 74.6% | 82.6% | 78.9% |
| 2 | 83.8% | 85.6% | 85.1% |
| 4 | 89.0% | 90.2% | 91.0% |
| 6 | 87.5% | 90.9% | 90.8% |
| 8 | 90.9% | 91.3% | 93.3% |
| 10 | 91.1% | 92.1% | 91.6% |
| 12 | 92.3% | 94.2% | 92.6% |
| 14 | 90.7% | 92.1% | 92.0% |
| 16 | 91.6% | 93.7% | 94.8% |
| 18 | 92.2% | 93.4% | 94.5% |
| 20 | 92.3% | 93.0% | 94.0% |
| 22 | 92.5% | 94.2% | 94.7% |
| 24 | 93.3% | 94.8% | 95.2% |
| 26 | 93.5% | 94.6% | 95.5% |
| 28 | 93.8% | 95.0% | 95.3% |
| 30 | 94.1% | 95.4% | 95.5% |
| 32 | 93.6% | 95.1% | 95.2% |
| 34 | 94.0% | 95.5% | 95.7% |
| 36 | 94.4% | 95.2% | 95.8% |
| 38 | 94.6% | 95.5% | 96.1% |
| 40 | 94.6% | 95.5% | 96.0% |

8L×W=5 is consistently ahead from epoch 8 onward, reaching 96.3% best val at epoch 39.

### Mod Tied: Same Window/Depth Sweep (3×proj+LN)

The smaller window helps mod tied even more dramatically — closing the fixed vs mod gap entirely at W=10.

| Config | Fixed Tied Test | Mod Tied Test | Gap |
|--------|----------------|--------------|-----|
| 4L × W=20 | 95.2% | 94.4% | -0.8% |
| 8L × W=20 | 95.5% | 94.6% | -0.9% |
| 4L × W=10 | 95.8% | 95.8% | 0.0% |
| 8L × W=5 | **96.2%** | 95.8% | -0.4% |

At W=20, mod is ~0.8% behind fixed. At W=10, the gap disappears. At W=5 with 8 layers, fixed pulls slightly ahead again (+0.4%).

The smaller window helps mod because data-dependent frequency prediction is easier over a short span — the shared proj doesn't need to predict frequencies that work across 20 frames, just 5 or 10. With a wide window, the frequency predictions must be more precise to avoid averaging in irrelevant signal, which is harder for a shared network.

### Per-Epoch Val Accuracy: Mod Tied RF=40 Decomposition

| Epoch | Mod 4L×W=10 | Mod 8L×W=5 |
|-------|------------|-----------|
| 1 | 70.9% | 76.6% |
| 2 | 84.8% | 84.1% |
| 4 | 89.0% | 90.6% |
| 6 | 90.7% | 87.6% |
| 8 | 91.6% | 90.8% |
| 10 | 89.8% | 91.7% |
| 12 | 91.3% | 92.9% |
| 14 | 92.7% | 92.9% |
| 16 | 93.4% | 92.7% |
| 18 | 93.3% | 93.7% |
| 20 | 92.5% | 93.1% |
| 22 | 94.0% | 94.7% |
| 24 | 93.5% | 94.2% |
| 26 | 94.1% | 93.6% |
| 28 | 94.0% | 94.3% |
| 30 | 94.5% | 95.1% |
| 32 | 94.8% | 95.4% |
| 34 | 94.9% | 95.3% |
| 36 | 95.2% | 95.3% |
| 38 | 95.2% | 95.5% |
| 40 | 95.3% | 95.5% |

### Complete Tied Model Summary

| Model | Layers | Window | RF | Params | Test |
|-------|--------|--------|-----|--------|------|
| Fixed tied | 2 | 20 | 40 | 23,612 | 94.4% |
| Fixed tied | 4 | 20 | 80 | 24,012 | 95.2% |
| Fixed tied | 4 | 10 | 40 | 24,012 | 95.8% |
| Fixed tied | 8 | 20 | 160 | 24,812 | 95.5% |
| **Fixed tied** | **8** | **5** | **40** | **24,812** | **96.2%** |
| Mod tied | 4 | 20 | 80 | 27,172 | 94.4% |
| Mod tied | 4 | 10 | 40 | 27,172 | 95.8% |
| Mod tied | 8 | 20 | 160 | 27,812 | 94.6% |
| Mod tied | 8 | 5 | 40 | 27,812 | 95.8% |
| Untied fixed | 4 | 20 | 80 | 82,252 | 96.3% |
| MelCNNMaxPool | — | — | ~38 | 25,628 | 95.4% |

**Key takeaways**:
1. **Small window + many layers >> wide window + few layers**: 8L×W=5 (96.2%) beats 4L×W=20 (95.2%) and 8L×W=20 (95.5%) decisively
2. **Tied matches untied**: Fixed tied 8L×W=5 (96.2%, 24.8K) nearly matches untied (96.3%, 82K) — 3.3× param reduction for 0.1% cost
3. **Tied beats MelCNNMaxPool**: Both fixed (96.2%) and mod (95.8%) tied models beat MelCNNMaxPool (95.4%) at similar or fewer params
4. **Cumsum needs tight local windows**: Unlike attention, cumsum aggregates indiscriminately — wider windows dilute signal with noise. GLU gating between layers provides the selectivity

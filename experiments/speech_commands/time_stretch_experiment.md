# Time-Stretch & Perturbation Experiments: Fixed vs Mod

## Hypothesis
Data-dependent frequencies (Mod) should adapt better when the temporal basis varies across examples. We tested this with multiple perturbation types:
1. Uniform time-stretch (whole waveform)
2. Nonlinear distortion (x + alpha*x^3)
3. Split-stretch (each half of waveform stretched independently)

## Summary of Findings

**Mod shows no meaningful advantage over Fixed under any perturbation regime.** Across uniform stretch, distortion, and split-stretch, both models degrade nearly identically. The one weak signal — Mod's slight slow-side advantage on uniform stretch (~+0.35%) — is within noise for a single seed.

---

## Experiment 1: Uniform Time-Stretch

### Setup
- All runs: n_embed=80, hop=80, epochs=40, `--time_stretch` augmentation (0.8x-1.2x random during training)
- Eval: 9 fixed stretch factors applied uniformly to entire test set

### Results

| Stretch | Fixed w10 l8 | Mod w10 l8 | Mod w8 l5 | Mod w15 l8 |
|---------|-------------|-----------|----------|-----------|
| 0.80 | 93.29% | 92.84% | 92.33% | 91.12% |
| 0.85 | 94.70% | 94.35% | 94.03% | 93.17% |
| 0.90 | 95.44% | 95.19% | 94.74% | 93.76% |
| 0.95 | 95.70% | 95.58% | 95.32% | 94.09% |
| **1.00** | **95.72%** | **95.83%** | **95.46%** | **94.27%** |
| 1.05 | 95.17% | 95.48% | 95.03% | 93.86% |
| 1.10 | 94.56% | 94.68% | 94.41% | 92.84% |
| 1.15 | 93.56% | 94.17% | 93.84% | 91.69% |
| 1.20 | 91.86% | 92.23% | 91.86% | 89.71% |

### Mod - Fixed delta (w10 l8)
| Stretch | Delta |
|---------|-------|
| 0.80 | -0.45% |
| 0.85 | -0.35% |
| 0.90 | -0.25% |
| 0.95 | -0.12% |
| 1.00 | +0.11% |
| 1.05 | +0.31% |
| 1.10 | +0.12% |
| 1.15 | +0.61% |
| 1.20 | +0.37% |

Mod advantage on slow side (1.05-1.20): avg +0.35%
Fixed advantage on fast side (0.80-0.95): avg -0.29%

### Window size sweep
- w=10 is optimal for both models
- w=15 hurts clean accuracy by ~1.5% and degrades worse at all stretch levels
- w=8 l=5 slightly lower than w=10 l=8 across the board
- The slow-side asymmetry persists at all window sizes, ruling out window-to-content ratio as the cause

### Per-Epoch Training (val_acc, `--time_stretch`)

| Epoch | Fixed w10 l8 | Mod w10 l8 | Mod w8 l5 |
|-------|-------------|-----------|----------|
| 1 | .7661 | .6403 | .6993 |
| 5 | .9140 | .8827 | .8884 |
| 10 | .9156 | .8913 | .9028 |
| 15 | .9223 | .9088 | .9129 |
| 20 | .9370 | .9235 | .9133 |
| 25 | .9424 | .9406 | .9397 |
| 30 | .9525 | .9453 | .9417 |
| 35 | .9530 | .9500 | .9473 |
| 40 | .9534 | .9516 | .9480 |

Fixed converges faster early (epoch 1-15 gap ~1-2%), but Mod nearly catches up by epoch 40.

---

## Experiment 2: Nonlinear Distortion

### Setup
- Training: `--distortion` augmentation (normalize, apply x + alpha*x^3 with random alpha~U(0,5), clip to [-1,1])
- Eval: fixed alpha values applied to entire test set

### Results

| Alpha | Fixed | Mod | Delta |
|-------|-------|-----|-------|
| 0.0 | 93.76% | 90.38% | -3.38% |
| 0.5 | 96.03% | 95.64% | -0.39% |
| 1.0 | 95.74% | 95.50% | -0.24% |
| 2.0 | 95.42% | 95.50% | +0.08% |
| 3.0 | 95.01% | 95.19% | +0.18% |
| 5.0 | 94.72% | 94.66% | -0.06% |
| 8.0 | 94.13% | 93.94% | -0.19% |
| 10.0 | 93.90% | 93.74% | -0.16% |

### Key observations
- **Mod's clean accuracy cratered** (90.4% vs 93.8%) — distortion augmentation hurt Mod much more during training
- **Both models improve with mild distortion** (alpha=0.5 > clean) — likely because normalization standardizes volume levels
- At alpha >= 0.5, both degrade similarly — Mod shows no spectral adaptation advantage
- Distortion robustness is essentially identical once you're in the distorted regime

---

## Experiment 3: Split-Stretch (Independent per-half stretch)

### Motivation
The strongest test for adaptive frequencies: split each waveform in half, apply independent random time-stretch to each half, reassemble. The model must handle a mid-sequence tempo change. Mod should theoretically adapt its frequencies at the boundary while Fixed cannot.

### Setup
- Training: `--split_stretch` (each half gets independent 0.8x-1.2x stretch)
- Eval: `--eval_split` with increasing max_stretch values

### Results

| max_stretch | Fixed | Mod | Delta |
|-------------|-------|-----|-------|
| 0.00 | 95.74% | 95.58% | -0.16% |
| 0.05 | 95.62% | 95.56% | -0.06% |
| 0.10 | 95.32% | 95.27% | -0.05% |
| 0.15 | 95.09% | 94.95% | -0.14% |
| 0.20 | 94.58% | 94.41% | -0.17% |
| 0.30 | 92.66% | 92.64% | -0.02% |

### Conclusion
**No difference whatsoever.** Fixed and Mod degrade identically under split-stretch, even when trained with it. Mod's adaptive frequencies are not helping with mid-sequence tempo changes.

---

## Experiment 4: Dual-Command (for completeness)

### Setup
- Training: `--time_stretch` (normal single-command training)
- Eval: two full 16000-sample commands concatenated to 32000, each with independent random stretch, maxpool per half, both must be correct

### Results

| max_stretch | Fixed | Mod | Delta |
|-------------|-------|-----|-------|
| 0.00 | 90.96% | 90.75% | -0.21% |
| 0.05 | 90.57% | 90.67% | +0.10% |
| 0.10 | 90.36% | 90.73% | +0.37% |
| 0.15 | 89.98% | 90.02% | +0.04% |
| 0.20 | 88.83% | 89.03% | +0.20% |

No meaningful difference — both models were trained on single commands so this eval is equally out-of-distribution for both.

---

## Overall Conclusions

1. **Mod's adaptive frequencies provide no robustness advantage** over Fixed's learned-but-static frequencies, across all perturbation types tested
2. **Fixed converges faster** during training (1-2% val gap in early epochs)
3. **Clean accuracy is essentially tied** (~95.7% for both at w10 l8)
4. The slight slow-side stretch advantage for Mod (+0.35% avg) is the only signal, and likely noise
5. Speech Commands may be too uniform a dataset — short commands, similar recording conditions — to stress-test adaptive frequencies
6. All results are single-seed

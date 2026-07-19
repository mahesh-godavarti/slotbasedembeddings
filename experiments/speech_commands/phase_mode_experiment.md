# Phase Mode & Stretch Regime Experiments

## Overview
Two questions tested:
1. What phase semantics work best for Mod's cumulative phase rotation?
2. How does the training stretch range affect Fixed vs Mod comparison?

---

## Experiment 1: Phase Modes (0.8-1.2x stretch training)

### Phase Modes Tested

| Mode | Derotation | Rerotation | Semantics |
|------|-----------|-----------|-----------|
| **default (past-phase)** | Φ(t) | Φ(t) | Current frame's frequency affects how it sees the past AND how the output is projected |
| **derot_prev** | Φ(t-1) | Φ(t) | Frame doesn't know its own freq when being derotated; output uses current freq |
| **both_prev** | Φ(t-1) | Φ(t-1) | Both derot and rerot use previous frame's phase |

Where Φ(t) = cumsum of instantaneous frequencies up to time t.

### Setup
- All runs: n_embed=80, hop=80, epochs=40, `--time_stretch` (0.8x-1.2x), seed=42
- `elementwise_affine=True` (default LayerNorm), `freq_bias=True`
- All 4 models run from same code state, same script (verified bit-for-bit match with Jul 17 results)

### Results

| Stretch | Fixed | Past-phase Mod | Derot-prev Mod | Both-prev Mod |
|---------|-------|---------------|----------------|---------------|
| 0.80 | 93.3% | 92.8% | 92.0% | 92.3% |
| 0.90 | 95.4% | 95.2% | 94.6% | 94.7% |
| **1.00** | **95.7%** | **95.8%** | **95.0%** | **95.6%** |
| 1.10 | 94.6% | 94.7% | 94.0% | 94.6% |
| 1.20 | 91.9% | 92.2% | 91.1% | 91.9% |
| 1.30 | 85.5% | 86.3% | 84.7% | 85.4% |
| 1.40 | 74.3% | 75.8% | 73.2% | 72.5% |
| 1.50 | 61.2% | 62.5% | 60.7% | 57.0% |
| 1.60 | 47.6% | 49.1% | 45.1% | 43.2% |
| 1.80 | 29.2% | 29.1% | 26.0% | 24.6% |
| 2.00 | 16.0% | 14.8% | 14.4% | 12.8% |

### Phase Mode Conclusions
- **Past-phase (default) is the best Mod variant** — slight advantage over Fixed at 1.1-1.6x
- **Derot-prev hurts** (~0.8% below past-phase everywhere)
- **Both-prev** matches on clean but degrades faster at extreme stretch
- Default Φ(t) semantics are optimal; don't shift

---

## Experiment 2: Training Stretch Range

### Setup
Same architecture and hyperparameters, varying only the stretch augmentation range during training. All use past-phase (default) Mod and Fixed.

### No Stretch Training

| Stretch | Fixed | Mod | Delta |
|---------|-------|-----|-------|
| 0.80 | 85.8% | 83.8% | -2.0% |
| 0.90 | 94.0% | 93.2% | -0.8% |
| **1.00** | **96.5%** | **95.6%** | **-0.9%** |
| 1.10 | 93.8% | 92.4% | -1.4% |
| 1.20 | 85.6% | 81.0% | -4.6% |
| 1.30 | 70.8% | 63.2% | -7.6% |
| 1.40 | 52.5% | 44.9% | -7.6% |
| 1.50 | 37.8% | 31.7% | -6.1% |
| 1.60 | 27.6% | 23.7% | -3.9% |
| 1.80 | 15.8% | 14.1% | -1.7% |
| 2.00 | 11.8% | 10.6% | -1.2% |

### Mild Stretch Training (0.8-1.2x)

| Stretch | Fixed | Mod | Delta |
|---------|-------|-----|-------|
| 0.80 | 93.3% | 92.8% | -0.5% |
| 0.90 | 95.4% | 95.2% | -0.2% |
| **1.00** | **95.7%** | **95.8%** | **+0.1%** |
| 1.10 | 94.6% | 94.7% | +0.1% |
| 1.20 | 91.9% | 92.2% | +0.3% |
| 1.30 | 85.5% | 86.3% | +0.8% |
| 1.40 | 74.3% | 75.8% | +1.5% |
| 1.50 | 61.2% | 62.5% | +1.3% |
| 1.60 | 47.6% | 49.1% | +1.5% |
| 1.80 | 29.2% | 29.1% | -0.1% |
| 2.00 | 16.0% | 14.8% | -1.2% |

### Extreme Stretch Training (0.5-2.0x)

| Stretch | Fixed | Mod | Delta |
|---------|-------|-----|-------|
| 0.80 | 91.7% | 90.4% | -1.3% |
| 0.90 | 92.3% | 90.7% | -1.6% |
| **1.00** | **92.8%** | **91.6%** | **-1.2%** |
| 1.10 | 92.0% | 91.3% | -0.7% |
| 1.20 | 91.2% | 90.0% | -1.2% |
| 1.30 | 89.3% | 88.5% | -0.8% |
| 1.40 | 88.1% | 87.3% | -0.8% |
| 1.50 | 85.9% | 84.8% | -1.1% |
| 1.60 | 83.9% | 82.8% | -1.1% |
| 1.80 | 77.2% | 76.7% | -0.5% |
| 2.00 | 65.6% | 62.8% | -2.8% |

---

## Summary

| Training Regime | Fixed clean | Mod clean | Mod advantage? |
|----------------|-----------|-----------|---------------|
| No stretch | 96.5% | 95.6% | **No** — Fixed wins everywhere (up to -7.6%) |
| 0.8-1.2x stretch | 95.7% | 95.8% | **Slight** — Mod +0.1-1.5% on slow side only |
| 0.5-2.0x stretch | 92.8% | 91.6% | **No** — Fixed wins everywhere (up to -2.8%) |

## Key Insight

**Mod needs mild exposure to stretch to learn useful adaptive behavior, but more exposure lets Fixed fit better.**

- Without any stretch training, Mod has no inherent robustness advantage — Fixed's static frequencies are actually MORE robust to unseen time-stretch (up to 7.6% gap at 1.3x).
- With mild stretch (0.8-1.2x), Mod learns a slight slow-side advantage (+1% at 1.3-1.6x).
- With extreme stretch (0.5-2.0x), the extra training signal benefits Fixed more — its static frequencies can learn a single robust setting, while Mod's adaptive mechanism adds optimization difficulty without payoff.

The adaptive frequency mechanism adds parameters and optimization complexity but doesn't translate into better generalization. Fixed's simpler optimization landscape wins when given sufficient training signal.

---

## Per-Epoch Training Data (0.8-1.2x stretch)

### val_acc comparison

| Epoch | Fixed | Past-phase Mod | Derot-prev Mod |
|-------|-------|---------------|----------------|
| 1 | .7661 | .6403 | .5931 |
| 5 | .9140 | .8827 | .8451 |
| 10 | .9156 | .8913 | .8803 |
| 15 | .9223 | .9088 | .9133 |
| 20 | .9370 | .9235 | .9091 |
| 25 | .9424 | .9406 | .9370 |
| 30 | .9525 | .9453 | .9467 |
| 35 | .9530 | .9500 | .9516 |
| 40 | .9534 | .9516 | .9514 |

Fixed converges fastest (epoch 1 gap: +12.6% over Mod). Mod catches up by epoch 25-30. Derot-prev starts slowest but reaches similar final val_acc.

---

## Code State
- `elementwise_affine=True` (default LayerNorm on freq_ln)
- `freq_bias=True` (bias in freq projection)
- `stretch_range` configurable via `--stretch_range` flag
- `phase_mode` configurable via `--phase_mode` flag (default, derot_prev, both_prev)
- All results single-seed (42)

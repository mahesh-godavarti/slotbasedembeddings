# ImageNet ViT-S Results: RoPE2D vs JoFormer

## Goal

Reproduce the RoPE2D paper (ECCV 2024) finding that RoPE positional encoding beats learned PE on Vision Transformers, then test whether JoFormer's V rotation provides additional benefit.

The paper showed:
- RoPE-Axial > learned PE on ViT-B at ImageNet scale
- RoPE-Mixed (combined) > RoPE-Axial
- Gap widens at resolution extrapolation (512x512)

This is for the NeurIPS resubmission of "Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning" — reviewers want ImageNet-scale results.

## Machine

- **instance-qmti92t1-main** on ThunderCompute
- 2x NVIDIA H100 PCIe (80GB each)
- PyTorch 2.10, torchvision 0.25
- AMP with fp16 (GradScaler) — not bf16; future runs should use bf16

## Model

- **ViT-S**: D=384, 12 layers, 6 heads, patch_size=16, img_size=224 (14x14 = 196 patches)
- ~22M parameters, 0 PE parameters (fixed frequencies for rope2d and joformer_old)
- Originally tried ViT-B (86M params) but too slow on single GPU (~12.5 days per model)

## Training Recipe (DeiT-III)

- AdamW, lr=1e-3, weight_decay=0.05
- Cosine LR with 5-epoch linear warmup
- 300 epochs, batch_size=1024, eval every 10 epochs
- RandAugment, Mixup (0.8), CutMix (1.0), Random Erasing (0.25)
- Label smoothing (0.1), stochastic depth (0.1)
- AMP (fp16 with GradScaler)
- seed=42

## PE Variants Being Compared

```
                        Fixed freqs              Learned freqs
                        ───────────              ─────────────
Axial, Q/K only:        rope2d                   monoidal_axial
Axial, Q/K/V+inv:       joformer_old             joformer_axial
Combined, Q/K only:     rope2dv2                 monoidal
Combined, Q/K/V+inv:    joformer_fixed           joformer
Baseline:               learned (additive PE)
```

**JoFormer** = rotates V in addition to Q/K, then inverse-rotates the output. Uses K@Q^T attention ordering (not Q@K^T) to match the journey operator.

---

## Experiment 1: Simplified Recipe (6-layer ViT-S, no augmentation)

**Config**: ViT-S (D=384, 6 layers, 6 heads), batch=1024, 300 epochs, AdamW lr=1e-3, cosine decay, 5-epoch warmup. No Mixup/CutMix/RandAugment. Interrupted at ~epoch 70-79 by machine preemption.

### Results (rope2d vs learned)

| Epoch | Learned Top-1 | Rope2d Top-1 | Gap |
|-------|--------------|-------------|-----|
| 1     | 9.06%        | 9.56%       | +0.50% |
| 10    | 48.42%       | 52.48%      | +4.06% |
| 20    | 55.35%       | 58.88%      | +3.53% |
| 30    | 57.36%       | 60.74%      | +3.38% |
| 40    | 58.48%       | 61.58%      | +3.10% |
| 50    | 59.29%       | 61.59%      | +2.30% |
| 60    | 59.89%       | 61.96%      | +2.07% |
| 70    | 60.37%       | 62.78%      | +2.41% |

**Top-5 at epoch 70**: Learned 82.58%, Rope2d 84.11% (+1.53%).

### Observations

1. **Rope2d consistently ahead** — reproduces the paper's finding (rope2d > learned PE).
2. **Gap narrows over training**: +4.06% at epoch 10 → +2.07% at epoch 60. Learned PE catches up as it learns what RoPE provides structurally from the start.
3. **Rope2d reached learned's epoch-40 accuracy (58.48%) at epoch 20** — 2x faster to reach the same performance.
4. **Absolute accuracy lower than paper** (~62% vs ~75% for ViT-S at 300 epochs): expected since we use no augmentation. The relative comparison is what matters.

---

## Experiment 2: DeiT-III Recipe (12-layer ViT-S, full augmentation)

**Config**: ViT-S (D=384, 12 layers, 6 heads), batch=1024, 300 epochs, full DeiT-III recipe (see above). Two runs in parallel on 2x H100.

### First attempt (interrupted by machine preemption at ~epoch 38-43)

No resume checkpoints existed. Data lost. Results matched closely with second attempt through the epochs completed.

### Second attempt (current, with resume checkpoints)

Crashed once at epoch 68 (rope2d, CUDA driver fault on GPU 0). Resumed successfully from checkpoint after GPU reset via `tnr modify` (ThunderCompute restart).

### Results: rope2d vs joformer_old (in progress, through epoch 210/170)

| Epoch | Rope2d Top-1 | Rope2d Top-5 | JoFormer_old Top-1 | JoFormer_old Top-5 | Gap (Top-1) |
|-------|-------------|-------------|--------------------|--------------------|-------------|
| 1     | 4.47%       | 13.41%      | 5.01%              | 14.56%             | +0.54% |
| 10    | 50.83%      | 75.54%      | 51.38%             | 76.18%             | +0.55% |
| 20    | 61.19%      | 83.93%      | 62.77%             | 84.87%             | +1.58% |
| 30    | 65.38%      | 86.72%      | 66.02%             | 87.23%             | +0.64% |
| 40    | 67.67%      | 88.10%      | 68.18%             | 88.53%             | +0.51% |
| 50    | 68.69%      | 89.00%      | 69.31%             | 89.35%             | +0.62% |
| 60    | 69.59%      | 89.40%      | 70.31%             | 89.98%             | +0.72% |
| 70    | 70.60%      | 90.09%      | 70.84%             | 90.39%             | +0.24% |
| 80    | 71.38%      | 90.54%      | 71.30%             | 90.73%             | -0.08% |
| 90    | 71.85%      | 91.03%      | 72.04%             | 91.04%             | +0.19% |
| 100   | 72.58%      | 91.36%      | 72.66%             | 91.47%             | +0.08% |
| 110   | 73.00%      | 91.70%      | 73.40%             | 91.80%             | +0.40% |
| 120   | 73.31%      | 91.79%      | 73.80%             | 92.15%             | +0.49% |
| 130   | 74.08%      | 92.27%      | 74.13%             | 92.40%             | +0.05% |
| 140   | 74.49%      | 92.49%      | 75.06%             | 92.70%             | +0.57% |
| 150   | 75.16%      | 92.71%      | 75.09%             | 92.87%             | -0.07% |
| 160   | 75.55%      | 93.04%      | 75.62%             | 93.08%             | +0.07% |
| 170   | 76.21%      | 93.23%      | 76.08%             | 93.38%             | -0.13% |
| 180   | 76.74%      | 93.55%      | —                  | —                  | — |
| 190   | 77.00%      | 93.65%      | —                  | —                  | — |
| 200   | 77.56%      | 94.01%      | —                  | —                  | — |
| 210   | 78.14%      | 94.26%      | —                  | —                  | — |

### Observations

1. **JoFormer_old and rope2d are essentially tied.** The gap fluctuates between -0.13% and +1.58%, with no consistent trend after epoch 50. V rotation does not provide a meaningful advantage at this scale.
2. **Early advantage for JoFormer_old** (epochs 10-60, gap +0.5-1.6%) vanishes by epoch 80+. This mirrors the CIFAR-100 finding that JoFormer starts slower but catches up — except here rope2d also catches up to JoFormer.
3. **12 layers + DeiT-III much stronger than 6 layers + no augmentation**: 78% at epoch 210 vs 63% at epoch 70.
4. **JoFormer_old is ~30% slower per epoch** (~1020s vs ~790s) due to extra V rotation and inverse rotation operations. Same accuracy for more compute — not a good tradeoff.
5. **Both runs still climbing** at ~0.4-0.5% per 10 epochs. Cosine schedule acceleration in late training should push final accuracy higher.

### Comparison with RoPE2D paper (ECCV 2024)

The paper reports ViT-S at ~79.4% on ImageNet with DeiT-III at 400 epochs. We're at 78.14% at epoch 210 with 300-epoch schedule — on track to land close to the paper's numbers, validating our implementation.

---

## Remaining Experiments

### Priority 1: Complete current runs
- rope2d: epoch ~214 of 300
- joformer_old: epoch ~171 of 300

### Priority 2: Baselines and other PE variants
- **learned** PE — to reproduce rope2d > learned at ImageNet scale (already shown in simplified recipe)
- **rope2dv2** (combined/mixed) — to reproduce rope2dv2 > rope2d from the paper
- **monoidal_axial** / **monoidal** — learned-frequency variants (per-head, per-layer)
- **joformer_axial** / **joformer** — V-rotation with learned frequencies

### Priority 3: Improvements for future runs
- Switch from fp16 to **bf16** (H100 has 2x bf16 TFLOPS, removes GradScaler overhead)
- Consider dropping JoFormer variants if V rotation continues showing no benefit at ImageNet scale

---

## Infrastructure Notes

### ImageNet download
- Used HF streaming API (`load_dataset("ILSVRC/imagenet-1k", streaming=True)`) to avoid disk space issues
- Saved as ImageFolder at `/home/ubuntu/cifar10_composition/data/imagenet/`
- 1,281,167 train images, 50,000 val images, 1000 classes

### Resume checkpoints
- `vit_imagenet.py --resume` saves `latest_<pe_type>.pt` every epoch (model + optimizer + scheduler + scaler + args)
- On restart with `--resume`, loads checkpoint and continues from next epoch
- Tested and working (recovered rope2d from epoch 68 crash)

### Known issues
- CUDA driver faults can corrupt GPU state, requiring machine restart (`tnr modify` to trigger reboot)
- `auto_resume.sh` and `auto-resume-training.service` exist but are unused — manual restart preferred

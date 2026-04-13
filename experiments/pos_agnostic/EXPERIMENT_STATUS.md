# Experiment Status — 2026-03-27

## Machine
- Instance: `qmti92t1`
- GPUs: 2x NVIDIA H100 PCIe 80GB
- vCPUs: 36, RAM: 180GB

## Completed Experiments

### 1. RoPE baseline (163M params)
- Config: n_embed=768, n_layers=16, n_heads=8, block_size=512, unwindowed, OWT
- 100K iters (lr=5e-4) + 100K continuation (lr=2e-4)
- Final val PPL: 31.94
- Extrapolation: 512:31.51, 1024:43.76, 2048:88.86, 4096:154.82
- **Degrades catastrophically beyond training length**

### 2. JoFormer Fixed baseline (193M params)
- Same config but with V rotation on fixed (RoPE) angles
- 100K iters + 100K continuation
- Final val PPL: 31.49
- Extrapolation: not evaluated at scale (small-scale results show flat with windowing)

### 3. JoFormer v2 from Fixed — lr=5e-4/angle=5e-5 (193M params)
- Converted JoFormer Fixed at 20K → JoFormer v2 (zero-init angles)
- 80K continuation iters at lr=5e-4, angle_lr=5e-5
- Final val PPL: 34.52
- Extrapolation: 512:35.07, 1024:32.99, 2048:32.41, 4096:34.68
- **Flat generalization, but PPL spike at start (43.02 at 1K), slow recovery**

### 4. JoFormer v2 from Fixed — lr=5e-5/angle=5e-5 (193M params) ← BEST
- Same conversion, but lr=5e-5 for both
- 80K continuation iters, monotonic improvement, no spike
- Final val PPL: 30.45
- Extrapolation: 512:30.68, 1024:29.21, 2048:28.32, 4096:29.48
- **Flat generalization. Beats RoPE at all lengths. Improves at 1024-2048.**
- Checkpoint: `checkpoints/scale_up_full/joformer2_from_fixed.pt_best.pt`

### 5. JoFormer v2 from Fixed — lr=5e-4/angle=5e-4 (FAILED)
- Both lr at 5e-4. PPL spiked to 115 at 5K. Killed.

### 6. JoFormer v2 from Fixed — lr=2e-4/angle=2e-4 (FAILED)
- Both lr at 2e-4. PPL spiked to 63.5, slow recovery. Killed at 6K.

### 7. JoFormer v2 from scratch — lr=2e-4/angle=2e-4 (FAILED)
- Very oscillatory (PPL bouncing 480-770). Killed at 15K.

### 8. JoFormer v2 from scratch — lr=5e-5/angle=5e-5 (FAILED)
- Too slow from random init. PPL 93 at 21K. Killed.

## Currently Running

### 9. JoFormer v2 continue (GPU 0)
- Continuing from experiment 4's best checkpoint (val PPL 30.47)
- lr=5e-5, 50K more iters
- Current: ~22K/50K, val PPL 29.79
- Extrapolation at 21K: 512:29.06, 1024:29.61, 2048:28.80, 4096:28.89
- **Still flat. Improving very slowly (~0.03 PPL/1K iters)**
- Checkpoint: `checkpoints/joformer2_continue/joformer2_from_fixed.pt_best.pt`

### 10. Monoidal2 from RoPE (GPU 1)
- Control experiment: data-dependent angles + cumsum, NO V rotation
- Converted RoPE at 65K → Monoidal2 (zero-init angles)
- lr=5e-5 for both, 80K iters
- Current: ~41K/80K, val PPL 28.86
- Extrapolation at 39K: 512:28.12, 1024:28.79, 2048:28.68, 4096:30.11
- **Also flat! V rotation is NOT required for length generalization.**
- Checkpoint: `checkpoints/scale_up_full/monoidal2_from_rope.pt_best.pt`

## Key Results

### Training Curve: RoPE vs JoFormer Fixed vs JoFormer v2

| Iter | RoPE  | JoFormer Fixed | JoFormer v2 (from fixed) |
|------|-------|----------------|--------------------------|
| 20K  | 39.51 | 39.70          | 39.70 (start)            |
| 25K  | 38.04 | 37.92          | 36.75                    |
| 30K  | 36.93 | 36.65          | 35.39                    |
| 35K  | 36.58 | 35.69          | 34.48                    |
| 40K  | 35.66 | 35.06          | 33.90                    |
| 45K  | 35.92 | 34.52          | 33.38                    |
| 50K  | 33.75 | 34.04          | 31.67                    |
| 55K  | 34.30 | 33.62          | 31.45                    |
| 60K  | 33.22 | 33.25          | 31.26                    |
| 65K  | 33.69 | 32.89          | 31.05                    |
| 70K  | 33.27 | 32.59          | 30.79                    |
| 75K  | 32.37 | 32.46          | 30.59                    |
| 80K  | 31.87 | 32.26          | 30.54                    |
| 85K  | 32.09 | 32.06          | 30.85                    |
| 90K  | 33.60 | 31.86          | 30.66                    |
| 95K  | 32.91 | 31.73          | 30.57                    |
| 100K | 31.94 | 31.49          | 30.45                    |

### Length Extrapolation (trained at 512, evaluated at 512-4096)

| Length | RoPE   | JoFormer v2 | Monoidal2 |
|--------|--------|-------------|-----------|
| 512    | 31.51  | 29.06       | 28.12     |
| 1024   | 43.76  | 29.61       | 28.79     |
| 2048   | 88.86  | 28.80       | 28.68     |
| 4096   | 154.82 | 28.89       | 30.11     |

### Key Findings

1. **Data-dependent angles + cumsum enable perfect length generalization** with full (unwindowed) attention
2. **V rotation is NOT required** — monoidal2 (no V rotation) generalizes just as well
3. **Warm-starting from a pretrained model works** — zero-init angles + low lr (5e-5) gives monotonic improvement
4. **Low lr is critical for conversion** — 5e-4 destabilizes, 2e-4 causes spike then slow recovery, 5e-5 is smooth
5. **JoFormer v2 beats RoPE by ~1.5 PPL** at matched iteration count, and doesn't degrade at long sequences

## Experiments Still To Do

### High Priority

1. **Let monoidal2 and joformer2 continue finish** — both still improving, ~2-3h left each

2. **Clean 200-iteration eval on all checkpoints** — the extrapolation numbers above are 50-iter evals. Need proper 200-iter eval for publication-quality numbers using `eval_all.py` (now fixed for split_angles).

3. **Monoidal2 from JoFormer Fixed at 20K** — current monoidal2 started from RoPE at 65K (different starting point than joformer2 which started from joformer_fixed at 20K). For a fair V-rotation ablation, run monoidal2 from the same 20K joformer_fixed checkpoint. Need a `convert_fixed_to_monoidal2.py`.

4. **Angle analysis** — examine the learned angle magnitudes and patterns. Are angles small (near-NoPE)? Do they encode meaningful content relationships? This would help explain WHY it works.

### Medium Priority

5. **Longer extrapolation** — test at 8192, 16384 tokens. Current results go to 4096 (8x training length). How far can it go?

6. **JoFormer v2 from scratch with proper lr schedule** — from-scratch at 5e-5 is too slow, at 5e-4 the angles are unstable. Try: lr=5e-4 main, angle_lr=5e-5 with angle warmup (freeze angles for first 20K, then unfreeze). This mimics the warm-start procedure but in a single training run.

7. **NoPE control at scale** — run NoPE with full attention at this scale (193M params) to confirm it degrades, establishing the baseline that data-dependent angles specifically fix.

### Lower Priority

8. **DataDep (no cumsum) control** — test data-dependent angles WITHOUT cumsum. If it also generalizes, the cumsum isn't needed either. If it degrades, cumsum is essential.

9. **Scaling laws** — do the gains from data-dependent angles grow or shrink with model size? Test at 400M+ params.

10. **Windowed attention comparison** — compare unwindowed JoFormer v2 vs windowed RoPE (which also generalizes). Which gives better PPL at matched params?

11. **Downstream tasks** — length generalization on actual tasks (long-context QA, summarization) rather than just PPL.

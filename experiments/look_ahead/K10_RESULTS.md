# K=10 Experiment: Higher K Helps D=1 Beat N=6

## Key Finding

At K=10 with fixed iterations (k_min=0), D=1 C=2048 **beats** N=6 C=1088 at FLOP parity starting from 65K iters. The gap grows to -0.61 by 80K. This is the first time D=1 has been ahead of N=6 at matched iterations.

Higher K is critical: K=10 with random K (k_min=2) performs similar to K=5 — the model needs consistent full-depth iterations to learn long correction chains.

## Setup

All experiments: block_size=1024, lr=2e-4, softmax, n_head=16, OWT data, ~85M inference FLOPs.

| Run | Model | K | k_min | batch | GPU |
|-----|-------|---|-------|-------|-----|
| N=6 b32 | roformer N=6 C=1088 | — | — | 32 | — |
| N=6 b16 | roformer N=6 C=1088 | — | — | 16 | — |
| D=1 K=5 b32 | block_head_corr_ffn_add D=1 C=2048 | 5 | 2 | 32 | — |
| D=1 K=10 k_min=0 | block_head_corr_ffn_add D=1 C=2048 | 10 | 0 | 16 | this machine |
| D=1 K=10 k_min=2 | block_head_corr_ffn_add D=1 C=2048 | 10 | 2 | 16 | qmt machine |

Note: K=5 uses batch=32, K=10 uses batch=16 (memory constraint). The gap columns compare each D=1 against its corresponding N=6 at the same batch size, so the gaps are comparable even though absolute PPLs differ.

## D=1 vs N=6 Gap Comparison

| Iter | Gap K=5 (b32) | Gap K=10 k_min=0 (b16) | Gap K=10 k_min=2 (b16) |
|------|--------------|----------------------|----------------------|
| 5K   | +9.41        | +4.34                | +6.56                |
| 10K  | +8.06        | +5.40                | +6.65                |
| 15K  | +6.87        | +3.73                | +6.15                |
| 20K  | +5.50        | +3.73                | +5.94                |
| 25K  | +4.63        | +2.60                | +5.29                |
| 30K  | +4.03        | +1.93                | +4.31                |
| 35K  | +3.47        | +1.35                | +3.59                |
| 40K  | +2.82        | +0.59                | +2.86                |
| 45K  | +2.96        | +0.82                | +3.04                |
| 50K  | +2.35        | +0.28                | +2.53                |
| 55K  | +2.29        | +0.28                | +2.49                |
| 60K  | +2.14        | +0.06                | +2.13                |
| 65K  | +1.88        | **-0.06**            | +2.06                |
| 70K  | +1.82        | **-0.57**            | +1.58                |
| 75K  | +1.48        | **-0.55**            | +1.39                |
| 80K  | +1.36        | **-0.61**            | +1.32                |
| 85K  | +1.33        | —                    | +1.09                |
| 90K  | +1.43        | —                    | +0.77                |
| 95K  | +1.08        | —                    | +0.90                |
| 100K | +0.99        | —                    | +0.91                |

## Raw PPL Values

### K=10 k_min=0 (batch=16) vs N=6 (batch=16)

| Iter | N=6 b16 | D=1 K=10 k_min=0 |
|------|---------|-----------------|
| 5K   | 93.64   | 97.98           |
| 10K  | 68.75   | 74.15           |
| 15K  | 58.39   | 62.12           |
| 20K  | 52.77   | 56.50           |
| 25K  | 49.19   | 51.79           |
| 30K  | 46.51   | 48.44           |
| 35K  | 44.77   | 46.12           |
| 40K  | 43.24   | 43.83           |
| 45K  | 41.87   | 42.69           |
| 50K  | 40.84   | 41.12           |
| 55K  | 39.93   | 40.21           |
| 60K  | 39.21   | 39.27           |
| 65K  | 38.55   | 38.49           |
| 70K  | 37.97   | 37.40           |
| 75K  | 37.42   | 36.87           |
| 80K  | 36.99   | 36.38           |

### K=5 (batch=32) vs N=6 (batch=32)

| Iter | N=6 b32 | D=1 K=5 b32 |
|------|---------|-------------|
| 5K   | 74.49   | 83.90       |
| 10K  | 55.87   | 63.93       |
| 15K  | 48.81   | 55.68       |
| 20K  | 45.16   | 50.66       |
| 25K  | 42.55   | 47.18       |
| 30K  | 40.76   | 44.79       |
| 35K  | 39.39   | 42.86       |
| 40K  | 38.36   | 41.18       |
| 45K  | 37.21   | 40.17       |
| 50K  | 36.49   | 38.84       |
| 55K  | 35.89   | 38.18       |
| 60K  | 35.30   | 37.44       |
| 65K  | 34.68   | 36.56       |
| 70K  | 34.31   | 36.13       |
| 75K  | 33.83   | 35.31       |
| 80K  | 33.50   | 34.86       |
| 85K  | 33.21   | 34.54       |
| 90K  | 32.86   | 34.29       |
| 95K  | 32.60   | 33.68       |
| 100K | 32.31   | 33.30       |

### K=10 k_min=2 (batch=16, from qmt machine) vs N=6 (batch=16)

| Iter | N=6 b16 | D=1 K=10 k_min=2 |
|------|---------|-----------------|
| 5K   | 93.64   | 100.20          |
| 10K  | 68.75   | 75.40           |
| 15K  | 58.39   | 64.54           |
| 20K  | 52.77   | 58.71           |
| 25K  | 49.19   | 54.48           |
| 30K  | 46.51   | 50.82           |
| 35K  | 44.77   | 48.36           |
| 40K  | 43.24   | 46.10           |
| 45K  | 41.87   | 44.91           |
| 50K  | 40.84   | 43.37           |
| 55K  | 39.93   | 42.42           |
| 60K  | 39.21   | 41.34           |
| 65K  | 38.55   | 40.61           |
| 70K  | 37.97   | 39.55           |
| 75K  | 37.42   | 38.81           |
| 80K  | 36.99   | 38.31           |
| 85K  | 36.50   | 37.59           |
| 90K  | 36.09   | 37.32*          |
| 95K  | 35.74   | 36.64           |
| 100K | 35.37   | 36.28           |
| 105K | —       | 35.72           |
| 110K | —       | 35.25           |
| 115K | —       | 34.97           |
| 120K | —       | 34.54           |

*N=6 b16 ran to 100K. k_min=2 gaps beyond 100K cannot be computed.

## Analysis

### K=10 k_min=0 is the winner

- **Crosses zero at 65K** — D=1 beats N=6 at FLOP parity
- Gap grows to **-0.61** by 80K
- K=5 never crosses zero — stabilizes around +1.0
- K=10 k_min=2 tracks close to K=5 — random lower K values are the bottleneck

### Why fixed K=10 works and random K doesn't

With k_min=2, the model samples K uniformly from {2, 3, ..., 10} each batch. Most samples have K < 10, so the correction chain rarely gets full-depth training. The model optimizes for the average K (~6), not for K=10.

With k_min=0 (fixed K=10), every batch trains the full 10-iteration correction chain. The model learns to use all 10 iterations, building longer and more effective correction chains.

This confirms the hypothesis: **K limits the effective training depth of the correction chain. Higher fixed K lets D=1 exploit longer block_size.**

### Why K matters more at large block_size

At block_size=256, K=5 was sufficient — 5 correction steps can cover a reasonable fraction of 256 positions. At block_size=1024, K=5 only covers 5/1024 of the sequence depth per iteration. K=10 doubles this, letting the model build corrections over longer ranges during training.

## Fine-tune Experiment

Also running: D=1 K=10 fine-tuned from N=1 C=2048 pretrained for 85K iters.

| ft iter | Total iter | D=1 K=10 ft PPL |
|---------|-----------|----------------|
| 0       | 85K       | 65.60          |
| 5K      | 90K       | 53.58          |
| 10K     | 95K       | 49.59          |
| 15K     | 100K      | 46.66          |
| 20K     | 105K      | 45.28          |
| 25K     | 110K      | 43.14          |
| 30K     | 115K      | 41.75          |
| 35K     | 120K      | 40.71          |
| 40K     | 125K      | 39.58          |
| 45K     | 130K      | 38.80          |
| 50K     | 135K      | 37.71          |
| 55K     | 140K      | 37.25          |
| 60K     | 145K      | 36.64          |
| 65K     | 150K      | 35.92          |
| 70K     | 155K      | 35.44          |
| 75K     | 160K      | 34.79          |
| 80K     | 165K      | 34.50          |
| 85K     | 170K      | 33.97          |
| 90K     | 175K      | 33.71          |
| 95K     | 180K      | 33.44          |
| 100K    | 185K      | 33.05          |

At 185K total iters, fine-tune is at 33.05 — well below N=6 b16's 100K result (35.37). But at matched total iters (e.g. 100K), fine-tune (46.66) is far behind N=6 (35.37). The correction mechanism needs many iters to adapt the pretrained representations.

## Active Experiments

- D=1 K=10 k_min=0 scratch (GPU 0, this machine) — at 80K, running to 400K
- D=1 K=10 fine-tune from N=1 (GPU 1, this machine) — at 100K ft, running to 400K
- D=1 K=10 k_min=2 scratch (GPU 0, qmt machine) — at 120K
- N=6 b32 bs1024 (GPU 0, this machine) — still running at 287K, was supposed to be stopped

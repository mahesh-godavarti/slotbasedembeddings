# D=1 K=10 Fine-tune from N=1 Roformer

## Experiment

Train a single-layer roformer (N=1), then convert to D=1 look-ahead with K=10 iterations and fine-tune. This tests whether:
1. K=10 (more training iterations per forward pass) helps D=1 beyond K=5
2. Fine-tuning from a pretrained single-layer transformer works for adding the correction mechanism

## Setup

### Phase 1: Pretrain N=1 roformer (completed)
- Model: roformer, N=1, C=2048
- block_size=1024, batch=16, lr=2e-4
- Trained for **85K iters** (stopped early, originally planned 100K)
- Final PPL: **65.60**
- Checkpoint: `checkpoints_n1_c2048_bs1024/roformer_latest.pt`
- Log: `logs/n1_c2048_bs1024.log`

### Phase 2: Convert and fine-tune D=1 K=10 (running)
- Converted using `convert_roformer_to_lookahead.py`
  - Copies token_embedding, block weights, ln_f, lm_head→head
  - Initializes corr_ffn output layer to zeros (correction starts as zero, so D=1 initially behaves like N=1)
  - ln_corr initialized to default (weight=1, bias=0)
- Model: block_head_corr_ffn_add, D=1, K=10, C=2048
- block_size=1024, batch=16, lr=2e-4, k_min=0 (fixed K=10 every batch)
- max_iters=400000 (iter counter restarted from 1 after conversion)
- Checkpoint: `checkpoints_d1_c2048_bs1024_k10_ft/`
- Log: `logs/d1_c2048_bs1024_k10_ft.log`
- GPU 1

### Baseline: N=6 C=1088 (running)
- Model: roformer, N=6, C=1088
- block_size=1024, batch=16, lr=2e-4
- max_iters=400000
- FLOP-matched to D=1 C=2048 at ~85M inference FLOPs
- Log: `logs/roformer_n6_c1088_bs1024_b16.log`
- GPU 0

## Context

### Previous results at K=5 bs1024

D=1 K=5 C=2048 matched N=6 C=1088 at bs1024 by ~245K iters (gap ~0.0). The question is whether K=10 provides additional benefit.

At K=5, the correction chain during training can only propagate K=5 steps. Positions more than 5 apart never see fully converged corrections during training — even though at sequential inference the chain is unlimited. With block_size=1024, most of the sequence is beyond the K=5 horizon.

K=10 doubles the training depth, letting the model learn to use longer correction chains within the 1024-position window. This should unlock more of the bs1024 benefit that K=5 couldn't fully exploit — and is likely why the bs256→bs512→bs1024 gains for D=1 were diminishing at K=5.

### Why fine-tune from N=1

- N=1 roformer and D=1 look-ahead share the same core: one transformer block + embeddings + head
- D=1 adds only the correction FFN (8C² params) and ln_corr
- By pretraining N=1, the block learns good representations first
- The correction mechanism is then added on top, initialized to zero (no disruption)
- This is faster than training D=1 K=10 from scratch (K=10 is ~2x slower per iter than K=5)

## Results

### N=1 pretraining curve

| Iter | N=1 C=2048 PPL |
|------|---------------|
| 5K   | 153.41        |
| 10K  | 118.07        |
| 15K  | 102.65        |
| 20K  | 93.83         |
| 25K  | 88.35         |
| 30K  | 83.85         |
| 35K  | 80.87         |
| 40K  | 78.26         |
| 45K  | 75.52         |
| 50K  | 73.66         |
| 55K  | 72.08         |
| 60K  | 70.77         |
| 65K  | 69.40         |
| 70K  | 68.42         |
| 75K  | 66.91         |
| 80K  | 66.55         |
| 85K  | **65.60**     |

### D=1 K=10 fine-tune (running)

TBD — iter counter starts from 1 after conversion, initial PPL 65.60.

### N=6 C=1088 baseline (running)

| Iter | N=6 PPL |
|------|---------|
| 5K   | 93.64   |
| 10K  | 68.75   |
| 15K  | 58.39   |
| 20K  | 52.77   |
| 25K  | 49.19   |
| 30K  | 46.51   |
| 35K  | 44.77   |

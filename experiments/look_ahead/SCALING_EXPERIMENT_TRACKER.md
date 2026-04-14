# Scaling Experiment Tracker

All experiments: C=1024, block_size=64, OWT data, lr=2e-4, softmax.

## Models and Checkpoints

### Roformers (N layers, separate weights)

| Model | n_head | Checkpoint (cycle 1) | Checkpoint (cycle 2) | Checkpoint (cycle 3) |
|-------|--------|---------------------|---------------------|---------------------|
| N=1 | 1 | checkpoints_scaling_n1/ | checkpoints_scaling_n1_cont/ | checkpoints_scaling_n1_cont2/ |
| N=2 | 16 | checkpoints_scaling_n2/ | checkpoints_scaling_n2_cont/ | checkpoints_scaling_n2_cont2/ |
| N=3 | 16 | checkpoints_scaling_n3/ | checkpoints_scaling_n3_cont/ | checkpoints_scaling_n3_cont2/ |
| N=6 | 16 | checkpoints_scaling_n6/ | checkpoints_scaling_n6_cont/ | checkpoints_scaling_n6_cont2/ |

### D fine-tunes (converted from cycle 1 roformer, K=2-5)

| Model | n_head | Checkpoint (cycle 1) | Checkpoint (cycle 2) | Checkpoint (cycle 3) | Checkpoint (round 2) |
|-------|--------|---------------------|---------------------|---------------------|---------------------|
| D=1 cont | 1 | checkpoints_scaling_d1/ | checkpoints_scaling_d1_cont/ | checkpoints_scaling_d1_cont2/ | checkpoints_scaling_d1_cont3/ |
| D=2 cont | 16 | checkpoints_scaling_d2/ | checkpoints_scaling_d2_cont/ | checkpoints_scaling_d2_cont2/ | checkpoints_scaling_d2_cont3/ |
| D=3 cont | 16 | checkpoints_scaling_d3/ | checkpoints_scaling_d3_cont/ | checkpoints_scaling_d3_cont2/ | checkpoints_scaling_d3_cont3/ |
| D=6 cont | 16 | checkpoints_scaling_d6/ | checkpoints_scaling_d6_cont/ | checkpoints_scaling_d6_cont2/ | checkpoints_scaling_d6_cont3/ |

### D fresh (converted from cycle 3 roformer, K=2-5)

| Model | n_head | Checkpoint |
|-------|--------|-----------|
| D=1 fresh | 1 | checkpoints_scaling_d1_fresh/ |
| D=2 fresh | 16 | checkpoints_scaling_d2_fresh/ |
| D=3 fresh | 16 | checkpoints_scaling_d3_fresh/ |
| D=6 fresh | 16 | checkpoints_scaling_d6_fresh/ |

## Token Budgets

Per cycle: roformers get 409M tokens, D fine-tunes get 102M (cycles 1-3) or 409M (round 2).

### Total tokens seen by each model

| Model | Roformer pretrain | FT tokens | Total |
|-------|------------------|-----------|-------|
| N=1 (3 cycles) | — | — | 1,227M |
| N=2 (3 cycles) | — | — | 1,227M |
| N=3 (3 cycles) | — | — | 1,227M |
| N=6 (3 cycles) | — | — | 1,227M |
| D=1 cont (3 FT cycles + round 2) | 409M | 306M + 409M = 715M | 1,124M |
| D=2 cont (3 FT cycles + round 2) | 409M | 306M + 409M = 715M | 1,124M |
| D=3 cont (3 FT cycles + round 2) | 409M | 306M + 409M = 715M | 1,124M |
| D=6 cont (3 FT cycles + round 2) | 409M | 306M + 409M = 715M | 1,124M |
| D=1 fresh (round 2) | 1,227M | 409M | 1,636M |
| D=2 fresh (round 2) | 1,227M | 409M | 1,636M |
| D=3 fresh (round 2) | 1,227M | 409M | 1,636M |
| D=6 fresh (round 2) | 1,227M | 409M | 1,636M |

## PPL Results

### After cycle 1 (409M roformer, 102M fine-tune)

| N | N PPL | D PPL | D - N |
|---|-------|-------|-------|
| 1 | 145.14 | 116.03 | -29.11 |
| 2 | 90.35 | 87.42 | -2.93 |
| 3 | 74.80 | 75.71 | +0.91 |
| 6 | 62.35 | 63.00 | +0.65 |

### After cycle 2 (818M roformer, 409M + 204M fine-tune)

| N | N PPL | D PPL | D - N |
|---|-------|-------|-------|
| 1 | 122.45 | 103.72 | -18.73 |
| 2 | 78.05 | 81.06 | +3.01 |
| 3 | 65.55 | 70.90 | +5.35 |
| 6 | 55.05 | 60.47 | +5.42 |

### After cycle 3 (1,227M roformer, 409M + 306M fine-tune)

| N | N PPL | D PPL | D - N |
|---|-------|-------|-------|
| 1 | 112.82 | 96.75 | -16.07 |
| 2 | 72.83 | 77.29 | +4.46 |
| 3 | 61.60 | 67.73 | +6.13 |
| 6 | 52.47 | 58.76 | +6.29 |

### After round 2 (in progress)

| N | N PPL (cycle 3) | D cont3 PPL | D fresh PPL |
|---|----------------|-------------|-------------|
| 1 | 112.82 | — | — |
| 2 | 72.83 | — | — |
| 3 | 61.60 | — | — |
| 6 | 52.47 | — | — |

## Log Files

### Roformers
- `logs/scaling_roformer_n{1,2,3,6}_c1024_bs64.log` (cycle 1)
- `logs/scaling_roformer_n{1,2,3,6}_c1024_bs64_cont.log` (cycle 2)
- `logs/scaling_roformer_n{1,2,3,6}_c1024_bs64_cont2.log` (cycle 3)

### D fine-tunes (cont path)
- `logs/scaling_finetune_d{1,2,3,6}_c1024_bs64.log` (cycle 1)
- `logs/scaling_finetune_d{1,2,3,6}_c1024_bs64_cont.log` (cycle 2)
- `logs/scaling_finetune_d{1,2,3,6}_c1024_bs64_cont2.log` (cycle 3)
- `logs/scaling_finetune_d{1,2,3,6}_c1024_bs64_cont3.log` (round 2 cont)

### D fine-tunes (fresh path)
- `logs/scaling_finetune_d{1,2,3,6}_c1024_bs64_fresh.log` (round 2 fresh)

## Scripts
- `run_scaling_experiment.sh` — cycle 1 (N=1 only, cuDNN error stopped rest)
- `run_scaling_experiment_resume.sh` — cycle 1 resume (N=2,3,6)
- `run_scaling_continuation.sh` — cycle 2
- `run_scaling_continuation2.sh` — cycle 3
- `run_scaling_finetune_round2.sh` — round 2 (fresh + cont3)
- `eval_scaling.sh` — final eval at batch=32 (not yet run)

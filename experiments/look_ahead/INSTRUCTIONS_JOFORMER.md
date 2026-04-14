# Instructions: JoFormer Look-Ahead Experiments

## Context

We have shown that D=1 look-ahead with roformer blocks steadily closes the gap against deeper roformer models. The joformer uses non-abelian attention (rotates V in addition to K and Q), which was shown to be superior to roformer in early small-scale experiments. We now want to test joformer look-ahead at scale.

All code is in `/home/ubuntu/look_ahead8/`. This is a copy of look_ahead7 with two model registrations added to models.py:
- `joformer_projected_mh` — N-layer joformer baseline (multi-head)
- `joformer_projected_corr_ffn_add` — D=1 look-ahead with joformer blocks (multi-head)

## FLOP accounting

JoFormer projected block = 16C² (4C² attention + 8C² FFN + C² vector_proj + 3C² angle_proj)
Roformer block = 12C²
corr_ffn = 8C²

D=1 joformer look-ahead = 16C² + 8C² = 24C²
N-layer roformer = 12N × C²

## Experiments

### Experiment 1: D=1 joformer FLOP-equivalent to N=12 roformer C=768 (~85M FLOPs)

N=12 roformer C=768: 144 × 768² = 85.0M → **37.83 PPL** (done)
D=1 roformer C=2048: 20 × 2048² = 83.9M → **39.94 PPL** (done)
D=1 joformer: 24C² = 144 × 768² → C = 1888 (n_head=16, head_dim=118, even). 24 × 1888² = 85.6M.

```bash
cd /home/ubuntu/look_ahead8
mkdir -p checkpoints_joformer_d1_c1888 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models joformer_projected_corr_ffn_add --n_embed 1888 --n_layers 5 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_joformer_d1_c1888 \
    --gpu 0 \
    --amp 2>&1 | tee logs/joformer_d1_c1888_scratch.log
```

### Experiment 2: D=1 joformer FLOP-equivalent to N=12 roformer C=1536 (~340M FLOPs)

N=12 roformer C=1536: 144 × 1536² = 340M → **29.01 PPL** (done)
D=1 roformer C=4128: 20 × 4128² = 341M → **33.40 PPL** (done)
D=1 joformer: 24C² = 144 × 1536² → C = 3776 (n_head=16, head_dim=236, even). 24 × 3776² = 342M.

```bash
cd /home/ubuntu/look_ahead8
mkdir -p checkpoints_joformer_d1_c3776 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models joformer_projected_corr_ffn_add --n_embed 3776 --n_layers 5 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_joformer_d1_c3776 \
    --gpu 1 \
    --amp 2>&1 | tee logs/joformer_d1_c3776_scratch.log
```

## Execution order

Launch both experiments immediately — Experiment 1 on GPU 0, Experiment 2 on GPU 1.

## Comparisons to make

### ~85M FLOPs

| Model | FLOPs | PPL |
|-------|-------|-----|
| N=12 roformer C=768 | 85.0M | 37.83 (done) |
| D=1 roformer C=2048 | 83.9M | 39.94 (done) |
| N=2 roformer C=1888 | 85.4M | 42.99 (done) |
| D=1 joformer C=1888 | 85.6M | ? |

### ~340M FLOPs

| Model | FLOPs | PPL |
|-------|-------|-----|
| N=12 roformer C=1536 | 340M | 29.01 (done) |
| D=1 roformer C=4128 | 341M | 33.40 (done) |
| N=2 roformer C=3776 | 342M | 36.10 (done) |
| D=1 joformer C=3776 | 342M | ? |

Key questions:
1. Does D=1 joformer beat D=1 roformer at the same FLOPs? (joformer's non-abelian attention should help the correction mechanism)
2. How close does D=1 joformer get to N=12 roformer?
3. Does the gap narrow steadily like D=1 roformer, or faster?

## Checking progress

```bash
bash check_progress.sh logs/LOGFILE.log
```

## Important notes

- Venv: `/home/ubuntu/exp8/venv/bin/python`
- Data: `/home/ubuntu/look_ahead/look_ahead/data_owt`
- All code runs from `/home/ubuntu/look_ahead8/` (NOT look_ahead6 or look_ahead7)
- Do NOT modify blocks.py or models.py
- Checkpoints auto-resume from checkpoint_dir if files exist

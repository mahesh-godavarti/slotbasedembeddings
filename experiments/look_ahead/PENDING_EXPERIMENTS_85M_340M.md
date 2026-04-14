# Pending Experiments

## ~85M FLOP budget (batch=64, block_size=256, 100K iters, OWT)

### Baselines (N=x)

| Model | FLOPs | C | Status |
|-------|-------|---|--------|
| N=2 C=1888 | 24 × 1888² = 85.4M | 1888 | Done: 42.99 |
| N=4 C=1344 | 48 × 1344² = 86.7M | 1344 | Done: 38.68 |
| N=6 C=1088 | 72 × 1088² = 85.2M | 1088 | Running (qmti92t1) |
| N=12 C=768 | 144 × 768² = 85.0M | 768 | Done: 37.83 |

### Look-ahead (D=x)

| Model | FLOPs | C | Status |
|-------|-------|---|--------|
| D=1 C=2048 | 20 × 2048² = 83.9M | 2048 | Done: 39.94 |
| D=3 C=1408 | 44 × 1408² = 87.2M | 1408 | Running (this machine, GPU 0) |
| D=5 C=1120 | 68 × 1120² = 85.3M | 1120 | **Pending** |
| D=6 C=1024 | 80 × 1024² = 83.9M | 1024 | **Pending** |
| D=11 C=768 | 140 × 768² = 82.6M | 768 | Running (this machine, GPU 1) |

### Pending commands (85M)

D=5 C=1120:
```bash
cd /home/ubuntu/look_ahead6
mkdir -p checkpoints_d5_c1120 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1120 --n_layers 25 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 5 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d5_c1120 \
    --gpu GPU_ID \
    --amp 2>&1 | tee logs/d5_c1120_scratch.log
```

D=6 C=1024:
```bash
cd /home/ubuntu/look_ahead6
mkdir -p checkpoints_d6_c1024 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 30 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 6 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d6_c1024 \
    --gpu GPU_ID \
    --amp 2>&1 | tee logs/d6_c1024_scratch.log
```

---

## ~341M FLOP budget (batch=32, block_size=256, 200K iters, OWT)

### Baselines (N=x)

| Model | FLOPs | C | Status |
|-------|-------|---|--------|
| N=2 C=3776 | 24 × 3776² = 342M | 3776 | Done: 36.10 |
| N=4 C=2656 | 48 × 2656² = 339M | 2656 | Done: 31.95 |
| N=6 C=2176 | 72 × 2176² = 341M | 2176 | Done: 30.35 |
| N=12 C=1536 | 144 × 1536² = 340M | 1536 | Done: 29.01 |
| N=24 C=1088 | 288 × 1088² = 341M | 1088 | Done: 28.68 |

### Look-ahead (D=x)

| Model | FLOPs | C | Status |
|-------|-------|---|--------|
| D=1 C=4128 | 20 × 4128² = 341M | 4128 | Done: 33.40 |
| D=3 C=2784 | 44 × 2784² = 341M | 2784 | Running (qmti92t1) |
| D=5 C=2240 | 68 × 2240² = 341M | 2240 | **Pending** |
| D=6 C=2048 | 80 × 2048² = 336M | 2048 | Done: 29.04 |
| D=11 C=1568 | 140 × 1568² = 344M | 1568 | **Pending** |

### SA (D=x)

| Model | FLOPs | C | Status |
|-------|-------|---|--------|
| SA D=1 C=3776 | 24 × 3776² = 342M | 3776 | Done: 33.32 |
| SA D=3 C=2656 | 48 × 2656² = 339M | 2656 | Stopped: 32.16 @ 140K |
| SA D=5 C=2176 | 72 × 2176² = 341M | 2176 | Done: 29.38 |

### Pending commands (341M)

D=5 C=2240:
```bash
cd /home/ubuntu/look_ahead6
mkdir -p checkpoints_d5_c2240 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2240 --n_layers 25 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 5 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d5_c2240 \
    --gpu GPU_ID \
    --amp 2>&1 | tee logs/d5_c2240_scratch.log
```

D=11 C=1568:
```bash
cd /home/ubuntu/look_ahead6
mkdir -p checkpoints_d11_c1568 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1568 --n_layers 55 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 11 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d11_c1568 \
    --gpu GPU_ID \
    --amp 2>&1 | tee logs/d11_c1568_scratch.log
```

---

## Width scaling: D=1 vs N=2 (block_size=64, Chinchilla tokens)

Script: `run_width_scaling_d1.sh` (transferred to qmti92t1)

| D=1 C | FLOP-matched N=2 C | N=2 PPL | D=1 Status |
|-------|-------------------|---------|------------|
| 280 | 256 | 158.83 | **Pending** |
| 560 | 512 | 95.48 | **Pending** |
| 1120 | 1024 | 72.83 | **Pending** |

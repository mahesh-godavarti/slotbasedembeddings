#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# 1. rope_lf: 100K at lr=5e-4
echo "$(date): rope_lf 100K on GPU 0"
/home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 \
    --batch_size 32 --lr 5e-4 --max_iters 100000 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models rope_lf \
    --checkpoint_dir checkpoints/rope_lf \
    --gpu 0

# 2. 50K at lr=2e-4
echo "$(date): rope_lf lr=2e-4 on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/rope_lf/rope_lf_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192 \
    --bf16 \
    --checkpoint_dir checkpoints/rope_lf_sched_150k \
    --gpu 0

# 3. 50K at lr=5e-5
echo "$(date): rope_lf lr=5e-5 on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/rope_lf_sched_150k/rope_lf_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192 \
    --bf16 \
    --checkpoint_dir checkpoints/rope_lf_sched_200k \
    --gpu 0

echo "$(date): GPU 0 done."

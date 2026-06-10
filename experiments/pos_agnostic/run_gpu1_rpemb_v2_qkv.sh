#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# 1. rpemb_v2 qkv: 100K at lr=5e-4
echo "$(date): rpemb_v2 qkv 100K from scratch on GPU 1"
/home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 \
    --batch_size 32 --lr 5e-4 --max_iters 100000 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models shared_rpemb_qkv \
    --checkpoint_dir checkpoints/shared_rpemb_qkv_v2 \
    --gpu 1

# 2. 50K at lr=2e-4
echo "$(date): rpemb_v2 qkv lr=2e-4 on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_rpemb_qkv_v2/shared_rpemb_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_rpemb_qkv_v2_sched_150k \
    --gpu 1

# 3. 50K at lr=5e-5
echo "$(date): rpemb_v2 qkv lr=5e-5 on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_rpemb_qkv_v2_sched_150k/shared_rpemb_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_rpemb_qkv_v2_sched_200k \
    --gpu 1

echo "$(date): GPU 1 done."

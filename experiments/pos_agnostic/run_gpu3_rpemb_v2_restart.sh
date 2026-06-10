#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# rpemb_v2 restart from scratch (original crashed from disk full)
# 1. 100K at lr=5e-4
echo "$(date): rpemb_v2 100K from scratch on GPU 3"
/home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 \
    --batch_size 32 --lr 5e-4 --max_iters 100000 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models shared_rpemb_qk \
    --checkpoint_dir checkpoints/shared_rpemb_qk_v2 \
    --gpu 3

# 2. 50K at lr=2e-4
echo "$(date): rpemb_v2 lr=2e-4 phase on GPU 3"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_rpemb_qk_v2/shared_rpemb_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_rpemb_qk_v2_sched_150k \
    --gpu 3

# 3. 50K at lr=5e-5
echo "$(date): rpemb_v2 lr=5e-5 phase on GPU 3"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_rpemb_qk_v2_sched_150k/shared_rpemb_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_rpemb_qk_v2_sched_200k \
    --gpu 3

echo "$(date): GPU 3 done."

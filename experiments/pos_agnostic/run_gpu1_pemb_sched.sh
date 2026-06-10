#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# 1. pemb_qk: 50K at lr=2e-4 (from 100K checkpoint)
echo "$(date): pemb_qk lr=2e-4 phase on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pemb_qk/shared_pemb_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pemb_qk_sched_150k \
    --gpu 1

# 2. pemb_qk: 50K at lr=5e-5 (from 150K checkpoint)
echo "$(date): pemb_qk lr=5e-5 phase on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pemb_qk_sched_150k/shared_pemb_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pemb_qk_sched_200k \
    --gpu 1

echo "$(date): GPU 1 done."

#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# Stage 1: 100K→150K at lr=2e-4
echo "=== ALiBi Stage 1: continuing at lr=2e-4 for 50K ==="
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/new_exp/alibi_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --eval_interval 5000 \
    --eval_batch_size 4 --bf16 \
    --checkpoint_dir checkpoints/alibi_150k \
    --gpu 0

# Stage 2: 150K→200K at lr=5e-5
echo "=== ALiBi Stage 2: continuing at lr=5e-5 for 50K ==="
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/alibi_150k/alibi_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --eval_interval 5000 \
    --eval_batch_size 4 --bf16 \
    --checkpoint_dir checkpoints/alibi_200k \
    --gpu 0

echo "Done: ALiBi 200K"

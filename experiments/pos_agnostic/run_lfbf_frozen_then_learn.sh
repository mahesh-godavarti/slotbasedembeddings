#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

GPU=3

# Phase 1: Train with frozen angles for 5K at lr=5e-4
echo "Phase 1: Training shared_lfbf_qk with frozen angles for 5K..."
/home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 \
    --lr 5e-4 --angle_lr 1e-30 --max_iters 5000 \
    --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 --angle_hidden_mult 1 \
    --models shared_lfbf_qk \
    --checkpoint_dir checkpoints/shared_lfbf_qk_h1_frozen \
    --gpu $GPU

echo "Phase 1 done. Resuming with angle learning..."

# Phase 2: Resume with both lr=5e-5
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_lfbf_qk_h1_frozen/shared_lfbf_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --angle_lr 5e-5 --max_iters 95000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_lfbf_qk_h1_learn \
    --gpu $GPU

echo "Phase 2 done."

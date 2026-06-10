#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

COMMON="--data_dir /home/ubuntu/look_ahead/look_ahead/data_owt --n_embed 768 --n_layers 1 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-4 --max_iters 50000 --eval_interval 5000 --extrap_interval 5000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16"

# rope and joformer_fixed (standard architecture)
echo "Running rope and joformer_fixed (1 layer)..."
/home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models rope joformer_fixed \
    --checkpoint_dir checkpoints/1layer_rope_jfixed \
    --gpu 3

# monoidal2 (datadep2, no V rotation)
echo "Running monoidal2 (1 layer)..."
/home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models monoidal2 --angle_lr 5e-4 \
    --checkpoint_dir checkpoints/1layer_monoidal2 \
    --gpu 3

# joformer2 (datadep2, V rotation)
echo "Running joformer2 (1 layer)..."
/home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models joformer2 --angle_lr 5e-4 \
    --checkpoint_dir checkpoints/1layer_joformer2 \
    --gpu 3

echo "All done."

#!/bin/bash
# Launch 4 parallel experiments on 4 GPUs (A100-80GB)
# Config matches scale-up: n_embed=768, n_layers=16, n_heads=8, block_size=512, lr=5e-4
# Eval at: 512, 1024, 2048, 4096, 8192, 16384

DATA_DIR="/home/ubuntu/look_ahead/look_ahead/data_owt"
COMMON="--data_dir $DATA_DIR --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-4 --max_iters 100000 --eval_interval 5000 --extrap_interval 10000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16"

cd /home/ubuntu/pos_agnostic

# GPU 0: shared_pos_qk (shared angle MLP, cumsum, Q/K only)
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_pos_qk \
    --checkpoint_dir checkpoints/new_exp \
    --gpu 0 \
    >> logs/pafl_shared_pos_qk.log 2>&1 &

# GPU 1: shared_pos_qkv (shared angle MLP, cumsum, Q/K/V + inverse)
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_pos_qkv \
    --checkpoint_dir checkpoints/new_exp \
    --gpu 1 \
    >> logs/pafl_shared_pos_qkv.log 2>&1 &

# GPU 2: random_pos_qk (random positive angles, cumsum, Q/K only)
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models random_pos_qk \
    --checkpoint_dir checkpoints/new_exp \
    --gpu 2 \
    >> logs/pafl_random_pos_qk.log 2>&1 &

# GPU 3: random_pos_qkv (random positive angles, cumsum, Q/K/V + inverse)
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models random_pos_qkv \
    --checkpoint_dir checkpoints/new_exp \
    --gpu 3 \
    >> logs/pafl_random_pos_qkv.log 2>&1 &

echo "Launched 4 experiments on GPUs 0-3"
echo "Logs:"
echo "  logs/pafl_shared_pos_qk.log"
echo "  logs/pafl_shared_pos_qkv.log"
echo "  logs/pafl_random_pos_qk.log"
echo "  logs/pafl_random_pos_qkv.log"

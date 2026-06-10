#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

COMMON="--data_dir /home/ubuntu/look_ahead/look_ahead/data_owt --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-4 --max_iters 100000 --eval_interval 5000 --extrap_interval 10000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16 --angle_hidden_mult 1"

# GPU 0: shared_ln_qk h1, angle_dropout=0.3
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_ln_qk --angle_dropout 0.3 \
    --checkpoint_dir checkpoints/shared_ln_qk_h1_adrop03 \
    --gpu 0 \
    >> logs/pafl_shared_ln_qk_h1_adrop03.log 2>&1 &

# GPU 1: shared_ln_qk h1, angle_dropout=0.5
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_ln_qk --angle_dropout 0.5 \
    --checkpoint_dir checkpoints/shared_ln_qk_h1_adrop05 \
    --gpu 1 \
    >> logs/pafl_shared_ln_qk_h1_adrop05.log 2>&1 &

# GPU 2: shared_ln_qkv h1, angle_dropout=0.3
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_ln_qkv --angle_dropout 0.3 \
    --checkpoint_dir checkpoints/shared_ln_qkv_h1_adrop03 \
    --gpu 2 \
    >> logs/pafl_shared_ln_qkv_h1_adrop03.log 2>&1 &

# GPU 3: shared_ln_qkv h1, angle_dropout=0.5
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_ln_qkv --angle_dropout 0.5 \
    --checkpoint_dir checkpoints/shared_ln_qkv_h1_adrop05 \
    --gpu 3 \
    >> logs/pafl_shared_ln_qkv_h1_adrop05.log 2>&1 &

echo "Launched 4 angle dropout experiments:"
echo "  GPU 0: shared_ln_qk  h1 angle_dropout=0.3"
echo "  GPU 1: shared_ln_qk  h1 angle_dropout=0.5"
echo "  GPU 2: shared_ln_qkv h1 angle_dropout=0.3"
echo "  GPU 3: shared_ln_qkv h1 angle_dropout=0.5"

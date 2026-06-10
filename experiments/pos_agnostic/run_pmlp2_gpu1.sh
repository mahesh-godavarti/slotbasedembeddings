#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "$(date): pmlp2 (stacked corrections) 100K on GPU 1"
/home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 \
    --batch_size 32 --lr 5e-4 --max_iters 100000 \
    --eval_interval 1000 --extrap_interval 1000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models shared_pmlp2_qk \
    --checkpoint_dir checkpoints/shared_pmlp2_qk \
    --gpu 1

echo "$(date): pmlp2 done."

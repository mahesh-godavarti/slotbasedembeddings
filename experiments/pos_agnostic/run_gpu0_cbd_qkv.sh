#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# Resume cbd K=4 qkv from 55K to 100K (45K more iters)
echo "$(date): cbd K=4 qkv resume from 55K to 100K on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_cbd_qkv_K4/shared_cbd_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-4 --max_iters 45000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_cbd_qkv_K4_100k \
    --gpu 0

echo "$(date): done."

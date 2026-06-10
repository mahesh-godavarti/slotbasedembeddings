#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 0 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

# 1. cbd K=4 qkv: 50K at lr=2e-4 (from 100K checkpoint)
echo "$(date): cbd K=4 qkv lr=2e-4 phase on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_cbd_qkv_K4_100k/shared_cbd_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_cbd_qkv_K4_sched_150k \
    --gpu 0

# 2. cbd K=4 qkv: 50K at lr=5e-5 (from 150K checkpoint)
echo "$(date): cbd K=4 qkv lr=5e-5 phase on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_cbd_qkv_K4_sched_150k/shared_cbd_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_cbd_qkv_K4_sched_200k \
    --gpu 0

echo "$(date): GPU 0 queue done."

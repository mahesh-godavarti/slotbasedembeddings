#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

# 1. lf_qkv: resume from 65K to 100K (35K more)
echo "$(date): lf_qkv resume to 100K on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_lf_qkv_h1_resume/shared_lf_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-4 --max_iters 35000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_lf_qkv_h1_100k \
    --gpu 1

# 2. 50K at lr=2e-4
echo "$(date): lf_qkv lr=2e-4 on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_lf_qkv_h1_100k/shared_lf_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_lf_qkv_h1_sched_150k \
    --gpu 1

# 3. 50K at lr=5e-5
echo "$(date): lf_qkv lr=5e-5 on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_lf_qkv_h1_sched_150k/shared_lf_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_lf_qkv_h1_sched_200k \
    --gpu 1

echo "$(date): GPU 1 done."

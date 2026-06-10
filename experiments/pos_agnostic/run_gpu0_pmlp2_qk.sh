#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 0 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

# 1. pmlp2_qk: resume from 21K to 100K (79K more iters)
echo "$(date): pmlp2_qk resume to 100K on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pmlp2_qk/shared_pmlp2_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-4 --max_iters 79000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pmlp2_qk_100k \
    --gpu 0

# 2. pmlp2_qk: 50K at lr=2e-4
echo "$(date): pmlp2_qk lr=2e-4 phase on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pmlp2_qk_100k/shared_pmlp2_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pmlp2_qk_sched_150k \
    --gpu 0

# 3. pmlp2_qk: 50K at lr=5e-5
echo "$(date): pmlp2_qk lr=5e-5 phase on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pmlp2_qk_sched_150k/shared_pmlp2_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pmlp2_qk_sched_200k \
    --gpu 0

echo "$(date): GPU 0 done."

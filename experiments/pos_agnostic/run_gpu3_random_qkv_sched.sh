#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

# 1. random_qkv: 50K at lr=2e-4 (from 100K checkpoint)
echo "$(date): random_qkv lr=2e-4 phase on GPU 3"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/new_exp/random_ln_indep_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/random_indep_qkv_sched_150k \
    --gpu 3

# 2. random_qkv: 50K at lr=5e-5 (from 150K checkpoint)
echo "$(date): random_qkv lr=5e-5 phase on GPU 3"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/random_indep_qkv_sched_150k/random_ln_indep_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/random_indep_qkv_sched_200k \
    --gpu 3

echo "$(date): GPU 3 done."

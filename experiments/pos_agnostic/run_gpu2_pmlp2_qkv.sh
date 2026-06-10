#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 2 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

# 1. pmlp2_qkv: 100K at lr=5e-4 from scratch
echo "$(date): pmlp2_qkv 100K from scratch on GPU 2"
/home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 \
    --batch_size 32 --lr 5e-4 --max_iters 100000 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models shared_pmlp2_qkv \
    --checkpoint_dir checkpoints/shared_pmlp2_qkv \
    --gpu 2

# 2. pmlp2_qkv: 50K at lr=2e-4
echo "$(date): pmlp2_qkv lr=2e-4 phase on GPU 2"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pmlp2_qkv/shared_pmlp2_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pmlp2_qkv_sched_150k \
    --gpu 2

# 3. pmlp2_qkv: 50K at lr=5e-5
echo "$(date): pmlp2_qkv lr=5e-5 phase on GPU 2"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pmlp2_qkv_sched_150k/shared_pmlp2_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pmlp2_qkv_sched_200k \
    --gpu 2

echo "$(date): GPU 2 done."

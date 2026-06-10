#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# GPU 1: joformer2 angle learning — wait for current run to finish, then continue 100K
echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
echo "Launching joformer2 angle learning 200K on GPU 1"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/joformer2_from_frozen_slowboth/joformer2_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --angle_lr 5e-5 --max_iters 100000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/joformer2_angle_200k \
    --gpu 1
echo "joformer2 angle learning 200K done."

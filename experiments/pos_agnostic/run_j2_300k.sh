#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# GPU 0: j2 control → 300K
echo "Waiting for GPU 0 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
echo "Launching j2 control 300K on GPU 0"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/joformer2_control_200k/joformer2_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --angle_lr 1e-30 --max_iters 100000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/joformer2_control_300k \
    --gpu 0
echo "j2 control 300K done."

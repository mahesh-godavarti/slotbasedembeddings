#!/bin/bash
set -e

echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then
        echo "GPU 3 is free (${MEM}MiB used). Starting."
        break
    fi
    sleep 60
done

cd /home/ubuntu/pos_agnostic

nohup /home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_lf_qkv_h1/shared_lf_qkv_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-4 --max_iters 65000 --eval_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --eval_batch_size 4 --bf16 \
    --checkpoint_dir checkpoints/shared_lf_qkv_h1_resume \
    --gpu 3 \
    >> logs/pafl_shared_lf_qkv_h1_resume.log 2>&1 &

echo "Resumed shared_lf_qkv (65K remaining) on GPU 3"

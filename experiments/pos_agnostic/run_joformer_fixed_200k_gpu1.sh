#!/bin/bash
set -e

echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then
        echo "GPU 1 is free (${MEM}MiB used). Starting."
        break
    fi
    sleep 60
done

cd /home/ubuntu/pos_agnostic

# Continue joformer_fixed from 150K at lr=5e-5 for 50K
nohup /home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/joformer_fixed_150k/joformer_fixed_150k.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --eval_interval 5000 \
    --eval_batch_size 4 --bf16 \
    --checkpoint_dir checkpoints/joformer_fixed_200k \
    --gpu 1 \
    >> logs/pafl_joformer_fixed_200k.log 2>&1 &

echo "Launched joformer_fixed 200K continuation on GPU 1"

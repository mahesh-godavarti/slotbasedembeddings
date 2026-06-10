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

nohup /home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 \
    --lr 5e-4 --angle_lr 5e-5 --max_iters 100000 --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --angle_hidden_mult 1 --angle_activation none \
    --models monoidal2 \
    --checkpoint_dir checkpoints/monoidal2_h1_noln \
    --gpu 3 \
    >> logs/pafl_monoidal2_h1_noln.log 2>&1 &

echo "Launched monoidal2 (h1, no output LN, angle_lr=5e-5) on GPU 3"

#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

COMMON="--data_dir /home/ubuntu/look_ahead/look_ahead/data_owt --n_embed 768 --n_layers 1 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-4 --max_iters 50000 --eval_interval 5000 --extrap_interval 5000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16 --angle_hidden_mult 1"

echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

echo "Running det_qk (1 layer)..."
/home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_det_qk \
    --checkpoint_dir checkpoints/1layer_det_qk \
    --gpu 3

echo "Running detb_qk (1 layer)..."
/home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_detb_qk \
    --checkpoint_dir checkpoints/1layer_detb_qk \
    --gpu 3

echo "Done."

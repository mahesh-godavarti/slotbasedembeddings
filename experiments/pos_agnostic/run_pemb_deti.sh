#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

COMMON="--data_dir /home/ubuntu/look_ahead/look_ahead/data_owt --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-4 --max_iters 100000 --eval_interval 5000 --extrap_interval 5000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16"

echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

echo "Running shared_pemb_qk (16 layers)..."
/home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_pemb_qk \
    --checkpoint_dir checkpoints/shared_pemb_qk \
    --gpu 3

echo "Done. deti_qk running on GPU 0 separately."

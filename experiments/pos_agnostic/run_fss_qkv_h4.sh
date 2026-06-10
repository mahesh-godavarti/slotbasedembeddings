#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

echo "Launching shared_fss_qkv h4 on GPU 3"
/home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 \
    --lr 5e-4 --max_iters 100000 \
    --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models shared_fss_qkv \
    --checkpoint_dir checkpoints/shared_fss_qkv_h4 \
    --gpu 3
echo "Done."

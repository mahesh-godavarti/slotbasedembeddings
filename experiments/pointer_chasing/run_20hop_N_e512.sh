#!/bin/bash
set -e

echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then
        echo "GPU 1 free. Starting 20-hop transformers e=512."
        break
    fi
    sleep 30
done

/home/ubuntu/exp8/venv/bin/python -u pointer_chasing.py \
    --n_hops 20 --n_keys 8 --n_values 16 --n_embed 512 --n_head 4 \
    --n_iters 10000 --batch_size 64 --lr 1e-3 --gpu 1 \
    --permutation --run N1,N5,N10,N15,N19,N20,N21 \
    2>&1 | tee logs/pointer_chasing_20hop_k8_e512_N_10k.log

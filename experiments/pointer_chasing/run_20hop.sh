#!/bin/bash
set -e

# GPU 0: 20-hop BPTT
echo "Waiting for GPU 0 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 1000 ]; then
        echo "GPU 0 free. Starting 20-hop BPTT."
        break
    fi
    sleep 30
done

/home/ubuntu/exp8/venv/bin/python -u pointer_chasing.py \
    --n_hops 20 --n_keys 8 --n_values 16 --n_embed 256 --n_head 4 \
    --n_iters 5000 --batch_size 64 --lr 1e-3 --gpu 0 \
    --permutation --run bptt \
    2>&1 | tee logs/pointer_chasing_20hop_k8_e256_bptt_5k.log

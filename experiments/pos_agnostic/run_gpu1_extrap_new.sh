#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

echo "$(date): Full extrap eval for new models on GPU 1"

/home/ubuntu/exp8/venv/bin/python eval_extrap.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --eval_lengths 512,1024,2048,4096,8192,16384,32768,65536 \
    --eval_batch_size 2 \
    --eval_iters 20 \
    --gpu 1 \
    --checkpoints \
        checkpoints/shared_rpemb_qk_v2_sched_200k/shared_rpemb_qk_best.pt \
        checkpoints/shared_rpemb4_qk_sched_200k/shared_rpemb4_qk_best.pt \
        checkpoints/shared_lf_qkv_h1_sched_200k/shared_lf_qkv_best.pt \
        checkpoints/shared_rpemb_qkv_v2_sched_200k/shared_rpemb_qkv_best.pt

echo "$(date): Done."

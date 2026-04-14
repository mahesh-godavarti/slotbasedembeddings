#!/bin/bash
# D=23 C=1024 h16 with K-schedule on GPU 1
# Waits for D=12 to finish, then launches.
# Uses flash attention via PYTHONPATH override (no file swapping).

set -e
cd /home/ubuntu/look_ahead6

# Wait for D=12 to finish
echo "$(date): Waiting for D=12 to finish..."
while pgrep -f 'd_block 12' > /dev/null 2>&1; do
    sleep 60
done
echo "$(date): D=12 done. Launching D=23 K-schedule on GPU 1."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/ubuntu/look_ahead6/flash_override:$PYTHONPATH
PYTHON=/home/ubuntu/exp8/venv/bin/python

$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 115 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 23 --n_head 16 \
    --k_schedule "0:1,40000:2,170000:2-5" \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d23_ksched \
    --gpu 1 \
    --amp 2>&1 | tee logs/corr_ffn_add_d23_c1024_h16_ksched_owt.log

#!/bin/bash
# D=6 C=2048 from scratch -- FLOP-matched to N=24 C=1024
# (12*6+8) * 2048^2 = 80 * 2048^2 at inference
# 72 * 2048^2 = 288 * 1024^2 for the blocks alone (matched to N=24)
# Same batch=32, block_size=256, 200K iters as N=24 C=1024
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

echo "$(date): Starting D=6 C=2048 from scratch"
mkdir -p checkpoints_d6_c2048_scratch
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2048 --n_layers 30 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 6 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_d6_c2048_scratch \
    --gpu $1 \
    --amp 2>&1 | tee logs/d6_c2048_scratch.log
echo "$(date): Finished D=6 C=2048 from scratch"

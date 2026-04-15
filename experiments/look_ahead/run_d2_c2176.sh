#!/bin/bash
# D=2 C=2176 from scratch -- FLOP-matched to N=12 C=1024
# 32 * 2176^2 = 151.5M ~ 144 * 1024^2 = 151.0M (0.3% over)
# Same batch=32, block_size=256, 200K iters as N=12 C=1024
# GPU 0
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

echo "$(date): Starting D=2 C=2176 from scratch (FLOP-matched to N=12 C=1024)"
mkdir -p checkpoints_d2_c2176
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2176 --n_layers 10 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 2 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_d2_c2176 \
    --gpu 0 \
    --amp 2>&1 | tee logs/d2_c2176_scratch.log
echo "$(date): Finished D=2 C=2176"

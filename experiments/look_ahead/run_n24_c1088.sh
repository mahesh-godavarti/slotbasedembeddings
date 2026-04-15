#!/bin/bash
# N=24 C=1088 -- FLOP-matched to D=6 C=2048
# 288 * 1088^2 = 341M vs 80 * 2048^2 = 336M
# Same batch=32, block_size=256, 200K iters
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

echo "$(date): Starting N=24 C=1088"
mkdir -p checkpoints_n24_c1088
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1088 --n_layers 24 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_n24_c1088 \
    --gpu 0 \
    --amp 2>&1 | tee logs/roformer_n24_c1088_h16_owt.log
echo "$(date): Finished N=24 C=1088"

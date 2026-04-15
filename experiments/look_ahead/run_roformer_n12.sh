#!/bin/bash
# Roformer N=12 C=1024 h16 baseline on GPU 1
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python

$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1024 --n_layers 12 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_n12 \
    --gpu 1 \
    --amp 2>&1 | tee logs/roformer_n12_c1024_h16_owt.log

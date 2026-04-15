#!/bin/bash
# N=1 C=1952 baseline -- to compare against D=1 C=1952
# Same tokens (1227M), same block_size=64
# batch=512, tokens/iter=32768, iters=37445
# GPU 1
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

echo "$(date): Starting N=1 C=1952"
mkdir -p checkpoints_n1_c1952
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1952 --n_layers 1 --block_size 64 --batch_size 512 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 37445 --eval_interval 500 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_n1_c1952 \
    --gpu 1 \
    --amp 2>&1 | tee logs/roformer_n1_c1952.log
echo "$(date): Finished N=1 C=1952"

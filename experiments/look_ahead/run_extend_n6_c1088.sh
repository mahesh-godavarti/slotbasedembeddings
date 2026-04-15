#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

# N=6 C=1088 resume from 100K to 200K
echo "$(date): Starting N=6 C=1088 extend to 200K"
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1088 --n_layers 6 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_n6_c1088 \
    --gpu $1 \
    --amp 2>&1 | tee logs/roformer_n6_c1088_ext200k.log
echo "$(date): Finished N=6 C=1088 extend"

#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

# GPU 0: D=1 C=2048 resume from 100K to 200K
echo "$(date): Starting D=1 C=2048 extend to 200K"
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2048 --n_layers 5 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_d1_c2048 \
    --gpu $1 \
    --amp 2>&1 | tee logs/d1_c2048_ext200k.log
echo "$(date): Finished D=1 C=2048 extend"

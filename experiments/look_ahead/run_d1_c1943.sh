#!/bin/bash
# D=1 C=1952 from scratch -- FLOP-matched to N=6 C=1024
# D=1 inference: 20 * 1952^2 = 72 * 1024^2 = N=6 C=1024
# 1227M tokens to match N=6 C=1024 scaling experiment
# batch=256, block_size=64, tokens/iter=16384, iters=74890
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

echo "$(date): Starting D=1 C=1952 from scratch (FLOP-matched to N=6 C=1024)"
mkdir -p checkpoints_d1_c1952
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1952 --n_layers 5 --block_size 64 --batch_size 256 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 74890 --eval_interval 1000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_d1_c1952 \
    --gpu 1 \
    --amp 2>&1 | tee logs/d1_c1952_scratch.log
echo "$(date): Finished D=1 C=1952"

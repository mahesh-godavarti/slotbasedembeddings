#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=0

# N=2 C=1888 at block_size=64
echo "$(date): === N=2 C=1888 block_size=64 ==="
mkdir -p checkpoints_n2_c1888_bs64 logs
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1888 --n_layers 2 --block_size 64 --batch_size 64 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_n2_c1888_bs64 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/roformer_n2_c1888_bs64.log
echo "$(date): Finished N=2 C=1888 bs64"

# D=1 C=2048 at block_size=64
echo "$(date): === D=1 C=2048 block_size=64 ==="
mkdir -p checkpoints_d1_c2048_bs64 logs
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2048 --n_layers 5 --block_size 64 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_d1_c2048_bs64 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/d1_c2048_bs64.log
echo "$(date): Finished D=1 C=2048 bs64"

echo "$(date): Block size test complete."

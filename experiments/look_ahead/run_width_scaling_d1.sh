#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=0
BS=64
LR=2e-4
EVAL=500

# D=1 C=280 FLOP-matched to N=2 C=256 (500M tokens, n_head=4)
echo "$(date): === D=1 C=280 ==="
mkdir -p checkpoints_width_d1_c280_scratch
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 280 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 4 --k_min 2 \
    --max_iters 7629 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c280_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c280_scratch.log
echo "$(date): Finished D=1 C=280"

# D=1 C=560 FLOP-matched to N=2 C=512 (900M tokens, n_head=8)
echo "$(date): === D=1 C=560 ==="
mkdir -p checkpoints_width_d1_c560_scratch
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 560 --n_layers 5 --block_size $BS --batch_size 512 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 8 --k_min 2 \
    --max_iters 27466 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c560_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c560_scratch.log
echo "$(date): Finished D=1 C=560"

# D=1 C=1120 FLOP-matched to N=2 C=1024 (1227M tokens, n_head=16)
echo "$(date): === D=1 C=1120 ==="
mkdir -p checkpoints_width_d1_c1120_scratch
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1120 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 18723 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c1120_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c1120_scratch.log
echo "$(date): Finished D=1 C=1120"

echo "$(date): All D=1 width scaling experiments complete."

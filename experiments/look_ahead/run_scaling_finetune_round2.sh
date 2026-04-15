#!/bin/bash
# Fine-tune round 2: Two D experiments per N, each gets 409M tokens (200K × 32 equiv)
#
# 1. "fresh": Convert latest roformer (cont2) to D, fine-tune from scratch
# 2. "cont": Continue existing D (cont2) for another cycle
#
# This tests: is converting a more-trained roformer better than continuing an earlier fine-tune?
# All on GPU 0.
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=0
BS=64
LR=2e-4
EVAL=500

run_fresh_finetune() {
    local D=$1 BATCH=$2 ITERS=$3 NHEAD=$4
    local K=5
    local NLAYERS=$((D * K))
    local ROFORMER_CKPT=checkpoints_scaling_n${D}_cont2/roformer_latest.pt
    local CKPT=checkpoints_scaling_d${D}_fresh
    local LOG=logs/scaling_finetune_d${D}_c1024_bs${BS}_fresh.log
    mkdir -p $CKPT

    echo "$(date): Converting latest N=$D (cont2) to D=$D fresh"
    $PYTHON convert_roformer_to_lookahead.py \
        --roformer_ckpt $ROFORMER_CKPT \
        --output_ckpt ${CKPT}/block_head_corr_ffn_add_latest.pt \
        --n_embed 1024 --n_layers $NLAYERS --d_block $D --n_head $NHEAD \
        --block_size $BS --vocab_size 32000

    echo "$(date): Starting D=$D fresh fine-tune batch=$BATCH iters=$ITERS"
    $PYTHON train_wiki_streaming.py train \
        --models block_head_corr_ffn_add --n_embed 1024 --n_layers $NLAYERS --block_size $BS --batch_size $BATCH \
        --lr $LR --softmax --convergence_weight 0.1 --d_block $D --n_head $NHEAD --k_min 2 \
        --max_iters $ITERS --eval_interval $EVAL \
        --data_dir $DATA \
        --checkpoint_dir $CKPT \
        --gpu $GPU \
        --amp 2>&1 | tee $LOG
    echo "$(date): Finished D=$D fresh fine-tune"
}

run_cont_finetune() {
    local D=$1 BATCH=$2 EXTRA_ITERS=$3 NHEAD=$4
    local K=5
    local NLAYERS=$((D * K))
    local SRC_CKPT=checkpoints_scaling_d${D}_cont2
    local CKPT=checkpoints_scaling_d${D}_cont3
    local LOG=logs/scaling_finetune_d${D}_c1024_bs${BS}_cont3.log
    mkdir -p $CKPT

    # Copy checkpoint to preserve cont2
    cp ${SRC_CKPT}/block_head_corr_ffn_add_latest.pt ${CKPT}/block_head_corr_ffn_add_latest.pt

    local ORIG_ITER=$($PYTHON -c "import torch; c=torch.load('${CKPT}/block_head_corr_ffn_add_latest.pt', map_location='cpu', weights_only=False); print(c['iter'])")
    local NEW_MAX=$((ORIG_ITER + EXTRA_ITERS + 1))

    echo "$(date): Continuing D=$D from iter $ORIG_ITER for $EXTRA_ITERS more iters (max=$NEW_MAX)"
    $PYTHON train_wiki_streaming.py train \
        --models block_head_corr_ffn_add --n_embed 1024 --n_layers $NLAYERS --block_size $BS --batch_size $BATCH \
        --lr $LR --softmax --convergence_weight 0.1 --d_block $D --n_head $NHEAD --k_min 2 \
        --max_iters $NEW_MAX --eval_interval $EVAL \
        --data_dir $DATA \
        --checkpoint_dir $CKPT \
        --gpu $GPU \
        --amp 2>&1 | tee $LOG
    echo "$(date): Finished D=$D cont3 fine-tune"
}

echo "============================================================"
echo "Fine-tune round 2: fresh conversion vs continued D"
echo "Each D gets 409M tokens (200K × 32 equiv)"
echo "============================================================"

# D=1 (n_head=1): 409M tokens = 12500 iters at batch=512
run_fresh_finetune 1 512 12500 1
run_cont_finetune 1 512 12500 1

# D=2: 409M tokens = 25000 iters at batch=256
run_fresh_finetune 2 256 25000 16
run_cont_finetune 2 256 25000 16

# D=3: 409M tokens = 50000 iters at batch=128
run_fresh_finetune 3 128 50000 16
run_cont_finetune 3 128 50000 16

# D=6: 409M tokens = 50000 iters at batch=128
run_fresh_finetune 6 128 50000 16
run_cont_finetune 6 128 50000 16

echo "$(date): All round 2 fine-tunes complete."

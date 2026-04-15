#!/bin/bash
# Third cycle: continue all models from *_cont checkpoints for another full token budget.
# 409M more tokens for roformers, 102M more for fine-tunes.
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

continue_roformer() {
    local N=$1 BATCH=$2 EXTRA_ITERS=$3 NHEAD=$4
    local SRC_CKPT=checkpoints_scaling_n${N}_cont
    local CKPT=checkpoints_scaling_n${N}_cont2
    local LOG=logs/scaling_roformer_n${N}_c1024_bs${BS}_cont2.log

    mkdir -p $CKPT
    cp ${SRC_CKPT}/roformer_latest.pt ${CKPT}/roformer_latest.pt

    local ORIG_ITER=$($PYTHON -c "import torch; c=torch.load('${CKPT}/roformer_latest.pt', map_location='cpu', weights_only=False); print(c['iter'])")
    local NEW_MAX=$((ORIG_ITER + EXTRA_ITERS + 1))

    echo "$(date): Continuing roformer N=$N from iter $ORIG_ITER for $EXTRA_ITERS more iters (max=$NEW_MAX)"
    $PYTHON train_wiki_streaming.py train \
        --models roformer --n_embed 1024 --n_layers $N --block_size $BS --batch_size $BATCH \
        --lr $LR --softmax --n_head $NHEAD \
        --max_iters $NEW_MAX --eval_interval $EVAL \
        --data_dir $DATA \
        --checkpoint_dir $CKPT \
        --gpu $GPU \
        --amp 2>&1 | tee $LOG
    echo "$(date): Finished roformer N=$N continuation2"
}

continue_finetune() {
    local D=$1 BATCH=$2 EXTRA_ITERS=$3 NHEAD=$4
    local K=5
    local NLAYERS=$((D * K))
    local SRC_CKPT=checkpoints_scaling_d${D}_cont
    local CKPT=checkpoints_scaling_d${D}_cont2
    local LOG=logs/scaling_finetune_d${D}_c1024_bs${BS}_cont2.log

    mkdir -p $CKPT
    cp ${SRC_CKPT}/block_head_corr_ffn_add_latest.pt ${CKPT}/block_head_corr_ffn_add_latest.pt

    local ORIG_ITER=$($PYTHON -c "import torch; c=torch.load('${CKPT}/block_head_corr_ffn_add_latest.pt', map_location='cpu', weights_only=False); print(c['iter'])")
    local NEW_MAX=$((ORIG_ITER + EXTRA_ITERS + 1))

    echo "$(date): Continuing D=$D fine-tune from iter $ORIG_ITER for $EXTRA_ITERS more iters (max=$NEW_MAX)"
    $PYTHON train_wiki_streaming.py train \
        --models block_head_corr_ffn_add --n_embed 1024 --n_layers $NLAYERS --block_size $BS --batch_size $BATCH \
        --lr $LR --softmax --convergence_weight 0.1 --d_block $D --n_head $NHEAD --k_min 2 \
        --max_iters $NEW_MAX --eval_interval $EVAL \
        --data_dir $DATA \
        --checkpoint_dir $CKPT \
        --gpu $GPU \
        --amp 2>&1 | tee $LOG
    echo "$(date): Finished D=$D fine-tune continuation2"
}

echo "============================================================"
echo "Scaling continuation cycle 3: another full token budget"
echo "============================================================"

# N=1 / D=1 (n_head=1, 409M/102M tokens)
continue_roformer 1 2048 3125 1
continue_finetune 1 512 3125 1

# N=2 / D=2 (409M/102M tokens)
continue_roformer 2 1024 6250 16
continue_finetune 2 256 6250 16

# N=3 / D=3 (409M/102M tokens)
continue_roformer 3 512 12500 16
continue_finetune 3 128 12500 16

# N=6 / D=6 (409M/102M tokens)
continue_roformer 6 256 25000 16
continue_finetune 6 128 12500 16

echo "$(date): All continuation2 complete."

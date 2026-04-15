#!/bin/bash
# Resume scaling experiment from N=2 (N=1/D=1 already done)
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=0
BS=64
LR=2e-4

run_roformer() {
    local N=$1 BATCH=$2 ITERS=$3 NHEAD=$4
    local EVAL=$((ITERS / 20))
    local CKPT=checkpoints_scaling_n${N}
    local LOG=logs/scaling_roformer_n${N}_c1024_bs${BS}.log
    mkdir -p $CKPT
    echo "$(date): Starting roformer N=$N batch=$BATCH iters=$ITERS nhead=$NHEAD"
    $PYTHON train_wiki_streaming.py train \
        --models roformer --n_embed 1024 --n_layers $N --block_size $BS --batch_size $BATCH \
        --lr $LR --softmax --n_head $NHEAD \
        --max_iters $ITERS --eval_interval $EVAL \
        --data_dir $DATA \
        --checkpoint_dir $CKPT \
        --gpu $GPU \
        --amp 2>&1 | tee $LOG
    echo "$(date): Finished roformer N=$N"
}

run_finetune() {
    local D=$1 BATCH=$2 ITERS=$3 NHEAD=$4
    local K=5
    local NLAYERS=$((D * K))
    local EVAL=$((ITERS / 20))
    local ROFORMER_CKPT=checkpoints_scaling_n${D}/roformer_latest.pt
    local CONVERTED_CKPT=checkpoints_scaling_d${D}/block_head_corr_ffn_add_latest.pt
    local LOG=logs/scaling_finetune_d${D}_c1024_bs${BS}.log
    mkdir -p checkpoints_scaling_d${D}

    echo "$(date): Converting N=$D to D=$D"
    $PYTHON convert_roformer_to_lookahead.py \
        --roformer_ckpt $ROFORMER_CKPT \
        --output_ckpt $CONVERTED_CKPT \
        --n_embed 1024 --n_layers $NLAYERS --d_block $D --n_head $NHEAD \
        --block_size $BS --vocab_size 32000

    echo "$(date): Starting D=$D fine-tune batch=$BATCH iters=$ITERS nhead=$NHEAD"
    $PYTHON train_wiki_streaming.py train \
        --models block_head_corr_ffn_add --n_embed 1024 --n_layers $NLAYERS --block_size $BS --batch_size $BATCH \
        --lr $LR --softmax --convergence_weight 0.1 --d_block $D --n_head $NHEAD --k_min 2 \
        --max_iters $ITERS --eval_interval $EVAL \
        --data_dir $DATA \
        --checkpoint_dir checkpoints_scaling_d${D} \
        --gpu $GPU \
        --amp 2>&1 | tee $LOG
    echo "$(date): Finished D=$D fine-tune"
}

echo "============================================================"
echo "Scaling experiment resume: N=2,3,6 (N=1/D=1 already done)"
echo "============================================================"

# N=2 / D=2
run_roformer 2 1024 6250 16
run_finetune 2 256 6250 16

# N=3 / D=3
run_roformer 3 512 12500 16
run_finetune 3 128 12500 16

# N=6 / D=6
run_roformer 6 256 25000 16
run_finetune 6 128 12500 16

echo "$(date): All training complete."

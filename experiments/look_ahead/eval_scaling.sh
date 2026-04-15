#!/bin/bash
# Evaluate all scaling experiment checkpoints at batch=32 for fair comparison.
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=0
BS=64          # block_size
BATCH=32       # eval batch size — same for all

echo "============================================================"
echo "Scaling eval: all models at batch=$BATCH, block_size=$BS"
echo "============================================================"

# Roformers
for N in 1 2 3 6; do
    CKPT=checkpoints_scaling_n${N}/roformer_latest.pt
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Roformer N=$N ==="
        $PYTHON train_wiki_streaming.py train \
            --models roformer --n_embed 1024 --n_layers $N --block_size $BS --batch_size $BATCH \
            --lr $LR --softmax --n_head 16 \
            --max_iters 1 --eval_interval 1 \
            --data_dir $DATA \
            --checkpoint_dir checkpoints_scaling_n${N} \
            --gpu $GPU \
            --amp 2>&1 | grep -E 'val_ppl|PPL|Resumed'
    fi
done

# Fine-tuned look-ahead models
for D in 1 2 3 6; do
    K=5
    NLAYERS=$((D * K))
    CKPT=checkpoints_scaling_d${D}/block_head_corr_ffn_add_latest.pt
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== D=$D fine-tuned ==="
        $PYTHON train_wiki_streaming.py train \
            --models block_head_corr_ffn_add --n_embed 1024 --n_layers $NLAYERS --block_size $BS --batch_size $BATCH \
            --lr $LR --softmax --convergence_weight 0.1 --d_block $D --n_head 16 --k_min 2 \
            --max_iters 1 --eval_interval 1 \
            --data_dir $DATA \
            --checkpoint_dir checkpoints_scaling_d${D} \
            --gpu $GPU \
            --amp 2>&1 | grep -E 'val_ppl|PPL|Resumed'
    fi
done

echo ""
echo "$(date): Eval complete"

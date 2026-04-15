#!/bin/bash
# Extend all width scaling runs to 37,500 iters (2x the C=1024 budget)
# N=2 C=256: was 7,629 → 37,500
# N=2 C=512: was 27,466 → 37,500
# N=2 C=1024: was 18,750 → 37,500
# D=1 C=280: was 7,629 → 37,500
# D=1 C=560: was 27,466 → 37,500
# D=1 C=1120: was 18,723 → 37,500
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=$1
BS=64
LR=2e-4
EVAL=500

# N=2 C=256 (resume from 7,629)
echo "$(date): === N=2 C=256 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 256 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 4 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_n2_c256 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c256_ext.log
echo "$(date): Finished N=2 C=256 extend"

# N=2 C=512 (resume from 27,466)
echo "$(date): === N=2 C=512 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 512 --n_layers 2 --block_size $BS --batch_size 512 \
    --lr $LR --softmax --n_head 8 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_n2_c512 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c512_ext.log
echo "$(date): Finished N=2 C=512 extend"

# N=2 C=1024 (resume from 18,750)
echo "$(date): === N=2 C=1024 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1024 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 16 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_scaling_n2_cont2 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c1024_ext.log
echo "$(date): Finished N=2 C=1024 extend"

# D=1 C=280 (resume from 7,629)
echo "$(date): === D=1 C=280 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 280 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 4 --k_min 2 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c280_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c280_ext.log
echo "$(date): Finished D=1 C=280 extend"

# D=1 C=560 (resume from 27,466)
echo "$(date): === D=1 C=560 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 560 --n_layers 5 --block_size $BS --batch_size 512 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 8 --k_min 2 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c560_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c560_ext.log
echo "$(date): Finished D=1 C=560 extend"

# D=1 C=1120 (resume from 18,723)
echo "$(date): === D=1 C=1120 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1120 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c1120_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c1120_ext.log
echo "$(date): Finished D=1 C=1120 extend"

echo "$(date): All width scaling extensions complete."

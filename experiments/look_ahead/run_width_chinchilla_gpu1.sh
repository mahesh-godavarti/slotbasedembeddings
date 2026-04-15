#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=1
BS=64
LR=2e-4
EVAL=500

# D=1 C=560 (resume from 37,500 → 82,500)
echo "$(date): === D=1 C=560 extend to 82,500 ==="
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 560 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 8 --k_min 2 \
    --max_iters 82500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c560_b1024 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c560_chinchilla.log
echo "$(date): Finished D=1 C=560"

# N=2 C=1024 (resume from 37,500 → 189,750)
echo "$(date): === N=2 C=1024 extend to 189,750 ==="
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1024 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 16 \
    --max_iters 189750 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_scaling_n2_cont2 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c1024_chinchilla.log
echo "$(date): Finished N=2 C=1024"

echo "$(date): GPU 1 Chinchilla extensions complete."

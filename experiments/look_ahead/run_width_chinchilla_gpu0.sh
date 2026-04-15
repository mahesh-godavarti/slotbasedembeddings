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

# N=2 C=512 (resume from 37,500 → 82,500)
echo "$(date): === N=2 C=512 extend to 82,500 ==="
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 512 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 8 \
    --max_iters 82500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_n2_c512_b1024 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c512_chinchilla.log
echo "$(date): Finished N=2 C=512"

# D=1 C=1120 (resume from 37,500 → 189,750)
echo "$(date): === D=1 C=1120 extend to 189,750 ==="
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1120 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 189750 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c1120_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c1120_chinchilla.log
echo "$(date): Finished D=1 C=1120"

echo "$(date): GPU 0 Chinchilla extensions complete."

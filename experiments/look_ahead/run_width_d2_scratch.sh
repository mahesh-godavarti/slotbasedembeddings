#!/bin/bash
# D=2 from scratch at C=256 and C=512
# Same token budget as the roformers (Chinchilla 20:1)
# C=256: 500M tokens, C=512: 900M tokens
# GPU 1
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=1
BS=64
LR=2e-4
EVAL=500

# D=2 C=256: 500M tokens, batch=512, tokens/iter=32768, iters=15259
echo "$(date): === D=2 C=256 from scratch ==="
mkdir -p checkpoints_width_d2_c256_scratch
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 256 --n_layers 10 --block_size $BS --batch_size 512 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 2 --n_head 4 --k_min 2 \
    --max_iters 15259 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d2_c256_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d2_c256_scratch.log
echo "$(date): Finished D=2 C=256 from scratch"

# D=2 C=512: 900M tokens, batch=256, tokens/iter=16384, iters=54932
echo "$(date): === D=2 C=512 from scratch ==="
mkdir -p checkpoints_width_d2_c512_scratch
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 512 --n_layers 10 --block_size $BS --batch_size 256 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 2 --n_head 8 --k_min 2 \
    --max_iters 54932 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d2_c512_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d2_c512_scratch.log
echo "$(date): Finished D=2 C=512 from scratch"

echo "$(date): All D=2 from scratch experiments complete."

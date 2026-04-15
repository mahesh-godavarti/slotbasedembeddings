#!/bin/bash
# D=2 C=1536 from scratch -- FLOP-matched to N=6 C=1024
# 32 * 1536^2 = 72 * 1024^2 (exact match)
# Same tokens (1227M), same batch=256, same block_size=64 as N=6 C=1024
# tokens/iter=16384, iters=74890
# GPU 0
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

echo "$(date): Starting D=2 C=1536 from scratch (FLOP-matched to N=6 C=1024)"
mkdir -p checkpoints_d2_c1536
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1536 --n_layers 10 --block_size 64 --batch_size 256 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 2 --n_head 16 --k_min 2 \
    --max_iters 74890 --eval_interval 1000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_d2_c1536 \
    --gpu 0 \
    --amp 2>&1 | tee logs/d2_c1536_scratch.log
echo "$(date): Finished D=2 C=1536"

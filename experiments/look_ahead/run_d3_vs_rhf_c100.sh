#!/bin/bash
# A10G #2: D=3 concat v2 C=96 FLOP-matched vs roformer_head_ffn N=3 C=100
# 48 × 96² = 442,368 vs 44 × 100² = 440,000

LOGDIR=/home/ubuntu/look_ahead5/logs
mkdir -p $LOGDIR

echo "=== D=3 concat v2 C=96 (FLOP-matched vs roformer_head_ffn N=3 C=100) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models block_head_corr_ffn_concat \
    --n_embed 96 --n_layers 15 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp --d_block 3 --k_min 2 \
    2>&1 | tee $LOGDIR/d3_concat_v2_c96.log

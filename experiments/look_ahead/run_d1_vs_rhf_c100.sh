#!/bin/bash
# A10G #3: D=1 concat v2 C=136 FLOP-matched vs roformer_head_ffn N=3 C=100
# 24 × 136² = 443,904 vs 44 × 100² = 440,000

LOGDIR=/home/ubuntu/look_ahead5/logs
mkdir -p $LOGDIR

echo "=== D=1 concat v2 C=136 (FLOP-matched vs roformer_head_ffn N=3 C=100) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models block_head_corr_ffn_concat \
    --n_embed 136 --n_layers 5 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp --k_min 2 \
    2>&1 | tee $LOGDIR/d1_concat_v2_c136.log

#!/bin/bash
# roformer_head_ffn N=3 C=100 baseline
# 44 × 100² = 440,000 FLOPs

LOGDIR=/home/ubuntu/look_ahead5/logs
mkdir -p $LOGDIR

echo "=== roformer_head_ffn N=3 C=100 (baseline) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models roformer_head_ffn \
    --n_embed 100 --n_layers 3 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp \
    2>&1 | tee $LOGDIR/roformer_head_ffn_n3_c100.log

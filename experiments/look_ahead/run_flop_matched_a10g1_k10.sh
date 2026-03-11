#!/bin/bash
# A10G #1: D=1 concat v2 FLOP-matched, K=10 (k_min=2), eval sweeps K=1..10
# Experiment 1: C=62 (vs roformer N=3 C=50, 36C²=90K FLOPs)
# Experiment 2: C=68 (vs roformer_head_ffn N=3 C=50, 44C²=110K FLOPs)

LOGDIR=/home/ubuntu/look_ahead5/logs
mkdir -p $LOGDIR

echo "=== D=1 concat v2 C=62 K=10 (FLOP-matched vs roformer N=3 C=50) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models block_head_corr_ffn_concat \
    --n_embed 62 --n_layers 10 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp --k_min 2 \
    2>&1 | tee $LOGDIR/d1_concat_v2_c62_k10.log

echo "=== D=1 concat v2 C=68 K=10 (FLOP-matched vs roformer_head_ffn N=3 C=50) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models block_head_corr_ffn_concat \
    --n_embed 68 --n_layers 10 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp --k_min 2 \
    2>&1 | tee $LOGDIR/d1_concat_v2_c68_k10.log

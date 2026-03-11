#!/bin/bash
# Param-matched: roformer_head_ffn N=6 C=100 vs D=1 concat v2 C=115
# roformer_head_ffn N=6 C=100: 4,024,700 params, 800,000 FLOPs
# D=1 concat v2 C=115:         4,015,930 params, 317,400 FLOPs

LOGDIR=/home/ubuntu/look_ahead5/logs
mkdir -p $LOGDIR

echo "=== roformer_head_ffn N=6 C=100 ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models roformer_head_ffn \
    --n_embed 100 --n_layers 6 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp \
    2>&1 | tee $LOGDIR/roformer_head_ffn_n6_c100.log

echo "=== D=1 concat v2 C=115 (param-matched vs N=6) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models block_head_corr_ffn_concat \
    --n_embed 115 --n_layers 5 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp --k_min 2 \
    2>&1 | tee $LOGDIR/d1_concat_v2_c115.log

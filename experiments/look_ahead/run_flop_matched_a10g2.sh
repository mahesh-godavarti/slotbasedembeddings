#!/bin/bash
# A10G #2: D=3 concat v2 FLOP-matched vs both baselines
# Experiment 1: C=43->44 (vs roformer N=3 C=50, 36C²=90K FLOPs)
# Experiment 2: C=48 (vs roformer_head_ffn N=3 C=50, 44C²=110K FLOPs)

LOGDIR=/home/ubuntu/look_ahead5/logs
mkdir -p $LOGDIR

echo "=== D=3 concat v2 C=44 (FLOP-matched vs roformer N=3 C=50) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models block_head_corr_ffn_concat \
    --n_embed 44 --n_layers 15 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp --d_block 3 --k_min 2 \
    2>&1 | tee $LOGDIR/d3_concat_v2_c44.log

echo "=== D=3 concat v2 C=48 (FLOP-matched vs roformer_head_ffn N=3 C=50) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models block_head_corr_ffn_concat \
    --n_embed 48 --n_layers 15 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp --d_block 3 --k_min 2 \
    2>&1 | tee $LOGDIR/d3_concat_v2_c48.log

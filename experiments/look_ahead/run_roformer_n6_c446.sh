#!/bin/bash
# roformer_head_ffn N=6 C=446 — how much does doubling depth buy?
# N=3 C=446 baseline: 25.78 PPL (from big machine)

LOGDIR=/home/ubuntu/look_ahead5/logs
mkdir -p $LOGDIR

echo "=== roformer_head_ffn N=6 C=446 ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead5/train_wiki_streaming.py train \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
    --models roformer_head_ffn \
    --n_embed 446 --n_layers 6 --block_size 256 --batch_size 64 \
    --max_iters 100000 --eval_interval 5000 \
    --softmax --lr 0.0002 --amp \
    2>&1 | tee $LOGDIR/roformer_head_ffn_n6_c446.log

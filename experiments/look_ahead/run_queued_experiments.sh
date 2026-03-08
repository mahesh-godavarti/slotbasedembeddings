#!/bin/bash
# Sequential experiment runner — remaining queued runs.
# Launch with: nohup bash run_queued_experiments.sh > queued_runs.log 2>&1 &

set -e
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead

COMMON="--data_dir look_ahead/data_full --block_size 64 --batch_size 64 --lr 2e-4 --max_iters 10000 --eval_interval 500 --seed 42 --softmax --generate_len 200"

echo "========================================"
echo "Run 1: joformer_learned_look_ahead_nocat (causal angle fix)"
echo "Started: $(date)"
echo "========================================"
python train_wiki_streaming.py train \
    --models joformer_learned_look_ahead_nocat \
    --n_embed 768 --n_layers 10 \
    $COMMON

echo "========================================"
echo "Run 2: roformer_look_ahead_nocat C=1786 (param-matched to roformer C=768 N=10)"
echo "Started: $(date)"
echo "========================================"
python train_wiki_streaming.py train \
    --models roformer_look_ahead_nocat \
    --n_embed 1786 --n_layers 10 \
    $COMMON

echo "========================================"
echo "All runs complete: $(date)"
echo "========================================"

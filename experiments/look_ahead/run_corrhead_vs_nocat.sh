#!/bin/bash
# Head comparison: correction[t] vs processed_x[t]
# Same params (5.89M), same block, K=10, C=300, vocab=8000
# correction_head sees x[t] (self-inclusive), nocat sees past-only context

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

DATA_DIR="look_ahead/data_v8k"
COMMON="--n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 10000 --softmax --convergence_weight 0.1 --data_dir $DATA_DIR --eval_interval 5000"

echo "=========================================="
echo "Experiment 1: nocat (head = processed_x[t], past-only)"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_look_ahead_nocat \
  --n_layers 10 \
  $COMMON

echo "=========================================="
echo "Experiment 2: corrhead (head = correction[t], self-inclusive)"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_look_ahead_corrhead \
  --n_layers 10 \
  $COMMON

echo "=========================================="
echo "All experiments complete."
echo "=========================================="

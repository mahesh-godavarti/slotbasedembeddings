#!/bin/bash
# Sequential experiments: baseline → look-ahead → stacked look-ahead
# All use: C=50, block_size=64, batch_size=64, lr=2e-4, 100K iters, softmax, conv_weight=0.1
# Data: look_ahead/data_full (983M tokens, BPE vocab 16000)

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

COMMON="--n_embed 50 --block_size 64 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full"

echo "=========================================="
echo "Experiment 1: Roformer N=1 baseline"
echo "=========================================="
python /home/ubuntu/look_ahead2/train_wiki_streaming.py train \
  --models roformer \
  --n_layers 1 \
  --eval_interval 5000 \
  $COMMON

echo "=========================================="
echo "Experiment 2: D=1, K=10 look-ahead (roformer_look_ahead_nocat)"
echo "=========================================="
python /home/ubuntu/look_ahead2/train_wiki_streaming.py train \
  --models roformer_look_ahead_nocat \
  --n_layers 10 \
  --eval_interval 5000 \
  $COMMON

echo "=========================================="
echo "Experiment 3: Stacked N=10, K=10 (roformer_stacked_look_ahead_nocat)"
echo "=========================================="
python /home/ubuntu/look_ahead2/train_wiki_streaming.py train \
  --models roformer_stacked_look_ahead_nocat \
  --n_layers 100 --n_units 10 \
  --eval_interval 5000 \
  $COMMON

echo "=========================================="
echo "All experiments complete."
echo "=========================================="

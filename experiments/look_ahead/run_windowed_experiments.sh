#!/bin/bash
# Fair comparison: stacked look-ahead (corrhead) vs roformer baseline
# SAME params (1,769,350), block_size=256, window_size=64, n_embed=50
# corrhead: head uses correction[t] (self-inclusive, size C) — no param advantage
# Data: look_ahead/data_full (983M tokens, BPE vocab 16000)

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

COMMON="--n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full --eval_interval 5000 --window_size 64"

echo "=========================================="
echo "Experiment 1: Roformer N=5 baseline (windowed, w=64) — 1,769,350 params"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_windowed \
  --n_layers 5 \
  $COMMON

echo "=========================================="
echo "Experiment 2: Stacked N=5, K=10 corrhead (windowed, w=64) — 1,769,350 params"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_stacked_look_ahead_corrhead_windowed \
  --n_layers 50 --n_units 5 \
  $COMMON

echo "=========================================="
echo "All experiments complete."
echo "=========================================="

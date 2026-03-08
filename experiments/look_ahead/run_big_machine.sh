#!/bin/bash
# Big machine experiment: stacked projhead vs roformer at C=300
# Stacked N=5 K=10 projhead: ~15.2M params (+1.2% vs roformer)
# Roformer N=5:              ~15.0M params
# Nearly param-matched, fair comparison.
#
# Requires: preprocessed data in look_ahead/data_full (vocab=16000)

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

COMMON="--n_embed 300 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full --eval_interval 5000"

echo "=========================================="
echo "Experiment 1: Roformer N=5 baseline"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer \
  --n_layers 5 \
  $COMMON

echo "=========================================="
echo "Experiment 2: Stacked N=5 K=10 projhead"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_stacked_look_ahead_projhead \
  --n_layers 50 --n_units 5 \
  $COMMON

echo "=========================================="
echo "All experiments complete."
echo "=========================================="

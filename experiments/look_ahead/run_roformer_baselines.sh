#!/bin/bash
# Roformer N=1 and N=2 baselines at block_size=256
# These are the fair baselines for D=1 K=10 look-ahead heads

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer \
  --n_layers 1 \
  --n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full --eval_interval 5000

python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer \
  --n_layers 2 \
  --n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full --eval_interval 5000

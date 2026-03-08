#!/bin/bash
# Concat vs projhead: 100K iters, C=50, K=10, block_size=256, vocab=16000
# concat: 2,446,850 params (2C→vocab head)
# projhead: 1,651,800 params (Linear 2C→C then C→vocab head)

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_look_ahead_projhead roformer_look_ahead \
  --n_layers 10 \
  --n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full --eval_interval 5000

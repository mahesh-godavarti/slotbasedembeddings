#!/bin/bash
# nocat 100K to complete the head comparison set
# Same settings as projhead/concat/corrhead runs

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_look_ahead_nocat \
  --n_layers 10 \
  --n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full --eval_interval 5000

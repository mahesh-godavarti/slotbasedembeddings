#!/bin/bash
# Machine 2 (remote): Variant 3 (block_head_ffn) and nocat baseline
# block_head_ffn = standard block + extra FFN at head
# roformer_look_ahead_nocat = the baseline to beat (processed_x head)

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

python /home/ubuntu/look_ahead4/train_wiki_streaming.py train \
  --models block_head_ffn roformer_look_ahead_nocat \
  --n_layers 10 \
  --n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 \
  --max_iters 100000 --softmax --convergence_weight 0.1 \
  --data_dir look_ahead/data_full --eval_interval 5000

#!/bin/bash
# Machine 1 (local): Variant 1 (attn_corr_ffn) and Variant 2 (attn_head_ffn)
# Compare split-block attention-only variants against nocat baseline

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

python /home/ubuntu/look_ahead4/train_wiki_streaming.py train \
  --models attn_corr_ffn attn_head_ffn \
  --n_layers 10 \
  --n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 \
  --max_iters 100000 --softmax --convergence_weight 0.1 \
  --data_dir look_ahead/data_full --eval_interval 5000

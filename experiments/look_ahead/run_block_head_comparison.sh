#!/bin/bash
# Run block_head D=3 C=446, then block_head_corr_ffn D=3 C=446
# Both with AMP, output to separate log files

set -e

cd /home/ubuntu/look_ahead

echo "=== Run 1: block_head D=3 C=446 K=5 ==="
/home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models block_head \
  --n_embed 446 --n_layers 15 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 3 \
  --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/block_head_d3_c446.log

echo "=== Run 2: block_head_corr_ffn D=3 C=446 K=5 ==="
/home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models block_head_corr_ffn \
  --n_embed 446 --n_layers 15 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 3 \
  --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/block_head_corr_ffn_d3_c446.log

echo "=== All runs complete ==="

#!/bin/bash
set -e

echo "=== Run 1: stacked_block_head_corr_ffn N=3 C=446 K=5 ==="
cd /home/ubuntu/look_ahead && /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models stacked_block_head_corr_ffn \
  --n_embed 446 --n_layers 15 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --n_units 3 \
  --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/stacked_corr_ffn_n3_c446_k5_kmin2.log

echo "=== Run 2: stacked_block_head_corr_ffn_add N=3 C=446 K=5 ==="
cd /home/ubuntu/look_ahead && /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models stacked_block_head_corr_ffn_add \
  --n_embed 446 --n_layers 15 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --n_units 3 \
  --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/stacked_corr_ffn_add_n3_c446_k5.log

echo "=== All runs complete ==="

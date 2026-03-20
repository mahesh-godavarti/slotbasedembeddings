#!/bin/bash
set -e

echo "=== Run 1: block_head_corr_ffn_add D=4 C=446 K=5 200K ==="
cd /home/ubuntu/look_ahead && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models block_head_corr_ffn_add --n_embed 446 --n_layers 20 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 4 \
  --max_iters 200000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_add_d4_c446_200k.log

echo "=== Run 2: roformer N=6 C=446 200K ==="
cd /home/ubuntu/look_ahead && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models roformer --n_embed 446 --n_layers 6 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax \
  --max_iters 200000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/roformer_n6_c446_200k.log

echo "=== All runs complete ==="

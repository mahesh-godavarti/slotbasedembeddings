#!/bin/bash
set -e

echo "=== Run 1: roformer N=6 C=768 OWT ==="
cd /home/ubuntu/look_ahead6 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models roformer --n_embed 768 --n_layers 6 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/roformer_n6_c768_owt.log

echo "=== Run 2: block_head_corr_ffn_add D=3 C=800 OWT ==="
cd /home/ubuntu/look_ahead6 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models block_head_corr_ffn_add --n_embed 800 --n_layers 15 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 3 \
  --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_add_d3_c800_owt.log

echo "=== All runs complete ==="

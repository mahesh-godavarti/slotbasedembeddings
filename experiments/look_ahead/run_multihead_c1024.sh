#!/bin/bash
set -e

# Experiment: corr_ffn_add D=12 C=1024 n_head=16 OWT on GPU 1
# batch_size=32 (D=12 OOMs at batch=64 with 16 heads)
# 200K iters to match total tokens with batch=32
echo "=== block_head_corr_ffn_add D=12 C=1024 n_head=16 OWT ==="
cd /home/ubuntu/look_ahead6 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models block_head_corr_ffn_add --n_embed 1024 --n_layers 60 --block_size 256 --batch_size 32 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 12 --n_head 16 \
  --max_iters 200000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
  --checkpoint_dir /home/ubuntu/look_ahead6/checkpoints \
  --gpu 1 \
  --amp 2>&1 | tee -a /home/ubuntu/look_ahead6/logs/corr_ffn_add_d12_c1024_h16_owt.log

echo "=== Done ==="

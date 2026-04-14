#!/bin/bash
set -e

# Experiment: roformer N=24 C=1024 n_head=16 OWT on GPU 0
# batch_size=32, 200K iters to match total tokens
echo "=== roformer N=24 C=1024 n_head=16 OWT ==="
cd /home/ubuntu/look_ahead6 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models roformer --n_embed 1024 --n_layers 24 --block_size 256 --batch_size 32 \
  --lr 2e-4 --softmax --n_head 16 \
  --max_iters 200000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
  --checkpoint_dir /home/ubuntu/look_ahead6/checkpoints \
  --gpu 0 \
  --amp 2>&1 | tee -a /home/ubuntu/look_ahead6/logs/roformer_n24_c1024_h16_owt.log

echo "=== Done ==="

#!/bin/bash
set -e

echo "=== roformer N=12 C=768 OWT (144C² FLOPs) ==="
cd /home/ubuntu/look_ahead6 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models roformer --n_embed 768 --n_layers 12 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
  --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/roformer_n12_c768_owt.log

echo "=== Done ==="

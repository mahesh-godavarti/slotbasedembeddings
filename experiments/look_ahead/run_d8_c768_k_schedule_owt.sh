#!/bin/bash
set -e

echo "=== block_head_corr_ffn_add D=8 C=768 K-schedule OWT (104C² FLOPs, compiled) ==="
echo "Schedule: K=1 (0-50K), K=2 (50K-90K), K=random(2,5) (90K-100K)"
cd /home/ubuntu/look_ahead6 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train \
  --models block_head_corr_ffn_add --n_embed 768 --n_layers 40 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 0 --d_block 8 \
  --max_iters 100000 --eval_interval 5000 \
  --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
  --amp \
  --k_schedule "0:1,50000:2,90000:2-5" \
  2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_add_d8_c768_k_schedule_owt.log

echo "=== Done ==="

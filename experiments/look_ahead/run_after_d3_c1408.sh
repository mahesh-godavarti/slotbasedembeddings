#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for D=3 C=1408 to finish..."
while pgrep -f 'n_embed 1408.*n_layers 15' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): D=3 C=1408 done. Starting D=6 C=1024."

mkdir -p checkpoints_d6_c1024 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 30 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 6 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d6_c1024 \
    --gpu 0 \
    --amp 2>&1 | tee logs/d6_c1024_scratch.log

#!/bin/bash
set -e
cd /home/ubuntu/look_ahead7

echo "$(date): Waiting for D=1 C=4128 to finish..."
while pgrep -f 'n_embed 4128.*n_layers 5' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): D=1 C=4128 done. Starting SA D=11 C=768."

mkdir -p checkpoints_sa_d11_c768
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_sa_corr_ffn_add --n_embed 768 --n_layers 55 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 11 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_sa_d11_c768 \
    --gpu 1 \
    --amp 2>&1 | tee logs/sa_d11_c768.log

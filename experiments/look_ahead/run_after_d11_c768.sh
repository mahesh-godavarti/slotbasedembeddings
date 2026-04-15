#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for D=11 C=768 to finish..."
while pgrep -f 'n_embed 768.*n_layers 55' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): D=11 C=768 done. Starting D=5 C=1120."

mkdir -p checkpoints_d5_c1120 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1120 --n_layers 25 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 5 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d5_c1120 \
    --gpu 1 \
    --amp 2>&1 | tee logs/d5_c1120_scratch.log

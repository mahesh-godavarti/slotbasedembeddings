#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for D=1 C=2048 bs512 to finish..."
while pgrep -f 'n_embed 2048.*n_layers 5.*block_size 512' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): Testing D=1 C=2048 bs1024 batch=32."

mkdir -p checkpoints_d1_c2048_bs1024 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2048 --n_layers 5 --block_size 1024 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d1_c2048_bs1024 \
    --gpu 1 \
    --amp 2>&1 | tee logs/d1_c2048_bs1024.log
echo "$(date): Finished D=1 C=2048 bs1024"

#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for N=1 C=2656 to finish..."
while pgrep -f 'n_embed 2656.*n_layers 1' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): Done. Starting D=1 pure C=2048."

mkdir -p checkpoints_d1_pure_c2048 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add_pure --n_embed 2048 --n_layers 5 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d1_pure_c2048 \
    --gpu 1 \
    --amp 2>&1 | tee logs/d1_pure_c2048_scratch.log
echo "$(date): Finished D=1 pure C=2048"

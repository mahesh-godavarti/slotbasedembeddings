#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Extending N=6 C=1088 bs1024 to 400K."

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models roformer --n_embed 1088 --n_layers 6 --block_size 1024 --batch_size 32 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 400000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_n6_c1088_bs1024 \
    --gpu 0 \
    --amp 2>&1 | tee -a logs/roformer_n6_c1088_bs1024.log
echo "$(date): Finished N=6 C=1088 bs1024 400K"

#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for N=24 C=1088 to finish..."
while pgrep -f 'n_embed 1088.*n_layers 24' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): N=24 C=1088 done. Starting N=4 C=2656."

mkdir -p checkpoints_n4_c2656
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models roformer --n_embed 2656 --n_layers 4 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_n4_c2656 \
    --gpu 0 \
    --amp 2>&1 | tee logs/roformer_n4_c2656_h16_owt.log

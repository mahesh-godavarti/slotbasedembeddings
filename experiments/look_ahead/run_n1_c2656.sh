#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for D=1 C=4128 lr=1e-5 to finish..."
while pgrep -f 'n_embed 4128.*lr 1e-5' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): Done. Starting N=1 C=2656."

mkdir -p checkpoints_n1_c2656 logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models roformer --n_embed 2656 --n_layers 1 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 100000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_n1_c2656 \
    --gpu 1 \
    --amp 2>&1 | tee logs/roformer_n1_c2656.log
echo "$(date): Finished N=1 C=2656"

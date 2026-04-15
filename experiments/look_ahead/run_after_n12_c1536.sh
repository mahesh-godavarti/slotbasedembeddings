#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for N=12 C=1536 to finish..."
while pgrep -f 'n_embed 1536.*n_layers 12' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): N=12 C=1536 done. Starting N=2 C=3776."

mkdir -p checkpoints_n2_c3776
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models roformer --n_embed 3776 --n_layers 2 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_n2_c3776 \
    --gpu 1 \
    --amp 2>&1 | tee logs/roformer_n2_c3776_h16_owt.log

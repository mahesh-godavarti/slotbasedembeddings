#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for D=5 C=1120 to finish..."
while pgrep -f 'n_embed 1120.*n_layers 25' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): D=5 C=1120 done. Starting width scaling extensions on GPU 1."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=1
BS=64
LR=2e-4
EVAL=500

# N=2 C=256 (resume from 7,629 → 37,500)
echo "$(date): === N=2 C=256 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 256 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 4 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_n2_c256 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c256_ext.log
echo "$(date): Finished N=2 C=256 extend"

# N=2 C=512 (from scratch, batch=1024)
echo "$(date): === N=2 C=512 fresh ==="
mkdir -p checkpoints_width_n2_c512_b1024
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 512 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 8 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_n2_c512_b1024 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c512_b1024.log
echo "$(date): Finished N=2 C=512 fresh"

# D=1 C=280 (resume from 7,629 → 37,500)
echo "$(date): === D=1 C=280 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 280 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 4 --k_min 2 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c280_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c280_ext.log
echo "$(date): Finished D=1 C=280 extend"

# D=1 C=560 (from scratch, batch=1024)
echo "$(date): === D=1 C=560 fresh ==="
mkdir -p checkpoints_width_d1_c560_b1024
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 560 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 8 --k_min 2 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c560_b1024 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c560_b1024.log
echo "$(date): Finished D=1 C=560 fresh"

echo "$(date): GPU 1 width scaling extensions complete."

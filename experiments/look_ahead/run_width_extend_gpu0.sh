#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for D=6 C=1024 to finish..."
while pgrep -f 'n_embed 1024.*n_layers 30' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): D=6 C=1024 done. Starting width scaling extensions on GPU 0."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=0
BS=64
LR=2e-4
EVAL=500

# N=2 C=1024 (resume from 18,750 → 37,500)
echo "$(date): === N=2 C=1024 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 1024 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 16 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_scaling_n2_cont2 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c1024_ext.log
echo "$(date): Finished N=2 C=1024 extend"

# D=1 C=1120 (resume from 18,723 → 37,500)
echo "$(date): === D=1 C=1120 extend ==="
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1120 --n_layers 5 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 37500 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d1_c1120_scratch \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_d1_c1120_ext.log
echo "$(date): Finished D=1 C=1120 extend"

echo "$(date): GPU 0 width scaling extensions complete."

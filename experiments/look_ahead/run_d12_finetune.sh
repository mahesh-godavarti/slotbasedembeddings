#!/bin/bash
# Fine-tune converted roformer N=12 as look-ahead D=12 at K=2 on GPU 1
# Run convert_roformer_to_lookahead.py first to create the checkpoint.
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/ubuntu/look_ahead6/flash_override:$PYTHONPATH
PYTHON=/home/ubuntu/exp8/venv/bin/python

# K-schedule: start at K=2 (model already has good representations from roformer pretraining)
# Brief K=2-5 at end for convergence robustness
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 60 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 12 --n_head 16 \
    --k_schedule "0:2-5" \
    --max_iters 50000 --eval_interval 2000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d12_converted \
    --gpu 1 \
    --amp 2>&1 | tee logs/corr_ffn_add_d12_c1024_h16_finetune_owt.log

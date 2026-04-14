#!/bin/bash
# D=23 C=1024 h16 with Flash Attention
# Temporarily renames blocks.py so blocks_flash.py can import from it,
# then makes blocks_flash.py available as blocks.py.
# Resumes from checkpoint in checkpoints_d23/.

set -e
cd /home/ubuntu/look_ahead6

# Move original blocks.py aside, symlink flash version as blocks.py
mv blocks.py blocks_original.py
# blocks_flash.py imports from blocks_original via "from blocks import *"
# so we need to update it. Instead, let's use a simpler approach:
# rename original to _blocks_base.py, update flash to import from that.
mv blocks_original.py _blocks_base.py
ln -sf blocks_flash.py blocks.py

# Restore on exit
trap "rm -f blocks.py; mv _blocks_base.py blocks.py" EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python

$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 115 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 23 --n_head 16 \
    --max_iters 200000 --eval_interval 10000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d23 \
    --gpu 0 \
    --amp 2>&1 | tee logs/corr_ffn_add_d23_c1024_h16_flash_owt.log

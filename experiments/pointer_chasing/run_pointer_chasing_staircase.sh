#!/bin/bash
# Pointer chasing: 10-hop staircase with windowed attention (no shuffle)
# Shows depth separation: N layers solves ~N levels, BPTT solves all
#
# Settings: k=5, v=10, e=256, window=38, no shuffle, permutation
# Results: N=1→1, N=3→3, N=5→6, N=10→7, N=11→8, N=12→all 11

source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead6

GPU=${1:-0}

# Transformer staircase
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u pointer_chasing.py \
    --n_hops 10 --n_keys 5 --n_values 10 \
    --n_embed 256 --n_head 4 --n_iters 50000 --batch_size 64 --lr 1e-4 \
    --gpu $GPU --permutation --run N1,N3,N5,N10,N11,N12 \
    --window 38 --no_shuffle \
    --checkpoint_dir checkpoints_staircase_e256 \
    2>&1 | tee logs/staircase_e256_50k.log

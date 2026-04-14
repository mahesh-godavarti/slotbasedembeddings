#!/bin/bash
# Pointer chasing: BPTT (D=1 look-ahead) with windowed attention
# Solves all 11 levels (10-hop) with a single shared block
#
# Two variants: e=128 (lr=1e-3, solves in ~20K) and e=256 (lr=1e-4, solves in ~16K)

source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead6

GPU=${1:-0}
EMBED=${2:-128}

if [ "$EMBED" = "256" ]; then
    LR=1e-4
else
    LR=1e-3
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u pointer_chasing.py \
    --n_hops 10 --n_keys 5 --n_values 10 \
    --n_embed $EMBED --n_head 4 --n_iters 100000 --batch_size 64 --lr $LR \
    --gpu $GPU --permutation --run bptt \
    --window 38 --no_shuffle \
    --checkpoint_dir checkpoints_bptt_e${EMBED} \
    2>&1 | tee logs/bptt_e${EMBED}_100k.log

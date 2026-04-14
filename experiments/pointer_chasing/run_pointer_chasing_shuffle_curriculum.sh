#!/bin/bash
# Pointer chasing: shuffled with curriculum (no-RoPE + windowed + key/hop curriculum)
# The only combination that works for multi-hop composition with shuffling
#
# Slow key curriculum: 2→3→4→5 with 50K per step at 3-hop phase
# L1 transfers from k=2 to k=5, L2 partially transfers

source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead6

GPU=${1:-0}

python -u pointer_chasing.py \
    --n_hops 3 --n_keys 5 --n_values 10 \
    --n_embed 128 --n_head 4 --n_iters 300000 --batch_size 64 --lr 1e-3 \
    --gpu $GPU --permutation --run N4 \
    --no_rope --window 38 \
    --hop_curriculum "0:2,100000:3" \
    --key_curriculum "0:2,20000:5,100000:2,150000:3,200000:4,250000:5" \
    --checkpoint_dir checkpoints_shuffle_curriculum \
    2>&1 | tee logs/shuffle_curriculum_300k.log

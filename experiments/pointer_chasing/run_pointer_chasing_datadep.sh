#!/bin/bash
# Pointer chasing: data-dependent positional encoding variants
#
# Available variants:
#   datadep    - content-dependent angles, Q/K only
#   datadepv   - content-dependent angles, Q/K/V + inverse
#   monoidal   - cumsum angles, Q/K only
#   joformer   - cumsum angles, Q/K/V + inverse
#   datadep3   - MLP angles, Q/K only
#   datadep2   - angles flow through layers, no cumsum
#   monoidal2  - angles flow through layers, cumsum
#   joformer2  - angles flow through layers, cumsum, rotate V
#
# Usage: bash run_pointer_chasing_datadep.sh [gpu] [variant]
# Example: bash run_pointer_chasing_datadep.sh 0 datadepv

source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead6

GPU=${1:-0}
VARIANT=${2:-datadep}

python -u pointer_chasing.py \
    --n_hops 3 --n_keys 5 --n_values 10 \
    --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
    --gpu $GPU --permutation --run ${VARIANT}_N4 \
    --window 38 \
    --checkpoint_dir checkpoints_${VARIANT}_N4 \
    2>&1 | tee logs/${VARIANT}_N4_100k.log

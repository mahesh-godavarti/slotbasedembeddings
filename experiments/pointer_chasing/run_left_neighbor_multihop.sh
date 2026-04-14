#!/bin/bash
# Left neighbor multi-hop task
# Each level is a shuffled permutation, "left neighbor" is the mapping
# Simpler encoding than pointer chasing — no triplets, just tokens in a row
#
# Usage: bash run_left_neighbor_multihop.sh [gpu] [K] [n_hops] [models]
# Example: bash run_left_neighbor_multihop.sh 0 8 3 N3

source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead6

GPU=${1:-0}
K=${2:-8}
NHOPS=${3:-3}
MODELS=${4:-N3}

python -u left_neighbor_multihop.py \
    --K $K --n_hops $NHOPS \
    --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
    --gpu $GPU --run $MODELS \
    2>&1 | tee logs/left_neighbor_K${K}_${NHOPS}hop_${MODELS}_100k.log

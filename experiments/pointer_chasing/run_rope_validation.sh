#!/bin/bash
# RoPE vs no-RoPE validation tasks
# Three tasks that verify our no-RoPE implementation:
#   1. min_element  — order-invariant (RoPE = no-RoPE)
#   2. copy_back2   — positional (RoPE >> no-RoPE)
#   3. left_neighbor — content + positional (RoPE > no-RoPE)

source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead6

GPU=${1:-0}

echo "=== Min Element (order-invariant, expect RoPE = no-RoPE) ==="
python -u min_element.py --V 20 --N 10 --n_embed 128 --n_head 4 --n_layers 3 \
    --n_iters 5000 --batch_size 64 --lr 1e-3 --gpu $GPU

echo ""
echo "=== Copy Back 2 (positional, expect RoPE >> no-RoPE) ==="
python -u copy_back2.py --V 20 --N 10 --n_embed 128 --n_head 4 --n_layers 3 \
    --n_iters 5000 --batch_size 64 --lr 1e-3 --gpu $GPU

echo ""
echo "=== Left Neighbor (content + positional, expect RoPE > no-RoPE) ==="
python -u left_neighbor.py --V 20 --N 10 --n_embed 128 --n_head 4 --n_layers 3 \
    --n_iters 10000 --batch_size 64 --lr 1e-3 --gpu $GPU

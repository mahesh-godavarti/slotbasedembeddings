#!/bin/bash
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/cifar10_composition

for pe in learned rope2d joformer_old monoidal_axial joformer_axial rope2dv2 monoidal joformer joformer_fixed; do
    echo "========== $pe =========="
    python vit_cifar10.py --dataset cifar100 --pe_type $pe \
        --embed_dim 128 --n_layers 4 --n_heads 4 \
        --epochs 300 --seed 42 --lr 1e-3 --cosine_decay --weight_decay 0.1
    echo ""
done

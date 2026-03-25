#!/bin/bash
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/cifar10_composition

for pe in learned rope2d joformer_old monoidal_axial joformer_axial monoidal_axial_perlayer joformer_axial_perlayer rope2dv2 monoidal joformer joformer_fixed monoidal_perlayer joformer_perlayer; do
    echo "========== $pe =========="
    python vit_cifar10.py --dataset cifar100 --pe_type $pe \
        --embed_dim 256 --n_layers 5 --n_heads 8 \
        --epochs 600 --seed 42 --lr 1e-3 --cosine_decay \
        --weight_decay 0.05 --dropout 0.1 --label_smoothing 0.1 \
        --warmup_epochs 10 --warmup_lr 2e-4 \
        --high_epochs 20 --high_lr 1e-3 \
        --resume
    echo ""
done

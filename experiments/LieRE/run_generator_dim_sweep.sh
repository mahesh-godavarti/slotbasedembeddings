#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

for gdim in 2 8; do
    echo "========== generator_dim=$gdim Q/K only =========="
    python run_cifar100.py --model_size tiny --epochs 200 --gpu 1 --generator_dim $gdim
    echo ""

    echo "========== generator_dim=$gdim Q/K/V =========="
    python run_cifar100.py --model_size tiny --epochs 200 --gpu 1 --generator_dim $gdim --rotate_v
    echo ""
done

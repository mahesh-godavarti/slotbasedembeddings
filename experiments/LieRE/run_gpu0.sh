#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

GPU=0

echo "========== LieRE axial (dense) Q/K only (200ep) =========="
python run_cifar100_axial_dense.py --model_size tiny --epochs 200 --gpu $GPU
echo ""

echo "========== Axial butterfly Q/K only (200ep) =========="
python run_cifar100_axial_butterfly.py --model_size tiny --epochs 200 --gpu $GPU
echo ""

echo "========== Random-mix n_rounds=2 Q/K only (200ep) =========="
python run_cifar100_randmix.py --model_size tiny --epochs 200 --gpu $GPU --n_rounds 2
echo ""

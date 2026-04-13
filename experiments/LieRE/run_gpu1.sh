#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

GPU=1

echo "========== LieRE axial (dense) Q/K/V (200ep) =========="
python run_cifar100_axial_dense.py --model_size tiny --epochs 200 --gpu $GPU --rotate_v
echo ""

echo "========== Axial butterfly Q/K/V (200ep) =========="
python run_cifar100_axial_butterfly.py --model_size tiny --epochs 200 --gpu $GPU --rotate_v
echo ""

echo "========== Random-mix n_rounds=2 Q/K/V (200ep) =========="
python run_cifar100_randmix.py --model_size tiny --epochs 200 --gpu $GPU --n_rounds 2 --rotate_v
echo ""

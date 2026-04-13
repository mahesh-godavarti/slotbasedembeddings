#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

EPOCHS=200
GPU=1

echo "========== Random-mix n_rounds=2 Q/K only =========="
python run_cifar100_randmix.py --model_size tiny --epochs $EPOCHS --gpu $GPU --n_rounds 2
echo ""

echo "========== Random-mix n_rounds=2 Q/K/V =========="
python run_cifar100_randmix.py --model_size tiny --epochs $EPOCHS --gpu $GPU --n_rounds 2 --rotate_v
echo ""

echo "========== Axial butterfly Q/K only =========="
python run_cifar100_axial_butterfly.py --model_size tiny --epochs $EPOCHS --gpu $GPU
echo ""

echo "========== Axial butterfly Q/K/V =========="
python run_cifar100_axial_butterfly.py --model_size tiny --epochs $EPOCHS --gpu $GPU --rotate_v
echo ""

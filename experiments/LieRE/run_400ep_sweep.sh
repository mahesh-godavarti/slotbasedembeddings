#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

EPOCHS=400
GPU=1

echo "========== LieRE64 Q/K only (400ep) =========="
python run_cifar100.py --model_size tiny --epochs $EPOCHS --gpu $GPU --generator_dim 64
echo ""

echo "========== LieRE64 Q/K/V (400ep) =========="
python run_cifar100.py --model_size tiny --epochs $EPOCHS --gpu $GPU --generator_dim 64 --rotate_v
echo ""

echo "========== Butterfly Q/K only (400ep) =========="
python run_cifar100_butterfly.py --model_size tiny --epochs $EPOCHS --gpu $GPU
echo ""

echo "========== Butterfly Q/K/V (400ep) =========="
python run_cifar100_butterfly.py --model_size tiny --epochs $EPOCHS --gpu $GPU --rotate_v
echo ""

echo "========== Block 2x2 Q/K only (400ep) =========="
python run_cifar100_block_v.py --model_size tiny --epochs $EPOCHS --gpu $GPU
echo ""

echo "========== Block 2x2 Q/K/V (400ep) =========="
python run_cifar100_block_v.py --model_size tiny --epochs $EPOCHS --gpu $GPU --rotate_v
echo ""

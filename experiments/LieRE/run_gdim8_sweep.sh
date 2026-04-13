#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

echo "========== generator_dim=8 Q/K only =========="
python run_cifar100.py --model_size tiny --epochs 200 --gpu 1 --generator_dim 8
echo ""

echo "========== generator_dim=8 Q/K/V =========="
python run_cifar100.py --model_size tiny --epochs 200 --gpu 1 --generator_dim 8 --rotate_v
echo ""

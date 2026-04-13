#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

echo "========== Butterfly Q/K only =========="
python run_cifar100_butterfly.py --model_size tiny --epochs 200 --gpu 1
echo ""

echo "========== Butterfly Q/K/V =========="
python run_cifar100_butterfly.py --model_size tiny --epochs 200 --gpu 1 --rotate_v
echo ""

#!/bin/bash
source /home/ubuntu/LieRE/venv/bin/activate
cd /home/ubuntu/LieRE
export WANDB_MODE=offline

echo "========== Block-diagonal Q/K only =========="
python run_cifar100_block_v.py --model_size tiny --epochs 200 --gpu 1
echo ""

echo "========== Block-diagonal Q/K/V + inverse (JoFormer) =========="
python run_cifar100_block_v.py --model_size tiny --epochs 200 --gpu 1 --rotate_v
echo ""

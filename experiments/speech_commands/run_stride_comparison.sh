#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== CumsumE2E (linear) stride=320 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e_s320 --n_embed 40 --n_layers 4 --window_l1 400 --window 20 --epochs 40 2>&1 | tee cumsum_e2e_s320.log

echo "=== CumsumE2E (linear) stride=640 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e_s640 --n_embed 40 --n_layers 4 --window_l1 400 --window 20 --epochs 40 2>&1 | tee cumsum_e2e_s640.log

echo "=== CumsumE2EMag (log) stride=320 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e_mag_s320 --n_embed 40 --n_layers 4 --window_l1 400 --window 20 --epochs 40 2>&1 | tee cumsum_e2e_mag_s320.log

echo "=== CumsumE2EMag (log) stride=640 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e_mag_s640 --n_embed 40 --n_layers 4 --window_l1 400 --window 20 --epochs 40 2>&1 | tee cumsum_e2e_mag_s640.log

#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== CumsumE2E readout=mag (baseline) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e --n_embed 40 --n_layers 4 --window 20 --readout mag --epochs 40 2>&1 | tee cumsum_e2e_mag_readout.log

echo "=== CumsumE2E readout=mlp ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e --n_embed 40 --n_layers 4 --window 20 --readout mlp --epochs 40 2>&1 | tee cumsum_e2e_mlp_readout.log

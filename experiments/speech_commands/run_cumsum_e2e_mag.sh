#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== CumsumE2EMag (fixed, mag+log, l1=400, w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e_mag --n_embed 40 --n_layers 4 --window_l1 400 --window 20 --epochs 40 2>&1 | tee cumsum_e2e_mag_n40_l4_w20.log

echo "=== CumsumE2EMagMod (mod, mag+log, l1=400, w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_e2e_mag_mod --n_embed 40 --n_layers 4 --window_l1 400 --window 20 --epochs 40 2>&1 | tee cumsum_e2e_mag_mod_n40_l4_w20.log

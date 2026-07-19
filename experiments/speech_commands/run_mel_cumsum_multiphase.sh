#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== MelCumsumFixed MultiPhase 2 (hop=80, w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_mp2 --n_embed 80 --n_layers 4 --window 20 --epochs 40 2>&1 | tee mel_cumsum_fixed_mp2_n80_l4_w20.log

echo "=== MelCumsumFixed MultiPhase 4 (hop=40, w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_mp4 --n_embed 80 --n_layers 4 --window 20 --epochs 40 2>&1 | tee mel_cumsum_fixed_mp4_n80_l4_w20.log

echo "=== MelCumsumMod MultiPhase 2 (hop=80, w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_mp2 --n_embed 80 --n_layers 4 --window 20 --epochs 40 2>&1 | tee mel_cumsum_mod_mp2_n80_l4_w20.log

echo "=== MelCumsumMod MultiPhase 4 (hop=40, w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_mp4 --n_embed 80 --n_layers 4 --window 20 --epochs 40 2>&1 | tee mel_cumsum_mod_mp4_n80_l4_w20.log

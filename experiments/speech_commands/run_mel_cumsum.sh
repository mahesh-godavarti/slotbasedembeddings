#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== MelCumsumFixed (full seq) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed --n_embed 80 --n_layers 4 --epochs 40 2>&1 | tee mel_cumsum_fixed_n80_l4.log

echo "=== MelCumsumMod (full seq) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod --n_embed 80 --n_layers 4 --epochs 40 2>&1 | tee mel_cumsum_mod_n80_l4.log

echo "=== MelCumsumFixed windowed (w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_w --n_embed 80 --n_layers 4 --window 20 --epochs 40 2>&1 | tee mel_cumsum_fixed_w20_n80_l4.log

echo "=== MelCumsumMod windowed (w=20) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_w --n_embed 80 --n_layers 4 --window 20 --epochs 40 2>&1 | tee mel_cumsum_mod_w20_n80_l4.log

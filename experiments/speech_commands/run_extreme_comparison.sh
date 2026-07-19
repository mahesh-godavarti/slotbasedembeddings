#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== Fixed w10 l8, time_stretch 0.5-2.0 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --time_stretch --stretch_range 0.5 2.0 --eval_grid

echo ""
echo "=== Past-phase Mod w10 l8, time_stretch 0.5-2.0 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --time_stretch --stretch_range 0.5 2.0 --eval_grid

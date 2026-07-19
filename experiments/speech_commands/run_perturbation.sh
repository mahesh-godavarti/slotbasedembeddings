#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== Fixed tied ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --time_stretch --pitch_shift --eval_grid

echo ""
echo "=== Mod tied ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --time_stretch --pitch_shift --eval_grid

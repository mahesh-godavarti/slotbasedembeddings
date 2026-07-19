#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== Both-prev Mod w10 l8, time_stretch 0.8-1.2 (both Φ(t-1)) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --time_stretch --eval_grid --phase_mode both_prev

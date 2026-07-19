#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# Wait for GPU to be free
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    sleep 30
done

echo "=== Future-phase Mod w10 l8, NO time_stretch ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --eval_grid --future_phase

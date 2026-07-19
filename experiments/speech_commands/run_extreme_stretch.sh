#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# Wait for GPU to be free
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    sleep 30
done

echo "=== Fixed w10 l8 extreme time_stretch + eval_grid ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --time_stretch --eval_grid

echo ""
echo "=== Mod w10 l8 extreme time_stretch + eval_grid ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --time_stretch --eval_grid

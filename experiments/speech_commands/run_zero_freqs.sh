#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# Wait for GPU to be free
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | awk '{if ($1+0 > 50) exit 0; else exit 1}'; do
    sleep 30
done

echo "=== Fixed w10 l8 ZERO FREQS, split_stretch + eval_split ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --split_stretch --eval_split --zero_freqs

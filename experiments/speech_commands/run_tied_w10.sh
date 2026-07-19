#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# Wait for GPU to be free
while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -gt 10 ]; do
    sleep 30
done

echo "=== Fixed Tied 4L W=10 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 4 --window 10 --epochs 40 2>&1 | tee mel_cumsum_fixed_tied_w10_l4.log

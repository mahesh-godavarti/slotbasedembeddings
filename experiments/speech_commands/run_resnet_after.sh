#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
# Wait for current GPU job to finish
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -qv "^0 %"; do
    sleep 30
done
sleep 5
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_resnet --window 3 --epochs 40 2>&1 | tee mel_cumsum_resnet_w3.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_magdeep_tied --n_embed 80 --n_layers 8 --window 5 --epochs 40 2>&1 | tee mel_cumsum_magdeep_tied_w5_l8.log

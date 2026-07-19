#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -qv "^0 %"; do
    sleep 30
done
sleep 5
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_spec_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 5 --window_l1 400 --hop 160 --epochs 40 2>&1 | tee cumsum_spec_cumsum_mod_tied_n80_w5_l8.log

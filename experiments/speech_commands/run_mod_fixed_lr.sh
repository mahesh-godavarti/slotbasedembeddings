#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -qv "^0 %"; do
    sleep 30
done
sleep 5
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 80 --lr 3e-4 2>&1 | tee mel_cumsum_fixed_tied_w10_l8_hop80_fixedlr3e-4_80ep.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 80 --lr 3e-4 2>&1 | tee mel_cumsum_mod_tied_w10_l8_hop80_fixedlr3e-4_80ep.log

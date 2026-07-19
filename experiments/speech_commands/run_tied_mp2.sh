#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -qv "^0 %"; do
    sleep 30
done
sleep 5

# FixedTied MP2, n=80, W=10, L=8, hop=80, 40ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied_mp2 --n_embed 80 --n_layers 8 --window 10 --epochs 40 2>&1 | tee mel_cumsum_fixed_tied_mp2_n80_w10_l8_40ep.log

# ModTied MP2, n=80, W=10, L=8, hop=80, 40ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied_mp2 --n_embed 80 --n_layers 8 --window 10 --epochs 40 2>&1 | tee mel_cumsum_mod_tied_mp2_n80_w10_l8_40ep.log

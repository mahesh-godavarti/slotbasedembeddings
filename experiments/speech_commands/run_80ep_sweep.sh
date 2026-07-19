#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -qv "^0 %"; do
    sleep 30
done
sleep 5

# n=100, W=10, L=8, hop=80, 80ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 100 --n_layers 8 --window 10 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_fixed_tied_n100_w10_l8_hop80_80ep.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 100 --n_layers 8 --window 10 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_mod_tied_n100_w10_l8_hop80_80ep.log

# n=120, W=10, L=8, hop=80, 80ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 120 --n_layers 8 --window 10 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_fixed_tied_n120_w10_l8_hop80_80ep.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 120 --n_layers 8 --window 10 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_mod_tied_n120_w10_l8_hop80_80ep.log

# n=80, W=40, L=2, hop=80, 80ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 2 --window 40 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_fixed_tied_n80_w40_l2_hop80_80ep.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 2 --window 40 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_mod_tied_n80_w40_l2_hop80_80ep.log

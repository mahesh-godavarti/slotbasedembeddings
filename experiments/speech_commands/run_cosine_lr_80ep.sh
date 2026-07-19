#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# Cosine 3e-4 → 0 over 80 epochs
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 80 --lr 3e-4 2>&1 | tee mel_cumsum_fixed_tied_w10_l8_hop80_cosine3e-4_80ep.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 80 --lr 3e-4 2>&1 | tee mel_cumsum_mod_tied_w10_l8_hop80_cosine3e-4_80ep.log

# Cosine 1e-3 → 0 over 80 epochs
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_fixed_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_fixed_tied_w10_l8_hop80_cosine1e-3_80ep.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 80 2>&1 | tee mel_cumsum_mod_tied_w10_l8_hop80_cosine1e-3_80ep.log

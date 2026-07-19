#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# ModTied no freq bias, n=80, W=10, L=8, hop=80, 40ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --no_freq_bias 2>&1 | tee mel_cumsum_mod_tied_n80_w10_l8_hop80_40ep_nofreqbias.log

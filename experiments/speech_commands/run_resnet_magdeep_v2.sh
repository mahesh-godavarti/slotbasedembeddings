#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_resnet --n_embed 24 --window 3 --epochs 40 2>&1 | tee mel_cumsum_resnet_w3_ch24.log

PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_magdeep_tied --n_embed 90 --n_layers 8 --window 5 --epochs 40 2>&1 | tee mel_cumsum_magdeep_tied_n90_w5_l8.log

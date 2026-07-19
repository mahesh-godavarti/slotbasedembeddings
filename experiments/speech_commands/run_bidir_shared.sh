#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -qv "^0 %"; do
    sleep 30
done
sleep 5

# Bidirectional Tied (shared params), n=80, W=10, L=8, hop=80, 40ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_bidir_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 2>&1 | tee mel_cumsum_bidir_tied_shared_n80_w10_l8_hop80_40ep.log

#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands
while nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | grep -qv "^0 %"; do
    sleep 30
done
sleep 5

# ModTied bottleneck k=20, n=80, W=10, L=8, hop=80, 40ep
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_cumsum_mod_tied --n_embed 80 --n_layers 8 --window 10 --hop 80 --epochs 40 --freq_bottleneck 20 2>&1 | tee mel_cumsum_mod_tied_n80_w10_l8_hop80_40ep_bottleneck20.log

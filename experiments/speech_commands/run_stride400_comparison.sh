#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== MelCNNMaxPool hop=400 n_mels=80 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_maxpool --hop 400 --n_embed 80 --epochs 40 2>&1 | tee mel_maxpool_hop400_n80.log

echo "=== LearnedSpecCNN (mag only) hop=400 n_freqs=80 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec --hop 400 --n_embed 80 --epochs 40 2>&1 | tee learned_spec_hop400_n80.log

echo "=== LearnedSpecMagReImCNN hop=400 n_freqs=80 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_magreim --hop 400 --n_embed 80 --epochs 40 2>&1 | tee learned_spec_magreim_hop400_n80.log

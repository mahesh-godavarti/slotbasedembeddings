#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== MelCNNMaxPool (no SA) hop=400 n_mels=80 time_shift=200 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_maxpool_nosa --hop 400 --n_embed 80 --time_shift 200 --epochs 40 2>&1 | tee mel_maxpool_nosa_hop400_n80_ts200.log

echo "=== LearnedSpecFrozen hop=400 n_freqs=80 time_shift=200 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_frozen --hop 400 --n_embed 80 --time_shift 200 --epochs 40 2>&1 | tee learned_spec_frozen_hop400_n80_ts200.log

echo "=== LearnedSpecMagReImFrozen hop=400 n_freqs=80 time_shift=200 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_magreim_frozen --hop 400 --n_embed 80 --time_shift 200 --epochs 40 2>&1 | tee learned_spec_magreim_frozen_hop400_n80_ts200.log

#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== MelMultiPhase hop=80 (2 phases) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_multiphase_80 --epochs 40 2>&1 | tee mel_multiphase_80.log

echo "=== MelMultiPhase hop=40 (4 phases) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_multiphase_40 --epochs 40 2>&1 | tee mel_multiphase_40.log

#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

echo "=== MelMaxPool hop=160 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_maxpool_160 --epochs 40 2>&1 | tee mel_maxpool_160.log

echo "=== MelMaxPool hop=80 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_maxpool_80 --epochs 40 2>&1 | tee mel_maxpool_80.log

echo "=== MelMaxPool hop=40 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_maxpool_40 --epochs 40 2>&1 | tee mel_maxpool_40.log

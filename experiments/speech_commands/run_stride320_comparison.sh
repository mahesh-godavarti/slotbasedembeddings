#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# Wait for GPU to be free
while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -gt 10 ]; do
    sleep 30
done

echo "=== MelCNNMaxPool hop=320 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model mel_maxpool --hop 320 --epochs 40 2>&1 | tee mel_maxpool_hop320.log

echo "=== LearnedSpecCNN (mag only) hop=320 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec --hop 320 --epochs 40 2>&1 | tee learned_spec_hop320.log

echo "=== LearnedSpecMagReImCNN hop=320 ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_magreim --hop 320 --epochs 40 2>&1 | tee learned_spec_magreim_hop320.log

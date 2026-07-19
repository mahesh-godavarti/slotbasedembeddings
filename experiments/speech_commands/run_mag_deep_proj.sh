#!/bin/bash
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/speech_commands

# Wait for GPU to be free
while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -gt 10 ]; do
    sleep 30
done

echo "=== CumsumMagDeep with proj (n_embed=44) ==="
PYTHONUNBUFFERED=1 python speech_commands.py --model cumsum_mag_deep_proj --n_embed 44 --n_layers 4 --window 20 --epochs 40 2>&1 | tee cumsum_mag_deep_proj_n44.log

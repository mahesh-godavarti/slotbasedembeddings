#!/bin/bash
source ~/experiments/venv/bin/activate
cd ~/speech_commands

echo "Starting MultiLayerV2 (40 epochs)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model multi_layer_v2 --n_embed 40 --window 400 --n_layers 4 --epochs 40 2>&1 | tee /tmp/multi_layer_v2.log

echo "V2 done. Starting MultiLayerModV2 (40 epochs)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model multi_layer_mod_v2 --n_embed 40 --window 400 --n_layers 4 --epochs 40 2>&1 | tee /tmp/multi_layer_mod_v2.log

echo "All done."

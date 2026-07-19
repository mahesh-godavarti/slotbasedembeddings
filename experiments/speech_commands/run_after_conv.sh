#!/bin/bash
source ~/experiments/venv/bin/activate
cd ~/speech_commands

while pgrep -f "speech_commands.py" > /dev/null; do
    sleep 10
done

echo "Conv done. Starting LearnedSpecCNN W=200..."
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec --window 200 --epochs 40 2>&1 | tee /tmp/learned_spec_w200.log

echo "W=200 done. Starting LearnedSpecCNN W=80..."
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec --window 80 --epochs 40 2>&1 | tee /tmp/learned_spec_w80.log

echo "W=80 done. Starting FilterbankCNN (k=80)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model filterbank --window 80 --epochs 40 2>&1 | tee /tmp/filterbank.log

echo "FilterbankCNN done. Starting MultiLayerMod..."
PYTHONUNBUFFERED=1 python speech_commands.py --model multi_layer_mod --n_embed 64 --window 400 --ds_factor 10 --n_layers 3 --epochs 40 2>&1 | tee /tmp/multi_layer_mod.log

echo "All done."

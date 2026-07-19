#!/bin/bash
# Wait for MultiLayerMod to finish, then run smoke tests + queued models
source ~/experiments/venv/bin/activate
cd ~/speech_commands

# Wait for any running speech_commands process to finish
while pgrep -f "speech_commands.py" > /dev/null; do
    sleep 10
done

echo "MultiLayerMod done. Smoke testing V2 models..."
PYTHONUNBUFFERED=1 python speech_commands.py --model multi_layer_v2 --n_embed 40 --window 400 --n_layers 4 --smoke 2>&1 | tee /tmp/multi_layer_v2_smoke.log

PYTHONUNBUFFERED=1 python speech_commands.py --model multi_layer_mod_v2 --n_embed 40 --window 400 --n_layers 4 --smoke 2>&1 | tee /tmp/multi_layer_mod_v2_smoke.log

echo "Smoke tests done. Starting FilterbankMelInitCNN (k=80)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model filterbank_mel --window 80 --epochs 40 2>&1 | tee /tmp/filterbank_mel.log

echo "FilterbankMelInit done. Starting LearnedSpecLinearCNN (W=200)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_linear --window 200 --epochs 40 2>&1 | tee /tmp/learned_spec_linear.log

echo "All done."

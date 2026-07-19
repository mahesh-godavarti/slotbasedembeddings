#!/bin/bash
source ~/experiments/venv/bin/activate
cd ~/speech_commands

echo "Starting MultiLayerV2 (stride 2 fix, 40 epochs)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model multi_layer_v2 --n_embed 40 --window 400 --n_layers 4 --epochs 40 2>&1 | tee /tmp/multi_layer_v2.log

echo "Starting MultiLayerModV2 (stride 2 fix, 40 epochs)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model multi_layer_mod_v2 --n_embed 40 --window 400 --n_layers 4 --epochs 40 2>&1 | tee /tmp/multi_layer_mod_v2.log

echo "Starting LearnedSpecMulti (2 per bin, W=200)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_multi --window 200 --epochs 40 2>&1 | tee /tmp/learned_spec_multi.log

echo "Starting ConvCumsumV2..."
PYTHONUNBUFFERED=1 python speech_commands.py --model conv_cumsum_v2 --n_embed 40 --window 400 --n_layers 4 --epochs 40 2>&1 | tee /tmp/conv_cumsum_v2.log

echo "Starting ConvCumsumModV2..."
PYTHONUNBUFFERED=1 python speech_commands.py --model conv_cumsum_mod_v2 --n_embed 40 --window 400 --n_layers 4 --epochs 40 2>&1 | tee /tmp/conv_cumsum_mod_v2.log

echo "Starting FilterbankSinCos..."
PYTHONUNBUFFERED=1 python speech_commands.py --model filterbank_sincos --window 400 --epochs 40 2>&1 | tee /tmp/filterbank_sincos.log

echo "Starting FilterbankSinCosLinear..."
PYTHONUNBUFFERED=1 python speech_commands.py --model filterbank_sincos_linear --window 400 --epochs 40 2>&1 | tee /tmp/filterbank_sincos_linear.log

echo "Starting FilterbankLinear..."
PYTHONUNBUFFERED=1 python speech_commands.py --model filterbank_linear --window 80 --epochs 40 2>&1 | tee /tmp/filterbank_linear.log

echo "Starting FilterbankMelInitLinear..."
PYTHONUNBUFFERED=1 python speech_commands.py --model filterbank_mel_linear --window 80 --epochs 40 2>&1 | tee /tmp/filterbank_mel_linear.log

echo "All done."

#!/bin/bash
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/exp8

python -u word_experiment.py \
    --models A \
    --n_embed 100 --n_layers 20 \
    --iters 10000 \
    --seeds 1

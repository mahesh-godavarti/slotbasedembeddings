#!/bin/bash
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/exp8

python -u word_experiment.py \
    --models "E'" "I'" "K'" \
    --n_embed 100 --n_layers 4 \
    --iters 1000000 \
    --seeds 1 \
    --wiki_lines 10000 \
    --causal_kg \
    --resume_training

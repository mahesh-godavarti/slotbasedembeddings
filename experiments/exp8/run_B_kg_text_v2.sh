#!/bin/bash
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/exp8

python -u word_experiment.py \
    --models B \
    --n_embed 100 --n_layers 4 \
    --iters 50000 \
    --seeds 1 \
    --wiki_lines 100000 \
    --kg_as_text

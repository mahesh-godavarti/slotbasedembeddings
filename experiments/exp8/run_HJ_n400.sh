#!/bin/bash
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/exp8

python -u word_experiment.py \
    --models H "H'" J "J'" \
    --n_embed 400 --n_layers 4 \
    --iters 500000 \
    --seeds 1 \
    --wiki_lines 10000 \
    --causal_kg \
    --checkpoint_dir checkpoints_HJ_n400

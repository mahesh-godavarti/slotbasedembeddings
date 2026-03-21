#!/bin/bash
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/joformer

python kg_text_experiment.py \
    --models roformer roformer_kg roformer_text_kg \
    --n_embed 100 --n_layers 2 \
    --max_iters 100000 --eval_interval 5000 \
    --kg_weight 0.1

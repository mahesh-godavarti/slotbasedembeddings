#!/bin/bash
# Sweep kg_weight values: 0.1, 0.3, 1.0
# Each run trains roformer (baseline), roformer_kg, roformer_text_kg
# n_embed=100, n_layers=2, 100K iters

set -e
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/joformer

for KW in 0.1 0.3 1.0; do
    echo "=============================================="
    echo "  kg_weight = $KW"
    echo "=============================================="
    python kg_text_experiment.py \
        --models roformer roformer_kg roformer_text_kg \
        --n_embed 100 --n_layers 2 \
        --max_iters 100000 --eval_interval 5000 \
        --kg_weight $KW
    echo ""
done

echo "ALL DONE"

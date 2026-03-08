#!/bin/bash
# Compare roformer baseline (10 layers) vs 4 look-ahead nocat variants (50 iters) at C=50, 100K iters
# Launch with: nohup bash run_c50_comparison.sh > c50_comparison.log 2>&1 &

set -e
source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead

COMMON="--data_dir look_ahead/data_full --n_embed 50 --block_size 64 --batch_size 64 --lr 2e-4 --max_iters 100000 --eval_interval 10000 --seed 42 --softmax --generate_len 200"

echo "========================================"
echo "Run 1/5: roformer (baseline, 10 separate layers)"
echo "Started: $(date)"
echo "========================================"
python train_wiki_streaming.py train --models roformer --n_layers 10 $COMMON

echo "========================================"
echo "Run 2/5: roformer_look_ahead_nocat (50 iters)"
echo "Started: $(date)"
echo "========================================"
python train_wiki_streaming.py train --models roformer_look_ahead_nocat --n_layers 50 $COMMON

echo "========================================"
echo "Run 3/5: joformer_fixed_look_ahead_nocat (50 iters)"
echo "Started: $(date)"
echo "========================================"
python train_wiki_streaming.py train --models joformer_fixed_look_ahead_nocat --n_layers 50 $COMMON

echo "========================================"
echo "Run 4/5: joformer_learned_look_ahead_nocat (50 iters)"
echo "Started: $(date)"
echo "========================================"
python train_wiki_streaming.py train --models joformer_learned_look_ahead_nocat --n_layers 50 $COMMON

echo "========================================"
echo "Run 5/5: joformer_projected_look_ahead_nocat (50 iters)"
echo "Started: $(date)"
echo "========================================"
python train_wiki_streaming.py train --models joformer_projected_look_ahead_nocat --n_layers 50 $COMMON

echo "========================================"
echo "All runs complete: $(date)"
echo "========================================"

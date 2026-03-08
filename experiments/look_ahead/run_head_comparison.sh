#!/bin/bash
# Head variant comparison: how to combine processed_x[t] and correction[t]
# All use D=1 K=10, C=50, block_size=256, vocab=16000, 10K iters
# Single invocation so all results saved in one JSON file.

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_look_ahead_nocat roformer_look_ahead_corrhead roformer_look_ahead_addhead roformer_look_ahead_projhead roformer_look_ahead \
  --n_layers 10 \
  --n_embed 50 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 10000 --softmax --convergence_weight 0.1 --data_dir look_ahead/data_full --eval_interval 2000

echo "=========================================="
echo "Head comparison complete."
echo "=========================================="

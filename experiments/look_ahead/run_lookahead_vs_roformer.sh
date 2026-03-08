#!/bin/bash
# Fair comparison: D=1 K=10 look-ahead (concat) vs roformer N=3
# C=300, vocab=8000, block_size=256, batch_size=64, ~8M params each
# Data: look_ahead/data_v8k (vocab=8000, full wiki)

set -e
cd /home/ubuntu/look_ahead
source /home/ubuntu/exp8/venv/bin/activate

DATA_DIR="look_ahead/data_v8k"

# Step 0: Preprocess data with vocab=8000 if not already done
if [ ! -f "$DATA_DIR/wiki_tokens.bin" ]; then
    echo "=========================================="
    echo "Preprocessing: BPE vocab=8000, full wiki"
    echo "=========================================="
    python /home/ubuntu/look_ahead3/train_wiki_streaming.py preprocess \
        --vocab_size 8000 \
        --data_dir "$DATA_DIR"
fi

COMMON="--n_embed 300 --block_size 256 --batch_size 64 --lr 0.0002 --max_iters 100000 --softmax --convergence_weight 0.1 --data_dir $DATA_DIR --eval_interval 5000"

echo "=========================================="
echo "Experiment 1: Roformer N=3 baseline (~8.06M params)"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer \
  --n_layers 3 \
  $COMMON

echo "=========================================="
echo "Experiment 2: D=1 K=10 look-ahead concat (~8.29M params)"
echo "=========================================="
python /home/ubuntu/look_ahead3/train_wiki_streaming.py train \
  --models roformer_look_ahead \
  --n_layers 10 \
  $COMMON

echo "=========================================="
echo "All experiments complete."
echo "=========================================="

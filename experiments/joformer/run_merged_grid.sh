#!/bin/bash
# Grid sweep: joformer_projected_merged only
# Same settings as grid v3: softmax, lr=2e-4, 200k iters, full wiki
# 9 configs x 3 vocab sizes = 27 runs

cd /home/ubuntu/joformer
source /home/ubuntu/exp8/venv/bin/activate

MODEL="--models joformer_projected_merged"
COMMON="--softmax --lr 2e-4 --max_iters 200000"

# --- vocab=8000 ---
DATA="--data_dir /home/ubuntu/joformer/data_v8k"
echo "=== vocab=8000 ==="

echo "=== v8k n100 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 2
echo "=== v8k n100 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 4
echo "=== v8k n100 L6 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 6
echo "=== v8k n100 L8 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 8
echo "=== v8k n200 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 2
echo "=== v8k n200 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 4
echo "=== v8k n200 L6 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 6
echo "=== v8k n500 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 500 --n_layers 2
echo "=== v8k n500 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 500 --n_layers 4

echo "=== vocab=8000 DONE ==="

# --- vocab=16000 ---
DATA="--data_dir /home/ubuntu/OTHER_STUFF/joformer/data_full"
echo "=== vocab=16000 ==="

echo "=== v16k n100 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 2
echo "=== v16k n100 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 4
echo "=== v16k n100 L6 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 6
echo "=== v16k n100 L8 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 8
echo "=== v16k n200 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 2
echo "=== v16k n200 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 4
echo "=== v16k n200 L6 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 6
echo "=== v16k n500 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 500 --n_layers 2
echo "=== v16k n500 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 500 --n_layers 4

echo "=== vocab=16000 DONE ==="

# --- vocab=32000 ---
DATA="--data_dir /home/ubuntu/OTHER_STUFF/joformer/data_full_v32k"
echo "=== vocab=32000 ==="

echo "=== v32k n100 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 2
echo "=== v32k n100 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 4
echo "=== v32k n100 L6 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 6
echo "=== v32k n100 L8 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 100 --n_layers 8
echo "=== v32k n200 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 2
echo "=== v32k n200 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 4
echo "=== v32k n200 L6 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 200 --n_layers 6
echo "=== v32k n500 L2 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 500 --n_layers 2
echo "=== v32k n500 L4 ==="
python -u train_wiki_streaming.py train $COMMON $MODEL $DATA --n_embed 500 --n_layers 4

echo "=== vocab=32000 DONE ==="
echo "=== Full grid complete ==="

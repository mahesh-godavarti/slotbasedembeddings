#!/bin/bash
# Wait for random K experiment to finish, then launch K=5 training
echo "Waiting for random K experiment to finish..."
while pgrep -f "train_wiki_streaming.py.*k_min" > /dev/null 2>&1; do
    sleep 30
done
echo "Random K experiment done. Launching K=5 training..."
cd /home/ubuntu/look_ahead && source /home/ubuntu/exp8/venv/bin/activate && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/ubuntu/look_ahead5/train_wiki_streaming.py train --data_dir /home/ubuntu/look_ahead/look_ahead/data_full --models block_head_corr_ffn --n_embed 50 --n_layers 5 --block_size 256 --batch_size 64 --max_iters 100000 --eval_interval 5000 --softmax --lr 0.0002 --amp

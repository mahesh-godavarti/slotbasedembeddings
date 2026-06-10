#!/bin/bash
set -e

echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then
        echo "GPU 3 is free (${MEM}MiB used). Starting."
        break
    fi
    sleep 60
done

cd /home/ubuntu/pos_agnostic

# joformer2 (Q/K/V, tanh*pi, split_angles) from scratch
# main lr=5e-4, angle_lr=5e-5
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 \
    --lr 5e-4 --angle_lr 5e-5 --max_iters 100000 --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models joformer2 \
    --checkpoint_dir checkpoints/joformer2_scratch_slowangle \
    --gpu 3 \
    >> logs/pafl_joformer2_scratch_slowangle.log 2>&1 &

echo "Launched joformer2 from scratch (lr=5e-4, angle_lr=5e-5) on GPU 3"

# After joformer2 finishes, run monoidal2 (Q/K only)
echo "Waiting for joformer2 to finish..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    # Wait for it to start then finish
    sleep 60
    if [ "$MEM" -lt 1000 ]; then
        # Check if the joformer2 log has "final" in it
        if grep -q "final" logs/pafl_joformer2_scratch_slowangle.log 2>/dev/null; then
            echo "joformer2 done. Starting monoidal2."
            break
        fi
    fi
done

nohup /home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 \
    --lr 5e-4 --angle_lr 5e-5 --max_iters 100000 --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --window_size 999999 --dropout 0.1 --bf16 \
    --models monoidal2 \
    --checkpoint_dir checkpoints/monoidal2_scratch_slowangle \
    --gpu 3 \
    >> logs/pafl_monoidal2_scratch_slowangle.log 2>&1 &

echo "Launched monoidal2 from scratch (lr=5e-4, angle_lr=5e-5) on GPU 3"

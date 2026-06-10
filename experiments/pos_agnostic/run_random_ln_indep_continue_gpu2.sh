#!/bin/bash
set -e

echo "Waiting for GPU 2 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    if [ "$MEM" -lt 1000 ]; then
        echo "GPU 2 is free (${MEM}MiB used). Starting."
        break
    fi
    sleep 60
done

cd /home/ubuntu/pos_agnostic

# Stage 1: 100K→150K at lr=2e-4
echo "=== Stage 1: continuing at lr=2e-4 for 50K ==="
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/new_exp/random_ln_indep_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 --eval_interval 5000 \
    --eval_batch_size 4 --bf16 \
    --checkpoint_dir checkpoints/random_ln_indep_150k \
    --gpu 2

# Stage 2: 150K→200K at lr=5e-5
echo "=== Stage 2: continuing at lr=5e-5 for 50K ==="
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/random_ln_indep_150k/random_ln_indep_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --eval_interval 5000 \
    --eval_batch_size 4 --bf16 \
    --checkpoint_dir checkpoints/random_ln_indep_200k \
    --gpu 2

echo "Done: random_ln_indep_qk 200K"

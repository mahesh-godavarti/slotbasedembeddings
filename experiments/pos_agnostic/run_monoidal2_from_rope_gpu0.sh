#!/bin/bash
set -e

echo "Waiting for GPU 0 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 1000 ]; then
        echo "GPU 0 is free (${MEM}MiB used). Starting."
        break
    fi
    sleep 60
done

cd /home/ubuntu/pos_agnostic

# Convert 150K RoPE to monoidal2 (split_angles, tanh*pi, zero-init angles)
echo "=== Converting RoPE 150K to monoidal2 ==="
mkdir -p checkpoints/monoidal2_from_rope_150k
/home/ubuntu/exp8/venv/bin/python convert_rope_to_monoidal2.py \
    --src checkpoints/scale_up_continue_rope/rope_best.pt \
    --dst checkpoints/monoidal2_from_rope_150k/monoidal2_converted.pt

# Continue as monoidal2 for 50K at lr=5e-5 (both main and angle lr)
echo "=== Continuing as monoidal2 at lr=5e-5 for 50K ==="
nohup /home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/monoidal2_from_rope_150k/monoidal2_converted.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --angle_lr 5e-5 --max_iters 50000 --eval_interval 5000 \
    --eval_batch_size 4 --bf16 \
    --checkpoint_dir checkpoints/monoidal2_from_rope_200k \
    --gpu 0 \
    >> logs/pafl_monoidal2_from_rope_200k.log 2>&1 &

echo "Launched monoidal2 from RoPE continuation on GPU 0"

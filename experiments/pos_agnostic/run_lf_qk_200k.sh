#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

# Wait for the 150K run to finish on GPU 1
echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

echo "GPU 1 free, launching lf_qk 150K→200K at lr=5e-5"
nohup /home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_lf_qk_h1_150k_v2/shared_lf_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-5 --max_iters 50000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 10000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_lf_qk_h1_200k \
    --gpu 1 \
    >> logs/pafl_shared_lf_qk_h1_200k.log 2>&1 &
echo "Launched lf_qk 200K PID: $!"

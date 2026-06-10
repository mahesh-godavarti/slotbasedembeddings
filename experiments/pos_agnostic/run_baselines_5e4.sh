#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

COMMON="--data_dir /home/ubuntu/look_ahead/look_ahead/data_owt --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-4 --max_iters 200000 --eval_interval 5000 --extrap_interval 5000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16"

# GPU 0: RoPE
echo "Waiting for GPU 0 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
echo "Launching RoPE 5e-4 200K on GPU 0"
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models rope \
    --checkpoint_dir checkpoints/rope_5e4_200k \
    --gpu 0 \
    >> logs/pafl_rope_5e4_200k.log 2>&1 &
echo "RoPE PID: $!"

# GPU 1: joformer_fixed
echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
echo "Launching joformer_fixed 5e-4 200K on GPU 1"
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models joformer_fixed \
    --checkpoint_dir checkpoints/jfixed_5e4_200k \
    --gpu 1 \
    >> logs/pafl_jfixed_5e4_200k.log 2>&1 &
echo "jfixed PID: $!"

# GPU 2: random_ln_indep_qk
echo "Waiting for GPU 2 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
echo "Launching random_ln_indep_qk 5e-4 200K on GPU 2"
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models random_ln_indep_qk \
    --checkpoint_dir checkpoints/random_indep_5e4_200k \
    --gpu 2 \
    >> logs/pafl_random_indep_5e4_200k.log 2>&1 &
echo "random PID: $!"

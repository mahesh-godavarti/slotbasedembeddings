#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

COMMON="--data_dir /home/ubuntu/look_ahead/look_ahead/data_owt --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-4 --max_iters 100000 --eval_interval 5000 --extrap_interval 10000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16 --angle_hidden_mult 1 --detach_v"

# GPU 0: lf_qkv detach_v (learned freq, Uniform noise, V rotation, detached)
echo "Waiting for GPU 0 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_lf_qkv \
    --checkpoint_dir checkpoints/shared_lf_qkv_h1_detachv \
    --gpu 0 \
    >> logs/pafl_shared_lf_qkv_h1_detachv.log 2>&1 &
echo "Launched shared_lf_qkv detach_v on GPU 0"

# GPU 1: lfb_qkv detach_v (learned freq, Bernoulli noise, V rotation, detached)
echo "Waiting for GPU 1 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_lfb_qkv \
    --checkpoint_dir checkpoints/shared_lfb_qkv_h1_detachv \
    --gpu 1 \
    >> logs/pafl_shared_lfb_qkv_h1_detachv.log 2>&1 &
echo "Launched shared_lfb_qkv detach_v on GPU 1"

# GPU 2: detb_qkv detach_v (fixed freq, Bernoulli noise, V rotation, detached)
echo "Waiting for GPU 2 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_detb_qkv \
    --checkpoint_dir checkpoints/shared_detb_qkv_detachv \
    --gpu 2 \
    >> logs/pafl_shared_detb_qkv_detachv.log 2>&1 &
echo "Launched shared_detb_qkv detach_v on GPU 2"

# GPU 3: lfds_qkv detach_v (learned freq, fixed signs, V rotation, detached)
echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    $COMMON --models shared_lfds_qkv \
    --checkpoint_dir checkpoints/shared_lfds_qkv_h1_detachv \
    --gpu 3 \
    >> logs/pafl_shared_lfds_qkv_h1_detachv.log 2>&1 &
echo "Launched shared_lfds_qkv detach_v on GPU 3"

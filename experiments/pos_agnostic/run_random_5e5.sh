#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 2 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

echo "Launching random_ln_indep_qk on GPU 2"
nohup /home/ubuntu/exp8/venv/bin/python train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt --n_embed 768 --n_layers 16 --n_heads 8 --block_size 512 --batch_size 32 --lr 5e-5 --max_iters 200000 --eval_interval 5000 --extrap_interval 5000 --eval_lengths 512,1024,2048,4096,8192,16384 --window_size 999999 --dropout 0.1 --bf16 \
    --models random_ln_indep_qk \
    --checkpoint_dir checkpoints/random_indep_5e5_200k \
    --gpu 2 \
    >> logs/pafl_random_indep_5e5_200k.log 2>&1 &
echo "random_ln_indep_qk PID: $!"

#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "Waiting for GPU 3 to be free..."
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$MEM" -lt 1000 ]; then break; fi
    sleep 60
done

echo "Extending pemb_qk to 200K on GPU 3"
/home/ubuntu/exp8/venv/bin/python continue_training.py \
    --checkpoint checkpoints/shared_pemb_qk/shared_pemb_qk_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 5e-4 --max_iters 100000 --batch_size 32 \
    --eval_interval 5000 --extrap_interval 5000 \
    --eval_lengths 512,1024,2048,4096,8192,16384 \
    --bf16 \
    --checkpoint_dir checkpoints/shared_pemb_qk_200k \
    --gpu 3
echo "Done."

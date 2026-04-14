#!/bin/bash
# Launcher for both training jobs. Designed to be called from cron @reboot
# or manually. Each model runs in a detached screen session.
# Auto-resumes from checkpoint if available.

cd /home/ubuntu/look_ahead6
mkdir -p logs checkpoints

# Wait for GPUs to be ready
for i in $(seq 1 30); do
    nvidia-smi > /dev/null 2>&1 && break
    sleep 2
done

# Kill any existing training processes
pkill -f "train_wiki_streaming.py" 2>/dev/null
sleep 2

# Remove stale iter-0-only checkpoints (< 10MB = not real training progress)
for f in checkpoints/*_latest.pt; do
    [ -f "$f" ] || continue
    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
    # A real checkpoint with training progress will be large; iter-0 ones are also large
    # so we can't filter by size. Just keep all checkpoints.
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
SCRIPT=/home/ubuntu/look_ahead6/train_wiki_streaming.py
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
CKPT=/home/ubuntu/look_ahead6/checkpoints
TS=$(date +%Y%m%d_%H%M%S)

# GPU 0: roformer N=24 C=1024 h16
ROFORMER_LOG=/home/ubuntu/look_ahead6/logs/roformer_n24_c1024_h16_owt_${TS}.log
screen -dmS roformer bash -c "
  $PYTHON $SCRIPT train \
    --models roformer --n_embed 1024 --n_layers 24 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir $CKPT \
    --gpu 0 \
    --amp 2>&1 | tee $ROFORMER_LOG
"

# GPU 1: corr_ffn_add D=12 C=1024 h16
CORRFFN_LOG=/home/ubuntu/look_ahead6/logs/corr_ffn_add_d12_c1024_h16_owt_${TS}.log
screen -dmS corr_ffn bash -c "
  $PYTHON $SCRIPT train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 60 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 12 --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir $CKPT \
    --gpu 1 \
    --amp 2>&1 | tee $CORRFFN_LOG
"

echo "$(date): Launched training sessions (roformer → $ROFORMER_LOG, corr_ffn → $CORRFFN_LOG)" >> /home/ubuntu/look_ahead6/logs/launcher.log

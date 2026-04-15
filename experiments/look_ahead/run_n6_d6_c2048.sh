#!/bin/bash
# N=6 C=2048 vs D=6 C=2048 -- FLOP-matched to N=24 C=1024
# N=6 C=2048: 72 * 2048^2 = 288 * 1024^2 = same FLOPs as N=24 C=1024
# D=6 C=2048: (72+8) * 2048^2 = 80 * 2048^2 at inference (11% more than N=6)
#
# Step 1: Train N=6 C=2048 for 200K iters at batch=32
# Step 2: Convert to D=6, fine-tune at K=2-5
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
LR=2e-4

# Step 1: N=6 C=2048
echo "$(date): Starting N=6 C=2048"
mkdir -p checkpoints_n6_c2048
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 2048 --n_layers 6 --block_size 256 --batch_size 32 \
    --lr $LR --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_n6_c2048 \
    --gpu $1 \
    --amp 2>&1 | tee logs/roformer_n6_c2048_h16_owt.log
echo "$(date): Finished N=6 C=2048"

# Step 2: Convert to D=6
echo "$(date): Converting N=6 to D=6 C=2048"
mkdir -p checkpoints_d6_c2048
$PYTHON convert_roformer_to_lookahead.py \
    --roformer_ckpt checkpoints_n6_c2048/roformer_latest.pt \
    --output_ckpt checkpoints_d6_c2048/block_head_corr_ffn_add_latest.pt \
    --n_embed 2048 --n_layers 30 --d_block 6 --n_head 16 \
    --block_size 256 --vocab_size 32000

# Step 3: Fine-tune D=6 at K=2-5 for 50K iters
echo "$(date): Starting D=6 C=2048 fine-tune"
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2048 --n_layers 30 --block_size 256 --batch_size 32 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 6 --n_head 16 --k_min 2 \
    --max_iters 50000 --eval_interval 2000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_d6_c2048 \
    --gpu $1 \
    --amp 2>&1 | tee logs/finetune_d6_c2048_h16_owt.log
echo "$(date): Finished D=6 C=2048 fine-tune"

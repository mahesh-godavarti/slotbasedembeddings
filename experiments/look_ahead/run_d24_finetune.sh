#!/bin/bash
# Convert roformer N=24 to look-ahead D=24, fine-tune at K=2-5 on GPU 1
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/ubuntu/look_ahead6/flash_override:$PYTHONPATH
PYTHON=/home/ubuntu/exp8/venv/bin/python

# Step 1: Convert roformer N=24 checkpoint to D=24 look-ahead
mkdir -p checkpoints_d24_converted
$PYTHON convert_roformer_to_lookahead.py \
    --roformer_ckpt checkpoints/roformer_latest.pt \
    --output_ckpt checkpoints_d24_converted/block_head_corr_ffn_add_latest.pt \
    --n_embed 1024 --n_layers 120 --d_block 24 --n_head 16 \
    --block_size 256 --vocab_size 32000

# Step 2: Fine-tune at K=2-5
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 120 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 24 --n_head 16 \
    --k_schedule "0:2-5" \
    --max_iters 50000 --eval_interval 2000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d24_converted \
    --gpu 1 \
    --amp 2>&1 | tee logs/corr_ffn_add_d24_c1024_h16_finetune_owt.log

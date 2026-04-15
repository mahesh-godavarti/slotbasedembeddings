#!/bin/bash
# Ablation: block_head (no corr_ffn) fine-tune from N=24
# Tests whether iteration alone helps, or if corr_ffn is essential
# GPU 0
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt

# Convert N=24 to block_head D=24 (no corr_ffn)
echo "$(date): Converting N=24 to block_head D=24"
mkdir -p checkpoints_blockhead_d24
$PYTHON convert_roformer_to_blockhead.py \
    --roformer_ckpt checkpoints/roformer_latest.pt \
    --output_ckpt checkpoints_blockhead_d24/block_head_latest.pt \
    --n_embed 1024 --n_layers 120 --d_block 24 --n_head 16 \
    --block_size 256 --vocab_size 32000

# Fine-tune at K=2-4 (same as D=24 corr_ffn_add experiment)
echo "$(date): Starting block_head D=24 fine-tune (no corr_ffn)"
$PYTHON train_wiki_streaming.py train \
    --models block_head --n_embed 1024 --n_layers 120 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 24 --n_head 16 --k_min 2 \
    --max_iters 50000 --eval_interval 2000 \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_blockhead_d24 \
    --gpu 0 \
    --amp 2>&1 | tee logs/blockhead_d24_c1024_finetune.log
echo "$(date): Finished block_head D=24 fine-tune"

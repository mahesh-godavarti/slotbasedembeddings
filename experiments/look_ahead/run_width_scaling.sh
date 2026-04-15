#!/bin/bash
# Width scaling: N=2 vs D=2 at C=256, 512
# We already have C=1024 from the scaling experiment.
# Token budgets: Chinchilla 20:1 ratio.
# All on GPU 1.
set -e
cd /home/ubuntu/look_ahead6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/ubuntu/exp8/venv/bin/python
DATA=/home/ubuntu/look_ahead/look_ahead/data_owt
GPU=1
BS=64
LR=2e-4
EVAL=500

# C=256, N=2: ~25M params, 500M optimal tokens
# batch=1024, tokens/iter=1024*64=65536, iters=500M/65536=7629
echo "$(date): === C=256 N=2 ==="
mkdir -p checkpoints_width_n2_c256
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 256 --n_layers 2 --block_size $BS --batch_size 1024 \
    --lr $LR --softmax --n_head 4 \
    --max_iters 7629 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_n2_c256 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c256.log
echo "$(date): Finished N=2 C=256"

# Convert and fine-tune D=2 C=256: 125M tokens = 1907 iters at batch=1024
echo "$(date): === C=256 D=2 ==="
mkdir -p checkpoints_width_d2_c256
$PYTHON convert_roformer_to_lookahead.py \
    --roformer_ckpt checkpoints_width_n2_c256/roformer_latest.pt \
    --output_ckpt checkpoints_width_d2_c256/block_head_corr_ffn_add_latest.pt \
    --n_embed 256 --n_layers 10 --d_block 2 --n_head 4 \
    --block_size $BS --vocab_size 32000
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 256 --n_layers 10 --block_size $BS --batch_size 512 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 2 --n_head 4 --k_min 2 \
    --max_iters 3815 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d2_c256 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_finetune_d2_c256.log
echo "$(date): Finished D=2 C=256"

# C=512, N=2: ~45M params, 900M optimal tokens
# batch=512, tokens/iter=512*64=32768, iters=900M/32768=27466
echo "$(date): === C=512 N=2 ==="
mkdir -p checkpoints_width_n2_c512
$PYTHON train_wiki_streaming.py train \
    --models roformer --n_embed 512 --n_layers 2 --block_size $BS --batch_size 512 \
    --lr $LR --softmax --n_head 8 \
    --max_iters 27466 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_n2_c512 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_roformer_n2_c512.log
echo "$(date): Finished N=2 C=512"

# Convert and fine-tune D=2 C=512: 225M tokens = 6867 iters at batch=512
echo "$(date): === C=512 D=2 ==="
mkdir -p checkpoints_width_d2_c512
$PYTHON convert_roformer_to_lookahead.py \
    --roformer_ckpt checkpoints_width_n2_c512/roformer_latest.pt \
    --output_ckpt checkpoints_width_d2_c512/block_head_corr_ffn_add_latest.pt \
    --n_embed 512 --n_layers 10 --d_block 2 --n_head 8 \
    --block_size $BS --vocab_size 32000
$PYTHON train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 512 --n_layers 10 --block_size $BS --batch_size 256 \
    --lr $LR --softmax --convergence_weight 0.1 --d_block 2 --n_head 8 --k_min 2 \
    --max_iters 13733 --eval_interval $EVAL \
    --data_dir $DATA \
    --checkpoint_dir checkpoints_width_d2_c512 \
    --gpu $GPU \
    --amp 2>&1 | tee logs/width_finetune_d2_c512.log
echo "$(date): Finished D=2 C=512"

echo "$(date): All width scaling experiments complete."

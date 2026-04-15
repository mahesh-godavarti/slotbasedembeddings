#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for N=1 C=2048 bs1024 to finish..."
while pgrep -f 'n_embed 2048.*n_layers 1.*block_size 1024' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): N=1 done. Converting to D=1 K=10."

# Convert
mkdir -p checkpoints_d1_c2048_bs1024_k10_ft
/home/ubuntu/exp8/venv/bin/python convert_roformer_to_lookahead.py \
    --roformer_ckpt checkpoints_n1_c2048_bs1024/roformer_latest.pt \
    --output_ckpt checkpoints_d1_c2048_bs1024_k10_ft/block_head_corr_ffn_add_latest.pt \
    --n_embed 2048 --n_layers 10 --d_block 1 --n_head 16 \
    --block_size 1024 --vocab_size 32000

echo "$(date): Conversion done. Starting D=1 K=10 fine-tune for 300K."

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2048 --n_layers 10 --block_size 1024 --batch_size 16 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 0 \
    --max_iters 400000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d1_c2048_bs1024_k10_ft \
    --gpu 1 \
    --amp 2>&1 | tee logs/d1_c2048_bs1024_k10_ft.log
echo "$(date): Finished D=1 K=10 fine-tune"

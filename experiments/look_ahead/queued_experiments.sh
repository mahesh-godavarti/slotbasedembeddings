#!/bin/bash
# Queued experiments: D>2 corr_ffn_add and corr_ffn_concat at C=446 K=5 k_min=2
# Run one at a time as GPUs become free

# corr_ffn_add D=4 C=446 K=5 (56C² — between roformer N=4 and N=5)
cd /home/ubuntu/look_ahead && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train --models block_head_corr_ffn_add --n_embed 446 --n_layers 20 --block_size 256 --batch_size 64 --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 4 --max_iters 100000 --eval_interval 5000 --data_dir /home/ubuntu/look_ahead/look_ahead/data_full --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_add_d4_c446_k5.log

# corr_ffn_add D=5 C=446 K=5 (68C² — between roformer N=5 and N=6)
cd /home/ubuntu/look_ahead && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train --models block_head_corr_ffn_add --n_embed 446 --n_layers 25 --block_size 256 --batch_size 64 --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 5 --max_iters 100000 --eval_interval 5000 --data_dir /home/ubuntu/look_ahead/look_ahead/data_full --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_add_d5_c446_k5.log

# corr_ffn_add D=6 C=446 K=5 (80C² — = roformer_head_ffn N=6)
cd /home/ubuntu/look_ahead && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train --models block_head_corr_ffn_add --n_embed 446 --n_layers 30 --block_size 256 --batch_size 64 --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 6 --max_iters 100000 --eval_interval 5000 --data_dir /home/ubuntu/look_ahead/look_ahead/data_full --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_add_d6_c446_k5.log

# corr_ffn_concat D=4 C=446 K=5 (60C² — = roformer N=5)
cd /home/ubuntu/look_ahead && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train --models block_head_corr_ffn_concat --n_embed 446 --n_layers 20 --block_size 256 --batch_size 64 --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 4 --max_iters 100000 --eval_interval 5000 --data_dir /home/ubuntu/look_ahead/look_ahead/data_full --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_concat_d4_c446_k5.log

# corr_ffn_concat D=5 C=446 K=5 (72C² — = roformer N=6)
cd /home/ubuntu/look_ahead && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/ubuntu/exp8/venv/bin/python /home/ubuntu/look_ahead6/train_wiki_streaming.py train --models block_head_corr_ffn_concat --n_embed 446 --n_layers 25 --block_size 256 --batch_size 64 --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 2 --d_block 5 --max_iters 100000 --eval_interval 5000 --data_dir /home/ubuntu/look_ahead/look_ahead/data_full --amp 2>&1 | tee /home/ubuntu/look_ahead6/logs/corr_ffn_concat_d5_c446_k5.log

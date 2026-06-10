#!/bin/bash
set -e
cd /home/ubuntu/pos_agnostic

echo "$(date): Full extrapolation eval on GPU 2 (eval_iters=100, batch=4)"

/home/ubuntu/exp8/venv/bin/python eval_extrap.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --eval_lengths 65536 \
    --eval_batch_size 2 \
    --eval_iters 20 \
    --gpu 2 \
    --checkpoints \
        checkpoints/rope_200k/rope_best.pt \
        checkpoints/joformer_fixed_200k/joformer_fixed_150k.pt_best.pt \
        checkpoints/shared_pemb_qk_sched_200k/shared_pemb_qk_best.pt \
        checkpoints/shared_pemb_qkv_sched_200k/shared_pemb_qkv_best.pt \
        checkpoints/shared_cbd_qk_K4_sched_200k/shared_cbd_qk_best.pt \
        checkpoints/shared_cbd_qkv_K4_sched_200k/shared_cbd_qkv_best.pt \
        checkpoints/shared_pmlp_qk_sched_200k/shared_pmlp_qk_best.pt \
        checkpoints/shared_pmlp_qkv_sched_200k/shared_pmlp_qkv_best.pt \
        checkpoints/random_ln_indep_200k/random_ln_indep_qk_best.pt \
        checkpoints/random_indep_qkv_sched_200k/random_ln_indep_qkv_best.pt \
        checkpoints/shared_lf_qk_h1_200k/shared_lf_qk_best.pt

echo "$(date): Done."

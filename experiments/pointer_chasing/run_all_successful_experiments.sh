#!/bin/bash
# All successful pointer chasing experiments — commands that produced key results
#
# Usage: Uncomment the experiment you want to run, set GPU variable
# Each experiment is independent — run one at a time per GPU

source /home/ubuntu/exp8/venv/bin/activate
cd /home/ubuntu/look_ahead6

GPU=0

# ============================================================================
# 1. WINDOWED NO-SHUFFLE STAIRCASE (e=256, lr=1e-4) — CLEAN DEPTH SEPARATION
# ============================================================================
# Result: N=1→1, N=3→3, N=5→6, N=10→7, N=11→8, N=12→all 11 levels
# Time: ~50K iters each, N=12 solves at 50K

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 5 --n_values 10 \
#     --n_embed 256 --n_head 4 --n_iters 50000 --batch_size 64 --lr 1e-4 \
#     --gpu $GPU --permutation --run N1,N3,N5,N10,N11,N12 \
#     --window 38 --no_shuffle \
#     --checkpoint_dir checkpoints_staircase_e256 \
#     2>&1 | tee logs/staircase_e256_50k.log

# ============================================================================
# 2. BPTT WINDOWED (e=128, lr=1e-3) — SOLVES ALL 11 LEVELS IN 20K
# ============================================================================
# Result: 100% all levels at ~20K iters
# Critical: window must be passed to BPTT model

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run bptt \
#     --window 38 --no_shuffle \
#     --checkpoint_dir checkpoints_bptt_e128_windowed \
#     2>&1 | tee logs/bptt_e128_windowed_100k.log

# ============================================================================
# 3. BPTT WINDOWED (e=256, lr=1e-4) — SOLVES ALL 11 LEVELS IN 16K
# ============================================================================

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 5 --n_values 10 \
#     --n_embed 256 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-4 \
#     --gpu $GPU --permutation --run bptt \
#     --window 38 --no_shuffle \
#     --checkpoint_dir checkpoints_bptt_e256_windowed \
#     2>&1 | tee logs/bptt_e256_windowed_100k.log

# ============================================================================
# 4. BPTT WINDOWED k=10 (e=128, lr=1e-3) — LARGER KEY SPACE
# ============================================================================
# Result: 99.7% all levels at ~23K. Window=52 for k=10.

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 10 --n_values 20 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run bptt \
#     --window 52 --no_shuffle \
#     --checkpoint_dir checkpoints_bptt_k10_windowed \
#     2>&1 | tee logs/bptt_k10_windowed_100k.log

# ============================================================================
# 5. SHUFFLE 2-HOP k=5 N=3 — COMPOSITION WITH SHUFFLING (77K iters)
# ============================================================================
# Result: L0-L2=100% at 77K iters (first shuffled composition success)

# python -u pointer_chasing.py \
#     --n_hops 2 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N3 \
#     --checkpoint_dir checkpoints_2hop_k5_shuffle \
#     2>&1 | tee logs/2hop_k5_shuffle_N3_100k.log

# ============================================================================
# 6. SHUFFLE 3-HOP k=2 N=4 — 3-HOP WORKS AT k=2 (11.5K iters)
# ============================================================================

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 2 --n_values 4 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N4 \
#     2>&1 | tee logs/3hop_k2_shuffle_N4_100k.log

# ============================================================================
# 7. HYBRID RoPE+NoPE — SHUFFLE 3-HOP SOLVED AT 9K
# ============================================================================
# Result: 100% all levels at 9K iters
# N=6: 3 RoPE layers (window=3) + 3 NoPE layers (window=54)

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N6 \
#     --window 3 --hybrid_k 3 --hybrid_window 54 \
#     --checkpoint_dir checkpoints_hybrid_3hop \
#     2>&1 | tee logs/hybrid_3hop_N6_100k.log

# ============================================================================
# 8. ADAPTIVE CURRICULUM — SHUFFLE 3-HOP k=5 SOLVED AT 69K ★★★
# ============================================================================
# Result: 100% all levels at 69K iters
# Adaptive per-level key curriculum: waits for 90% accuracy before advancing k
# L1: k=2→5 in 10K. L2: k=2→5 in 48.5K. Converged at 69K.

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 500000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N4 \
#     --no_rope --window 38 \
#     --adaptive_curriculum --adaptive_threshold 0.9 --adaptive_consecutive 3 \
#     --checkpoint_dir checkpoints_adaptive_3hop \
#     2>&1 | tee logs/adaptive_3hop_N4_500k.log

# ============================================================================
# 9. ADAPTIVE CURRICULUM — SHUFFLE 4-HOP (IN PROGRESS)
# ============================================================================

# python -u pointer_chasing.py \
#     --n_hops 4 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 500000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N5 \
#     --no_rope --window 38 \
#     --adaptive_curriculum --adaptive_threshold 0.9 --adaptive_consecutive 3 \
#     --checkpoint_dir checkpoints_adaptive_4hop \
#     2>&1 | tee logs/adaptive_4hop_N5_500k.log

# ============================================================================
# 10. NO-SHUFFLE WINDOWED 3-HOP — QUICK SOLVE (instant)
# ============================================================================

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 200000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N4 \
#     --window 38 --no_shuffle \
#     --checkpoint_dir checkpoints_3hop_noshuffle_windowed \
#     2>&1 | tee logs/3hop_noshuffle_windowed_N4_200k.log

# ============================================================================
# 11. ROPE VALIDATION — THREE DIAGNOSTIC TASKS
# ============================================================================
# min_element: RoPE=100% no-RoPE=100% (order-invariant)
# copy_back2:  RoPE=100% no-RoPE=98% (positional)
# left_neighbor: RoPE=100% no-RoPE=99.6% (content+positional)

# python -u min_element.py --V 20 --N 10 --n_embed 128 --n_head 4 --n_layers 3 \
#     --n_iters 5000 --batch_size 64 --lr 1e-3 --gpu $GPU

# python -u copy_back2.py --V 20 --N 10 --n_embed 128 --n_head 4 --n_layers 3 \
#     --n_iters 5000 --batch_size 64 --lr 1e-3 --gpu $GPU

# python -u left_neighbor.py --V 20 --N 10 --n_embed 128 --n_head 4 --n_layers 3 \
#     --n_iters 10000 --batch_size 64 --lr 1e-3 --gpu $GPU

# ============================================================================
# 12. MULTI-Q HELPER SECTIONS — L1 BREAKTHROUGH FOR 3-HOP
# ============================================================================
# Result: L1=100% at 80K for 3-hop N=4 (first time without curriculum)

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N4 --multi_q \
#     --checkpoint_dir checkpoints_3hop_multiq \
#     2>&1 | tee logs/3hop_k5_multiq_N4_100k.log

# ============================================================================
# 13. DATADEP VARIANTS — DATA-DEPENDENT POSITIONAL ENCODING
# ============================================================================
# Available: datadep, datadepv, monoidal, joformer, datadep2, monoidal2, joformer2
# None solved shuffled 3-hop k=5 without curriculum

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run datadepv_N4 \
#     --window 38 \
#     --checkpoint_dir checkpoints_datadepv \
#     2>&1 | tee logs/datadepv_N4_100k.log

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
# 2. WINDOWED NO-SHUFFLE STAIRCASE (e=128, lr=1e-3) — ALSO WORKS
# ============================================================================
# Result: N=1→1, N=3→3, N=5→4 (e=128 slightly worse staircase)
# Time: 200K iters each

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 200000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N1,N3,N5 \
#     --window 38 --no_shuffle \
#     --checkpoint_dir checkpoints_staircase_e128 \
#     2>&1 | tee logs/staircase_e128_200k.log

# ============================================================================
# 3. BPTT WINDOWED (e=128, lr=1e-3) — SOLVES ALL 11 LEVELS IN 20K
# ============================================================================
# Result: 100% all levels at ~20K iters
# Critical: window must be passed to BPTT (was a bug — see notes)

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run bptt \
#     --window 38 --no_shuffle \
#     --checkpoint_dir checkpoints_bptt_e128_windowed \
#     2>&1 | tee logs/bptt_e128_windowed_100k.log

# ============================================================================
# 4. BPTT WINDOWED (e=256, lr=1e-4) — SOLVES ALL 11 LEVELS IN 16K
# ============================================================================
# Result: 100% all levels at ~16K iters

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 5 --n_values 10 \
#     --n_embed 256 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-4 \
#     --gpu $GPU --permutation --run bptt \
#     --window 38 --no_shuffle \
#     --checkpoint_dir checkpoints_bptt_e256_windowed \
#     2>&1 | tee logs/bptt_e256_windowed_100k.log

# ============================================================================
# 5. BPTT WINDOWED k=10 (e=128, lr=1e-3) — LARGER KEY SPACE
# ============================================================================
# Result: wave propagates through all levels (slower than k=5)
# Window=52 for k=10 (one level block)

# python -u pointer_chasing.py \
#     --n_hops 10 --n_keys 10 --n_values 20 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run bptt \
#     --window 52 --no_shuffle \
#     --checkpoint_dir checkpoints_bptt_k10_windowed \
#     2>&1 | tee logs/bptt_k10_windowed_100k.log

# ============================================================================
# 6. SHUFFLE 2-HOP k=5 N=3 — COMPOSITION WITH SHUFFLING (77K iters)
# ============================================================================
# Result: L0-L2=100% at 77K iters (the first shuffled composition success)
# No window, no curriculum, just RoPE + shuffle + Q-format

# python -u pointer_chasing.py \
#     --n_hops 2 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N3 \
#     --checkpoint_dir checkpoints_2hop_k5_shuffle \
#     2>&1 | tee logs/2hop_k5_shuffle_N3_100k.log

# ============================================================================
# 7. SHUFFLE 2-HOP k=2 N=3 — EASY SHUFFLED COMPOSITION (1K iters)
# ============================================================================
# Result: 100% at ~1K iters (binary permutations, trivial)

# python -u pointer_chasing.py \
#     --n_hops 2 --n_keys 2 --n_values 4 \
#     --n_embed 128 --n_head 4 --n_iters 5000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N3 \
#     2>&1 | tee logs/2hop_k2_shuffle_N3_5k.log

# ============================================================================
# 8. SHUFFLE 3-HOP k=2 N=4 — 3-HOP WORKS AT k=2 (11.5K iters)
# ============================================================================
# Result: L0-L2=100%, LF=96% at ~11.5K

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 2 --n_values 4 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N4 \
#     2>&1 | tee logs/3hop_k2_shuffle_N4_100k.log

# ============================================================================
# 9. SHUFFLE + CURRICULUM + NO-ROPE + WINDOW — L1 SOLVED AT k=5
# ============================================================================
# Result: L1=100% (first time for shuffled k=5 3-hop!)
# L2=90% at k=2, drops when k increases

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 300000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N4 \
#     --no_rope --window 38 \
#     --hop_curriculum "0:2,100000:3" \
#     --key_curriculum "0:2,20000:5,100000:2,150000:3,200000:4,250000:5" \
#     --checkpoint_dir checkpoints_shuffle_slow_curriculum \
#     2>&1 | tee logs/shuffle_slow_curriculum_300k.log

# ============================================================================
# 10. NO-SHUFFLE WINDOWED 3-HOP — QUICK SOLVE (instant)
# ============================================================================
# Result: 100% all levels almost immediately (positional patterns through window)

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
# 12. LEFT NEIGHBOR MULTI-HOP — ALTERNATIVE TASK (K=8, shuffled)
# ============================================================================
# Result: L1=100%, L2=57%, L3=58% at 16K (promising but K=20 fails)

# python -u left_neighbor_multihop.py \
#     --K 8 --n_hops 3 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --run N3 \
#     2>&1 | tee logs/left_neighbor_K8_3hop_N3_100k.log

# ============================================================================
# 13. MULTI-Q HELPER SECTIONS — L1 BREAKTHROUGH FOR 3-HOP
# ============================================================================
# Result: L1=100% at 80K for 3-hop N=4 (first time without curriculum)
# Multiple Q sections per level: Q1(1-hop), Q2(2-hop), etc.

# python -u pointer_chasing.py \
#     --n_hops 3 --n_keys 5 --n_values 10 \
#     --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
#     --gpu $GPU --permutation --run N4 --multi_q \
#     --checkpoint_dir checkpoints_3hop_multiq \
#     2>&1 | tee logs/3hop_k5_multiq_N4_100k.log

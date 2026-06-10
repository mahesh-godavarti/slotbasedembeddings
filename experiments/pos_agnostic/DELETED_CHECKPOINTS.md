# Deleted Checkpoints

Deleted on 2026-06-08 to free disk space (was 100% full, all experiments crashed).

## Early failures (PPL in thousands, never converged)
- shared_pmlp_qk: iter=5000, val_ppl=10325 (v1, dead gradient from zero-init)
- shared_pmlp_qk_v2: iter=0, val_ppl=36716 (also failed)
- shared_pmlp_qk_v3: iter=2000, val_ppl=891 (also failed)
- shared_lfbf_qk_h1: iter=5000, val_ppl=1363 (tanh*π+rope without LN)
- shared_lfnl_qk_h1: iter=5000, val_ppl=1566 (learned freq, no LN, never converged)
- shared_lfnl_qkv_h1: iter=5000, val_ppl=4275 (learned freq, no LN, never converged)
- joformer2_from_scratch_noanglr: iter=0, val_ppl=38008
- joformer2_from_scratch_v2: iter=0, val_ppl=38033
- monoidal2_from_scratch_noanglr: iter=0, val_ppl=38350
- monoidal2_from_scratch_v2: iter=0, val_ppl=37819
- joformer2_h1_tanh_pi: iter=5000, val_ppl=976
- joformer2_h4_ln_consistent: iter=5000, val_ppl=221
- rope_control: iter=0, val_ppl=37963
- mixed_50k: iter=0, val_ppl=18597
- shared_pcb_qk_K32: iter=0, val_ppl=37857 (K=32 codebook, never started)
- shared_cbd_qk_K4_v2: iter=0, val_ppl=37483

## Short experiments stopped at 5-10K (exploratory, not extended)
- shared_fsnt_qk_h1: iter=5000 (sign+noise+tanh variant)
- shared_fsnt_qkv_h1: iter=5000
- shared_fssd_qk_h1: iter=5000 (sign+detach variant)
- shared_fssr_qk: iter=5000
- shared_fssx_qk: iter=5000
- shared_fssa_qk: iter=10000
- shared_pcb_qk_K4: iter=5000 (factored codebook, superseded by cbd)
- shared_pcb_qk_K4_v2: iter=5000
- shared_pemb_qk_1k_eval: iter=7000 (pemb with 1K eval interval, test run)
- shared_deti_qk_gpu0: iter=5000 (duplicate of deti_qk)
- joformer2_h1_ln_consistent: iter=5000
- joformer2_h1_tanh_freq: iter=5000
- joformer2_h1_tanh_lfreq: iter=5000
- joformer2_h4_tanh_lfreq: iter=5000
- joformer2_h4_tanh_lfreq_frozen_emb: iter=5000
- joformer2_h4_tanh_lfreq_random_emb: iter=5000
- joformer2_h4_lntf_random_emb: iter=5000
- joformer2_frozen_angles: iter=5000
- joformer2_from_frozen_v2: iter=5000
- joformer2_from_frozen_fastangle: iter=0
- monoidal2_from_frozen: iter=0
- monoidal2_from_frozen_fastboth: iter=0
- monoidal2_from_frozen_v2: iter=0
- monoidal2_frozen_angles: iter=5000
- scale_up_continue: iter=5000

## Duplicate/superseded checkpoints
- joformer2_from_fixed_200k_cont: iter=0 (duplicate of joformer2_from_fixed_200k)
- joformer2_from_fixed_200k_cont_v2: iter=0 (duplicate)
- shared_pmlp_qk_v4: iter=4000 (superseded by v5)
- shared_pmlp2_qkv_sched_150k: iter=45000 (intermediate, crashed from disk full)
- shared_rpemb_qk_sched_150k: iter=40000 (intermediate, crashed from disk full)
- shared_pemb2_qk: load failed (crashed at start from disk full)

## Previously deleted (earlier in session)
- shared_pmlp2_qk, shared_pmlp2_qk_100k, shared_pmlp2_qk_sched_150k, shared_pmlp2_qk_sched_200k (pmlp2 stacked corrections, extrap blew up to 1.97x)
- shared_pmlp2_qkv (pmlp2 qkv, extrap blew up to 1.74x)
- shared_rpemb3_qk (rpemb v3, no rope_base, extrap 5.07x at 5K)
- shared_cbd_qk_K8, shared_cbd_qk_K8_200k (K=8 codebook, worse than K=4)
- shared_cbd_qkv_K8 (K=8 qkv)
- shared_pmlp_qk_sched_150k, shared_pmlp_qkv_sched_150k (intermediate schedule checkpoints)
- shared_pemb_qk_sched_150k, shared_pemb_qkv_sched_150k (intermediate)
- shared_cbd_qkv_K4_sched_150k, shared_cbd_qk_K4_sched_150k (intermediate)
- shared_rpemb_qk (v1, no LN, extrap 1.45x at 5K)
- shared_cbd_qk_K32 (K=32, early experiment)

# 20-Hop Experiment Tracking

## Final settings: e=256, lr=1e-4, n_head=4, k=8, v=16, permutation, gradient clipping, blocks2.py

### Completed transformer results

| Model | Levels solved (>90%) | L4 | Status | Source log |
|-------|---------------------|-----|--------|-----------|
| N=1 | L0 (1 level) | 25% | Done | pointer_chasing_20hop_k8_e256_lr4_clip_N_20k.log |
| N=5 | L0-L3 (4 levels) | 25% | Done (converged by 5K) | pointer_chasing_20hop_k8_e256_lr4_clip_N_20k.log |
| N=10 | — | — | Running | pointer_chasing_20hop_k8_e256_lr4_clip_N_rest_20k.log |
| N=15 | — | — | Queued | pointer_chasing_20hop_k8_e256_lr4_clip_N_rest_20k.log |
| N=19 | — | — | Queued | pointer_chasing_20hop_k8_e256_lr4_clip_N_rest_20k.log |
| N=20 | — | — | Queued | pointer_chasing_20hop_k8_e256_lr4_clip_N_rest_20k.log |
| N=21 | — | — | Queued | pointer_chasing_20hop_k8_e256_lr4_clip_N_rest_20k.log |

### BPTT results

| Setting | Status | Best result | Source log |
|---------|--------|-------------|-----------|
| e=256 lr=1e-4 (no clip) | Crashed at 4K | L0-L19=71-100% at 3.5K, then collapsed | pointer_chasing_20hop_k8_e256_lr4_bptt_50k.log |
| e=256 lr=5e-5 (no clip) | Crashed at 6.5K | L0-L19=88-100% at 6K, then collapsed | pointer_chasing_20hop_k8_e256_lr5e5_bptt_50k.log |
| e=256 lr=1e-4 (clipped) | Running | L1=92% at 2K | pointer_chasing_20hop_k8_e256_lr4_clip_bptt_50k.log |
| e=512 lr=1e-4 (no clip) | **Solved** | **100% all 21 levels at 4K** | pointer_chasing_20hop_k8_e512_lr4_bptt_50k.log |

### Expected staircase at e=256

Based on completed results and previous experiments:
- N=1: 1 level
- N=5: 4 levels (confirmed)
- N=10: ~9 levels (expected)
- N=15: ~14 levels (expected)
- N=19: ~18 levels (expected)
- N=20: ~19 levels (expected)
- N=21: ~20 levels (expected)
- BPTT: all 21 levels (confirmed at e=512, in progress at e=256)

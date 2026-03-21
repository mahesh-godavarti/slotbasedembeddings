# Running Processes

## Currently running
| Task ID | Group | Log file | Status |
|---------|-------|----------|--------|
| bqushqulz | Dual I/I'/A' softmax 100K | logs/dual_100K_softmax_I_Ip_Ap.log | Just launched |

Config: n_embed=500, n_layers=2, max_iters=100000, softmax, seeds=1, dual_objective=True

## Completed (100K non-dual)
- bipfyywvw: MLM A/A'/F/F'/G/G' → results in expanded_summary
- bplu6t8p5: MLM E/E'/H/H' → results in expanded_summary
- b7a22655b: Causal Ec/Ec'/Hc/Hc' → results in expanded_summary
- bewn06t5n: kg_as_text B/B'/C/C' → results in expanded_summary
- bt3idr9v6: MLM I/I' softmax → results in expanded_summary

## Completed (dual 100K)
- bb3nybrre: Dual E/E'/H/H'/I/I' softplus → results in dual_summary (I/I' NaN — dead)
- bv7deq00d: Dual B/B'/C/C' (kg_as_text) → results in dual_summary
- begvfszz2: Dual A/A'/F/F'/G/G'/J/J' softplus → results in dual_summary (A' NaN at 68.5K)

## Completed (dual short)
- bx95ytftx: Dual A/A'/J/J' 10K iters (random 1 obj/iter)
- bnvt6s6qu: Dual A/A'/J/J' 1K iters (sum-all-4) → `exp7_results_dual_1000ch_n500_l2_1Kiters_summary.md`

## Completed (other)
- l8 softplus 10K → `exp7_results_1000ch_n500_l8_10Kiters_expanded_summary.md`
- l2 softmax 10K → `exp7_results_1000ch_n500_l2_10Kiters_softmax_summary.md`
- l4 softplus 10K → `exp7_results_1000ch_n500_l4_10Kiters_expanded_summary.md`
- l2 softplus 10K → `exp7_results_1000ch_n500_l2_10Kiters_expanded_summary.md`

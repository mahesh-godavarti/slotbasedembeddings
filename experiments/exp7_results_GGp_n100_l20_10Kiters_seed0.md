# G/G' Results — n_embed=100, n_layers=20, 10K iters, seed 0

Source: `b530a1e.output` (run killed mid-seed-1; only seed 0 complete)

## G — Text Eval (h@5 / PPL)

| Tier | h@5 | PPL |
|------|-----|-----|
| memorization | 0.097 | 5.53 |
| transfer | 0.033 | 5.83 |
| generalization | 0.122 | 5.44 |
| kg_excl_mem | 0.067 | 6.94 |
| kg_excl_gen | 0.067 | 6.09 |
| text_excl_mem | 0.033 | 5.92 |
| text_excl_gen | 0.067 | 6.44 |

## G — KG Eval (h@5 / PPL)

| Tier | h@5 | PPL |
|------|-----|-----|
| memorization | 0.124 | 5.15 |
| transfer | 0.133 | 5.24 |
| generalization | 0.189 | 5.09 |
| kg_excl_mem | 0.067 | 5.72 |
| kg_excl_gen | 0.100 | 5.25 |
| text_excl_mem | 0.083 | 6.68 |
| text_excl_gen | 0.117 | 6.10 |

## G' — Text Eval (h@5 / PPL)

| Tier | h@5 | PPL |
|------|-----|-----|
| memorization | 0.098 | 5.50 |
| transfer | 0.078 | 5.69 |
| generalization | 0.100 | 5.58 |
| kg_excl_mem | 0.067 | 7.03 |
| kg_excl_gen | 0.067 | 6.60 |
| text_excl_mem | 0.033 | 5.88 |
| text_excl_gen | 0.100 | 6.59 |

## G' — KG Eval (h@5 / PPL)

| Tier | h@5 | PPL |
|------|-----|-----|
| memorization | 0.148 | 4.93 |
| transfer | 0.189 | 4.96 |
| generalization | 0.133 | 5.07 |
| kg_excl_mem | 0.150 | 5.36 |
| kg_excl_gen | 0.133 | 5.19 |
| text_excl_mem | 0.067 | 5.92 |
| text_excl_gen | 0.083 | 6.33 |

## Key Observations (n100/l20 vs prior n500/l2)

- G and G' are much closer here — V rotation advantage is drastically reduced at depth (G' KG mem h@5=.148 vs G .124, compared to .764 vs .129 at n500/l2)
- Text h@5 is now non-zero (~.097) vs essentially zero at n500/l2 — depth helps text
- PPL is healthy across all tiers (~5-7) — no catastrophic kg_excl PPL like n500/l2 had (3134!)
- Cross-pollination present: text_excl on KG h@5=.067-.117 for G, kg_excl on text h@5=.067 for both
- Only 1 seed — run was killed before seed 1 completed

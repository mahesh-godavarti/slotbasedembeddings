# Exp 7a Results: 1000 teaching chains, n_embed=180, 3000 iters (resumed from 2000), 1 seed

## Text Evaluation — Final (hit@5 / ppl)

| Tier | A | A' | B | B' | C | C' | D | D' | E | E' |
|------|---|----|----|----|----|-----|---|----|----|-----|
| **mem** | .080/12 | .085/9 | .081/9 | .093/7 | .093/11 | **.118**/7 | .076/22 | .087/11 | .074/23 | .083/14 |
| **transfer** | .089/19 | .056/15 | .011/10 | .100/7 | .089/12 | **.122**/6 | .100/15 | .022/16 | .067/19 | .044/14 |
| **gen** | .100/8 | .122/6 | .044/7 | .078/6 | .067/10 | .078/6 | .056/12 | .056/8 | .044/13 | .100/8 |
| **kg_excl_mem** | .067/20 | .083/22 | .067/21 | .083/15 | .050/24 | .000/26 | .067/23 | .017/19 | .050/27 | .067/22 |
| **kg_excl_gen** | .083/29 | .000/12 | .100/22 | .017/10 | .017/18 | .050/12 | .083/22 | .100/12 | .033/56 | .133/21 |
| **text_excl_mem** | .033/9 | .083/13 | .067/9 | .050/9 | .000/8 | **.183**/6 | .100/10 | .067/11 | .017/11 | .100/11 |
| **text_excl_gen** | .150/22 | .167/17 | .100/18 | .067/12 | .067/28 | .050/10 | .050/40 | .067/17 | .033/39 | .067/40 |

## KG Evaluation — Final (hit@5 / ppl)

| Tier | A | A' | D | D' | E | E' |
|------|---|-----|---|------|---|------|
| **mem** | .079/11 | .096/8 | .075/11 | .096/8 | .076/10 | **.105**/8 |
| **transfer** | .033/11 | .111/8 | .100/12 | .078/9 | .078/10 | **.144**/7 |
| **gen** | .067/9 | .122/6 | .067/9 | .122/7 | .078/7 | .111/7 |
| **kg_excl_mem** | .100/11 | .067/9 | .100/11 | **.117**/8 | .033/10 | .067/15 |
| **kg_excl_gen** | .133/9 | .017/9 | .050/11 | .083/9 | .000/9 | .083/8 |
| **text_excl_mem** | .067/8 | .083/12 | .083/12 | .083/10 | .067/8 | .033/29 |
| **text_excl_gen** | .067/21 | .033/11 | .117/20 | .033/10 | .017/15 | .050/29 |

## Cross-Pollination at n180/3K

| Direction | Best models | Signal |
|-----------|------------|--------|
| **KG→Text** (kg_excl_mem on text eval) | B'=.083, A'=.083, B=.067, A=.067, E'=.067, D=.067 | Present |
| **Text→KG** (text_excl_mem on KG eval) | A'=.083, D=.083, D'=.083, E=.067, A=.067 | Present |

## Progression: n180 ppl on memorization

| Model | n180/1K | n180/2K | n180/3K | n90/10K target |
|-------|---------|---------|---------|----------------|
| B' text | 8.37 | 6.98 | **6.52** | 6.24 |
| C' text | 8.82 | 7.31 | **6.94** | 5.90 |
| E' KG | 9.35 | 8.07 | **7.92** | 6.32 |
| A' KG | 9.86 | 9.29 | **8.26** | 6.56 |
| D' KG | 8.75 | 7.54 | **7.54** | — |

## Key Findings

1. **C' dominates text**: mem .118 (best), transfer .122 (best), text_excl_mem .183 (best, ppl=5.60)
2. **E' leads KG**: mem .105 (best), transfer .144 (best) — strongest KG model
3. **B' nearly at n90/10K target**: text mem ppl 6.52 vs 6.24 target
4. **C' text_excl_mem = .183**: strongest exclusive-tier signal, suggesting strong text learning
5. **Primed > unprimed holds**: B'>B, C'>C, A'>A, D'>D, E'>E on respective strengths
6. **Cross-pollination present but noisy**: small sample sizes (n=60) make exclusive tiers unreliable

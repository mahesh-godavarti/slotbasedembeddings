# Exp 7a Results: n_embed=90, 10000 iters, 1 seed

## KG Evaluation — Final (hit@5 / ppl)

| Tier | A | A' | D | D' | E | **E'** |
|------|---|-----|---|------|---|--------|
| **memorization** | .722/3.3 | .925/2.0 | .708/3.7 | 1.000/1.3 | .847/2.8 | **1.000/1.3** |
| **transfer** | .789/3.2 | .956/1.9 | .778/3.7 | .989/1.3 | .889/3.1 | **1.000/1.3** |
| **generalization** | .544/4.5 | .922/3.1 | .433/5.5 | .933/2.0 | .600/4.2 | **.978/1.9** |
| **kg_excl_mem** | .767/3.2 | 1.000/1.9 | .583/4.0 | 1.000/1.3 | .817/3.1 | **1.000/1.3** |
| **kg_excl_gen** | .433/7.0 | .817/4.1 | .367/6.9 | .933/2.1 | .500/6.0 | **.933/2.0** |
| **text_excl_mem** | .000/20 | .133/45 | .067/28 | .083/78 | .167/29 | .033/82 |
| **text_excl_gen** | .133/25 | .100/67 | .067/28 | .100/101 | .183/36 | .117/132 |

## Text Evaluation — Final (hit@5 / ppl)

| Tier | A | A' | B | B' | C | **C'** | D | D' | E | E' |
|------|---|----|----|----|----|------|---|----|----|-----|
| **mem** | .250/4.5 | .258/4.3 | .329/3.0 | .600/2.4 | .483/2.5 | **.742/2.0** | .283/4.3 | .308/3.9 | .237/4.4 | .333/4.1 |
| **transfer** | .100/7.0 | .217/6.5 | .333/3.0 | .650/2.5 | .333/2.6 | **.717/1.9** | .283/6.6 | .217/9.5 | .183/7.5 | .150/9.8 |
| **gen** | .133/7.3 | .133/6.1 | .167/6.3 | .300/5.4 | .350/5.8 | **.467/6.2** | .233/6.1 | .150/6.2 | .150/8.4 | .283/7.0 |
| **kg_excl_mem** | .000/59 | .000/65 | .050/187 | .000/435 | .025/474 | .000/374 | .000/52 | .050/61 | .050/43 | .050/64 |
| **kg_excl_gen** | .000/89 | .000/51 | .000/352 | .000/337 | .000/1019 | .000/2991 | .000/114 | .025/78 | .000/81 | .025/87 |
| **text_excl_mem** | .325/4.6 | .275/4.2 | .225/3.3 | .700/2.4 | .500/2.4 | **.850/1.9** | .350/4.9 | .425/3.9 | .375/4.4 | .325/4.5 |
| **text_excl_gen** | .175/8.4 | .100/7.2 | .150/5.1 | .200/4.7 | .100/6.5 | **.475/4.3** | .100/6.6 | .150/7.5 | .200/12 | .050/11 |

## Key Findings

1. **E' and D' tied as KG champions** — both hit@5=1.000 on memorization and kg_excl_mem (ppl=1.3), E' edges out on transfer (1.000 vs .989) and generalization (.978 vs .933)
2. **C' is the dominant text champion** — hit@5=.742 on mem, .850 on text_excl_mem, .717 on transfer, .475 on text_excl_gen
3. **B' is the surprise improver at 10K** — text mem .321→.600, text_excl_mem .425→.700 (vs 5K)
4. **Primed consistently beats unprimed**: E'>E, D'>D, A'>A, C'>C, B'>B on respective strengths
5. **Zero cross-pollination**: kg_excl tiers ~0 on text eval; text_excl tiers ~0 on KG eval
6. **Asymmetric leakage**: KG eval on text_excl shows weak signal (E: .167/.183) while text eval on kg_excl is truly zero — bidirectional attention can partially exploit text-learned patterns but causal attention cannot exploit KG-learned patterns

## Comparison: 5K vs 10K iters

### KG improvements (hit@5)
- A: .525→.722 (mem), .578→.789 (transfer)
- A': .831→.925 (mem), .889→.956 (transfer), .900→1.000 (kg_excl_mem)
- D': .969→1.000 (mem), .983→1.000 (kg_excl_mem), .683→.933 (kg_excl_gen)
- E': .975→1.000 (mem), .989→1.000 (transfer), .900→.978 (gen), 1.000→1.000 (kg_excl_mem)

### Text improvements (hit@5)
- B': .321→.600 (mem), .350→.650 (transfer), .425→.700 (text_excl_mem)
- C': .438→.742 (mem), .467→.717 (transfer), .450→.850 (text_excl_mem), .175→.475 (text_excl_gen)
- C: .375→.483 (mem), .383→.333 (transfer — slight drop), .425→.500 (text_excl_mem)

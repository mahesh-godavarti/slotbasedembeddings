# Exp 7a Results: 1000 chains (expanded_names), n_embed=500, n_layers=4, 10K iters, 1 seed

Config: n_embed=500, n_layers=4, max_iters=10000, batch_size=32, lr=0.0005, device=cuda
Seeds: 1
MLM KG: A, A', E, E', F, F', G, G', H, H'
Causal KG: Ec, Ec', Hc, Hc' (--causal_kg)
kg_as_text: B, B', C, C' (--kg_as_text)

Note: E/E'/H/H' = MLM KG training (default). Ec/Ec'/Hc/Hc' = causal KG training (--causal_kg).

## Text Evaluation (h@5 / PPL)

| Tier | A | A' | B (kat) | B' (kat) | C (kat) | **C' (kat)** | E | E' | Ec | **Ec'** | F | F' | G | G' | H | H' | Hc | Hc' |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mem** | .009/8.58 | .011/8.83 | .137/5.03 | .544/2.66 | .566/2.84 | **.953/1.31** | .022/9.92 | .019/9.08 | .457/2.74 | .819/1.66 | .010/8.83 | .015/8.55 | .013/8.65 | .011/8.79 | .010/9.27 | .011/8.96 | .683/2.16 | .776/1.81 |
| **trn** | .000/10.80 | .000/9.95 | .156/5.44 | .467/2.95 | .656/2.77 | **.944/1.46** | .011/14.44 | .011/12.44 | .333/3.44 | .522/2.79 | .000/11.55 | .000/11.14 | .000/10.81 | .000/9.50 | .000/10.39 | .000/10.97 | .489/3.00 | .467/2.98 |
| **gen** | .000/9.07 | .000/10.38 | .044/7.06 | .322/5.48 | .411/5.46 | **.633/3.86** | .011/17.31 | .000/11.30 | .256/6.41 | .533/4.53 | .000/11.20 | .000/10.31 | .000/9.07 | .000/10.10 | .000/12.69 | .011/10.88 | .411/4.93 | .456/4.71 |
| **kg_ex_m** | .000/1406 | .000/536 | .000/10.29 | .033/9.64 | .117/7.10 | .200/6.25 | .000/7590 | .000/7109 | .050/5.39 | .017/9.02 | .000/1021 | .000/706 | .000/4015 | .000/824 | .000/697 | .000/421 | **.100/4.61** | .100/5.45 |
| **kg_ex_g** | .000/2100 | .000/1064 | .017/13.24 | .033/16.39 | .083/18.23 | .200/14.28 | .000/7968 | .000/8011 | .050/15.12 | .067/14.66 | .000/4766 | .000/1321 | .000/3950 | .000/2425 | .000/1281 | .000/788 | **.100/12.59** | .033/13.41 |
| **tx_ex_m** | .000/10.29 | .000/11.44 | .017/7.53 | .317/3.88 | .283/4.51 | **.850/1.87** | .000/16.14 | .017/11.06 | .200/5.63 | .700/2.55 | .000/11.60 | .050/11.21 | .017/12.11 | .033/13.52 | .000/11.63 | .033/11.45 | .267/3.64 | .533/3.24 |
| **tx_ex_g** | .000/13.98 | .000/16.80 | .050/12.36 | .167/6.90 | .117/11.71 | **.550/6.39** | .017/28.10 | .000/22.16 | .183/22.60 | .383/9.62 | .017/13.76 | .000/20.46 | .000/14.77 | .000/25.37 | .000/22.61 | .000/22.33 | .350/8.86 | .367/7.59 |

(kat) = kg_as_text mode. E/E'/H/H' = MLM KG. Ec/Ec'/Hc/Hc' = causal KG.

## KG Evaluation (h@5 / PPL)

B/B'/C/C' linearized; A/A'/E/E'/F/F'/G/G'/H/H' MLM; Ec/Ec'/Hc/Hc' causal.

| Tier | A | A' | B (lin) | B' (lin) | C (lin) | **C' (lin)** | E | E' | Ec | Ec' | F | F' | G | G' | H | **H'** | Hc | **Hc'** |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mem** | .715/2.77 | .933/1.48 | .137/4.81 | .476/2.85 | .569/2.85 | .954/1.32 | .021/57.09 | .898/1.53 | .670/2.23 | .764/2.04 | .353/6.14 | .768/2.22 | .124/24.62 | .925/1.55 | .829/2.18 | **.948/1.43** | .785/2.09 | .807/1.93 |
| **trn** | .578/3.47 | .911/1.54 | .144/5.04 | .400/3.12 | .583/2.80 | **.967/1.44** | .011/59.48 | .789/1.65 | .744/2.09 | .700/2.12 | .344/6.73 | .689/2.62 | .122/23.64 | .878/1.50 | .844/2.25 | .956/1.54 | .789/2.00 | .778/2.02 |
| **gen** | .567/4.85 | .744/2.56 | .061/6.65 | .306/5.32 | .361/5.57 | .628/3.85 | .011/111.31 | .789/2.70 | .444/8.40 | .589/7.14 | .211/13.52 | .489/3.60 | .089/35.21 | **.800/2.10** | .567/3.94 | .789/2.02 | .556/5.68 | .522/4.78 |
| **kg_ex_m** | .283/10.92 | .567/3.34 | .000/8.64 | .167/5.81 | .108/6.13 | .717/2.35 | .000/115.97 | .550/4.00 | .700/2.56 | .833/2.01 | .083/24.25 | .383/6.82 | .000/72.95 | .633/3.00 | .417/5.73 | .767/2.32 | .900/1.96 | **.933/1.84** |
| **kg_ex_g** | .083/70.08 | .300/13.69 | .008/11.64 | .092/11.58 | .108/15.93 | .533/7.06 | .000/239.33 | .267/12.76 | .617/13.17 | .633/20.03 | .083/95.64 | .217/25.66 | .000/175.78 | .300/16.10 | .083/38.12 | .283/9.06 | **.650/9.79** | .633/5.75 |
| **tx_ex_m** | .000/1819 | .000/1888 | .025/7.23 | .150/5.05 | .267/4.83 | **.492/3.81** | .000/502 | .000/2968 | .017/33.73 | .017/51.13 | .000/744 | .000/1519 | .000/518 | .000/3102 | .000/2216 | .000/1315 | .017/18.49 | .025/15.87 |
| **tx_ex_g** | .000/2690 | .000/3915 | .017/11.96 | .158/8.21 | .117/12.28 | **.392/9.14** | .000/741 | .000/2623 | .000/84.96 | .000/133.90 | .000/1422 | .000/1929 | .000/560 | .000/4586 | .000/3248 | .000/2621 | .000/35.88 | .017/21.47 |

## Key Observations

### C' l4 is extraordinary
- **Text mem**: .953/1.31 — nearly perfect h@5, PPL barely above 1. Massive jump from l2 (.783/2.02)
- **Text transfer**: .944/1.46 — almost as good as memorization
- **Text_excl_mem**: .850/1.87 — text-exclusive facts nearly perfectly learned
- **Lin KG mem**: .954/1.32 — linearized KG essentially memorized perfectly
- **Lin KG kg_excl_mem**: .717/2.35 — huge improvement from l2 (.517/3.57). C' learning KG-exclusive facts well at depth
- **Lin KG tx_ex_mem**: .492/3.81 — text-exclusive facts on linearized KG improving (was .167/7.33 at l2)

### Ec' is a revelation at l4
- **Text mem**: .819/1.66 — second-best text model after C', without kg_as_text mode
- **Text_excl_mem**: .700/2.55 — text-exclusive facts well learned
- **KG kg_excl_mem**: .833/2.01 — strong KG generalization to unseen facts
- Massive jump from l2 (.329/3.36 text mem). Learned cumsum + V rotation + causal + depth = strong dual learner

### Hc/Hc' strong on both text and KG
- **Hc' text**: .776/1.81 mem — third best text model. Up from l2 (.489/2.67)
- **Hc text**: .683/2.16 mem — fourth best. Up from l2 (.343/3.14)
- **Hc' KG kg_excl_mem**: .933/1.84 — best KG generalization of any model
- **Hc KG kg_excl_mem**: .900/1.96 — second best KG generalization
- Fixed cumsum + causal training excels at both modalities with depth

### Depth dramatically helps kg_as_text models
- **C (no V rotation)**: text mem .566/2.84 vs .296/4.25 at l2. Depth compensates significantly for missing V rotation
- **B'**: text mem .544/2.66 vs .365/3.52 at l2. Standard RoPE + V rotation benefits from depth
- **B**: text mem .137/5.03 vs .064/5.86 at l2. Even B without V rotation doubles h@5
- Standard RoPE (B) is fundamentally weak — even at l4 it's far behind C (.566) and B' (.544)

### MLM KG models: excellent KG, zero text
- **H' best KG mem PPL**: .948/1.43 — best of any model
- **A'**: .933/1.48 — close second
- **G'**: .925/1.55 — V rotation transforms G from broken (.124/24.62) to top-tier
- **E'**: .898/1.53 — strong but not dominant at this config
- **All MLM models fail at text**: PPL 8-14, near-zero h@5. MLM bidirectional KG training prevents text learning regardless of depth

### V rotation remains critical
- **G vs G'**: .124/24.62 vs .925/1.55 KG mem — per-relation slots without V rotation are catastrophic even at l4
- **E vs E'**: .021/57.09 vs .898/1.53 KG mem — learned cumsum without V rotation is broken for MLM
- **A vs A'**: .715/2.77 vs .933/1.48 KG mem — shared slots without V rotation recovers with depth but V rotation still helps
- **H vs H'**: .829/2.18 vs .948/1.43 KG mem — fixed cumsum partially recovers without V rotation at depth
- **F vs F'**: .353/6.14 vs .768/2.22 KG mem — flat RoPE needs V rotation

### Depth narrows the V-rotation gap for some models
- **A**: .185→.715 KG mem (l2→l4). A' .860→.933. Gap narrows from .675 to .218
- **H**: .473→.829 (l2→l4). H' .899→.948. Gap narrows from .426 to .119
- **F**: .077→.353 (l2→l4). F' .562→.768. Gap narrows from .485 to .415
- **E**: .018→.021 (l2→l4). E' .907→.898. Depth doesn't help E without V rotation — learned cumsum is fundamentally broken without it
- **G**: .094→.124 (l2→l4). G' .807→.925. Depth barely helps G without V rotation either

### Causal vs MLM KG training at l4
- **Text**: Causal models (Ec/Ec'/Hc/Hc') learn text (PPL 1.66-2.74); MLM models completely fail (PPL 8-14)
- **KG mem with V rotation**: MLM E' (.898/1.53) and H' (.948/1.43) beat causal Ec' (.764/2.04) and Hc' (.807/1.93). MLM still better for KG memorization
- **KG exclusive**: Causal Hc' (.933/1.84) and Hc (.900/1.96) dominate. Causal training generalizes better to unseen KG facts
- **Trade-off confirmed**: MLM = better KG memorization, Causal = better text + better KG generalization

### Cross-pollination: still weak but improving
- **kg_excl on text**: Hc .100/4.61, Hc' .100/5.45 — modest h@5 at l4 (was .033/.083 at l2). Best of any model on this tier
- **C' kg_excl on text**: .200/6.25 — better h@5 but higher PPL than Hc. C' linearization helps h@5 but isn't true cross-pollination
- **tx_ex on KG**: Causal models 15-134 PPL (better than MLM 500-4586) but still very high. Near-zero h@5
- **Cross-pollination remains the bottleneck** — even at l4, models don't transfer knowledge between modalities effectively

### Text leaderboard at l4
| Rank | Model | Text mem h@5 | Text mem PPL |
|------|-------|-------------|-------------|
| 1 | **C'** (kat) | .953 | **1.31** |
| 2 | **Ec'** | .819 | 1.66 |
| 3 | **Hc'** | .776 | 1.81 |
| 4 | **Hc** | .683 | 2.16 |
| 5 | C (kat) | .566 | 2.84 |
| 6 | B' (kat) | .544 | 2.66 |
| 7 | Ec | .457 | 2.74 |
| 8 | B (kat) | .137 | 5.03 |

### KG leaderboard at l4 (mem)
| Rank | Model | KG mem h@5 | KG mem PPL |
|------|-------|-----------|-----------|
| 1 | C' (lin) | .954 | 1.32 |
| 2 | **H'** (MLM) | .948 | **1.43** |
| 3 | A' (MLM) | .933 | 1.48 |
| 4 | G' (MLM) | .925 | 1.55 |
| 5 | E' (MLM) | .898 | 1.53 |
| 6 | H (MLM) | .829 | 2.18 |
| 7 | Hc' (causal) | .807 | 1.93 |
| 8 | Hc (causal) | .785 | 2.09 |

### KG leaderboard at l4 (kg_excl_mem — generalization)
| Rank | Model | kg_excl h@5 | kg_excl PPL |
|------|-------|------------|------------|
| 1 | **Hc'** (causal) | .933 | **1.84** |
| 2 | Hc (causal) | .900 | 1.96 |
| 3 | Ec' (causal) | .833 | 2.01 |
| 4 | H' (MLM) | .767 | 2.32 |
| 5 | C' (lin) | .717 | 2.35 |
| 6 | Ec (causal) | .700 | 2.56 |
| 7 | G' (MLM) | .633 | 3.00 |
| 8 | A' (MLM) | .567 | 3.34 |

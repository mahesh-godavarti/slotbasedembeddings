# Exp 7a Results: 1000 chains (expanded_names), n_embed=500, n_layers=8, 10K iters, 1 seed

Config: n_embed=500, n_layers=8, max_iters=10000, batch_size=32, lr=0.0005, device=cuda
Seeds: 1
MLM KG: A, A', E, E', F, F', G, G', H, H'
Causal KG: Ec, Ec', Hc, Hc' (--causal_kg)
kg_as_text: B, B', C, C' (--kg_as_text)

Note: E/E'/H/H' = MLM KG training (default). Ec/Ec'/Hc/Hc' = causal KG training (--causal_kg).
B collapsed to NaN at iter 7500. Hc collapsed to NaN.

## Text Evaluation (h@5 / PPL)

| Tier | A | A' | B (kat) | B' (kat) | C (kat) | **C' (kat)** | E | E' | Ec | **Ec'** | F | F' | G | G' | H | H' | Hc | **Hc'** |
|------|---|---|---------|----------|---------|-------------|---|---|---|---------|---|---|---|---|---|---|---|------|
| **mem** | .010/8.42 | .014/7.98 | NaN | .629/2.34 | .608/2.59 | **.988/1.12** | .038/9.90 | .032/8.54 | .629/2.26 | **.982/1.15** | .015/8.27 | .015/8.36 | .012/8.25 | .014/8.09 | .012/8.62 | .012/8.60 | NaN | **.911/1.50** |
| **trn** | .000/10.23 | .022/9.30 | NaN | .667/2.21 | .556/2.84 | **.978/1.18** | .000/19.61 | .044/11.79 | .289/3.24 | **.689/2.07** | .000/9.28 | .000/9.30 | .011/8.96 | .011/9.22 | .000/10.09 | .000/10.61 | NaN | **.633/2.16** |
| **gen** | .022/9.36 | .011/9.63 | NaN | .389/5.01 | .378/5.29 | **.678/3.71** | .000/15.63 | .022/11.07 | .389/7.22 | **.644/3.71** | .000/10.00 | .000/10.86 | .022/9.51 | .011/10.27 | .000/11.07 | .000/9.33 | NaN | **.600/4.46** |
| **kg_ex_m** | .000/1387 | .000/464 | NaN | .067/9.05 | .167/8.23 | **.683/2.99** | .000/7821 | .000/36092 | .033/5.58 | .050/10.54 | .000/2794 | .000/546 | .000/3488 | .000/915 | .000/1006 | .000/416 | NaN | **.133/5.07** |
| **kg_ex_g** | .000/2116 | .000/827 | NaN | .083/11.91 | .117/14.69 | **.267/15.26** | .000/3982 | .000/24398 | .050/19.78 | .100/21.41 | .000/3365 | .000/892 | .000/7852 | .000/1657 | .000/1147 | .000/335 | NaN | **.117/9.27** |
| **tx_ex_m** | .033/8.81 | .017/10.06 | NaN | .433/3.07 | .350/4.46 | **.983/1.27** | .033/12.55 | .017/10.48 | .450/4.17 | **.917/1.28** | .017/9.10 | .000/11.14 | .000/10.14 | .033/8.59 | .033/8.76 | .017/11.29 | NaN | **.833/1.68** |
| **tx_ex_g** | .017/12.44 | .000/12.65 | NaN | .233/7.62 | .250/11.46 | **.617/5.10** | .033/42.52 | .050/15.70 | .283/12.65 | **.667/6.08** | .000/11.73 | .017/14.08 | .017/11.82 | .000/12.00 | .000/15.56 | .000/17.07 | NaN | **.500/7.82** |

(kat) = kg_as_text mode. E/E'/H/H' = MLM KG. Ec/Ec'/Hc/Hc' = causal KG.

## KG Evaluation (h@5 / PPL)

B/B'/C/C' linearized; A/A'/E/E'/F/F'/G/G'/H/H' MLM; Ec/Ec'/Hc/Hc' causal.

| Tier | A | A' | B (lin) | B' (lin) | C (lin) | **C' (lin)** | E | E' | Ec | **Ec'** | F | F' | G | G' | H | **H'** | Hc | **Hc'** |
|------|---|---|---------|----------|---------|-------------|---|---|---|---------|---|---|---|---|---|---|---|------|
| **mem** | .755/2.63 | **.966/1.29** | NaN | .579/2.51 | .609/2.58 | **.990/1.12** | .020/54.36 | .867/1.67 | .729/2.09 | .791/2.00 | .508/4.51 | .866/1.72 | .141/17.78 | .936/1.47 | .844/2.03 | **.976/1.23** | NaN | .825/1.90 |
| **trn** | .800/2.57 | **.978/1.28** | NaN | .650/2.45 | .556/2.83 | **.994/1.13** | .000/70.49 | .844/1.66 | .711/2.22 | .822/2.02 | .478/4.57 | .811/1.83 | .089/19.55 | .978/1.50 | .833/1.97 | **.978/1.21** | NaN | .844/1.84 |
| **gen** | .589/4.66 | .967/1.61 | NaN | .372/5.17 | .383/5.13 | **.667/3.26** | .000/101.29 | .644/3.00 | .544/7.56 | .511/7.85 | .278/9.86 | .633/2.97 | .056/39.31 | .733/2.10 | .711/2.95 | **.833/1.63** | NaN | .611/5.37 |
| **kg_ex_m** | .233/8.52 | .800/2.32 | NaN | .275/5.04 | .250/6.22 | **.967/1.27** | .000/127.73 | .600/4.11 | .883/1.94 | **.917/1.81** | .150/19.42 | .400/5.82 | .033/48.84 | .633/2.92 | .533/5.26 | .767/2.24 | NaN | **.983/1.67** |
| **kg_ex_g** | .100/39.83 | .450/7.34 | NaN | .175/8.11 | .150/11.19 | **.600/7.29** | .000/217.04 | .383/14.91 | .600/15.92 | .600/37.39 | .117/52.95 | .267/24.87 | .000/154.76 | .267/12.84 | .250/15.41 | .483/5.80 | NaN | **.633/6.84** |
| **tx_ex_m** | .000/3798 | .000/2307 | NaN | .242/4.00 | .275/5.28 | **.642/2.64** | .000/606 | .000/3050 | .000/23.48 | .017/35.98 | .000/1698 | .000/1459 | .000/492 | .000/3074 | .000/2758 | .000/5267 | NaN | .033/10.11 |
| **tx_ex_g** | .000/2723 | .000/3807 | NaN | .150/8.50 | .158/11.18 | **.392/7.03** | .000/679 | .000/4410 | .000/48.74 | .017/69.65 | .000/1764 | .000/2792 | .000/873 | .000/4029 | .000/6989 | .000/7308 | NaN | .033/26.44 |

## Key Observations

### Ec' is the revelation at l8
- **Text mem**: .982/1.15 — essentially perfect. Massive jump from l4 (.819/1.66)
- **Text tx_ex_m**: .917/1.28 — text-exclusive facts nearly perfectly learned
- **Text tx_ex_g**: .667/6.08 — strong generalization to unseen text-exclusive facts
- **KG kg_ex_m**: .917/1.81 — excellent KG generalization
- Learned cumsum + V rotation + causal KG + depth = the best dual learner

### C' remains the text champion
- **Text mem**: .988/1.12 — near-perfect, up from .953/1.31 at l4
- **Text trn**: .978/1.18 — nearly perfect transfer
- **Text tx_ex_m**: .983/1.27 — text-exclusive facts perfectly learned
- **Lin KG mem**: .990/1.12 — linearized KG essentially perfect
- **Lin KG kg_ex_m**: .967/1.27 — KG-exclusive facts via linearization near-perfect
- C' at l8 achieves near-ceiling on almost every metric

### Hc' strong third on text, best on KG generalization
- **Text mem**: .911/1.50 — up from .776/1.81 at l4
- **Text tx_ex_m**: .833/1.68 — strong text-exclusive learning
- **KG kg_ex_m**: .983/1.67 — best KG generalization of any model
- **KG kg_ex_g**: .633/6.84 — strong KG gen generalization

### NaN collapses at l8
- **B** (standard RoPE, no V rotation, kg_as_text): NaN at iter 7500. Same as previous l8 run.
- **Hc** (fixed cumsum, no V rotation, causal KG): NaN collapse. New.
- Both models lack V rotation. V rotation may stabilize training at deeper layers.

### MLM models: excellent KG, zero text (unchanged)
- **H'**: .976/1.23 KG mem — best MLM KG model, up from .948/1.43 at l4
- **A'**: .966/1.29 KG mem — excellent, up from .933/1.48 at l4
- **G'**: .936/1.47 — up from .925/1.55 at l4
- **F'**: .866/1.72 — up from .768/2.22 at l4
- **E'**: .867/1.67 — down from .898/1.53 at l4 (unusual)
- **All MLM models still fail at text**: PPL 8-11, near-zero h@5. Bidirectional KG training prevents text learning regardless of depth.

### Depth dramatically helps causal and kg_as_text models

| Model | l2 text mem | l4 text mem | l8 text mem |
|-------|------------|------------|------------|
| C' (kat) | .783/2.02 | .953/1.31 | .988/1.12 |
| Ec' | .329/3.36 | .819/1.66 | .982/1.15 |
| Hc' | .489/2.67 | .776/1.81 | .911/1.50 |
| B' (kat) | .365/3.52 | .544/2.66 | .629/2.34 |
| C (kat) | .296/4.25 | .566/2.84 | .608/2.59 |
| Ec | .249/3.51 | .457/2.74 | .629/2.26 |

### Text leaderboard at l8
| Rank | Model | Text mem h@5 | Text mem PPL |
|------|-------|-------------|-------------|
| 1 | **C'** (kat) | .988 | **1.12** |
| 2 | **Ec'** | .982 | 1.15 |
| 3 | **Hc'** | .911 | 1.50 |
| 4 | B' (kat) | .629 | 2.34 |
| 5 | Ec | .629 | 2.26 |
| 6 | C (kat) | .608 | 2.59 |

### KG leaderboard at l8 (mem)
| Rank | Model | KG mem h@5 | KG mem PPL |
|------|-------|-----------|-----------|
| 1 | C' (lin) | .990 | **1.12** |
| 2 | H' (MLM) | .976 | 1.23 |
| 3 | A' (MLM) | .966 | 1.29 |
| 4 | G' (MLM) | .936 | 1.47 |
| 5 | F' (MLM) | .866 | 1.72 |
| 6 | E' (MLM) | .867 | 1.67 |
| 7 | H (MLM) | .844 | 2.03 |
| 8 | Hc' (causal) | .825 | 1.90 |

### KG leaderboard at l8 (kg_excl_mem — generalization)
| Rank | Model | kg_excl h@5 | kg_excl PPL |
|------|-------|------------|------------|
| 1 | **Hc'** (causal) | .983 | **1.67** |
| 2 | C' (lin) | .967 | 1.27 |
| 3 | Ec' (causal) | .917 | 1.81 |
| 4 | Ec (causal) | .883 | 1.94 |
| 5 | A' (MLM) | .800 | 2.32 |
| 6 | H' (MLM) | .767 | 2.24 |
| 7 | G' (MLM) | .633 | 2.92 |
| 8 | E' (MLM) | .600 | 4.11 |

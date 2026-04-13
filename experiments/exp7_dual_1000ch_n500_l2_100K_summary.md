# Exp 7a Results: Dual Objective — 1000 chains (default names), n_embed=500, n_layers=2, 100K iters, 1 seed

Config: n_embed=500, n_layers=2, max_iters=100000, batch_size=32, lr=0.0005, device=cuda, **dual_objective=True**
Seeds: 1
Script: kg_text_experiment_dual.py (random one-objective-per-iter)
Attention: softplus (default) for all models except I, I', A', K, K' which use softmax

## Text Evaluation (h@5 / PPL)

| Tier | B (kat) | B' (kat) | C (kat) | C' (kat) | E | E' | F | F' | G | G' | H | H' | I | I' | A | A' | J | J' | K (smx) | **K' (smx)** |
|------|---------|----------|---------|----------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---------|----------|
| **mem** | .102/5.26 | .107/5.22 | .105/5.20 | .119/5.11 | .100/5.25 | .147/5.08 | .100/5.50 | .106/5.26 | .100/5.34 | .103/5.23 | .102/5.41 | .103/5.25 | .110/5.17 | .134/5.13 | .101/5.46 | .101/5.24 | .100/5.38 | .103/5.28 | .103/5.30 | **.594/3.49** |
| **trn** | .111/5.31 | .067/5.27 | .067/5.23 | .078/5.10 | .111/5.28 | .178/5.02 | .044/5.79 | .000/5.45 | .044/5.73 | .100/5.53 | .144/5.54 | .078/5.39 | .078/5.24 | .056/5.52 | .178/5.63 | .100/5.23 | .067/5.53 | .011/5.44 | .111/5.35 | **.478/3.83** |
| **gen** | .089/5.28 | .100/5.20 | .078/5.11 | .044/5.17 | .133/5.14 | .078/5.01 | .144/5.41 | .089/5.36 | .122/5.47 | .089/5.44 | .089/5.49 | .111/5.22 | .067/5.30 | .122/5.14 | .122/5.36 | .222/5.30 | .078/5.33 | .122/5.16 | .078/5.29 | **.500/4.26** |
| **kg_ex_m** | .017/5.99 | .067/5.77 | .033/5.91 | .150/5.28 | .067/5.41 | .050/6.29 | .133/6.07 | .050/5.93 | .083/6.31 | .083/5.76 | .067/5.67 | .000/5.64 | .083/5.35 | .183/6.02 | .117/5.98 | .150/5.37 | .050/6.11 | .067/5.97 | .167/5.59 | **.350/4.73** |
| **kg_ex_g** | .000/5.89 | .033/5.71 | .017/5.68 | .000/5.63 | .017/5.83 | .117/6.38 | .000/6.65 | .050/5.55 | .083/6.41 | .050/5.71 | .017/6.15 | .050/5.79 | .117/5.62 | .050/6.99 | .067/5.91 | .117/5.85 | .100/6.30 | .117/6.03 | .100/6.49 | **.267/5.27** |
| **tx_ex_m** | .050/5.48 | .067/5.37 | .067/5.33 | .083/5.11 | .083/5.45 | .150/5.10 | .033/5.70 | .050/5.34 | .067/5.66 | .083/5.33 | .017/5.83 | .083/5.36 | .117/5.41 | .033/5.33 | .117/5.68 | .000/5.38 | .017/5.86 | .017/5.44 | .017/5.42 | .133/5.37 |
| **tx_ex_g** | .133/5.94 | .050/5.84 | .083/5.80 | .083/5.48 | .050/6.40 | .133/6.08 | .033/6.94 | .017/6.39 | .033/6.34 | .033/6.32 | .067/6.81 | .000/6.60 | .067/6.66 | .033/5.75 | .033/6.80 | .000/6.24 | .117/6.38 | .050/6.09 | .067/6.70 | .217/6.13 |

(kat) = kg_as_text mode. Dual text-only models alternate between text_causal and text_mlm (50/50).

## Linearized KG Evaluation (h@5 / PPL)

| Tier | B (lin) | B' (lin) | C (lin) | C' (lin) |
|------|---------|----------|---------|----------|
| **mem** | .109/5.16 | .117/5.15 | .104/5.17 | .152/4.96 |
| **trn** | .144/5.18 | .094/5.13 | .044/5.22 | .133/4.99 |
| **gen** | .106/5.04 | .128/5.10 | .106/5.08 | .072/5.10 |
| **kg_ex_m** | .050/5.56 | .025/5.61 | .042/5.62 | .208/4.97 |
| **kg_ex_g** | .033/5.48 | .033/5.63 | .000/5.54 | .083/5.26 |
| **tx_ex_m** | .017/5.62 | .083/5.41 | .058/5.36 | .033/5.22 |
| **tx_ex_g** | .067/6.20 | .008/5.67 | .033/5.72 | .075/5.66 |

## KG Evaluation (h@5 / PPL)

| Tier | B (lin) | B' (lin) | C (lin) | C' (lin) | E | **E'** | F | F' | G | G' | H | H' | I | I' | A | A' | J | J' | K (smx) | **K' (smx)** |
|------|---------|----------|---------|----------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---------|----------|
| **mem** | .109/5.16 | .117/5.15 | .104/5.17 | .152/4.96 | .166/4.71 | .777/2.42 | .095/6.08 | .176/4.68 | .123/5.31 | .184/4.50 | .108/5.68 | .184/4.49 | .172/6.11 | **.862/2.33** | .095/5.82 | .198/4.44 | .114/5.52 | .197/4.57 | .172/4.82 | **.777/2.67** |
| **trn** | .144/5.18 | .094/5.13 | .044/5.22 | .133/4.99 | .189/4.54 | .844/2.49 | .089/6.09 | .133/4.92 | .133/5.54 | .111/4.76 | .111/5.94 | .233/4.52 | .156/6.40 | **.856/2.33** | .067/5.91 | .189/4.90 | .133/5.27 | .222/4.47 | .144/5.05 | **.722/2.71** |
| **gen** | .106/5.04 | .128/5.10 | .106/5.08 | .072/5.10 | .156/4.95 | .589/4.08 | .111/5.68 | .133/5.14 | .078/5.43 | .111/4.63 | .022/5.89 | .100/4.86 | .100/6.71 | **.689/3.17** | .033/5.75 | .156/4.88 | .078/5.67 | .100/4.97 | .122/4.96 | **.611/3.65** |
| **kg_ex_m** | .050/5.56 | .025/5.61 | .042/5.62 | .208/4.97 | .083/5.64 | .867/2.41 | .150/6.95 | .067/5.24 | .083/5.75 | .083/5.14 | .033/6.94 | .100/4.80 | .167/6.55 | **.800/2.86** | .033/6.09 | .283/4.58 | .067/6.28 | .117/5.04 | .100/5.32 | **.717/3.21** |
| **kg_ex_g** | .033/5.48 | .033/5.63 | .000/5.54 | .083/5.26 | .067/5.22 | .583/3.85 | .067/6.71 | .067/5.34 | .067/5.64 | .083/5.07 | .100/6.54 | .050/5.19 | .117/6.93 | **.650/3.52** | .067/6.39 | .117/5.39 | .033/5.79 | .100/5.76 | .100/5.92 | **.550/4.26** |
| **tx_ex_m** | .017/5.62 | .083/5.41 | .058/5.36 | .033/5.22 | .067/6.30 | .117/10.62 | .117/7.28 | .067/5.67 | .050/5.89 | .133/5.43 | .083/6.99 | .033/5.45 | .067/7.81 | .083/6.56 | .100/6.20 | .033/5.63 | .083/6.23 | .017/5.62 | .050/5.80 | .100/6.42 |
| **tx_ex_g** | .067/6.20 | .008/5.67 | .033/5.72 | .075/5.66 | .017/6.26 | .217/7.55 | .083/7.17 | .183/5.49 | .067/6.98 | .167/5.90 | .100/7.84 | .117/5.45 | .033/9.30 | .250/5.82 | .083/7.96 | .133/6.39 | .067/7.03 | .150/5.50 | .100/6.68 | .167/5.76 |

## Key Observations (in progress)

### Dual text-only models are slower than non-dual
- C' non-dual 100K: 1.000/1.01 text mem — perfect
- C' dual 100K: .119/5.11 text mem — still early
- Dual text-only alternates 50/50 between causal and MLM, so each objective gets ~50K effective updates
- Non-dual gets 100K pure causal updates — dual needs ~2x more iterations to match
- C' dual shows best PPL (4.96 lin KG) and kg_ex_m (.208) among kat models — V rotation + learned angles still the best architecture

### C' is still the best kat model in dual
- C' text mem .119 vs B .102, B' .107, C .105
- C' lin KG mem .152 vs others ~.104-.117
- C' kg_ex_m lin KG .208 — best cross-pollination among kat models

### Dual objective prevents E' catastrophic forgetting
- **E' non-dual KG mem**: .907 at 10K → **.238 at 100K** — catastrophic forgetting
- **E' dual KG mem**: **.777/2.42 at 100K** — KG knowledge preserved
- **E' dual kg_ex_m KG**: .867/2.41 — near-perfect KG-exclusive memorization
- **E' dual text mem**: .147/5.08 — comparable to non-dual E' (.143/4.96)
- The random alternation between causal and MLM objectives on both modalities prevents the causal text gradient from overwhelming the MLM KG gradient
- This is the key result: dual training stabilizes MLM KG learning over long training

### E dual also benefits on KG
- **E dual KG mem**: .166/4.71 vs non-dual E: .205/4.28 — slightly worse PPL but still learning
- E dual KG is slower than E' dual — V rotation matters even in dual setting

### H/H' dual KG is worse than non-dual
- **H dual KG mem**: .108/5.68 vs non-dual H: .174/4.48 — dual is worse
- **H' dual KG mem**: .184/4.49 vs non-dual H': .231/4.03 — dual is worse
- Fixed cumsum models don't benefit from dual training on KG — the MLM objective gets diluted to 25% of iterations
- Non-dual H/H' already had stable (if slow) MLM KG learning, so there was nothing to "fix"

### Softplus NaN kills A', I, I' — softmax fixes them
- **I**: NaN at iter 1000, **I'**: NaN at iter 1500, **A'**: NaN at iter 68500 (all softplus)
- Root cause: `exp(wei)` overflow in softplus attention (`log(exp(x)+1)`)
- **Softmax fix**: I/I'/A' dual with softmax all train cleanly to 100K — no NaN
- I softmax dual KG mem: .172/6.11 — comparable to E (.166/4.71) but worse PPL
- **I' softmax dual KG mem: .862/2.33** — new dual KG champion (beats E' .777/2.42)
- A' softmax dual KG mem: .198/4.44 — strong, comparable to J' (.197/4.57)

### I' (softmax) is the new dual KG champion
- **I' dual KG mem**: .862/2.33 vs E' .777/2.42 — better h@5 and PPL
- **I' dual KG trn**: .856/2.33 vs E' .844/2.49
- **I' dual KG gen**: .689/3.17 vs E' .589/4.08 — much better generalization
- **I' dual kg_ex_m**: .800/2.86 vs E' .867/2.41 — E' has slightly better h@5 but I' has better PPL
- **I' dual kg_ex_g**: .650/3.52 vs E' .583/3.85
- **I' dual text mem**: .134/5.13 — comparable to E' (.147/5.08)
- Per-layer angle projectors from residual stream (I) + V rotation + softmax + dual objective = best KG learning
- Unlike E' which catastrophically forgets KG in non-dual mode, I' is stable in both modes

### G' is the best dual KG model (softplus, excluding E')
- **G' dual KG mem**: .184/4.50 — best among non-E' dual models
- **G' dual KG gen**: .111/4.63 — best generalization PPL
- F' also strong: .176/4.68 KG mem
- G' dual text mem: .103/5.23 — best text PPL among dual mixed models

### F/G dual KG is comparable to non-dual
- **F dual KG mem**: .095/6.08 vs non-dual F: .193/4.33 — dual is slower
- **F' dual KG mem**: .176/4.68 vs non-dual F': .209/4.07 — dual is slower but closer
- **G dual KG mem**: .123/5.31 vs non-dual G: .179/4.50 — dual is slower
- **G' dual KG mem**: .184/4.50 vs non-dual G': .248/3.94 — dual is slower
- All dual models get ~25% KG MLM updates vs 50% for non-dual

### A' softmax dual is strong
- **A' dual KG mem**: .198/4.44 — comparable to J' (.197/4.57), better PPL
- **A' dual kg_ex_m KG**: .283/4.58 — best among non-E'/I' dual models
- **A' dual text gen**: .222/5.30 — unusually high for a dual model
- **A dual KG mem**: .095/5.82 — much weaker without V rotation
- A' softplus died at 68.5K; A' softmax trains cleanly to 100K

### J' dual is strong, V rotation gives big KG boost
- **J' dual KG mem**: .197/4.57 — #3 among dual mixed models (after E' .777, H' .184... actually J' beats H')
- **J' dual KG trn**: .222/4.47 — best transfer among non-E' dual models
- **J dual KG mem**: .114/5.52 — mid-tier without V rotation
- J→J' KG mem: .114→.197 (1.7x), KG trn: .133→.222 (1.7x) — V rotation nearly doubles KG h@5
- **J' dual text**: .103/5.28 mem — comparable to G' (.103/5.23)
- J' KG PPL (4.57) beats G' (4.50) on mem but J' has better trn PPL (4.47 vs 4.76)

### K' (softmax) is the new dual text champion and #3 dual KG
- **K' dual text mem**: **.594/3.49** — 4x better than any other dual model (next best: E' .147/5.08)
- **K' dual text trn**: .478/3.83 — 2.7x better than E' (.178/5.02)
- **K' dual text gen**: .500/4.26 — 2.3x better than A' (.222/5.30)
- **K' dual KG mem**: .777/2.67 — ties E' (.777/2.42), behind I' (.862/2.33)
- **K' dual KG trn**: .722/2.71 — behind I' (.856) but ahead of E' (.844) on h@5... no, behind E'. But best PPL after I'.
- **K' dual KG gen**: .611/3.65 — #3 behind I' (.689) and E' (.589 but 4.08 PPL). K' has better PPL than E'.
- **K' dual kg_ex_m KG**: .717/3.21 — behind I' (.800) and E' (.867)
- **K' dual kg_ex_m text**: .350/4.73 — **new cross-pollination champion** among dual models (I' was .183)
- V rotation is massive: K (.103 text, .172 KG) → K' (.594 text, .777 KG) — 5.8x text, 4.5x KG
- K = I's projected angles + J's slot angles. The combination is exceptionally powerful with V rotation + softmax + dual objective.
- K' uniquely excels at BOTH text and KG in dual mode — other models (I', E') are strong on KG but weak on text.

### Dual helps models at risk of forgetting, not all models
- E' (which catastrophically forgot KG): .238 → .777 — massive improvement
- H/H' (which had slow but stable KG): .174/.231 → .108/.184 — slightly worse
- A', I, I': dead from softplus NaN — need normalized softplus or softmax
- The dual objective is medicine for the forgetting problem, not a universal accelerator

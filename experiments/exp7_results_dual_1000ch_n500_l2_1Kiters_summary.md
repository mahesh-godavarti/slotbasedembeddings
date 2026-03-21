# Exp 7a Results: Dual Objective — 1000 chains, n_embed=500, n_layers=2, 1 seed

Config: n_embed=500, n_layers=2, batch_size=32, lr=0.0005, device=cuda, softplus, **dual_objective=True**
Seeds: 1
Script: kg_text_experiment_dual.py
Models: A, A', J, J' (all MLM KG with dual objective)

## Dual Objective

Each iteration randomly picks ONE of 4 objectives:
1. Text → causal NTP (standard)
2. Text → bidirectional MLM (new)
3. KG → bidirectional MLM (standard)
4. KG → slot-causal NTP (new)

Each objective gets ~25% of iterations (for mixed models; text-only models alternate between 1 and 2).

Earlier version (1K run only) summed all 4 losses per iteration: Loss = text_causal + text_mlm + kg_mlm + kg_causal

## Model J/J' (new)
- Like G/G' (per-relation slot angles) but native KG format without REL token
- 2 slot angles per relation (HEAD, TAIL) instead of 3
- Relation identity encoded entirely in the slot angles

## Text Evaluation (h@5 / PPL)

| Tier | A | A' | J | J' |
|------|---|---|---|---|
| **mem** | .081/6.64 | .084/6.29 | .083/6.70 | .088/6.29 |
| **trn** | .044/7.01 | .067/6.51 | .078/7.06 | .044/6.65 |
| **gen** | .089/5.81 | .100/5.84 | .111/6.25 | .078/5.90 |
| **kg_ex_m** | .100/7.96 | .067/6.65 | .050/7.42 | .083/7.02 |
| **kg_ex_g** | .083/6.83 | .067/6.66 | .083/6.76 | .033/6.37 |
| **tx_ex_m** | .033/6.56 | .017/6.86 | .033/6.46 | .083/6.50 |
| **tx_ex_g** | .100/8.29 | .067/7.40 | .033/8.70 | .067/7.61 |

## KG Evaluation (h@5 / PPL)

| Tier | A | A' | J | J' |
|------|---|---|---|---|
| **mem** | .086/6.46 | .098/6.03 | .072/8.08 | .091/6.57 |
| **trn** | .078/6.54 | .133/6.31 | .078/8.23 | .033/6.89 |
| **gen** | .111/5.95 | .111/5.62 | .044/7.69 | .044/6.10 |
| **kg_ex_m** | .033/7.22 | .033/6.29 | .050/8.27 | .017/7.02 |
| **kg_ex_g** | .017/6.84 | .033/6.52 | .050/8.42 | .083/6.45 |
| **tx_ex_m** | .017/6.47 | .067/6.28 | .117/7.10 | .050/6.70 |
| **tx_ex_g** | .133/7.34 | .083/6.73 | .050/10.36 | .033/7.80 |

## Key Observations

### Dual objective converges faster on PPL than single-objective MLM
- A dual text PPL at 1K: 6.64 — vs A non-dual text PPL at **10K**: 10.16
- A dual KG PPL at 1K: 6.46 — vs A non-dual KG PPL at **10K**: 11.42
- The dual objective (adding causal signal to KG and MLM signal to text) accelerates early convergence
- However, h@5 is still near-zero everywhere at 1K — PPL improvement hasn't translated to accurate predictions yet

### All four models are similar at 1K
- h@5 ranges from .033 to .133 across all models and tiers — no clear winner
- A' has slightly better PPL than A across the board (V rotation helps)
- J without V rotation has worse KG PPL (8.08) vs A (6.46) and J' (6.57)
- J' with V rotation closes the gap with A/A'

### V rotation helps J more than A
- J vs J' KG mem: .072/8.08 vs .091/6.57 — V rotation drops KG PPL by 1.5
- A vs A' KG mem: .086/6.46 vs .098/6.03 — V rotation drops KG PPL by 0.4
- J's per-relation native KG format (no REL token) benefits more from V rotation

### Too early to assess dual objective's key question
- The main hypothesis: does dual objective prevent the catastrophic forgetting seen with E' (MLM KG .907→.238 from 10K to 100K)?
- Need to run to 10K+ iters to test this — at 1K, models haven't learned enough to forget

---

## 10K Results (random one-objective-per-iter)

### Text Evaluation (h@5 / PPL)

| Tier | A | A' | J | J' |
|------|---|---|---|---|
| **mem** | .092/6.06 | .092/5.76 | .092/6.10 | .094/5.86 |
| **trn** | .100/6.19 | .089/6.21 | .122/6.49 | .122/5.74 |
| **gen** | .156/5.60 | .078/5.86 | .067/5.88 | .067/5.58 |
| **kg_ex_m** | .000/7.02 | .117/6.45 | .017/6.71 | .033/6.95 |
| **kg_ex_g** | .050/6.57 | .050/6.56 | .000/7.26 | .100/6.30 |
| **tx_ex_m** | .017/6.36 | .050/6.10 | .067/6.64 | .100/6.08 |
| **tx_ex_g** | .117/7.93 | .033/7.31 | .067/8.99 | .067/6.95 |

### KG Evaluation (h@5 / PPL)

| Tier | A | A' | J | J' |
|------|---|---|---|---|
| **mem** | .078/8.42 | .104/5.88 | .081/6.83 | .097/5.96 |
| **trn** | .067/8.41 | .133/6.11 | .056/7.52 | .111/5.83 |
| **gen** | .122/7.63 | .089/5.72 | .067/6.36 | .122/5.46 |
| **kg_ex_m** | .033/8.85 | .017/6.68 | .100/7.13 | .050/6.18 |
| **kg_ex_g** | .050/8.84 | .033/6.38 | .117/7.11 | .083/6.30 |
| **tx_ex_m** | .067/8.95 | .050/6.23 | .100/6.57 | .117/6.35 |
| **tx_ex_g** | .100/9.61 | .083/6.11 | .067/9.18 | .150/6.79 |

### 10K Observations

#### Still near-zero h@5 at 10K
- All models ~.09-.10 text mem h@5 — same as non-dual A at 100K (.097)
- With random 1-of-4, each objective gets ~2500 effective updates at 10K
- Text PPL improved from 1K (6.64→6.06 for A) but h@5 hasn't moved

#### V rotation consistently helps PPL
- A' KG PPL: 5.88 vs A: 8.42 — V rotation cuts KG PPL by 30%
- J' KG PPL: 5.96 vs J: 6.83 — similar benefit
- Text PPL also better with V rotation (5.76/5.86 vs 6.06/6.10)

#### A dual KG PPL regressed from 1K to 10K
- A KG mem PPL: 6.46 at 1K → 8.42 at 10K — getting worse
- Same forgetting pattern as E' (MLM KG degrading as text training continues)
- Random alternation does NOT prevent forgetting for A without V rotation
- A' KG mem PPL: 6.03 at 1K → 5.88 at 10K — V rotation stabilizes

#### J' has best overall KG PPL
- J' KG mem: 5.96 — slightly better than A' (5.88) on gen (5.46 vs 5.72)
- Per-relation native KG format may have a slight edge with V rotation

#### 100K sweep will be the definitive test
- Full model sweep with dual objective at 100K launched (3 groups)
- Key question: does dual training prevent E'-style catastrophic forgetting at scale?

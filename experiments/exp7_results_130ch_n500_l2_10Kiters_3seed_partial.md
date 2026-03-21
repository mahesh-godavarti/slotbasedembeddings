# Exp 7a Results: 130ch, n_embed=500, n_layers=2, 10K iters, 3 seeds — PARTIAL (seed 0 complete for most)

Config: n_embed=500, n_layers=2, max_iters=10000, batch_size=32, lr=0.0005, device=cuda
Seeds: 3 (seed 0 shown below; seeds 1-2 still running)
MLM KG models: A, A', F (seed 0 done), F', G, G' (training)
Causal KG models: E, E', H, H' (all seed 0 done, seed 1 in progress)
kg_as_text models: B, B', C, C' (all seed 0 done, seed 2 in progress)

## Text Evaluation (h@5 / PPL) — seed 0

| Tier | A | A' | B (kat) | B' (kat) | C (kat) | C' (kat) | E | E' | F | H | H' |
|------|---|---|---|---|---|---|---|---|---|---|---|
| **mem** | .096/5.66 | .098/5.47 | .101/5.31 | .119/5.20 | .099/5.37 | **.258/4.72** | .104/5.25 | .276/4.52 | .096/5.62 | .104/5.23 | .124/5.14 |
| **transfer** | .044/5.80 | .089/5.90 | .122/5.24 | .100/5.22 | .189/5.27 | **.333/4.54** | .033/5.47 | .189/4.78 | .078/5.77 | .089/5.50 | .111/5.20 |
| **gen** | .111/5.43 | .156/5.39 | .044/5.46 | .211/5.15 | .044/5.28 | **.222/5.03** | .122/5.21 | .244/4.59 | .078/5.36 | .100/5.22 | .122/5.12 |
| **kg_excl_mem** | .033/7.10 | .017/8.20 | .100/5.63 | .067/5.80 | .083/6.14 | .083/5.79 | .067/5.77 | **.217/5.45** | .050/7.22 | .050/5.39 | .050/5.60 |
| **kg_excl_gen** | .067/7.09 | .017/7.72 | .067/6.03 | .017/5.98 | .000/6.04 | .083/5.85 | .000/6.52 | .083/6.34 | .017/7.12 | .067/5.69 | .033/5.50 |
| **text_excl_mem** | .000/6.23 | .017/5.63 | .067/5.62 | .067/5.34 | .050/5.58 | **.150/4.90** | .033/5.69 | .100/5.25 | .100/5.81 | .050/5.67 | .033/5.54 |
| **text_excl_gen** | .117/6.67 | .033/6.39 | .067/6.02 | .033/6.13 | .033/6.80 | .133/6.01 | **.133/6.56** | .083/6.03 | .067/7.08 | .083/6.27 | **.150/5.81** |

(kat) = kg_as_text mode. F', G, G' not yet complete.

## KG Evaluation (h@5 / PPL) — seed 0

Note: A/A'/F use MLM; E/E'/H/H' use causal (both directions). B/B'/C/C' use linearized KG eval (separate table below).

| Tier | A | A' | E | E' | F | H | H' |
|------|---|---|---|---|---|---|---|
| **mem** | .093/5.85 | .138/5.41 | .126/5.74 | **.794/2.14** | .094/5.71 | .122/5.77 | .143/5.57 |
| **transfer** | .067/5.93 | .178/5.38 | .078/5.68 | **.750/2.13** | .056/5.79 | .100/5.72 | .117/5.54 |
| **gen** | .067/5.58 | .111/5.49 | .078/5.89 | **.556/3.67** | .100/5.47 | .078/5.86 | .133/5.72 |
| **kg_excl_mem** | .083/6.23 | .100/5.49 | .050/5.76 | **.783/2.05** | .067/5.96 | .100/5.50 | .183/5.62 |
| **kg_excl_gen** | .050/6.07 | .133/5.99 | .050/6.22 | **.533/4.28** | .033/5.79 | .092/5.97 | .050/5.75 |
| **text_excl_mem** | .000/6.13 | .033/7.28 | .033/6.93 | .042/13.70 | .067/5.88 | .033/6.42 | .000/6.58 |
| **text_excl_gen** | .033/6.77 | .117/6.48 | .100/8.08 | .092/18.58 | .100/6.09 | .067/7.64 | .075/7.70 |

F', G, G' not yet complete.

## Linearized KG Evaluation (kg_as_text models) — seed 0

| Tier | B (lin) | B' (lin) | C (lin) | C' (lin) |
|------|---|---|---|---|
| **mem** | .108/5.28 | .111/5.19 | .104/5.33 | **.286/4.54** |
| **transfer** | .083/5.19 | .061/5.21 | .178/5.22 | **.289/4.51** |
| **gen** | .056/5.38 | .206/5.16 | .050/5.25 | **.183/4.98** |
| **kg_excl_mem** | .075/5.48 | .050/5.79 | .092/5.80 | **.242/5.29** |
| **kg_excl_gen** | .117/5.93 | .042/5.58 | .025/5.74 | **.242/5.33** |
| **text_excl_mem** | .083/5.55 | .058/5.40 | .083/5.58 | .067/5.53 |
| **text_excl_gen** | .050/5.94 | .008/6.13 | .033/6.95 | .108/6.54 |

## Key Observations (seed 0)

### Two champions: E' (KG) and C' (text)
- **E' dominates KG**: .794/2.14 mem, .783/2.05 kg_excl_mem — massive lead over all others
- **C' dominates text**: .258/4.72 mem, .333/4.54 transfer, .150/4.90 text_excl_mem — best text PPL of any model
- **C' also leads linearized KG**: .286/4.54 mem, .289/4.51 transfer — significantly ahead of B/B'/C

### V rotation is transformative for learned angles
- **E vs E'**: KG mem .126/5.74 vs .794/2.14 — night and day
- **C vs C'**: text mem .099/5.37 vs .258/4.72 — V rotation drops PPL by 0.65
- **B vs B'**: text mem .101/5.31 vs .119/5.20 — marginal (0.11 PPL drop). Standard RoPE doesn't benefit much
- **H vs H'**: text mem .104/5.23 vs .124/5.14 — small improvement. Fixed cumsum + V rotation is modest

### A/A' and F underperform at n500/l2 with 130 chains
- A' KG mem .138 here vs .875 at n500/l2/10K with 1000 chains — 130 chains is too little data for slotted models
- A/A'/F kg_excl text PPL is worst of all models (7.10-8.20) — slotted/flat RoPE models heavily penalize KG-exclusive text
- F KG is unremarkable (.094/5.71 mem) — unlike at n100/l20 where F was competitive

### H' shows promise on KG
- H' kg_excl_mem on KG = .183/5.62 — second best after E' (.783/2.05)
- H' gen on KG = .133/5.72 — third best after E' (.556/3.67) and A' (.111/5.49)
- Fixed cumsum + V rotation is the second-best KG approach at this config

### Cross-pollination
- **E' kg_excl_mem on text** = .217/5.45 — KG-only facts leaking to text predictions. Strongest signal.
- **C' text_excl_mem on text** = .150/4.90 — text-only facts well-modeled, but this is same-modality
- **E' text_excl on KG** = .042/13.70 — catastrophic PPL, E' specializes KG heavily
- **C' kg_excl on linearized KG** = .242/5.29 — C' handles KG-exclusive facts better than B/B' (.050-.075)
- **text→KG transfer near zero** for all native KG models (.000-.033 h@5 on text_excl)

### Linearized KG: C' pulls ahead, B/B' flat
- C' lin KG PPL (4.54 mem) is ~0.7 lower than B/B'/C (~5.2-5.3) — learned angles + V rotation helps
- But still far from E' native KG (2.14) — linearization cannot match native structure
- C' kg_excl on lin KG (.242/5.29) vs B (.075/5.48) — C' differentiates KG-exclusive facts better

## Still Running
- MLM group: F' seed 0, then G, G' seed 0, then seeds 1-2 for all
- Causal group: seed 1 (E in progress)
- kg_as_text group: seed 2 (B in progress)

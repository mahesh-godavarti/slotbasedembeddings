# Exp 7a Results: 130ch, n_embed=100, n_layers=20, 10K iters — kg_as_text B/B'/C/C' vs Standard Models

B/B'/C/C' run: kg_as_text mode, seeds=1, device=cuda
Other models: standard mode (from partial_summary + GGp file), seeds=1 (seed 0 only), device=cpu
Config: n_embed=100, n_layers=20, max_iters=10000, batch_size=32, lr=0.0005

## Text Evaluation (h@5 / PPL)

| Tier | A | A' | **B** | **B'** | **C** | **C'** | E | E' | F | F' | G | G' | H | H' |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mem** | .093/5.62 | .098/5.52 | .100/5.15 | .117/5.11 | .106/5.16 | .179/**4.92** | .157/4.97 | .142/5.05 | .092/5.63 | .100/5.49 | .097/5.53 | .098/5.50 | .129/5.05 | .154/5.00 |
| **transfer** | .133/5.74 | .089/5.76 | .156/5.18 | .122/5.24 | .067/5.23 | .278/**4.82** | .144/5.06 | .122/5.30 | .056/5.91 | .044/5.64 | .033/5.83 | .078/5.69 | .089/5.14 | .200/4.99 |
| **gen** | .122/5.45 | .100/5.39 | .067/5.14 | .122/5.16 | .089/5.08 | .100/5.12 | .100/5.06 | .067/5.19 | .056/5.49 | .078/5.55 | .122/5.44 | .100/5.58 | .167/5.14 | .089/**4.93** |
| **kg_excl_mem** | .033/6.89 | .033/7.08 | .017/5.58 | .033/5.76 | .083/5.69 | .083/6.00 | .067/5.34 | .100/**5.34** | .183/6.64 | .133/6.53 | .067/6.94 | .067/7.03 | .117/**5.21** | .067/5.35 |
| **kg_excl_gen** | .050/6.72 | .033/6.96 | .033/5.58 | .067/5.64 | .133/5.75 | .067/5.53 | .067/5.69 | .067/5.81 | .050/5.98 | .067/6.44 | .067/6.09 | .067/6.60 | .100/**5.37** | .033/5.52 |
| **text_excl_mem** | .000/5.95 | .117/5.42 | .067/5.34 | .083/5.34 | .083/5.23 | .183/**4.95** | .050/5.34 | .150/5.25 | .083/5.81 | .067/5.58 | .033/5.92 | .033/5.88 | .067/5.30 | .083/5.29 |
| **text_excl_gen** | .067/6.57 | .167/7.00 | .100/5.36 | .083/5.82 | .100/6.05 | .050/5.61 | .050/6.05 | .083/5.75 | .000/6.91 | .067/6.53 | .067/6.44 | .100/6.59 | .117/**5.81** | .050/5.83 |

## KG Evaluation (h@5 / PPL)

Note: A/E/F/H use native KG eval (MLM/causal); B/B' use linearized KG eval (causal). Not perfectly apples-to-apples but shows KG fact retrieval ability.

| Tier | A | A' | **B** (lin) | **B'** (lin) | **C** (lin) | **C'** (lin) | E | E' | F | F' | G | G' | H | H' |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mem** | .139/5.01 | .161/4.87 | .101/5.14 | .116/5.11 | .111/5.14 | .201/**4.86** | .323/4.53 | .809/**2.37** | .148/4.96 | .158/4.84 | .124/5.15 | .148/4.93 | .146/5.54 | .172/5.45 |
| **transfer** | .122/5.08 | .189/4.63 | .150/5.18 | .089/5.23 | .100/5.19 | .189/**4.82** | .278/4.50 | .806/**2.32** | .111/4.97 | .122/5.11 | .133/5.24 | .189/4.96 | .172/5.40 | .178/5.28 |
| **gen** | .156/4.94 | .111/4.96 | .106/5.13 | .078/5.15 | .067/5.07 | .167/5.02 | .272/5.04 | .633/**4.34** | .144/4.97 | .122/5.09 | .189/5.09 | .133/5.07 | .133/5.69 | .072/5.74 |
| **kg_excl_mem** | .067/5.71 | .167/5.06 | .025/5.52 | .017/5.66 | .108/5.55 | .158/**5.20** | .317/4.65 | .858/**2.17** | .117/5.25 | .217/5.44 | .067/5.72 | .150/5.36 | .175/5.43 | .117/5.47 |
| **kg_excl_gen** | .100/5.39 | .067/5.08 | .025/5.37 | .033/5.57 | .117/5.51 | .100/**5.20** | .150/5.18 | .625/**3.69** | .167/4.99 | .117/5.49 | .100/5.25 | .133/5.19 | .083/5.76 | .100/5.67 |
| **text_excl_mem** | .033/6.08 | .050/6.08 | .067/5.28 | .075/5.36 | .067/5.27 | .083/**5.25** | .017/7.12 | .050/12.12 | .083/5.81 | .067/5.72 | .083/6.68 | .067/5.92 | .067/6.05 | .025/6.07 |
| **text_excl_gen** | .133/5.86 | .150/5.92 | .067/5.42 | .100/5.73 | .033/6.15 | .042/5.81 | .092/8.10 | .108/15.67 | .100/5.94 | .133/**5.70** | .117/6.10 | .083/6.33 | .067/6.99 | .050/6.72 |

## Key Observations

### Text PPL
- **C' is the text PPL champion**: 4.92 mem, 4.82 transfer — best of any model. Learned per-token angles + V rotation + extra linearized KG data is a strong combination.
- **C (no V rotation) is unremarkable**: 5.16 mem — no better than B (5.15). V rotation is essential for C to shine.
- **B/B' are solid but not standout**: 5.15/5.11 mem — ahead of A/F/G (~5.5-5.6) but behind E (4.97), H' (5.00), and now C' (4.92).
- **G/G' cluster with A/F**: G 5.53, G' 5.50 mem PPL. Per-relation slot angles don't help text at depth. kg_excl PPL is worst of all models (6.94/7.03).
- **C' text_excl_mem PPL is excellent** (4.95): The model generalizes well even to text-exclusive facts, suggesting genuine language modeling improvement rather than just memorization.
- **kg_excl text PPL varies**: C' (6.00) and G/G' (6.94/7.03) are worst. B/B' (5.58/5.76) and E/H (~5.2-5.4) are best. Learned/per-relation angles penalize KG-exclusive text.

### KG (Linearized) PPL
- **C' linearized KG PPL is best among kg_as_text models**: 4.86 mem, 4.82 transfer — noticeably lower than B/B' (~5.1) and C (~5.1). V rotation helps C model linearized facts.
- **But still far from E' native KG PPL**: C' 4.86 vs E' 2.37 on mem. Linearization cannot match native KG structure for factual compression.
- **C' shows some tier differentiation**: kg_excl PPL (5.20) vs mem PPL (4.86) — a 0.34 gap. B/B' show almost no gap (~0.4). Still much less than E' (2.17 vs 2.37 — inverted, kg_excl is actually easier).
- **B/B' linearized KG PPL is flat ~5.1-5.7**: These models treat KG linearizations as generic text with no factual compression.

### Cross-Pollination (PPL perspective)
- **C' has best text_excl_mem PPL on text** (4.95): Tied with E' (5.25) as models that handle text-exclusive facts well. C' actually beats E' here.
- **kg_excl text PPL**: B/B' (5.58/5.76) are healthier than C' (6.00), A (6.89), F (6.64). Linearizing KG helps text PPL on KG-exclusive facts for B/B' but C' pays a slight penalty from its learned angles.
- **E' text_excl on KG has catastrophic PPL** (12.12/15.67): E' heavily specializes its KG head. C' avoids this (5.25 text_excl on linearized KG) by having no separate KG modality.

### V Rotation Effects
- **C vs C' is a large gap**: C' mem text PPL 4.92 vs C 5.16 — V rotation drops PPL by 0.24. On linearized KG: 4.86 vs 5.14. Learned angles benefit greatly from V rotation.
- **B vs B' is marginal**: B' mem text PPL 5.11 vs B 5.15 — only 0.04 improvement. Standard RoPE doesn't benefit much from V rotation in this mode.
- **G vs G' is minimal on text** (5.53 vs 5.50) **but visible on KG** (5.15 vs 4.93 mem). Per-relation slot angles + V rotation helps KG slightly.
- This mirrors the standard-mode finding: V rotation is transformative for learned angles (E→E', C→C') but marginal for fixed/slot-based approaches (B→B', F→F', G→G').

### Overall Assessment
- **C' is the clear kg_as_text winner** and the best text model overall (4.92 mem PPL, 4.82 transfer PPL), surpassing even E (4.97) and H' (5.00).
- **Linearization still cannot match native KG**: C' linearized KG PPL (4.86) is far from E' native KG PPL (2.37). Structure matters for factual compression.
- **The kg_as_text approach benefits text modeling**: Extra linearized KG data improves text PPL for all kg_as_text models, with C' benefiting most.
- **But it doesn't enable KG-exclusive learning**: kg_excl PPL on both text and linearized KG remains high for all kg_as_text models. The linearized format doesn't help models distinguish or compress KG-exclusive facts.
- **Two-modality specialization (E') vs one-modality generalism (C')**: E' gets far better KG PPL but worse text PPL on exclusive tiers. C' gets uniformly good text PPL but can't deeply learn structured facts.

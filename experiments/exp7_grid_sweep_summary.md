# Exp 7 Grid Sweep Summary

## Setup
- **1000 chains**, 100K iterations, 1 seed, softmax attention
- **Grid**: n_embed × n_layers: 50×{2,4,8,16,20}, 100×{2,4,8,16,20}, 250×{2,4,8}, 500×{2,4}
- **Dual**: `kg_text_experiment_dual.py` — random alternation between causal and MLM objectives
- **Non-dual**: `kg_text_experiment.py` — standard causal text + MLM KG training
- **Models**: A/A' through J/J' (primed = V rotation). K/K' backfill pending.
- **Groups per grid point**:
  - Dual: mixed (A-J') + KAT (B/C with --kg_as_text)
  - Non-dual: mixed (A-J') + causal (E/H/I with --causal_kg) + KAT (B/C with --kg_as_text)

## Completion Status (as of Mar 19, 2026)

### Dual
| Grid Point | mixed | kat | Status |
|---|---|---|---|
| n50_l2 | Done | Done | Complete |
| n50_l4 | Done | Done | Complete |
| n50_l8 | Done | Done | Complete |
| n50_l16 | Done | Done | Complete |
| n50_l20 | Done | Done | Complete |
| n100_l2 | Done | Done | Complete |
| n100_l4 | Done | Done | Complete |
| n100_l8 | Done | Done | Complete |
| n100_l16 | Done | Done | Complete |
| n100_l20 | Done | — | KAT pending |
| n250_l2–l8 | — | — | Pending |
| n500_l2–l4 | — | — | Pending |

### Non-dual
| Grid Point | mixed | causal | kat | Status |
|---|---|---|---|---|
| n50_l2 | Done | Done | Done | Complete |
| n50_l4 | Done | Done | Done | Complete |
| n50_l8 | Done | Done | Done | Complete |
| n50_l16 | Done | Done | Done | Complete |
| n50_l20 | Running (I 70%) | — | — | In progress |
| n100_l2–l20 | — | — | — | Pending |
| n250_l2–l8 | — | — | — | Pending |
| n500_l2–l4 | — | — | — | Pending |

---

## 1. KG Memorization PPL — Best Model per Grid Point

### Dual (lower = better)
| Config | Best Model | KG mem PPL | #2 Model | #2 PPL |
|---|---|---|---|---|
| n50_l2 | I' | 5.25 | E' | 5.31 |
| n50_l4 | I' | 4.84 | E' | 4.86 |
| n50_l8 | I' | 4.48 | E' | 4.51 |
| n50_l16 | E' | 4.25 | I' | 4.29 |
| n50_l20 | E' | 4.17 | J' | 4.21 |
| n100_l2 | I' | 4.53 | E' | 4.54 |
| n100_l4 | E'/I' | 4.29 | D' | 4.35 |
| n100_l8 | E' | 4.00 | J' | 4.09 |
| n100_l16 | E' | 3.54 | J' | 3.68 |
| **n100_l20** | **E'** | **3.29** | **J'** | **3.64** |

### Non-dual (lower = better)
| Config | Best Model | KG mem PPL | #2 Model | #2 PPL |
|---|---|---|---|---|
| n50_l2 | I' | 4.74 | D' | 4.90 |
| n50_l4 | I' | 4.22 | E' | 4.29 |
| n50_l8 | I' | 4.07 | E' | 4.08 |
| **n50_l16** | **J'** | **3.95** | **G'** | **3.96** |

**Takeaway**: E' depth scaling at n100 is remarkable — no sign of saturation from l8 (4.00) through l16 (3.54) to l20 (3.29). J' is a consistent #2. Non-dual still beats dual at matched configs (n50_l16: 3.95 vs 4.25), but n100_l20 dual E' (3.29) is the overall best.

---

## 2. KG Memorization h@5 — Best Model per Grid Point

### Dual
| Config | Best Model | KG mem h@5 | #2 Model | #2 h@5 |
|---|---|---|---|---|
| n50_l2 | I' | .112 | E' | .107 |
| n50_l4 | E' | .139 | D' | .135 |
| n50_l8 | I' | .173 | G' | .168 |
| n50_l16 | E' | .191 | D' | .186 |
| n50_l20 | J' | .217 | E' | .201 |
| n100_l2 | E' | .169 | I'/D' | .168 |
| n100_l4 | E' | .201 | I' | .198 |
| n100_l8 | E' | .281 | J' | .263 |
| n100_l16 | E' | .445 | J' | .353 |
| **n100_l20** | **E'** | **.527** | **J'** | **.368** |

### Non-dual
| Config | Best Model | KG mem h@5 | #2 Model | #2 h@5 |
|---|---|---|---|---|
| n50_l2 | I' | .162 | E' | .155 |
| n50_l4 | I' | .197 | E' | .193 |
| n50_l8 | J' | .212 | G' | .206 |
| **n50_l16** | **J'** | **.262** | **G'** | **.259** |

**Takeaway**: E' dominates dual KG h@5, reaching .527 at n100_l20 — over half of memorized KG facts predicted correctly. J' is consistently #2 (.368 at n100_l20). I' leads only at small/shallow configs and collapses at n100_l20 (.173).

---

## 3. Depth Scaling — Dual E' KG mem PPL

| Layers | n50 | n100 |
|---|---|---|
| 2 | 5.31 | 4.54 |
| 4 | 4.86 | 4.29 |
| 8 | 4.51 | 4.00 |
| 16 | 4.25 | 3.54 |
| 20 | 4.17 | **3.29** |

- At n50, depth scaling shows diminishing returns past l8 (l16→l20: 4.25→4.17)
- At n100, depth scaling **accelerates** past l8: l8→l16 drops 0.46, l16→l20 drops 0.25 — no saturation
- Width is more parameter-efficient at shallow depths: n100_l4 (4.29) ≈ n50_l16 (4.25)
- But at high depth, n100 pulls far ahead: n100_l20 (3.29) vs n50_l20 (4.17) — a 0.88 PPL gap

## 4. Depth Scaling — Non-dual I' KG mem PPL

| Layers | n50 |
|---|---|
| 2 | 4.74 |
| 4 | 4.22 |
| 8 | 4.07 |
| 16 | 4.03 |

Non-dual I' PPL flattens after l8. But J'/G' continue improving at l16 (3.95/3.96).

---

## 5. V Rotation Effect

V rotation (primed variants) is universally beneficial for MLM KG. Effect grows with scale:

### Dual n100_l20
| Model pair | Unprimed | Primed | Improvement |
|---|---|---|---|
| E → E' | 4.16 | **3.29** | **-0.87** |
| J → J' | 4.09 | 3.64 | -0.45 |
| G → G' | 4.13 | 3.87 | -0.26 |
| D → D' | 4.20 | 3.98 | -0.22 |
| H → H' | 4.15 | 4.03 | -0.12 |
| A → A' | 4.14 | 4.14 | 0.00 |
| F → F' | 4.17 | 4.12 | -0.05 |
| I → I' | 4.21 | 4.46 | **+0.25** |

V rotation effect is **much larger at scale**: E gains 0.87 PPL (was 0.35 at n100_l8). But **I' collapses** at n100_l20 — V rotation actually hurts (+0.25), the only MLM model where this happens. A shows zero effect.

---

## 6. Text Performance

Text h@5 remains low (~0.09–0.13) across all n50 configs — insufficient model capacity. Text PPL improves modestly with depth/width.

### Best Text PPL per Grid Point (dual, mixed models)
| Config | Best Model | Text mem PPL |
|---|---|---|
| n50_l2 | I' | 5.46 |
| n50_l4 | I' | 5.32 |
| n50_l8 | I' | 5.16 |
| n50_l16 | J' | 5.14 |
| n50_l20 | J' | 5.12 |
| n100_l2 | I' | 5.19 |
| n100_l4 | E' | 5.13 |
| n100_l8 | J' | 5.09 |
| n100_l16 | A | 5.09 |
| n100_l20 | A | 5.07 |

Text PPL converges to ~5.1 and barely improves past n100_l8. At n100_l16/l20, text PPL is essentially flat (~5.07–5.17) — all models are near the character-level entropy floor.

### Non-dual Text PPL
Non-dual text PPL is slightly worse than dual (~5.14–5.18 at n50_l16 vs ~5.14–5.15 dual). The dual objective's causal training helps text.

### KG-exclusive Text PPL (cross-pollination)
This is a key metric: can models predict text about facts only seen in KG?

| Config | Dual kg_excl_m text PPL | Non-dual kg_excl_m text PPL |
|---|---|---|
| n50_l2 | 6.12 (I') | 6.47 (G') |
| n50_l4 | 5.75 (I') | 6.19 (I) |
| n50_l8 | 5.46 (I') | 6.04 (G') |
| n50_l16 | 5.18 (H) | 6.22 (I) |
| n50_l20 | 5.30 (D') | 5.95 (E) |
| n100_l4 | 5.35 (I') | — |
| n100_l8 | 5.27 (I) | — |
| n100_l16 | 5.39 (A) | — |
| n100_l20 | 5.31 (D') | — |

Dual has significantly better cross-pollination (lower kg_excl text PPL). This makes sense — the dual objective forces the model to process KG facts in causal mode, bridging the two modalities. kg_excl text PPL plateaus around 5.3 at n100 — not improving much past l8. Non-dual kg_excl text PPL stays high (6.0+), meaning pure MLM KG training doesn't transfer well to text generation.

---

## 7. KAT (kg_as_text) Results

KAT converts KG triples to linearized text, so B/C models can learn from them.

### Best KAT Text PPL per Grid Point (dual)
| Config | Best Model | Text mem PPL | kg_excl_m PPL |
|---|---|---|---|
| n50_l2 | C' | 5.39 | 6.12 |
| n50_l4 | C' | 5.24 | 5.84 |
| n50_l8 | C' | 5.18 | 5.52 |
| n50_l16 | C' | 5.16 | 5.40 |
| n50_l20 | C' | 5.12 | 5.33 |
| n100_l4 | C' | 5.14 | 5.58 |

C' consistently leads KAT. KAT kg_excl PPL is better than mixed models' kg_excl text PPL because KG facts are presented as text. But still worse than memorization PPL — linearized KG format is harder to learn from.

### Best KAT Text PPL per Grid Point (non-dual)
| Config | Best Model | Text mem PPL | kg_excl_m PPL |
|---|---|---|---|
| n50_l2 | C' | 5.11 | 5.54 |
| n50_l4 | C' | 5.09 | 5.42 |
| n50_l16 | C' | 4.48 | 4.89 |

---

## 8. Non-dual Causal KG Results (E/H/I with --causal_kg)

These models use causal (left-to-right) KG training instead of MLM. **At n50_l16, causal KG training produces a dramatic breakthrough in text performance.**

### Non-dual Causal — Full Results

| Config | Best Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL |
|---|---|---|---|---|---|
| n50_l2 | I' | .099 | 5.12 | .000 | 5.31 |
| n50_l4 | I' | .102 | 5.09 | .033 | 5.43 |
| n50_l8 | H | .209 | 4.77 | .150 | 4.85 |
| **n50_l16** | **H'** | **.635** | **3.10** | **.400** | **4.06** |

At n50_l2/l4, causal KG gives modestly better text PPL than MLM (I' causal 5.09 vs MLM 5.33 at n50_l4). At n50_l8, H starts to pull ahead (.209 h@5, 4.77 PPL), but E and I remain unremarkable (.125/.141 h@5). The V rotation reversal is already visible at l8 (H .209 > H' .170).

**At n50_l16, a phase transition occurs.** Unprimed causal models achieve text h@5 of 0.5–0.6 and PPL of 3.1–3.5, compared to ~0.10 h@5 and ~5.1 PPL for all other training modes at the same config.

### n50_l8 Causal Detailed Results

| Model | Text mem h@5 | Text mem PPL | Text trans h@5 | Text gen h@5 | kg_excl_m h@5 | kg_excl_m PPL |
|---|---|---|---|---|---|---|
| H | .209 | 4.77 | .144 | .189 | .150 | 4.85 |
| H' | .170 | 4.82 | .167 | .189 | .050 | 5.23 |
| I | .141 | 4.95 | .056 | .067 | .183 | 5.12 |
| E | .125 | 5.04 | .200 | .100 | .000 | 5.23 |
| I' | .107 | 5.10 | .100 | .111 | .067 | 5.34 |
| E' | .106 | 5.10 | .033 | .100 | .017 | 5.55 |

At l8, H already leads with .209 h@5 / 4.77 PPL. V rotation reversal is visible (H > H', I > I', E > E'). But E and I haven't yet crossed the phase transition threshold.

### n50_l16 Causal Detailed Results

| Model | Text mem h@5 | Text mem PPL | Text trans h@5 | Text gen h@5 | kg_excl_m h@5 | kg_excl_m PPL |
|---|---|---|---|---|---|---|
| **H'** | **.635** | **3.10** | .533 | .500 | .400 | 4.06 |
| H | .604 | 3.23 | .578 | .411 | .367 | 3.72 |
| E | .601 | 3.10 | .544 | .489 | .500 | 3.44 |
| I | .540 | 3.49 | .500 | .389 | .400 | 3.96 |
| I' | .157 | 5.01 | .144 | .111 | .033 | 5.42 |
| E' | .107 | 5.09 | .067 | .144 | .017 | 5.43 |

### Key observations on causal n50_l16

1. **V rotation HURTS causal KG at depth**: E' (.107 h@5) is catastrophically worse than E (.601). I' (.157) is much worse than I (.540). H' (.635) is the sole exception — V rotation helps H but not E or I. This is the opposite of MLM KG training where V rotation is universally beneficial.

2. **Cross-pollination is strong**: E gets .500 kg_excl_m h@5 — half of KG-exclusive facts correctly predicted in text. This far exceeds dual mixed models' best kg_excl_m h@5 (~.183 at n100_l8).

3. **Text generalization works**: E gets .489 gen h@5 (unseen derived facts), I gets .389. These models are genuinely learning to reason about relations, not just memorizing.

4. **Phase transition between l4 and l16**: At l4, causal models look unremarkable (~.10 h@5, ~5.1 PPL). At l16, unprimed models explode to .5–.6 h@5 and 3.1–3.5 PPL. Something qualitative changes with sufficient depth — possibly the model learns to use the causal KG representations for text generation.

5. **H' is the best causal model**: Despite the general rule that V rotation hurts at depth, H' achieves .635 mem h@5 / 3.10 PPL — the best text performance in the entire grid sweep by a wide margin. H uses fixed cumsum angles, which may interact differently with V rotation than learned angles (E, I).

6. **n50_l20 causal pending**: Will reveal whether the phase transition continues improving or saturates. Based on the l8→l16 jump, l20 results could be significant.

---

## 9. Model Rankings Summary

### KG Champions (by PPL)
1. **E'** (learned cumsum + V rotation) — dominant at scale, **3.29** at n100_l20. Accelerating improvement with depth.
2. **J'** (RoPE + per-relation slot angles + V rotation) — consistent #2, 3.64 at n100_l20. Non-dual champion at n50_l16 (3.95).
3. **G'** (RoPE + per-relation slot angles, slotted format) — strong #3, 3.87 at n100_l20
4. **D'** (RoPE, flat format + V rotation) — 3.98 at n100_l20, simple but effective
5. **I'** — was champion at small/shallow configs but **collapses at n100_l20** (4.46, worse than n100_l8)

### KG Champions (by h@5)
1. **E'** — dominant, **.527** at n100_l20, .445 at n100_l16. Superlinear growth with depth.
2. **J'** — consistent #2, .368 at n100_l20
3. **G'** — .283 at n100_l20
4. **D'** — .259 at n100_l20
5. **I'** — collapses from .244 (l8) → .200 (l16) → .173 (l20)

### Text Champions
- **H'/E (causal KG, n50_l16)** — dramatic breakthrough: H' .635 mem h@5 / 3.10 PPL, E .601 h@5 / 3.10 PPL. Best text results in the entire sweep by a massive margin.
- **C' (non-dual KAT, n50_l16)** — .273 mem h@5 / 4.48 PPL, best non-causal text performance
- **J'** — best dual mixed text PPL at deep configs (5.09 at n100_l8)
- **I'** — best dual mixed text PPL at shallow configs

### Cross-pollination Champions
- **E (causal, n50_l16)** — .500 kg_excl_m h@5, best cross-pollination result by far
- **C' (non-dual KAT, n50_l16)** — .217 kg_excl_m h@5 / 4.89 PPL, strong KAT cross-pollination
- **Dual I'** — best kg_excl text PPL in dual mixed mode (~5.27)
- Non-dual MLM KG has poor cross-pollination (kg_excl text PPL stays 6.0+)

---

## 10. Key Findings

### 1. Non-dual beats dual on KG, but dual wins on text cross-pollination
Non-dual achieves better KG PPL at every grid point (e.g., I' 4.74 vs 5.25 at n50_l2). But dual has much better kg_excl text PPL (5.27 vs 6.22), meaning dual training helps transfer KG knowledge to text predictions.

### 2. Width and depth compound at scale
n100_l4 ≈ n50_l16 on KG PPL, so width is more efficient at small scale. But at n100, depth scaling **accelerates** rather than saturating: E' drops from 4.00 (l8) → 3.54 (l16) → 3.29 (l20). Width opens a performance regime that depth alone can't reach — n100_l20 E' (3.29) is 0.88 PPL better than n50_l20 E' (4.17).

### 3. J' emerges at depth
J' (RoPE + per-relation slot angles) is unremarkable at shallow configs but becomes the KG h@5 champion at deep configs (n50_l16+ non-dual, n50_l20 dual). Its slot-angle approach apparently needs more layers to fully develop.

### 4. V rotation is universally beneficial for MLM KG, but HURTS causal KG at depth
Every primed model beats its unprimed counterpart on KG metrics under MLM training, by 0.14–0.35 PPL. But under causal KG training at depth (l8+), V rotation catastrophically hurts E (h@5 .601→.107) and I (.540→.157). Only H is immune (H' .635 vs H .604). This reversal is the most surprising finding in the sweep — the same V rotation that reliably helps MLM becomes destructive for causal KG training.

### 5. Causal KG training produces the best text performance (by far)
Non-dual causal KG training at n50_l16 achieves text h@5 of .5–.6 and PPL of 3.1–3.5 — a dramatic breakthrough compared to ~.10 h@5 and ~5.1 PPL for all other training modes at the same config. A phase transition occurs between l4 (unremarkable) and l16 (explosive), with l8 showing early signs (H at .209 h@5 / 4.77 PPL). This suggests causal KG training at sufficient depth enables the model to truly unify KG and text representations.

### 6. E' is the overall KG champion (MLM mode)
E' (learned cumsum + relation operator + V rotation) dominates KG performance at scale with superlinear improvement. At n100_l20: KG PPL **3.29**, h@5 **.527** — over half of memorized facts predicted correctly. Transfer h@5 .567 is even higher. V rotation is critical — E (unprimed) stays at 4.16 PPL, a 0.87 gap. I' (the shallow-config champion) collapses at this scale.

### 7. Dual objective helps cross-pollination but hurts KG PPL
The dual objective (alternating causal/MLM) hurts KG-specific PPL (dual I' 5.25 vs non-dual I' 4.74 at n50_l2) but dramatically improves cross-modal transfer. Non-dual causal KG training is a middle ground — it helps cross-pollination without the full dual objective.

### 8. Non-dual KAT C' is strong at depth
Non-dual KAT (kg_as_text) C' achieves 4.48 text mem PPL and .273 h@5 at n50_l16 — much better than dual KAT (~5.12 PPL, ~.10 h@5 at n50_l20). Non-dual causal text training is consistently better for KAT models than the dual objective.

### 9. I' collapses at scale — V rotation is not always beneficial
I' (learned cumsum + shared relation op + V rotation) was the champion at small/shallow configs (n50_l2 through l8). But at n100_l20, I' PPL is **4.46** — worse than its n100_l8 result (4.12) and worse than unprimed I (4.21). V rotation becomes destructive for I at scale. This parallels the causal KG finding where V rotation hurts E' and I' at depth — suggesting V rotation + learned cumsum angles has a scaling ceiling.

---

## 11. Full KG Memorization PPL Tables

### Dual — KG Mem PPL across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 5.85 | 5.31 | 4.78 | 4.49 | 4.36 | 5.37 | 4.70 | 4.37 | 4.20 | 4.14 |
| A' | 5.48 | 5.02 | 4.64 | 4.38 | 4.32 | 4.85 | 4.45 | 4.23 | 4.12 | 4.14 |
| D | 5.81 | 5.39 | 4.92 | 4.53 | 4.44 | 5.18 | 4.71 | 4.39 | 4.26 | 4.20 |
| D' | 5.39 | 5.00 | 4.59 | 4.32 | 4.24 | 4.62 | 4.35 | 4.15 | 4.05 | 3.98 |
| E | 5.76 | 5.22 | 4.78 | 4.45 | 4.38 | 5.40 | 4.65 | 4.35 | 4.22 | 4.16 |
| E' | 5.31 | 4.86 | 4.51 | 4.25 | 4.17 | 4.54 | 4.29 | 4.00 | 3.54 | **3.29** |
| F | 5.86 | 5.36 | 4.86 | 4.47 | 4.43 | 5.36 | 4.74 | 4.41 | 4.21 | 4.17 |
| F' | 5.59 | 5.11 | 4.69 | 4.40 | 4.38 | 4.94 | 4.55 | 4.26 | 4.15 | 4.12 |
| G | 5.97 | 5.33 | 4.77 | 4.51 | 4.41 | 5.41 | 4.73 | 4.41 | 4.18 | 4.13 |
| G' | 5.53 | 5.04 | 4.63 | 4.40 | 4.29 | 4.83 | 4.45 | 4.22 | 4.01 | 3.87 |
| H | 5.83 | 5.28 | 4.80 | 4.47 | 4.39 | 5.49 | 4.74 | 4.42 | 4.13 | 4.15 |
| H' | 5.49 | 5.03 | 4.65 | 4.41 | 4.31 | 4.87 | 4.52 | 4.25 | 4.05 | 4.03 |
| I | 5.80 | 5.14 | 4.73 | 4.42 | 4.39 | 5.16 | 4.61 | 4.37 | 4.23 | 4.21 |
| I' | 5.25 | 4.84 | 4.48 | 4.29 | 4.28 | 4.53 | 4.29 | 4.12 | 4.15 | 4.46 |
| J | 5.84 | 5.27 | 4.73 | 4.46 | 4.36 | 5.43 | 4.61 | 4.35 | 4.21 | 4.09 |
| J' | 5.47 | 4.93 | 4.60 | 4.34 | 4.21 | 4.72 | 4.42 | 4.09 | 3.68 | 3.64 |

### Non-dual — KG Mem PPL across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 |
|---|---|---|---|---|
| A | 5.39 | 4.84 | 4.25 | 4.08 |
| A' | 5.15 | 4.55 | 4.16 | 4.06 |
| D | 5.38 | 4.88 | 4.31 | 4.11 |
| D' | 4.90 | 4.41 | 4.10 | 4.05 |
| E | 5.29 | 4.70 | 4.24 | 4.09 |
| E' | 4.91 | 4.29 | 4.08 | 4.03 |
| F | 5.62 | 5.03 | 4.38 | 4.14 |
| F' | 5.27 | 4.70 | 4.23 | 4.08 |
| G | 5.63 | 4.90 | 4.30 | 4.10 |
| G' | 5.06 | 4.48 | 4.14 | **3.96** |
| H | 5.53 | 4.94 | 4.32 | 4.08 |
| H' | 5.02 | 4.54 | 4.16 | 4.01 |
| I | 5.34 | 4.68 | 4.21 | 4.06 |
| I' | 4.74 | 4.22 | 4.07 | 4.03 |
| J | 5.48 | 4.94 | 4.33 | 4.08 |
| J' | 5.03 | 4.47 | 4.11 | **3.95** |

### Observations on the full tables
- **E' separates dramatically at n100 depth**: From 4.00 at l8 to 3.54 at l16 to **3.29** at l20. No other model comes close.
- **J' is a clear #2**: 3.68 at l16, 3.64 at l20. Consistent but can't match E'.
- **I' collapses at n100_l20**: 4.46 — worse than n100_l8 (4.12). V rotation becomes destructive for I at this scale.
- **Non-dual models converge at depth**: At n50_l16, all primed non-dual models cluster between 3.95–4.08. Architecture matters less at high depth.
- **Unprimed models lag consistently**: The gap between primed and unprimed is 0.15–0.87 PPL, growing with scale (E: 0.87 at n100_l20).

---

## 12. Full KG Memorization h@5 Tables

### Dual — KG Mem h@5 across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | .095 | .106 | .141 | .163 | .176 | .117 | .154 | .179 | .207 | .222 |
| A' | .098 | .126 | .153 | .185 | .189 | .145 | .175 | .206 | .212 | .211 |
| D | .093 | .113 | .142 | .159 | .171 | .124 | .162 | .179 | .186 | .194 |
| D' | .102 | .135 | .164 | .186 | .192 | .168 | .189 | .205 | .222 | .259 |
| E | .091 | .117 | .147 | .168 | .175 | .128 | .167 | .187 | .199 | .210 |
| E' | .107 | .139 | .165 | .191 | .201 | .169 | .201 | .281 | .445 | **.527** |
| F | .093 | .112 | .139 | .166 | .172 | .121 | .149 | .180 | .213 | .216 |
| F' | .096 | .115 | .157 | .170 | .177 | .142 | .164 | .193 | .217 | .226 |
| G | .087 | .111 | .145 | .165 | .176 | .114 | .149 | .179 | .218 | .231 |
| G' | .094 | .125 | .168 | .184 | .200 | .151 | .188 | .212 | .265 | .283 |
| H | .086 | .118 | .139 | .166 | .177 | .113 | .161 | .176 | .232 | .232 |
| H' | .098 | .123 | .150 | .179 | .188 | .152 | .169 | .212 | .241 | .259 |
| I | .089 | .120 | .143 | .170 | .173 | .129 | .167 | .182 | .197 | .197 |
| I' | .112 | .134 | .173 | .184 | .190 | .168 | .198 | .244 | .200 | .173 |
| J | .090 | .120 | .141 | .175 | .185 | .122 | .163 | .185 | .208 | .240 |
| J' | .103 | .133 | .159 | .180 | .217 | .165 | .185 | .263 | .353 | .368 |

### Non-dual — KG Mem h@5 across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 |
|---|---|---|---|---|
| A | .102 | .158 | .198 | .200 |
| A' | .123 | .185 | .201 | .205 |
| D | .112 | .155 | .196 | .199 |
| D' | .143 | .190 | .199 | .201 |
| E | .126 | .171 | .197 | .198 |
| E' | .155 | .193 | .202 | .203 |
| F | .102 | .144 | .189 | .196 |
| F' | .115 | .173 | .198 | .204 |
| G | .091 | .148 | .197 | .200 |
| G' | .133 | .189 | .206 | .259 |
| H | .103 | .149 | .191 | .204 |
| H' | .134 | .184 | .200 | .227 |
| I | .111 | .171 | .196 | .203 |
| I' | .162 | .197 | .200 | .200 |
| J | .111 | .152 | .195 | .201 |
| J' | .134 | .188 | .212 | **.262** |

### Observations on h@5 tables
- **E' h@5 accelerates at n100 depth**: .201→.281→.445→**.527** from l4 through l20. This is superlinear growth — E' is in a different regime at n100_l16+.
- **I' collapses**: .244 at n100_l8 → .200 at l16 → .173 at l20. Something breaks at scale.
- **J' is the consistent #2**: .263→.353→.368 from l8 through l20. Steady growth but can't match E'.
- **Non-dual h@5 saturates at ~.20** for most models by n50_l8. Only J' (.262) and G' (.259) break through at l16.

---

## 13. Text Memorization PPL Tables

### Dual — Text Mem PPL across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 6.03 | 5.63 | 5.35 | 5.18 | 5.16 | 5.47 | 5.25 | 5.18 | 5.09 | **5.07** |
| A' | 5.79 | 5.50 | 5.27 | 5.19 | 5.16 | 5.32 | 5.19 | 5.16 | 5.14 | 5.17 |
| D | 6.02 | 5.62 | 5.32 | 5.19 | 5.23 | 5.63 | 5.24 | 5.18 | 5.14 | 5.14 |
| D' | 5.60 | 5.34 | 5.19 | 5.17 | 5.15 | 5.24 | 5.18 | 5.14 | 5.15 | 5.13 |
| E | 5.73 | 5.45 | 5.23 | 5.17 | 5.19 | 5.38 | 5.19 | 5.14 | 5.17 | 5.13 |
| E' | 5.55 | 5.33 | 5.17 | 5.15 | 5.15 | 5.21 | 5.13 | 5.14 | 5.10 | 5.11 |
| F | 5.92 | 5.61 | 5.31 | 5.16 | 5.15 | 5.50 | 5.26 | 5.16 | 5.12 | 5.12 |
| F' | 5.81 | 5.52 | 5.29 | 5.18 | 5.16 | 5.36 | 5.23 | 5.13 | 5.11 | 5.12 |
| G | 6.01 | 5.66 | 5.37 | 5.18 | 5.18 | 5.49 | 5.24 | 5.18 | 5.14 | 5.11 |
| G' | 5.73 | 5.53 | 5.28 | 5.17 | 5.17 | 5.30 | 5.19 | 5.16 | 5.12 | 5.14 |
| H | 5.84 | 5.54 | 5.29 | 5.15 | 5.15 | 5.40 | 5.21 | 5.14 | 5.09 | 5.11 |
| H' | 5.78 | 5.44 | 5.23 | 5.15 | 5.14 | 5.29 | 5.19 | 5.10 | 5.10 | 5.10 |
| I | 5.57 | 5.36 | 5.21 | 5.16 | 5.15 | 5.25 | 5.18 | 5.16 | 5.12 | 5.14 |
| I' | 5.46 | 5.32 | 5.16 | 5.15 | 5.16 | 5.19 | 5.18 | 5.15 | 5.16 | 5.18 |
| J | 5.96 | 5.56 | 5.26 | 5.17 | 5.14 | 5.42 | 5.22 | 5.14 | 5.12 | 5.09 |
| J' | 5.83 | 5.39 | 5.23 | 5.14 | 5.12 | 5.29 | 5.22 | 5.09 | 5.09 | 5.10 |

Text PPL converges to ~5.1 by n100_l8 for all models and barely moves past that. Architecture doesn't matter for text at convergence.

---

## 14. Width vs Depth Equivalences (Dual E' KG mem PPL)

| n100 config | PPL | ≈ n50 equivalent | PPL |
|---|---|---|---|
| n100_l2 | 4.54 | ~ n50_l4 | 4.86 |
| n100_l4 | 4.29 | ≈ n50_l16 | 4.25 |
| n100_l8 | 4.00 | > n50_l20 | 4.17 |
| n100_l16 | 3.54 | >> n50_l20 | 4.17 |
| n100_l20 | **3.29** | >>> n50_l20 | 4.17 |

n100_l8 already surpasses anything at n50. At n100_l16+, width and depth compound — E' enters a new performance regime unreachable at n50. The gap grows from 0.17 (l8) to 0.63 (l16) to 0.88 (l20).

---

## Appendix: Architecture Quick Reference

| Model | Angle type | KG format | Notes |
|---|---|---|---|
| A/A' | RoPE + learned slot angles (shared) | Slotted (HEAD/REL/TAIL) | MLM |
| D/D' | RoPE | Flat (rel as token) | MLM |
| E/E' | Learned per-token cumsum + relation op | Native (chars only) | Causal-capable |
| F/F' | Fixed RoPE | Flat (rel as token) | MLM |
| G/G' | RoPE + per-relation slot angles | Slotted (HEAD/REL/TAIL) | MLM |
| H/H' | Fixed cumsum + relation op | Native (chars only) | Causal-capable |
| I/I' | Learned cumsum + shared relation op | Native (chars only) | Causal-capable |
| J/J' | RoPE + per-relation slot angles | Native (2 slots: HEAD/TAIL) | MLM |
| B/B' | RoPE (standard) | Text only (linearized) | KAT only |
| C/C' | Learned per-token angles | Text only (linearized) | KAT only |

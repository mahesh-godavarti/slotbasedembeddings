# Exp 7 Grid Sweep Summary

## Setup
- **1000 chains**, 100K iterations, 1 seed, softmax attention
- **Grid**: n_embed × n_layers: 50×{2,4,8,16,20}, 100×{2,4,8,16,20}, 250×{2,4,8}, 500×{2,4}
- **Dual**: `kg_text_experiment_dual.py` — random alternation between causal and MLM objectives
- **Non-dual**: `kg_text_experiment.py` — standard causal text + MLM KG training
- **Models**: A/A' through K/K' (primed = V rotation).
- **Groups per grid point**:
  - Dual: mixed (A-J') + KAT (B/C with --kg_as_text)
  - Non-dual: mixed (A-J') + causal (E/H/I with --causal_kg) + KAT (B/C with --kg_as_text)

## Completion Status (as of May 2, 2026)

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
| n100_l20 | Done | Done | Complete |
| n250_l2 | Done | Done | Complete |
| n250_l4 | Done | Done | Complete |
| n250_l8 | Done | Done | Complete |
| n500_l2 | Done | Done | Complete |
| n500_l4 | Done | Done | Complete |

### Non-dual
| Grid Point | mixed | causal | kat | Status |
|---|---|---|---|---|
| n50_l2 | Done | Done | Done | Complete |
| n50_l4 | Done | Done | Done | Complete |
| n50_l8 | Done | Done | Done | Complete |
| n50_l16 | Done | Done | Done | Complete |
| n50_l20 | Done | Done | Done | Complete |
| n100_l2 | Done | Done | Done | Complete |
| n100_l4 | Done | Done | Done | Complete |
| n100_l8 | Done | Done | Done | Complete |
| n100_l16 | Done | Done | Done | Complete |
| n100_l20 | Done | Done | Done | Complete |
| n250_l2 | Done | Done | Done | Complete |
| n250_l4 | Done | Done | Done | Complete |
| n250_l8 | Done | Done | Done | Complete |
| n500_l2 | Done | Done | Done | Complete |
| n500_l4 | Done | Done | Done | Complete |

### K/K' Backfill
| Grid Point | dual | non-dual | Status |
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
| n100_l20 | Done | Done | Complete |
| n250_l2 | Done | Done | Complete |
| n250_l4 | Done | Done | Complete |
| n250_l8 | Done | Done | Complete |
| n500_l2 | Done | Done | Complete |
| n500_l4 | Done | Done | Complete |

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
| n100_l20 | E' | 3.29 | J' | 3.64 |
| n250_l2 | I' | 3.76 | E' | 3.97 |
| n250_l4 | I' | 2.87 | E' | 3.03 |
| n250_l8 | E' | 1.88 | D' | 2.88 |
| n500_l2 | I' | 2.60 | E' | 2.96 |
| **n500_l4** | **E'** | **1.52** | **D'** | **2.71** |

### Non-dual (lower = better)
| Config | Best Model | KG mem PPL | #2 Model | #2 PPL |
|---|---|---|---|---|
| n50_l2 | I' | 4.74 | D' | 4.90 |
| n50_l4 | I' | 4.22 | E' | 4.29 |
| n50_l8 | I' | 4.07 | E' | 4.08 |
| n50_l16 | J' | 3.95 | G' | 3.96 |
| n50_l20 | J' | 3.83 | G' | 3.92 |
| n100_l2 | I' | 4.07 | E' | 4.09 |
| n100_l4 | J' | 3.98 | I' | 4.04 |
| n100_l8 | J' | 3.59 | G' | 3.73 |
| **n100_l16** | **J'** | **2.76** | **G'** | **3.07** |
| **n100_l20** | **J'** | **2.43** | **K'** | **2.45** |
| n250_l2 | I'/E'/G' | 4.00 | K' | 3.77 |
| n250_l4 | J'/G' | 3.34 | E' | 3.49 |
| n250_l8 | K' | 1.87 | E' | 2.34 |
| n500_l2 | E' | 3.23 | J' | 3.59 |
| **n500_l4** | **E'** | **1.88** | **D'** | **2.23** |

**Takeaway**: E' essentially solves KG memorization at n500_l4 (**1.52 PPL**, **.960** h@5 dual; **1.88 PPL**, **.887** h@5 non-dual). K' is the new #2: **1.87** non-dual at n250_l8, **2.24** dual at n500_l4 — survives where I'/J' diverge. I' dominates width (**2.60** at n500_l2) but diverges at depth. Non-dual J' 2.43 and K' 2.45 are nearly tied at n100_l20. J' survives n500_l4 in non-dual (3.27) but diverged in dual. D' non-dual 2.23 at n500_l4 beats dual 2.71.

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
| n100_l20 | E' | .527 | J' | .368 |
| n250_l2 | I' | .418 | E' | .310 |
| n250_l4 | I' | .704 | E' | .628 |
| n250_l8 | E' | .907 | D' | .674 |
| n500_l2 | I' | .805 | E' | .640 |
| **n500_l4** | **E'** | **.960** | **D'** | **.719** |

### Non-dual
| Config | Best Model | KG mem h@5 | #2 Model | #2 h@5 |
|---|---|---|---|---|
| n50_l2 | I' | .162 | E' | .155 |
| n50_l4 | I' | .197 | E' | .193 |
| n50_l8 | J' | .212 | G' | .206 |
| n50_l16 | J' | .262 | G' | .259 |
| n50_l20 | J' | .313 | G' | .278 |
| n100_l2 | I' | .202 | J' | .208 |
| n100_l4 | J' | .266 | A' | .219 |
| n100_l8 | J' | .373 | G' | .334 |
| **n100_l16** | **J'** | **.628** | **K'** | **.559** |
| **n100_l20** | **J'** | **.751** | **K'** | **.750** |
| n250_l2 | G' | .248 | I' | .245 |
| n250_l4 | J' | .441 | E' | .431 |
| n250_l8 | K' | .895 | E' | .777 |
| n500_l2 | E' | .537 | I' | .415 |
| **n500_l4** | **E'** | **.887** | **D'** | **.794** |

**Takeaway**: E' achieves **.960** dual / **.887** non-dual h@5 at n500_l4. K' is #2: **.895** non-dual at n250_l8, **.880** dual at n500_l4. D' non-dual .794 at n500_l4 beats dual .719. J' survives n500_l4 in non-dual (.443) but diverges in dual. I' diverges at n250_l8+ but achieves **.805** at n500_l2 via width.

---

## 3. Width × Depth Scaling — Dual KG mem PPL (best model per cell)

| Layers | n50 (best) | n100 (best) | n250 (best) | n500 (best) |
|---|---|---|---|---|
| 2 | 5.25 (I') | 4.53 (I') | 3.76 (I') | **2.60 (I')** |
| 4 | 4.84 (I') | 4.29 (E'/I') | 2.87 (I') | **1.52 (E')** |
| 8 | 4.48 (I') | 4.00 (E') | 1.88 (E') | — |
| 16 | 4.25 (E') | 3.54 (E') | — | — |
| 20 | 4.17 (E') | 3.29 (E') | — | — |

- **E' dominates depth scaling**: 4.00 (l8) → 3.54 (l16) → 3.29 (l20) at n100; 1.88 at n250_l8; **1.52** at n500_l4 — KG essentially solved
- **I' dominates width scaling at l2**: 5.25 → 4.53 → 3.76 → **2.60** at l2. Also 2.87 at n250_l4. But **diverges at n500_l4**.
- **I/I'/J/J' all diverge at n500_l4**: Only E', D', and the simpler models (A', F', G', H') survive at this scale.
- **D' is the robust #2**: 2.88 at n250_l8, **2.71** at n500_l4. Scales in both directions without failure.
- **E' at n500_l4 (.960 h@5)**: KG-exclusive h@5 .967 — even unseen KG facts predicted near-perfectly.

## 4. Depth Scaling — Non-dual KG mem PPL (best model)

| Layers | n50 | n100 | n250 | n500 |
|---|---|---|---|---|
| 2 | 4.74 (I') | 4.07 (I') | 4.00 (I'/E'/G') | 3.23 (E') |
| 4 | 4.22 (I') | 3.98 (J') | 3.34 (J'/G') | **1.88 (E')** |
| 8 | 4.07 (I') | 3.59 (J') | **1.87 (K')** | — |
| 16 | 3.95 (J') | 2.76 (J') | — | — |
| 20 | 3.83 (J') | 2.43 (J') | — | — |

Non-dual E' at n500_l4 achieves 1.88/.887 — nearly matching dual E' (1.52/.960). K' non-dual at n250_l8 (1.87/.895) outperforms E' dual (1.88/.907) at the same config! J' survives n500_l4 in non-dual (3.27/.443) but diverged in dual. I/I' diverge at n250_l8 (same as dual).

---

## 5. V Rotation Effect

V rotation (primed variants) is beneficial for MLM KG but can catastrophically fail at scale.

### Dual n250_l8 — V rotation at large scale
| Model pair | Unprimed | Primed | Improvement |
|---|---|---|---|
| E → E' | 4.25 | **1.88** | **-2.37** |
| D → D' | 4.31 | 2.88 | **-1.43** |
| J → J' | 4.26 | 3.37 | -0.89 |
| G → G' | 4.30 | 3.83 | -0.47 |
| H → H' | 4.29 | 3.94 | -0.35 |
| A → A' | 4.28 | 4.09 | -0.19 |
| F → F' | 4.31 | 4.13 | -0.18 |
| I → I' | 4.23 | **79.39** | **+75.16 (DIVERGED)** |

V rotation effect is **massive at n250_l8**: E' gains 2.37 PPL (was 0.87 at n100_l20, 0.35 at n100_l8). D' gains 1.43 — simple RoPE benefits enormously. But **I' completely diverges** (PPL 79.39 dual, billions non-dual). At n500_l4 dual, V rotation is even more critical: E' 1.52 vs E 4.48 (gain of **2.96**), D' 2.71 vs D 4.51 (gain of **1.80**). At n500_l4 non-dual: E' 1.88 vs E 4.09 (gain **2.21**), D' 2.23 vs D 4.10 (gain **1.87**). I diverges at n250_l8+ but I' survives at n500 non-dual (3.41) while diverging in dual.

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
| n250_l2 | D' | 5.19 |
| n250_l4 | J' | 5.03 |
| n250_l8 | H' | 4.84 |
| n500_l2 | I' | 4.14 |
| n500_l4 | E' | 4.21 |

Text PPL converges to ~5.1 at deep configs. But **I' at n500_l2 achieves 4.14** — a dramatic breakout. This is because I' is trained with width-scaled embeddings that also improve text. n250_l8 H' (4.84) and J' (4.54) also improve noticeably. I' at n250_l8 has catastrophic text PPL (34.06) due to its training divergence.

### Non-dual Text PPL
Non-dual text PPL is slightly worse than dual (~5.14–5.18 at n50_l16 vs ~5.14–5.15 dual). The dual objective's causal training helps text. At n50_l20, non-dual text PPL stays ~5.1 for most models.

### KG-exclusive Text PPL (cross-pollination)
This is a key metric: can models predict text about facts only seen in KG?

| Config | Dual kg_excl_m text PPL | Non-dual kg_excl_m text PPL |
|---|---|---|
| n50_l2 | 6.12 (I') | 6.47 (G') |
| n50_l4 | 5.75 (I') | 6.19 (I) |
| n50_l8 | 5.46 (I') | 6.04 (G') |
| n50_l16 | 5.18 (H) | 6.22 (I) |
| n50_l20 | 5.30 (D') | 6.31 (I) |
| n100_l2 | — | 6.49 (H') |
| n100_l4 | 5.35 (I') | 6.18 (J') |
| n100_l8 | 5.27 (I) | 6.54 (F/F'/I) |
| n100_l16 | 5.39 (A) | 6.07 (G') |
| n100_l20 | 5.31 (D') | 6.03 (I') |
| n250_l2 | 5.64 (D) | 6.54 (G') |
| n250_l4 | 5.27 (I) | 6.55 (F') |
| n250_l8 | 5.21 (I) | 6.65 (G/J) |
| n500_l2 | 5.22 (I') | 6.71 (D) |
| n500_l4 | 5.17 (H) | 6.88 (H) |

Dual cross-pollination (kg_excl text PPL) plateaus around 5.2–5.3 across all scales. Non-dual kg_excl text PPL stays high (6.0–7.0) and actually gets WORSE at n500 (6.88 vs 6.03 at n100_l20). Models that improve KG PPL at n500 (E', D', J') see their text kg_excl PPL deteriorate — the KG knowledge is being stored in angle mechanisms that don't help text generation.

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
| n100_l20 | C' | 5.03 | 5.22 |
| n250_l4 | C' | 4.96 | 5.28 |
| n250_l8 | C' | 4.05 | 4.52 |
| n500_l2 | C' | 4.82 | 5.02 |
| n500_l4 | C' | 4.12 | 4.61 |

C' consistently leads dual KAT. At n500_l4, C' achieves **4.12** text PPL and .351 h@5. KAT kg_excl PPL scales with model capacity (5.52 at n50_l8 → 4.52 at n250_l8 → 4.61 at n500_l4).

### Best KAT Text PPL per Grid Point (non-dual)
| Config | Best Model | Text mem PPL | kg_excl_m PPL |
|---|---|---|---|
| n50_l2 | C' | 5.11 | 5.54 |
| n50_l4 | C' | 5.09 | 5.42 |
| n50_l16 | C' | 4.48 | 4.89 |
| n50_l20 | C' | 3.62 | 4.50 |
| n100_l2 | C' | 4.91 | 5.25 |
| n100_l8 | C' | 1.68 | 2.59 |
| **n100_l16** | **C'** | **1.02** | **1.12** |
| n100_l20 | C' | 1.01 | 1.04 |
| n250_l2 | C' | 1.10 | 1.27 |
| n250_l4 | C' | 1.01 | 1.05 |
| **n250_l8** | **C'** | **1.00** | **1.01** |
| n500_l2 | C' | 1.01 | 1.02 |
| **n500_l4** | **C'** | **1.00** | **1.02** |

Non-dual KAT C' shows explosive scaling: 4.91 (n100_l2) → 1.68 (l8) → **1.02** (l16) → **1.00** (n250_l8, n500_l4). At n100_l16+, C' achieves **perfect 1.000 h@5** on all tiers. At n250_l8 and n500_l4, C' reaches **1.00 PPL** — the theoretical minimum. B/B' also reach .97–1.00 h@5 at n250_l8+. KAT is completely solved from n100_l16 onward.

---

## 8. Non-dual Causal KG Results (E/H/I with --causal_kg)

These models use causal (left-to-right) KG training instead of MLM. **At n50_l16, causal KG training produces a dramatic breakthrough in text performance.**

### Non-dual Causal — Full Results

| Config | Best Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL |
|---|---|---|---|---|---|
| n50_l2 | I' | .099 | 5.12 | .000 | 5.31 |
| n50_l4 | I' | .102 | 5.09 | .033 | 5.43 |
| n50_l8 | H | .209 | 4.77 | .150 | 4.85 |
| n50_l16 | H' | .635 | 3.10 | .400 | 4.06 |
| n50_l20 | H | .817 | 2.28 | .700 | 2.94 |
| n100_l2 | I | .137 | 4.90 | .050 | 5.04 |
| n100_l4 | I' | .884 | 1.76 | .817 | 2.05 |
| n100_l8 | H | .975 | 1.36 | .983 | 1.51 |
| **n100_l16** | **H'** | **.996** | **1.13** | **.983** | **1.37** |
| n100_l20 | H' | .998 | 1.10 | .983 | 1.18 |
| n250_l2 | I' | .945 | 1.20 | .933 | 1.36 |
| n250_l4 | I' | 1.000 | 1.06 | .917 | 1.24 |
| n250_l8 | E'/H'/I | 1.000/.999 | 1.01 | .950 | 1.25 |
| n500_l2 | E' | 1.000 | 1.01 | .767 | 2.18 |
| **n500_l4** | **I'/E'** | **1.000** | **1.01** | **.950** | **1.33** |

At n50_l2/l4, causal KG gives modestly better text PPL than MLM (I' causal 5.09 vs MLM 5.33 at n50_l4). At n50_l8, H starts to pull ahead (.209 h@5, 4.77 PPL). A phase transition occurs at n50_l16 where unprimed models explode to .5–.6 h@5 / 3.1–3.5 PPL, continuing to .8+ h@5 / 2.3 PPL at l20.

**At n100_l4, I' shatters records** with .884 text h@5 / 1.76 PPL. Width accelerates the phase transition: at n50, it needed l16+; at n100, it's already strong at l4. **At n100_l8, all models except E' converge to near-perfect text prediction** (.929–.975 h@5, 1.29–1.56 PPL). **At n100_l16/l20, the system is near-perfect**: H achieves **1.000 kg_excl_m h@5** (perfect cross-pollination), H/H' get .983 txtExcl_m (near-perfect reverse cross-pollination), and E' partially recovers (.672 at l20, up from .109 at l8).

**At n250, width unlocks further breakthroughs**: I' achieves .945 h@5 / 1.20 PPL at just l2 (2 layers!). At n250_l4, I' reaches **1.000 h@5 / 1.06 PPL** — perfect. At n250_l8, E/E'/H/H'/I all achieve .999–1.000 h@5, but **I' diverges** (PPL 15.49). **E' FULLY RECOVERS**: 1.000 h@5 / 1.01 PPL at n250_l8, up from .672 at n100_l20.

**At n500, causal is essentially solved**: E'/I' both achieve **1.000 h@5 / 1.01 PPL** at n500_l4. All models except I' (at n250_l8 only) reach .95+ h@5. I' recovers at n500 — .999/1.04 at l2, 1.000/1.01 at l4. The causal phase transition is complete.

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

### n50_l20 Causal Detailed Results

| Model | Text mem h@5 | Text mem PPL | Text trans h@5 | Text gen h@5 | kg_excl_m h@5 | kg_excl_m PPL |
|---|---|---|---|---|---|---|
| **H** | **.817** | **2.28** | .767 | .567 | .700 | 2.94 |
| E | .792 | 2.33 | .822 | .600 | **.717** | **2.74** |
| I | .784 | 2.41 | .700 | .589 | .617 | 2.93 |
| H' | .728 | 2.71 | .667 | .511 | .483 | 4.07 |
| I' | .709 | 2.90 | .611 | .533 | .667 | 3.49 |
| E' | .117 | 5.07 | .144 | .078 | .050 | 5.46 |

All unprimed models achieve excellent text performance (.784–.817 h@5, 2.28–2.41 PPL). I' recovers at l20 (.709 h@5, 2.90 PPL vs .157/5.01 at l16) — its l16 failure was temporary. But E' remains catastrophically broken (.117 h@5, 5.07 PPL). Cross-pollination is remarkable: E achieves .717 kg_excl_m h@5 / 2.74 PPL.

### n100_l2 Causal — Pre-transition

| Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL |
|---|---|---|---|---|
| I | .137 | 4.90 | .050 | 5.04 |
| I' | .117 | 5.04 | .067 | 5.53 |
| H | .112 | 5.04 | .117 | 5.12 |
| H' | .105 | 5.09 | .050 | 5.27 |
| E | .102 | 5.03 | .033 | 5.36 |
| E' | .100 | 5.13 | .050 | 5.43 |

At n100_l2, causal models are unremarkable — no phase transition yet. I has a slight edge (.137 h@5 / 4.90 PPL).

### n100_l4 Causal Detailed Results — **I' BREAKTHROUGH**

| Model | Text mem h@5 | Text mem PPL | Text trans h@5 | Text gen h@5 | kg_excl_m h@5 | kg_excl_m PPL |
|---|---|---|---|---|---|---|
| **I'** | **.884** | **1.76** | .844 | .611 | **.817** | **2.05** |
| I | .544 | 3.44 | .544 | .489 | .500 | 3.72 |
| H' | .542 | 3.39 | .478 | .433 | .283 | 4.13 |
| H | .521 | 3.51 | .444 | .467 | .417 | 4.07 |
| E | .418 | 3.87 | .389 | .300 | .317 | 4.09 |
| E' | .117 | 5.09 | .156 | .078 | .017 | 5.42 |

**I' at n100_l4 is the best text result in the entire grid sweep.** .884 h@5 / 1.76 PPL — nearly 90% of memorized facts predicted correctly in text. Cross-pollination is near-perfect: .817 kg_excl_m h@5 / 2.05 PPL. Key insights:

1. **V rotation HELPS I' at n100 causal**: I' .884 vs I .544 — a massive improvement. At n50_l16, V rotation hurt I' (.157 vs .540). The interaction between V rotation and causal training depends on width.
2. **Width accelerates the phase transition**: At n50, the causal breakthrough needed l16 (16 layers). At n100, it occurs at l4 (4 layers). This is a 4× reduction in depth requirement.
3. **I' surpasses H**: At n50, H dominated causal training. At n100, I' takes the lead — its shared relation operator benefits from wider embeddings.
4. **E' remains broken**: .117 h@5 / 5.09 PPL. Learned cumsum + V rotation is incompatible with causal KG at any scale tested.

### n100_l8 Causal Detailed Results — **ALL MODELS SOLVE TEXT**

| Model | Text mem h@5 | Text mem PPL | Text trans h@5 | Text gen h@5 | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|---|---|
| **H** | **.975** | **1.36** | .956 | .678 | **.983** | **1.51** | .667 |
| I | .958 | 1.53 | .933 | .700 | .933 | 1.74 | .450 |
| H' | .952 | 1.47 | .967 | .656 | .900 | 2.05 | .550 |
| E | .939 | 1.56 | .878 | .667 | .850 | 1.81 | .533 |
| **I'** | .929 | **1.29** | .900 | .667 | .917 | **1.37** | **.700** |
| E' | .109 | 5.11 | .044 | .122 | .117 | 5.84 | .050 |

**n100_l8 is the causal saturation point.** All models except E' achieve .929+ text h@5 and sub-1.6 PPL. Key developments:

1. **H reclaims the h@5 crown**: H .975 h@5 vs I' .929. At n100_l4, I' led (.884 vs H .521). H needed more layers to fully express its fixed cumsum advantage at n100.
2. **I' still has the best PPL**: I' 1.29 PPL — below H's 1.36. V rotation helps I' extract more information per prediction.
3. **Cross-pollination is essentially solved**: H achieves **.983 kg_excl_m h@5 / 1.51 PPL** — KG-exclusive facts are predicted as well as memorized facts. I' achieves .917 / 1.37.
4. **E breaks through at n100**: E was stuck at .418 h@5 at n100_l4. At l8 it reaches .939 — a massive jump. Width+depth combination unlocks E's learned cumsum for causal training.
5. **Text-exclusive transfer**: I' achieves .700 txtExcl_m h@5 — text-only facts are being learned through KG training. This is reverse cross-pollination.
6. **E' remains permanently broken** under causal KG + V rotation (.109 h@5, 5.11 PPL).

### n100_l16 Causal Detailed Results — **NEAR-PERFECT + E' RECOVERY**

| Model | Text mem h@5 | Text mem PPL | Text trans h@5 | Text gen h@5 | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|---|---|
| **H'** | **.996** | **1.13** | .989 | .678 | .983 | 1.37 | **.983** |
| **H** | **.995** | **1.12** | **1.000** | .678 | **1.000** | **1.19** | .983 |
| E | .991 | 1.15 | 1.000 | .689 | .950 | 1.35 | .917 |
| I | .987 | 1.16 | .978 | .689 | .983 | 1.26 | .900 |
| I' | .983 | 1.13 | .944 | .667 | .967 | 1.20 | .950 |
| E' | .604 | 2.34 | .556 | .367 | .483 | 2.87 | .567 |

Continued improvement from n100_l8 — all models now above .98 h@5 with sub-1.2 PPL. Key developments:

1. **H achieves perfect cross-pollination**: **1.000 kg_excl_m h@5** — every KG-exclusive fact correctly predicted in text. H also gets 1.000 transfer h@5.
2. **Reverse cross-pollination near-perfect**: H/H' .983 txtExcl_m h@5, I' .950. Text-only facts are being learned through KG training.
3. **E' partially recovers**: .604 h@5 / 2.34 PPL — up from .109/5.11 at l8. Like I' at n50 (broken at l16, recovered at l20), E' may need more depth to fully recover. Its l8 failure was transient.
4. **I' has best PPL**: I' 1.13 PPL (tied with H' 1.13), I' 1.20 kg_excl PPL. V rotation continues to help I' extract information.
5. **Generalization plateau**: All models at .667-.689 gen h@5. Text generalization doesn't scale past ~.7 — a hard ceiling on unseen derived facts.

### n100_l20 Causal — Continued Near-Perfection

| Model | Text mem h@5 | Text mem PPL | Text trans h@5 | Text gen h@5 | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|---|---|
| **H'** | **.998** | **1.10** | .967 | .733 | .983 | 1.36 | .950 |
| H | .992 | 1.10 | .967 | .667 | .983 | 1.18 | **1.000** |
| I' | .995 | **1.09** | .967 | .700 | .950 | 1.27 | .983 |
| E | .988 | 1.12 | .967 | .700 | .950 | 1.27 | .967 |
| I | .983 | 1.12 | .989 | .656 | .933 | 1.20 | .933 |
| E' | .672 | 2.10 | .589 | .489 | .550 | 2.76 | .633 |

H achieves **1.000 txtExcl_m h@5** — perfect reverse cross-pollination. E' continues recovery (.672, up from .604 at l16). All other models above .98.

### n250_l2 Causal — **I' WIDTH BREAKTHROUGH**

| Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|
| **I'** | **.945** | **1.20** | **.933** | **1.36** | **.967** |
| H' | .787 | 2.28 | .483 | 3.63 | .333 |
| E' | .658 | 2.15 | .533 | 2.60 | .500 |
| I | .539 | 3.25 | .400 | 3.79 | .167 |
| H | .529 | 3.01 | .483 | 3.45 | .183 |
| E | .346 | 4.16 | .300 | 4.39 | .067 |

**I' at just 2 layers achieves .945 h@5 / 1.20 PPL** — width alone enables the causal phase transition. E' also recovers significantly here (.658 vs .672 at n100_l20 but with only 2 layers). V rotation helps all models at n250.

### n250_l4 Causal — **I' PERFECT, ALL NEAR-PERFECT**

| Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|
| **I'** | **1.000** | **1.06** | .917 | 1.24 | **1.000** |
| H | .992 | 1.18 | **1.000** | **1.35** | .867 |
| H' | .992 | 1.14 | .967 | 1.49 | .967 |
| I | .989 | 1.27 | .950 | 1.45 | .783 |
| E | .979 | 1.35 | .900 | 1.70 | .717 |
| E' | .861 | 1.68 | .567 | 3.19 | .833 |

I' achieves **1.000 h@5 / 1.06 PPL** — perfection. E' continues recovery to .861. H again gets 1.000 kg_excl_m h@5.

### n250_l8 Causal — **E' FULLY RECOVERS, I' DIVERGES**

| Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|
| **E'** | **1.000** | **1.01** | .867 | 1.72 | **1.000** |
| H' | 1.000 | 1.03 | .967 | 1.33 | 1.000 |
| I | 1.000 | 1.07 | .950 | 1.32 | 1.000 |
| H | .999 | 1.04 | .933 | 1.25 | .983 |
| E | .999 | 1.08 | .950 | 1.35 | .983 |
| **I'** | **.006** | **15.49** | .000 | 17.20 | .000 |

**E' FULLY RECOVERS** to 1.000 h@5 / 1.01 PPL — its causal failure was transient, needing width (not just depth) to overcome. But **I' catastrophically diverges** (PPL 15.49). I' seems unstable at n250_l8 in both MLM (DIV) and causal modes.

### n500_l2 Causal — E'/I' LEAD

| Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|
| **E'** | **1.000** | **1.01** | .767 | 2.18 | **1.000** |
| **I'** | .999 | 1.04 | .867 | 1.32 | 1.000 |
| H' | .936 | 1.45 | .633 | 2.85 | .683 |
| H | .919 | 1.59 | .850 | 1.91 | .433 |
| I | .613 | 2.70 | .567 | 3.07 | .383 |
| E | .460 | 3.55 | .283 | 4.88 | .283 |

E' achieves 1.000/1.01 at just 2 layers with n500 width. I' recovers from its n250_l8 divergence (.999/1.04). V rotation helps enormously at this width.

### n500_l4 Causal — **CAUSAL FULLY SOLVED**

| Model | Text mem h@5 | Text mem PPL | kg_excl_m h@5 | kg_excl_m PPL | txtExcl_m h@5 |
|---|---|---|---|---|---|
| **I'** | **1.000** | **1.01** | .900 | 1.38 | **1.000** |
| **E'** | **1.000** | **1.01** | .900 | 1.60 | **1.000** |
| H' | .999 | 1.04 | .900 | 1.65 | 1.000 |
| H | .997 | 1.09 | .950 | 1.23 | .950 |
| I | .995 | 1.15 | .950 | 1.33 | .950 |
| E | .967 | 1.38 | .817 | 2.21 | .667 |

**Causal KG training is FULLY SOLVED at n500_l4.** All 6 models achieve .967+ h@5 with sub-1.4 PPL. I'/E' achieve perfect 1.000/1.01. Cross-pollination and reverse cross-pollination are near-universal (most models .90+ on both kg_excl and txtExcl). Only E (unprimed, no V rotation) lags slightly.

### Key observations on causal n50_l16

1. **V rotation HURTS causal KG at depth**: E' (.107 h@5) is catastrophically worse than E (.601). I' (.157) is much worse than I (.540). H' (.635) is the sole exception — V rotation helps H but not E or I. This is the opposite of MLM KG training where V rotation is universally beneficial.

2. **Cross-pollination is strong**: E gets .500 kg_excl_m h@5 — half of KG-exclusive facts correctly predicted in text. This far exceeds dual mixed models' best kg_excl_m h@5 (~.183 at n100_l8).

3. **Text generalization works**: E gets .489 gen h@5 (unseen derived facts), I gets .389. These models are genuinely learning to reason about relations, not just memorizing.

4. **Phase transition between l4 and l16**: At l4, causal models look unremarkable (~.10 h@5, ~5.1 PPL). At l16, unprimed models explode to .5–.6 h@5 and 3.1–3.5 PPL. Something qualitative changes with sufficient depth — possibly the model learns to use the causal KG representations for text generation.

5. **H is the best causal model at l20**: H (unprimed) surpasses H' to achieve .817 h@5 / 2.28 PPL. The V rotation reversal is now clear for all architectures except partial recovery for I' (.709 vs .157 at l16).

6. **l20 confirms continued improvement**: The l16→l20 improvement is large: H .635→.817 h@5, E .601→.792, I .540→.784. PPL drops from 3.1→2.3. The phase transition is not a one-time jump but sustained scaling.

7. **I' recovers at l20**: I' was catastrophically broken at l16 (.157 h@5, 5.01 PPL) but recovers at l20 (.709, 2.90). Its l16 failure was a transient instability, not permanent. E' remains broken (.117 h@5 at l20).

8. **Cross-pollination is extraordinary at l20**: E achieves .717 kg_excl_m h@5 / 2.74 PPL — nearly as good as memorization (.792/2.33). H achieves .700 kg_excl_m h@5. These models are truly transferring KG knowledge to text predictions.

---

## 9. Model Rankings Summary

### KG Champions (by PPL)
1. **E'** (learned cumsum + V rotation) — **overall champion**, **1.52** at n500_l4 dual, **1.88** at n500_l4 non-dual. KG memorization essentially solved. Superlinear scaling in both depth and width.
2. **K'** (RoPE + per-relation slot angles, native 2-slot + V rotation) — **1.87** non-dual / **2.00** dual at n250_l8. Scales like E' in both depth and width. **SURVIVES n500_l4** (2.24 dual). New top-tier model.
3. **I'** (learned cumsum + shared relation op + V rotation) — **width champion**, **2.60** at n500_l2. Diverges at l4+ (n250_l8, n500_l4).
4. **D'** (RoPE + V rotation) — **robust #2**, **2.71** at n500_l4 dual, **2.23** at n500_l4 non-dual (better than dual). Scales in both directions without catastrophic failure.
5. **J'** (RoPE + per-relation slot angles + V rotation) — **2.43** non-dual at n100_l20, 3.37 dual at n250_l8. DIVERGES at n500_l4 dual but **survives in non-dual** (3.27).
6. **G'** (RoPE + per-relation slot angles, slotted format) — 3.83 at n250_l8

### KG Champions (by h@5)
1. **E'** — **.960** at n500_l4. KG memorization essentially solved. KG-exclusive h@5 .967.
2. **K'** — **.895** non-dual / **.889** dual at n250_l8. **.880** at n500_l4 dual. Matches E' scaling trajectory.
3. **I'** — **.805** at n500_l2. Exceptional width scaling, but only at l2.
4. **J'** — **.751** non-dual at n100_l20. .450 at n250_l8 dual (diverges at n500_l4).
5. **D'** — **.719** at n500_l4. Consistent #2 at scale.
6. **G'** — .318 at n250_l8

### Text Champions
- **I'/E' (causal KG, n500_l4)** — **1.000** h@5 / **1.01** PPL — **perfect text prediction, causal fully solved**
- **C' (non-dual KAT, n250_l8+)** — **1.000** h@5 / **1.00** PPL — perfect on all tiers, KAT fully solved
- **I' (causal KG, n250_l2)** — .945 h@5 / 1.20 PPL — width alone achieves near-perfect at just 2 layers
- **E' (causal KG, n250_l8)** — 1.000/1.01 — **E' fully recovers** from causal failure at this width
- **E' (non-dual mixed, n500_l4)** — 4.27 text PPL, .350 h@5 — first meaningful text accuracy in mixed mode

### Cross-pollination Champions
- **H (causal, n250_l4)** — **1.000** kg_excl_m h@5 — perfect cross-pollination (also at n100_l16)
- **I'/E' (causal, n500_l4)** — .900 kg_excl_m h@5 / ~1.4 PPL — near-perfect at extreme scale
- **C' (non-dual KAT, n250_l8)** — **1.000** kg_excl_m h@5 / **1.01** PPL — perfect
- **H/I'/E' (causal, n500_l4)** — .950–1.000 txtExcl_m h@5 — perfect reverse cross-pollination
- **Dual I/I'** — best kg_excl text PPL in dual mixed mode (~5.2)
- Non-dual MLM KG has poor cross-pollination (kg_excl text PPL stays 6.0–7.0, gets worse at n500)

---

## 10. Key Findings

### 1. Non-dual beats dual on KG, but dual wins on text cross-pollination
Non-dual achieves better KG PPL at every grid point (e.g., I' 4.74 vs 5.25 at n50_l2). But dual has much better kg_excl text PPL (5.27 vs 6.22), meaning dual training helps transfer KG knowledge to text predictions.

### 2. E' scales in both directions; I' is width-only
E' scales superlinearly: 4.00 (l8) → 3.29 (l20) at n100; 1.88 at n250_l8; **1.52** at n500_l4 dual. Non-dual E' reaches **1.88/.887** at n500_l4 — nearly matching dual. It benefits from both width and depth. I' scales with width: 5.25 → 4.53 → 3.76 → **2.60** at l2, but **diverges** at depth (n250_l8: 79.39, n500_l4: DIV). I/J also diverge at n500_l4 — only E'/D' and simpler models survive at extreme scale.

### 3. J' emerges at depth with superlinear scaling
J' (RoPE + per-relation slot angles) is unremarkable at shallow configs but shows superlinear KG scaling at depth: 3.98 (n100_l4) → 3.59 (l8) → **2.76** (l16), with h@5 .266 → .373 → **.628**. At n100_l16, J' approaches dual E' at n250_l8 (1.88). G' follows the same pattern (3.07/.541 at n100_l16). Non-dual training with slot angles may rival dual E' at sufficient scale.

### 4. V rotation: massive benefit for MLM but catastrophic at scale for I' and causal E'
At n250_l8, V rotation gives E' a 2.37 PPL improvement (4.25→1.88) and D' a 1.43 improvement (4.31→2.88). But it **completely destroys I'** at the same config (PPL 79.39 vs 4.23 unprimed). Under causal KG training, V rotation catastrophically hurts E (h@5 .792→.117 at l20) while H and I are partially immune. The V rotation interaction is architecture-specific and can flip from massive benefit to catastrophic failure.

### 5. Causal KG training produces the best text performance — fully solved at n500_l4
At n100_l16, causal models achieve near-perfection. At n250_l4, I' reaches **1.000 h@5 / 1.06 PPL**. At n250_l8, **E' FULLY RECOVERS** (1.000/1.01) — its causal failure was transient, needing width to overcome. At n500_l4, **all models achieve .967+ h@5** with sub-1.4 PPL — causal is fully solved. I'/E' both reach 1.000/1.01. Width is the primary scaling axis: I' achieves .945/1.20 at n250_l2 (just 2 layers!). I' diverges at n250_l8 causal (PPL 15.49) but recovers at n500. Generalization plateaus at ~.7 h@5.

### 6. E' essentially solves KG memorization
E' (learned cumsum + relation operator + V rotation) achieves **1.52 PPL / .960 h@5** at n500_l4 — KG memorization is essentially solved. KG-exclusive h@5 is .967, meaning even unseen KG facts are predicted near-perfectly. E' also achieves .323 text h@5 / 4.21 text PPL at this config — the first time a dual mixed model achieves meaningful text accuracy beyond the ~.10 floor. E' scales in both depth and width without catastrophic failure, unlike I'/J' which diverge at n500_l4.

### 7. Dual objective helps cross-pollination but hurts KG PPL
The dual objective (alternating causal/MLM) hurts KG-specific PPL (dual I' 5.25 vs non-dual I' 4.74 at n50_l2) but dramatically improves cross-modal transfer. Non-dual causal KG training is a middle ground — it helps cross-pollination without the full dual objective.

### 8. Non-dual KAT C' achieves perfection at n100_l16
Non-dual KAT C' achieves **1.02 PPL / 1.000 h@5** at n100_l16 — perfect on memorization, kg_excl, AND txtExcl. The scaling is explosive: 4.91 (n100_l2) → 1.68 (l8) → **1.02** (l16). B/B' also reach 1.000/.999 h@5. KAT text prediction is fully solved at n100_l16.

### 9. Width accelerates the causal phase transition by 4×
At n50, causal KG models need l16+ to achieve the text breakthrough (H .635 h@5 at l16). At n100, I' achieves .884 h@5 at just l4. This is a **4× reduction in depth requirement**. Additionally, V rotation reverses from harmful (n50_l16: I' .157 vs I .540) to massively beneficial (n100_l4: I' .884 vs I .544). The causal training mechanism is highly sensitive to the width-depth interaction.

### 10. I' is a width specialist with a hard depth ceiling
I' scales excellently with width: 5.25 (n50) → 4.53 (n100) → 3.76 (n250) → **2.60** (n500) at l2. Also strong at l4: 4.84 → 4.29 → **2.87** at n250. But I' **diverges at n250_l8** (PPL 79.39) and **n500_l4** (both I and I'). The depth ceiling seems inversely proportional to width — deeper stacks are harder to stabilize with shared relation operators.

### 11. D' emerges as the robust #2
D' (simple RoPE + V rotation) was unremarkable at small scale but scales exceptionally: **2.71** PPL / **.719** h@5 at n500_l4. D (unprimed) diverges at n500_l2 (PPL 12.27) but recovers at n500_l4 (4.51) — the n500_l2 failure was likely a training instability, not a fundamental limit. D' benefits from both width and depth without catastrophic failure, making it the most robust architecture alongside E'.

### 12. n500_l4 reveals a survival threshold — non-dual stabilizes J'
At n500_l4 dual, I/I'/J/J' all diverge, while A/A'/D/D'/E/E'/F/F'/G/G'/H/H'/K/K' survive. But in non-dual, **J' survives n500_l4** (3.27/.443) — only I diverges (I' also survives at 3.41/.473). Non-dual training stabilizes models at extreme scale. K' surviving (2.24/.880 dual) while J' diverges in dual is notable — both use per-relation slot angles in native 2-slot format, but K's implementation differs. The surviving models all use either simple positional encodings (RoPE, fixed angles), E's learned cumsum, or K's stable slot-angle variant.

### 13. K' is a new top-tier model matching E' scaling
K' (RoPE + per-relation slot angles, native 2-slot, V rotation) scales like E' in both depth and width: **1.87/.895** non-dual at n250_l8 (matching E' 1.88/.907 dual), **2.24/.880** dual at n500_l4 (where I'/J' diverge). K' non-dual depth scaling at n100 is superlinear: 4.22 (l2) → 4.02 (l4) → 3.56 (l8) → 3.03 (l16) → **2.45** (l20). At n100_l20, K' (.750 h@5) essentially ties J' (.751). Non-dual K' slightly outperforms dual K' at n250_l8 (1.87 vs 2.00) but dual is far better at n500 (2.24 vs 3.07 at l4), suggesting dual training becomes critical at extreme width.

### 14. J' and K' are nearly tied at n100_l20 non-dual
J' 2.43/.751 and K' 2.45/.750 at n100_l20 non-dual — essentially identical. Both surpass dual D' at n500_l4 (2.71/.719). The key difference emerges at n500: K' survives (2.24/.880 dual) while J' diverges. This makes K' the safer choice for scaling, despite identical performance at moderate scale.

### 15. Non-dual training stabilizes models at extreme scale
Non-dual training produces better stability than dual at extreme configs. D' improves from 2.71 (dual) to **2.23** (non-dual) at n500_l4. J' **survives** n500_l4 in non-dual (3.27/.443) but DIVERGES in dual. I' survives n500_l4 in non-dual (3.41/.473) but diverges in dual. Only I diverges at n250_l8+ in non-dual. The dual objective's alternating training apparently introduces instability at extreme scale, while non-dual's consistent MLM KG objective is more stable.

---

## 11. Full KG Memorization PPL Tables

### Dual — KG Mem PPL across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 | n250_l2 | n250_l4 | n250_l8 | n500_l2 | n500_l4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 5.85 | 5.31 | 4.78 | 4.49 | 4.36 | 5.37 | 4.70 | 4.37 | 4.20 | 4.14 | 4.72 | 4.40 | 4.28 | 4.77 | 4.41 |
| A' | 5.48 | 5.02 | 4.64 | 4.38 | 4.32 | 4.85 | 4.45 | 4.23 | 4.12 | 4.14 | 4.29 | 4.23 | 4.09 | 4.25 | 4.20 |
| D | 5.81 | 5.39 | 4.92 | 4.53 | 4.44 | 5.18 | 4.71 | 4.39 | 4.26 | 4.20 | 4.81 | 4.47 | 4.31 | **12.27** | 4.51 |
| D' | 5.39 | 5.00 | 4.59 | 4.32 | 4.24 | 4.62 | 4.35 | 4.15 | 4.05 | 3.98 | 4.18 | 3.99 | 2.88 | 3.88 | 2.71 |
| E | 5.76 | 5.22 | 4.78 | 4.45 | 4.38 | 5.40 | 4.65 | 4.35 | 4.22 | 4.16 | 4.72 | 4.42 | 4.25 | 4.94 | 4.48 |
| E' | 5.31 | 4.86 | 4.51 | 4.25 | 4.17 | 4.54 | 4.29 | 4.00 | 3.54 | 3.29 | 3.97 | 3.03 | 1.88 | 2.96 | **1.52** |
| F | 5.86 | 5.36 | 4.86 | 4.47 | 4.43 | 5.36 | 4.74 | 4.41 | 4.21 | 4.17 | 4.88 | 4.42 | 4.31 | 4.74 | 4.45 |
| F' | 5.59 | 5.11 | 4.69 | 4.40 | 4.38 | 4.94 | 4.55 | 4.26 | 4.15 | 4.12 | 4.34 | 4.31 | 4.13 | 4.36 | 4.26 |
| G | 5.97 | 5.33 | 4.77 | 4.51 | 4.41 | 5.41 | 4.73 | 4.41 | 4.18 | 4.13 | 4.88 | 4.46 | 4.30 | 4.85 | 4.53 |
| G' | 5.53 | 5.04 | 4.63 | 4.40 | 4.29 | 4.83 | 4.45 | 4.22 | 4.01 | 3.87 | 4.35 | 4.11 | 3.83 | 4.26 | 4.24 |
| H | 5.83 | 5.28 | 4.80 | 4.47 | 4.39 | 5.49 | 4.74 | 4.42 | 4.13 | 4.15 | 4.76 | 4.45 | 4.29 | 4.87 | 4.41 |
| H' | 5.49 | 5.03 | 4.65 | 4.41 | 4.31 | 4.87 | 4.52 | 4.25 | 4.05 | 4.03 | 4.32 | 4.19 | 3.94 | 4.28 | 4.22 |
| I | 5.80 | 5.14 | 4.73 | 4.42 | 4.39 | 5.16 | 4.61 | 4.37 | 4.23 | 4.21 | 4.59 | 4.39 | 4.23 | 4.68 | DIV |
| I' | 5.25 | 4.84 | 4.48 | 4.29 | 4.28 | 4.53 | 4.29 | 4.12 | 4.15 | 4.46 | 3.76 | 2.87 | **79.39** | **2.60** | DIV |
| J | 5.84 | 5.27 | 4.73 | 4.46 | 4.36 | 5.43 | 4.61 | 4.35 | 4.21 | 4.09 | 4.76 | 4.43 | 4.26 | 4.74 | DIV |
| J' | 5.47 | 4.93 | 4.60 | 4.34 | 4.21 | 4.72 | 4.42 | 4.09 | 3.68 | 3.64 | 4.27 | 3.94 | 3.37 | 4.26 | DIV |
| K | 6.00 | 5.31 | 4.79 | 4.45 | 4.36 | 5.49 | 4.75 | 4.36 | 4.14 | 4.19 | 4.84 | 4.34 | 4.22 | 4.82 | 4.38 |
| K' | 5.42 | 4.91 | 4.54 | 4.26 | 4.16 | 4.75 | 4.39 | 3.69 | 3.48 | 3.16 | 4.16 | 2.64 | **2.00** | 2.67 | **2.24** |

### Non-dual — KG Mem PPL across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 | n250_l2 | n250_l4 | n250_l8 | n500_l2 | n500_l4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 5.39 | 4.84 | 4.25 | 4.08 | 4.07 | 4.77 | 4.18 | 4.08 | 4.01 | 3.98 | 4.20 | 4.09 | 4.02 | 4.19 | 4.14 |
| A' | 5.15 | 4.55 | 4.16 | 4.06 | 4.03 | 4.23 | 4.07 | 3.97 | 3.79 | 3.81 | 4.09 | 3.85 | 3.68 | 4.02 | 3.77 |
| D | 5.38 | 4.88 | 4.31 | 4.11 | 4.07 | 4.58 | 4.21 | 4.09 | 4.05 | 4.04 | 4.30 | 4.11 | 4.08 | 4.37 | 4.10 |
| D' | 4.90 | 4.41 | 4.10 | 4.05 | 4.02 | 4.13 | 4.06 | 4.04 | 3.98 | 3.93 | 4.06 | 3.93 | 2.95 | 3.73 | 2.23 |
| E | 5.29 | 4.70 | 4.24 | 4.09 | 4.06 | 4.48 | 4.14 | 4.07 | 4.01 | 4.03 | 4.21 | 4.08 | 4.05 | 4.30 | 4.09 |
| E' | 4.91 | 4.29 | 4.08 | 4.03 | 4.02 | 4.09 | 4.05 | 4.00 | 3.77 | 3.58 | 4.00 | 3.49 | 2.34 | 3.23 | **1.88** |
| F | 5.62 | 5.03 | 4.38 | 4.14 | 4.09 | 4.92 | 4.28 | 4.10 | 4.00 | 4.01 | 4.37 | 4.14 | 4.09 | 4.61 | 4.15 |
| F' | 5.27 | 4.70 | 4.23 | 4.08 | 4.06 | 4.42 | 4.11 | 4.03 | 3.91 | 3.89 | 4.08 | 3.93 | 3.82 | 4.13 | 3.95 |
| G | 5.63 | 4.90 | 4.30 | 4.10 | 4.05 | 4.83 | 4.16 | 4.06 | 3.96 | 3.80 | 4.28 | 4.05 | 3.98 | 4.31 | 4.13 |
| G' | 5.06 | 4.48 | 4.14 | 3.96 | 3.92 | 4.21 | 4.06 | 3.73 | 3.07 | 2.71 | 4.00 | 3.34 | 2.84 | 3.81 | 3.27 |
| H | 5.53 | 4.94 | 4.32 | 4.08 | 4.04 | 4.89 | 4.21 | 4.06 | 4.02 | 3.91 | 4.26 | 4.11 | 4.04 | 4.28 | 4.20 |
| H' | 5.02 | 4.54 | 4.16 | 4.01 | 3.95 | 4.26 | 4.07 | 3.90 | 3.55 | 3.36 | 4.02 | 3.59 | 3.02 | 3.98 | 3.58 |
| I | 5.34 | 4.68 | 4.21 | 4.06 | 4.06 | 4.53 | 4.17 | 4.07 | 4.04 | 4.03 | 4.24 | 4.08 | 25.07 | 20.63 | DIV |
| I' | 4.74 | 4.22 | 4.07 | 4.03 | 4.03 | 4.07 | 4.04 | 3.99 | 4.02 | 4.03 | 4.00 | 3.72 | DIV | 3.62 | 3.41 |
| J | 5.48 | 4.94 | 4.33 | 4.08 | 4.05 | 4.69 | 4.23 | 4.05 | 3.91 | 3.88 | 4.26 | 4.09 | 3.95 | 4.30 | 4.10 |
| J' | 5.03 | 4.47 | 4.11 | 3.95 | 3.83 | 4.18 | 3.98 | 3.59 | 2.76 | **2.43** | 4.04 | 3.34 | 2.39 | 3.59 | 3.27 |
| K | 5.91 | 4.81 | 4.23 | 4.07 | 4.05 | 4.76 | 4.17 | 4.06 | 3.84 | 3.89 | 4.23 | 4.07 | 4.02 | 4.29 | 4.12 |
| K' | 5.04 | 4.36 | 4.09 | 3.86 | 3.84 | 4.22 | 4.02 | 3.56 | 3.03 | **2.45** | 3.77 | 2.69 | **1.87** | 3.83 | 3.07 |

### Observations on the full tables
- **E' solves KG memorization**: **1.52** PPL / **.960** h@5 at n500_l4. Superlinear scaling in both depth and width. Never diverges.
- **I' width scaling is extreme but fragile**: 5.25 → 4.53 → 3.76 → **2.60** at l2. But diverges at n250_l8 (79.39) and n500_l4.
- **D' is the robust #2**: 2.88 at n250_l8, **2.71** at n500_l4. Scales in both directions. D (unprimed) diverges at n500_l2 but recovers at n500_l4.
- **I/J both diverge at n500_l4**: Both unprimed and primed variants. Only E'/D'/K' and the simpler models (A-H') survive.
- **Non-dual J'/K' show superlinear scaling at n100**: J' 3.98 (l4) → 3.59 (l8) → 2.76 (l16) → **2.43** (l20). K' 4.02 (l4) → 3.56 (l8) → 3.03 (l16) → **2.45** (l20). G' also breaks out: 3.73 (l8) → 3.07 (l16) → **2.71** (l20).
- **I' stalls at n100 depth in non-dual**: I' 4.07 (l2) → 4.04 (l4) → 3.99 (l8) → 4.02 (l16) → **4.03** (l20). I' shows zero depth scaling — a hard ceiling at ~4.0 PPL.
- **Non-dual E' nearly matches dual at n500_l4**: E' 1.88 non-dual vs 1.52 dual. The gap narrows dramatically at extreme scale.
- **K' non-dual beats dual at n250_l8**: K' 1.87 non-dual vs 2.00 dual. Non-dual is actually better for K' at this scale.
- **J' survives n500_l4 in non-dual**: J' 3.27 — it DIVERGED in dual at this config. Non-dual training stabilizes J'.
- **I/I' diverge at n250_l8 non-dual**: I PPL 25.07, I' catastrophically diverged (PPL in billions). Same pattern as dual (I' PPL 79.39).
- **D' non-dual 2.23 at n500_l4**: Better than dual D' (2.71). D' benefits from non-dual training at extreme scale.

---

## 12. Full KG Memorization h@5 Tables

### Dual — KG Mem h@5 across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 | n250_l2 | n250_l4 | n250_l8 | n500_l2 | n500_l4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | .095 | .106 | .141 | .163 | .176 | .117 | .154 | .179 | .207 | .222 | .159 | .182 | .193 | .169 | .180 |
| A' | .098 | .126 | .153 | .185 | .189 | .145 | .175 | .206 | .212 | .211 | .196 | .219 | .233 | .198 | .214 |
| D | .093 | .113 | .142 | .159 | .171 | .124 | .162 | .179 | .186 | .194 | .161 | .179 | .188 | .079 | .172 |
| D' | .102 | .135 | .164 | .186 | .192 | .168 | .189 | .205 | .222 | .259 | .203 | .287 | .674 | .352 | .719 |
| E | .091 | .117 | .147 | .168 | .175 | .128 | .167 | .187 | .199 | .210 | .165 | .184 | .200 | .166 | .176 |
| E' | .107 | .139 | .165 | .191 | .201 | .169 | .201 | .281 | .445 | .527 | .310 | .628 | .907 | .640 | **.960** |
| F | .093 | .112 | .139 | .166 | .172 | .121 | .149 | .180 | .213 | .216 | .159 | .177 | .200 | .179 | .182 |
| F' | .096 | .115 | .157 | .170 | .177 | .142 | .164 | .193 | .217 | .226 | .191 | .201 | .243 | .192 | .198 |
| G | .087 | .111 | .145 | .165 | .176 | .114 | .149 | .179 | .218 | .231 | .164 | .174 | .193 | .164 | .173 |
| G' | .094 | .125 | .168 | .184 | .200 | .151 | .188 | .212 | .265 | .283 | .196 | .262 | .318 | .198 | .207 |
| H | .086 | .118 | .139 | .166 | .177 | .113 | .161 | .176 | .232 | .232 | .169 | .188 | .193 | .167 | .184 |
| H' | .098 | .123 | .150 | .179 | .188 | .152 | .169 | .212 | .241 | .259 | .195 | .217 | .292 | .202 | .234 |
| I | .089 | .120 | .143 | .170 | .173 | .129 | .167 | .182 | .197 | .197 | .177 | .185 | .197 | .174 | DIV |
| I' | .112 | .134 | .173 | .184 | .190 | .168 | .198 | .244 | .200 | .173 | .418 | .704 | .002 | .805 | DIV |
| J | .090 | .120 | .141 | .175 | .185 | .122 | .163 | .185 | .208 | .240 | .172 | .182 | .198 | .173 | DIV |
| J' | .103 | .133 | .159 | .180 | .217 | .165 | .185 | .263 | .353 | .368 | .204 | .307 | .450 | .207 | DIV |
| K | .084 | .108 | .141 | .170 | .187 | .107 | .148 | .186 | .233 | .208 | .168 | .191 | .221 | .172 | .185 |
| K' | .101 | .140 | .169 | .217 | .243 | .158 | .193 | .420 | .468 | .543 | .270 | .767 | **.889** | .777 | **.880** |

### Non-dual — KG Mem h@5 across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 | n250_l2 | n250_l4 | n250_l8 | n500_l2 | n500_l4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | .102 | .158 | .198 | .200 | .205 | .178 | .197 | .207 | .225 | .239 | .203 | .205 | .233 | .197 | .201 |
| A' | .123 | .185 | .201 | .205 | .209 | .200 | .219 | .244 | .321 | .303 | .212 | .305 | .341 | .227 | .337 |
| D | .112 | .155 | .196 | .199 | .199 | .181 | .198 | .203 | .201 | .200 | .200 | .200 | .207 | .198 | .201 |
| D' | .143 | .190 | .199 | .201 | .204 | .198 | .201 | .205 | .240 | .244 | .208 | .286 | .593 | .361 | .794 |
| E | .126 | .171 | .197 | .198 | .199 | .190 | .199 | .202 | .219 | .206 | .193 | .208 | .217 | .197 | .207 |
| E' | .155 | .193 | .202 | .203 | .203 | .199 | .207 | .232 | .302 | .356 | .231 | .431 | .777 | .537 | **.887** |
| F | .102 | .144 | .189 | .196 | .201 | .158 | .202 | .203 | .230 | .232 | .198 | .201 | .203 | .197 | .198 |
| F' | .115 | .173 | .198 | .204 | .205 | .191 | .201 | .233 | .274 | .267 | .205 | .267 | .305 | .208 | .269 |
| G | .091 | .148 | .197 | .200 | .213 | .163 | .202 | .208 | .263 | .309 | .200 | .227 | .255 | .200 | .209 |
| G' | .133 | .189 | .206 | .259 | .278 | .197 | .216 | .334 | .541 | .655 | .248 | .411 | .581 | .290 | .436 |
| H | .103 | .149 | .191 | .204 | .204 | .165 | .203 | .205 | .216 | .263 | .201 | .207 | .216 | .204 | .205 |
| H' | .134 | .184 | .200 | .227 | .255 | .195 | .210 | .282 | .381 | .413 | .234 | .379 | .545 | .246 | .373 |
| I | .111 | .171 | .196 | .203 | .200 | .188 | .199 | .202 | .203 | .208 | .199 | .205 | .000 | .007 | DIV |
| I' | .162 | .197 | .200 | .200 | .200 | .202 | .211 | .242 | .209 | .204 | .245 | .370 | DIV | .415 | .473 |
| J | .111 | .152 | .195 | .201 | .216 | .182 | .201 | .214 | .271 | .281 | .204 | .223 | .260 | .199 | .213 |
| J' | .134 | .188 | .212 | .262 | .313 | .208 | .266 | .373 | .628 | **.751** | .231 | .441 | .743 | .344 | .443 |
| K | .097 | .158 | .197 | .208 | .206 | .175 | .199 | .218 | .293 | .270 | .207 | .215 | .240 | .204 | .209 |
| K' | .140 | .195 | .205 | .296 | .300 | .198 | .240 | .405 | .559 | **.750** | .327 | .690 | **.895** | .315 | .546 |

### Observations on h@5 tables
- **E' achieves .960 at n500_l4**: KG memorization essentially solved. .281→.445→.527→.628→.907→**.960**. Superlinear growth continues.
- **I' h@5 scales with width**: .112→.168→.418→.704→**.805** from n50 through n500 at l2/l4. But diverges at n250_l8 (.002) and n500_l4.
- **D' is the robust #2**: .205→.287→.674→**.719** from n100 through n500. Never diverges.
- **J' diverges at n500_l4**: Best non-E'/I'/D' at n250_l8 dual (.450) but can't survive extreme scale. Non-dual: .628 (l16) → **.751** (l20) at n100.
- **Non-dual J'/K'/G' show superlinear h@5 scaling**: J' .373 (n100_l8) → .628 (l16) → **.751** (l20). K' .405 (l8) → .559 (l16) → **.750** (l20). G' .334 → .541 → **.655**. Most other models remain stuck at ~.20–.30.
- **I' h@5 stalls in non-dual**: .242 (l8) → .209 (l16) → **.204** (l20) at n100. Depth provides no benefit — I' is width-only.
- **Non-dual E' .887 at n500_l4**: Nearly matches dual .960. E' non-dual h@5 scaling: .231 (n250_l2) → .431 (n250_l4) → .777 (n250_l8) → .537 (n500_l2) → **.887** (n500_l4).
- **K' non-dual .895 at n250_l8**: Beats dual K' (.889) and matches dual E' (.907). K' non-dual h@5 scaling: .327 (n250_l2) → .690 (n250_l4) → **.895** (n250_l8).
- **J' survives n500_l4 in non-dual with .443 h@5**: In dual, J' DIVERGED at n500_l4. Non-dual training stabilizes J'.
- **D' non-dual .794 at n500_l4**: Better than dual D' (.719). Non-dual training benefits D' at extreme scale.
- **I/I' diverge at n250_l8 non-dual**: I h@5 .000 (PPL 25.07), I' catastrophically diverged. Same instability as dual.

---

## 13. Text Memorization PPL Tables

### Dual — Text Mem PPL across grid
| Model | n50_l2 | n50_l4 | n50_l8 | n50_l16 | n50_l20 | n100_l2 | n100_l4 | n100_l8 | n100_l16 | n100_l20 | n250_l2 | n250_l4 | n250_l8 | n500_l2 | n500_l4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 6.03 | 5.63 | 5.35 | 5.18 | 5.16 | 5.47 | 5.25 | 5.18 | 5.09 | 5.07 | 5.30 | 5.19 | 5.14 | 5.24 | 5.25 |
| A' | 5.79 | 5.50 | 5.27 | 5.19 | 5.16 | 5.32 | 5.19 | 5.16 | 5.14 | 5.17 | 5.21 | 5.20 | 5.05 | 5.22 | 5.15 |
| D | 6.02 | 5.62 | 5.32 | 5.19 | 5.23 | 5.63 | 5.24 | 5.18 | 5.14 | 5.14 | 5.30 | 5.19 | 5.24 | 5.36 | 5.20 |
| D' | 5.60 | 5.34 | 5.19 | 5.17 | 5.15 | 5.24 | 5.18 | 5.14 | 5.15 | 5.13 | 5.19 | 5.23 | 5.20 | 5.17 | 4.89 |
| E | 5.73 | 5.45 | 5.23 | 5.17 | 5.19 | 5.38 | 5.19 | 5.14 | 5.17 | 5.13 | 5.23 | 5.17 | 5.17 | 5.28 | 5.18 |
| E' | 5.55 | 5.33 | 5.17 | 5.15 | 5.15 | 5.21 | 5.13 | 5.14 | 5.10 | 5.11 | 5.20 | 5.19 | 5.15 | 5.11 | **4.21** |
| F | 5.92 | 5.61 | 5.31 | 5.16 | 5.15 | 5.50 | 5.26 | 5.16 | 5.12 | 5.12 | 5.30 | 5.20 | 5.17 | 5.27 | 5.24 |
| F' | 5.81 | 5.52 | 5.29 | 5.18 | 5.16 | 5.36 | 5.23 | 5.13 | 5.11 | 5.12 | 5.21 | 5.19 | 5.03 | 5.23 | 5.20 |
| G | 6.01 | 5.66 | 5.37 | 5.18 | 5.18 | 5.49 | 5.24 | 5.18 | 5.14 | 5.11 | 5.34 | 5.16 | 5.20 | 5.23 | 5.24 |
| G' | 5.73 | 5.53 | 5.28 | 5.17 | 5.17 | 5.30 | 5.19 | 5.16 | 5.12 | 5.14 | 5.24 | 5.12 | 4.89 | 5.22 | 5.19 |
| H | 5.84 | 5.54 | 5.29 | 5.15 | 5.15 | 5.40 | 5.21 | 5.14 | 5.09 | 5.11 | 5.25 | 5.16 | 5.13 | 5.33 | 5.17 |
| H' | 5.78 | 5.44 | 5.23 | 5.15 | 5.14 | 5.29 | 5.19 | 5.10 | 5.10 | 5.10 | 5.19 | 5.13 | 4.84 | 5.21 | 5.09 |
| I | 5.57 | 5.36 | 5.21 | 5.16 | 5.15 | 5.25 | 5.18 | 5.16 | 5.12 | 5.14 | 5.24 | 5.18 | 5.17 | 5.22 | DIV |
| I' | 5.46 | 5.32 | 5.16 | 5.15 | 5.16 | 5.19 | 5.18 | 5.15 | 5.16 | 5.18 | 5.23 | 5.19 | **34.06** | **4.14** | DIV |
| J | 5.96 | 5.56 | 5.26 | 5.17 | 5.14 | 5.42 | 5.22 | 5.14 | 5.12 | 5.09 | 5.26 | 5.17 | 5.09 | 5.33 | DIV |
| J' | 5.83 | 5.39 | 5.23 | 5.14 | 5.12 | 5.29 | 5.22 | 5.09 | 5.09 | 5.10 | 5.24 | 5.03 | **4.54** | 5.21 | DIV |

Text PPL converges to ~5.1 for most configs. At n500_l2 I' achieves **4.14**, and at n500_l4 E' achieves **4.21** — both break through the floor. D' also improves to 4.89 at n500_l4. I/I'/J/J' all diverge at n500_l4.

---

## 14. Width vs Depth — Two Scaling Strategies

### E' (depth+width scaler) — KG mem PPL
| Config | PPL | h@5 | Notes |
|---|---|---|---|
| n50_l2 | 5.31 | .107 | |
| n100_l2 | 4.54 | .169 | |
| n250_l2 | 3.97 | .310 | |
| n500_l2 | 2.96 | .640 | |
| n100_l8 | 4.00 | .281 | |
| n100_l16 | 3.54 | .445 | |
| n100_l20 | 3.29 | .527 | |
| n250_l4 | 3.03 | .628 | |
| n250_l8 | 1.88 | .907 | near-perfect |
| **n500_l4** | **1.52** | **.960** | **KG memorization solved** |

### I' (width-only scaler) — KG mem PPL
| Config | PPL | h@5 | Notes |
|---|---|---|---|
| n50_l2 | 5.25 | .112 | |
| n100_l2 | 4.53 | .168 | |
| n250_l2 | 3.76 | .418 | |
| n250_l4 | 2.87 | .704 | |
| **n500_l2** | **2.60** | **.805** | **width champion** |
| n100_l8 | 4.12 | .244 | |
| n100_l16 | 4.15 | .200 | depth stalls |
| n100_l20 | 4.46 | .173 | depth hurts |
| n250_l8 | 79.39 | .002 | DIVERGED |
| n500_l4 | DIV | DIV | **DIVERGED** |

### D' (balanced scaler) — KG mem PPL
| Config | PPL | h@5 | Notes |
|---|---|---|---|
| n50_l2 | 5.39 | .102 | |
| n100_l2 | 4.62 | .168 | |
| n250_l2 | 4.18 | .203 | |
| n500_l2 | 3.88 | .352 | |
| n100_l20 | 3.98 | .259 | |
| n250_l4 | 3.99 | .287 | |
| n250_l8 | 2.88 | .674 | |
| **n500_l4** | **2.71** | **.719** | **robust at all scales** |

### K' (E'-like scaler) — KG mem PPL (non-dual)
| Config | PPL | h@5 | Notes |
|---|---|---|---|
| n50_l2 | 5.04 | .140 | |
| n100_l2 | 4.22 | .198 | |
| n250_l2 | 3.77 | .327 | |
| n500_l2 | 3.83 | .315 | width stalls at n500 (non-dual) |
| n100_l8 | 3.56 | .405 | |
| n100_l16 | 3.03 | .559 | |
| n100_l20 | 2.45 | .750 | |
| n250_l4 | 2.69 | .690 | |
| **n250_l8** | **1.87** | **.895** | **matches E' dual (1.88/.907)** |
| n500_l4 (dual) | 2.24 | .880 | **survives where J' diverges** |

Four scaling strategies, now fully resolved:
- **E'**: Scales in both depth and width. Best overall at n500_l4 (**1.52 PPL, .960 h@5**). KG memorization is essentially solved. Also achieves meaningful text accuracy (.323 h@5 / 4.21 PPL). Never diverges.
- **K'**: Scales like E' in both directions. **1.87/.895** non-dual at n250_l8 (matching E' dual). **2.24/.880** dual at n500_l4. Non-dual stalls at n500 (3.07/.546 at l4) but dual remains strong. Different scaling behavior from J' despite similar architecture.
- **I'**: Width-only. Exceptional at n500_l2 (2.60 PPL, .805 h@5). But has a **hard depth ceiling**: diverges at n250_l8 and n500_l4. Width scaling is the only safe dimension.
- **D'**: Balanced and robust. Scales in both directions without failure (**2.71** at n500_l4). Less extreme peaks than E' but never diverges. V rotation is critical — D (unprimed) diverges at n500_l2.

---

## 15. K/K' Results (Backfill)

K uses RoPE + per-relation slot angles in native 2-slot format (HEAD/TAIL, no REL token) — same as J but without learned cumsum. K' adds V rotation.

### Dual — K/K' KG Mem PPL/h@5
| Config | K PPL | K h@5 | K' PPL | K' h@5 |
|---|---|---|---|---|
| n50_l2 | 6.00 | .084 | 5.42 | .101 |
| n50_l4 | 5.31 | .108 | 4.91 | .140 |
| n50_l8 | 4.79 | .141 | 4.54 | .169 |
| n50_l16 | 4.45 | .170 | 4.26 | .217 |
| n50_l20 | 4.36 | .187 | 4.16 | .243 |
| n100_l2 | 5.49 | .107 | 4.75 | .158 |
| n100_l4 | 4.75 | .148 | 4.39 | .193 |
| n100_l8 | 4.36 | .186 | 3.69 | .420 |
| n100_l16 | 4.14 | .233 | 3.48 | .468 |
| n100_l20 | 4.19 | .208 | 3.16 | .543 |
| n250_l2 | 4.84 | .168 | 4.16 | .270 |
| n250_l4 | 4.34 | .191 | 2.64 | .767 |
| **n250_l8** | 4.22 | .221 | **2.00** | **.889** |
| n500_l2 | 4.82 | .172 | 2.67 | .777 |
| **n500_l4** | 4.38 | .185 | **2.24** | **.880** |

### Non-dual — K/K' KG Mem PPL/h@5
| Config | K PPL | K h@5 | K' PPL | K' h@5 |
|---|---|---|---|---|
| n50_l2 | 5.91 | .097 | 5.04 | .140 |
| n50_l4 | 4.81 | .158 | 4.36 | .195 |
| n50_l8 | 4.23 | .197 | 4.09 | .205 |
| n50_l16 | 4.07 | .208 | 3.86 | .296 |
| n50_l20 | 4.05 | .206 | 3.84 | .300 |
| n100_l2 | 4.76 | .175 | 4.22 | .198 |
| n100_l4 | 4.17 | .199 | 4.02 | .240 |
| n100_l8 | 4.06 | .218 | 3.56 | .405 |
| n100_l16 | 3.84 | .293 | 3.03 | .559 |
| n100_l20 | 3.89 | .270 | 2.45 | .750 |
| n250_l2 | 4.23 | .207 | 3.77 | .327 |
| n250_l4 | 4.07 | .215 | 2.69 | .690 |
| **n250_l8** | 4.02 | .240 | **1.87** | **.895** |
| n500_l2 | 4.29 | .204 | 3.83 | .315 |
| n500_l4 | 4.12 | .209 | 3.07 | .546 |

### K/K' Observations
- **K' is a new top-tier model**: At n250_l8, K' achieves **1.87/.895** non-dual and **2.00/.889** dual — matching E' (1.88/.907 dual). K' **survives n500_l4** (2.24/.880 dual) where I'/J' diverge.
- **K' scales like E' in both depth and width**: Non-dual depth scaling at n100: 4.22 → 4.02 → 3.56 → 3.03 → **2.45**. Width scaling at l8: 4.09 (n50) → 3.56 (n100) → **1.87** (n250).
- **Dual vs non-dual reversal at n500**: Non-dual K' is better at n250_l8 (1.87 vs 2.00), but **dual is far better at n500** — l2: 2.67 vs 3.83, l4: 2.24 vs 3.07. Dual training becomes critical at extreme width for K'.
- **K' nearly ties J' at n100_l20 non-dual**: K' 2.45/.750 vs J' 2.43/.751 — essentially identical. But K' survives n500_l4 while J' diverges.
- **K (unprimed) is consistently weak**: PPL stays 4.0+ across all configs. V rotation is essential for K to leverage its slot angles.
- **K' text PPL is poor**: K' at n250_l8 non-dual has 13.37 kg_excl text PPL — worse than dual mixed models. K' focuses KG knowledge in its slot-angle mechanism, which doesn't transfer to text.

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
| K/K' | RoPE + per-relation slot angles | Native (2 slots: HEAD/TAIL) | MLM. Scales like E', survives n500_l4 |
| B/B' | RoPE (standard) | Text only (linearized) | KAT only |
| C/C' | Learned per-token angles | Text only (linearized) | KAT only |

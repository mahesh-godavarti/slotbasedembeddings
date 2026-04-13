# Exp 7a Results: 1000 chains (default names), n_embed=500, n_layers=2, 100K iters, 1 seed

Config: n_embed=500, n_layers=2, max_iters=100000, batch_size=32, lr=0.0005, device=cuda, softplus (default)
Seeds: 1
MLM KG: A, A', E, E', F, F', G, G', H, H'
Causal KG: Ec, Ec', Hc, Hc' (--causal_kg)
kg_as_text: B, B', C, C' (--kg_as_text)
MLM KG (softmax): I, I', J, J', K, K' (--softmax, non-dual)

## Model Architectures

All models are character-level transformers. Primed variants (') add V rotation (rotate V and inverse-rotate output, making the operation non-commutative / operator-based).

| Model | Text angles | KG format | KG angles | Relation encoding | KG training |
|-------|------------|-----------|-----------|-------------------|-------------|
| **A/A'** | RoPE | Slotted: [HEAD][REL][TAIL] | RoPE + learned slot offsets (shared across relations) | REL token + 3 shared slot angle vectors | Bidir MLM |
| **B/B'** | RoPE | Text only (linearized) | N/A | Relation as text token (e.g. `<son_of>`) | Causal NTP (via text) |
| **C/C'** | Learned per-token angle embedding, cumsum | Text only (linearized) | N/A | Relation as text token | Causal NTP (via text) |
| **D/D'** | Learned per-token angle embedding, cumsum | Flat: [head_chars][rel_token][tail_chars] | Same cumsum (rel token has its own angle embedding) | REL is a full token (has embedding, attends, predicted by MLM) | Bidir MLM |
| **E/E'** | Learned per-token angle embedding, cumsum | Native: [head_chars][tail_chars] | Per-token angle embedding cumsum with relation angle inserted as gap between HEAD and TAIL | Learned angle vector per relation (not a token — no embedding, no attention, not predicted) | Bidir MLM |
| **F/F'** | Fixed RoPE | Flat: [head_chars][rel_token][tail_chars] | Fixed RoPE (same as text) | REL is a full token (same as D but with fixed angles) | Bidir MLM |
| **G/G'** | RoPE | Slotted: [HEAD][REL][TAIL] | RoPE + learned slot offsets (per-relation) | REL token + 3 per-relation slot angle vectors | Bidir MLM |
| **H/H'** | Fixed base_freq, cumsum | Native: [head_chars][tail_chars] | Fixed base_freq cumsum with learned relation angle inserted as gap | Learned angle vector per relation (same mechanism as E but fixed per-position angles) | Bidir MLM |
| **I/I'** | Per-layer MLP projector from residual stream, cumsum | Native: [head_chars][tail_chars] | Per-layer projected angles cumsum with relation angle inserted as gap | Learned angle vector per relation (same mechanism as E/H) | Bidir MLM |
| **J/J'** | RoPE | Native: [head_chars][tail_chars] | RoPE + learned slot offsets (per-relation, 2 slots) | 2 per-relation slot angle vectors (HEAD, TAIL) — no REL token, relation identity encoded entirely in slot angles | Bidir MLM |
| **K/K'** | Per-layer MLP projector + RoPE | Native: [head_chars][tail_chars] | Per-layer projected angles + RoPE (per-slot positions) + per-relation slot offsets (2 slots) | 2 per-relation slot angle vectors (HEAD, TAIL) — combines I's content-dependent angles with J's slot structure | Bidir MLM |

**Key architectural dimensions:**
- **Angle source**: Fixed RoPE (B, F) vs learned per-token embedding cumsum (C, D, E) vs fixed base_freq cumsum (H) vs per-layer MLP projector cumsum (I) vs RoPE + slot offsets (A, G, J) vs per-layer MLP projector + RoPE + slot offsets (K)
- **KG format**: Text-only/linearized (B, C) vs flat with REL token (D, F) vs slotted with REL token (A, G) vs native without REL token (E, H, I, J, K)
- **Relation encoding**: As text token (B, C), as full token with embeddings (D, F), as slot angle offsets (A, G, J, K), or as angle gap in cumsum (E, H, I)

## Text Evaluation (h@5 / PPL)

| Tier | B (kat) | B' (kat) | C (kat) | **C' (kat)** | E | Ec | **Ec'** | Hc | Hc' | A | A' | E' | F | F' | G | G' | H | H' | I (smx) | I' (smx) | J (smx) | J' (smx) | K (smx) | K' (smx) |
|------|---------|----------|---------|-------------|---|---|-----|---|-----|---|---|---|---|---|---|---|---|---|---------|----------|---------|----------|---------|----------|
| **mem** | .333/3.89 | .672/2.29 | .475/3.16 | **1.000/1.01** | .109/5.18 | .813/1.84 | **1.000/1.01** | **.917/1.55** | .969/1.32 | .097/5.13 | .123/5.05 | .143/4.96 | .104/5.18 | .116/5.09 | .101/5.13 | .140/5.01 | .105/5.12 | .118/5.09 | .106/5.14 | .128/5.09 | .105/5.16 | .114/5.11 | .101/5.20 | .167/5.01 |
| **trn** | .311/3.70 | .567/2.27 | .511/2.99 | **1.000/1.01** | .133/5.14 | .633/2.48 | **.944/1.32** | **.867/1.71** | .833/1.85 | .100/5.33 | .122/5.14 | .100/5.34 | .044/5.32 | .144/5.19 | .078/5.21 | .133/5.22 | .056/5.26 | .078/5.28 | .111/5.16 | .089/5.25 | .067/5.29 | .067/5.19 | .100/5.40 | .144/5.27 |
| **gen** | .278/4.53 | .456/4.09 | .289/4.54 | .700/4.29 | .100/5.14 | .622/4.30 | .733/5.48 | **.644/4.80** | .656/4.63 | .056/5.20 | .100/4.98 | .100/5.15 | .133/5.16 | .067/5.10 | .122/5.15 | .100/5.06 | .133/5.10 | .044/5.15 | .122/5.17 | .067/5.10 | .089/5.14 | .111/5.03 | .111/5.24 | .122/5.03 |
| **kg_ex_m** | .217/4.68 | .217/4.82 | .250/4.54 | **.983/1.19** | .033/9.41 | .733/2.34 | **.900/1.77** | **.800/1.81** | .717/2.90 | .067/7.75 | .050/8.42 | .033/8.43 | .067/8.46 | .017/7.85 | .117/7.46 | .017/7.73 | .000/7.89 | .050/7.45 | .117/7.48 | .033/7.87 | .117/7.72 | .017/7.67 | .067/8.26 | .050/7.26 |
| **kg_ex_g** | .100/6.04 | .233/6.58 | .183/5.92 | **.700/5.24** | .017/9.16 | .467/5.32 | .617/6.51 | **.617/4.42** | .517/7.41 | .050/9.51 | .000/8.86 | .050/8.83 | .017/8.32 | .050/7.63 | .017/8.56 | .033/7.93 | .017/8.02 | .067/7.74 | .050/7.91 | .033/8.88 | .083/8.09 | .033/7.85 | .017/8.37 | .033/8.72 |
| **tx_ex_m** | .167/4.65 | .550/2.97 | .300/4.09 | **1.000/1.05** | .017/5.32 | .483/3.53 | **1.000/1.05** | **.533/3.58** | .867/1.87 | .100/5.20 | .017/5.07 | .100/4.90 | .100/5.12 | .000/5.09 | .000/5.12 | .100/4.98 | .067/5.06 | .083/5.08 | .067/5.13 | .033/4.99 | .067/5.09 | .083/5.07 | .133/5.07 | .150/4.95 |
| **tx_ex_g** | .267/5.24 | .417/5.22 | .217/5.66 | .683/4.68 | .033/5.78 | .417/7.13 | .683/5.56 | **.433/7.28** | .617/6.04 | .100/5.43 | .117/5.42 | .067/5.33 | .100/5.54 | .067/5.31 | .117/5.38 | .100/5.56 | .117/5.25 | .117/5.29 | .067/5.34 | .067/5.43 | .133/5.31 | .083/5.56 | .050/5.54 | .133/5.33 |

(kat) = kg_as_text mode.

## KG Evaluation (h@5 / PPL)

B/B'/C/C' linearized; A/A'/E/E'/F/F'/G/G'/H/H' MLM; Ec/Ec'/Hc/Hc' causal.

| Tier | B (lin) | B' (lin) | C (lin) | **C' (lin)** | E | Ec | **Ec'** | Hc | Hc' | A | A' | E' | F | F' | G | G' | H | H' | I (smx) | **I' (smx)** | J (smx) | J' (smx) | K (smx) | **K' (smx)** |
|------|---------|----------|---------|-------------|---|---|-----|---|-----|---|---|---|---|---|---|---|---|---|---------|----------|---------|----------|---------|----------|
| **mem** | .339/3.86 | .664/2.30 | .497/3.05 | **1.000/1.01** | .205/4.28 | .892/1.80 | **.997/1.24** | **.980/1.58** | .993/1.28 | .183/4.41 | .206/4.06 | .238/3.98 | .193/4.33 | .209/4.07 | .179/4.50 | .248/3.94 | .174/4.48 | .231/4.03 | .199/4.31 | **.416/3.61** | .197/4.30 | .341/3.54 | .205/4.28 | **.419/3.47** |
| **trn** | .294/3.63 | .567/2.34 | .500/2.92 | **1.000/1.02** | .267/4.08 | .867/1.99 | **1.000/1.24** | **.967/1.63** | .978/1.29 | .222/4.54 | .200/3.90 | .256/3.98 | .178/4.22 | .211/3.97 | .156/4.49 | .244/3.77 | .233/4.32 | .233/4.14 | .256/4.13 | **.400/3.57** | .178/4.37 | .311/3.41 | .156/4.20 | **.422/3.38** |
| **gen** | .256/4.54 | .472/4.14 | .328/4.51 | .706/4.46 | .100/4.59 | .617/5.64 | .711/8.99 | **.689/5.57** | .678/5.32 | .100/4.59 | .211/4.35 | .189/4.18 | .067/4.70 | .178/4.36 | .100/4.98 | .189/4.35 | .156/4.74 | .144/4.24 | .189/4.62 | **.400/4.02** | .111/4.80 | .289/4.17 | .222/4.55 | **.322/4.14** |
| **kg_ex_m** | .267/4.44 | .358/3.60 | .325/3.76 | **1.000/1.03** | .167/4.23 | .900/1.64 | **1.000/1.08** | **.925/1.58** | 1.000/1.19 | .183/4.58 | .167/4.01 | .300/4.02 | .250/4.26 | .200/4.01 | .150/4.72 | .167/4.02 | .133/4.42 | .150/3.99 | .167/4.34 | **.467/3.61** | .200/4.21 | .350/3.49 | .167/4.35 | **.450/3.47** |
| **kg_ex_g** | .133/5.62 | .342/5.77 | .150/5.54 | .708/5.51 | .217/4.50 | .650/5.08 | .683/6.83 | **.700/4.31** | .700/6.98 | .200/4.65 | .167/4.08 | .217/4.14 | .133/4.53 | .217/4.32 | .200/4.57 | .200/4.04 | .167/4.43 | .150/4.28 | .250/4.74 | **.283/3.90** | .267/4.39 | .217/4.11 | .067/4.64 | **.317/3.96** |
| **tx_ex_m** | .175/4.79 | .458/3.63 | .267/4.58 | **1.000/1.07** | .050/9.14 | .342/5.37 | **.867/2.11** | **.467/5.57** | .467/7.99 | .067/7.02 | .000/7.49 | .017/11.01 | .017/7.09 | .000/7.59 | .017/6.92 | .017/8.30 | .033/7.06 | .017/9.94 | .033/6.79 | .067/8.62 | .067/7.70 | .017/8.94 | .017/6.63 | .083/10.37 |
| **tx_ex_g** | .292/5.36 | .275/6.48 | .208/5.76 | .683/5.25 | .017/11.52 | .342/11.41 | .642/10.71 | **.400/10.38** | .367/22.31 | .100/6.68 | .017/9.80 | .017/10.82 | .100/8.35 | .167/6.75 | .100/7.48 | .067/9.20 | .167/7.28 | .083/8.96 | .083/7.78 | .050/9.93 | .117/7.35 | .083/9.97 | .083/7.82 | .117/8.73 |

## Key Observations (in progress)

### C' and Ec' both achieve near-perfection at 100K
- **C' text mem**: 1.000/1.01, **Ec' text mem**: 1.000/1.01 — both perfect
- **C' KG mem**: 1.000/1.01, **Ec' KG mem**: .997/1.24 — both near-perfect
- **C' kg_ex_m text**: .983, **Ec' kg_ex_m text**: .900 — both strong cross-pollination
- **C' tx_ex_m text**: 1.000, **Ec' tx_ex_m text**: 1.000 — both perfect
- **Ec' tx_ex_m KG**: .867/2.11 — text-exclusive facts appearing in KG predictions (was .000 at 10K)
- C' uses linearized KG (kg_as_text), Ec' uses native causal KG — both approaches work at 100K
- Only weakness for both: generalization on unseen derived facts (~.700)

### Hc is the best non-V-rotation native KG model
- **Text mem**: .917/1.55 — surpasses Ec (.813/1.84)
- **KG mem**: .980/1.58 — surpasses Ec (.892/1.80)
- **kg_ex_m text**: .800/1.81 — strong cross-pollination, surpasses Ec (.733/2.34)
- **kg_ex_m KG**: .925/1.58 — near-perfect
- **tx_ex_m text**: .533/3.58 — text-exclusive facts appearing in text predictions
- Fixed cumsum (Hc) beats learned cumsum (Ec) on nearly every metric
- The fixed exponential decay pattern is a better inductive bias than learned angles for this task

### Hc' joins the top tier
- **Text mem**: .969/1.32 — #3 overall, behind only C' and Ec' (both 1.000)
- **KG mem**: .993/1.28 — #3 overall, near-perfect
- **kg_ex_m KG**: 1.000/1.19 — perfect KG-exclusive memorization
- **tx_ex_m text**: .867/1.87 — strong text-exclusive cross-pollination (matches Ec')
- V rotation lifts Hc (.917→.969 text, .980→.993 KG) but the gap is smaller than Ec→Ec'
- **Hc' tx_ex KG PPL is worse**: 7.99/22.31 vs Hc's 5.57/10.38 — V rotation hurts text→KG transfer

### I' (softmax) is the best MLM KG model
- **I' KG mem**: .416/3.61 — best h@5 and PPL among all MLM models by a wide margin
- **I' KG trn**: .400/3.57 — best transfer
- **I' KG gen**: .400/4.02 — best generalization (2x G' at .189)
- **I' kg_ex_m KG**: .467/3.61 — best KG-exclusive memorization among MLM models
- **I' kg_ex_g KG**: .283/3.90 — best KG-exclusive generalization
- V rotation doubles I's KG performance: .199→.416 mem, .189→.400 gen
- I' uses softmax attention (not softplus) — softplus causes NaN with learned per-layer angle projectors
- I without V rotation is comparable to E/A/F tier (.199 vs .205/.183/.193)
- **I' text mem**: .128/5.09 — #11 text, slightly behind G' (.140) but best text PPL among MLM models (tied with H'/F' at 5.09)
- Tradeoff: I' has worse cross-pollination (.033 kg_ex_m text vs G's .117) — V rotation sharpens KG at the cost of text transfer

### G' is the best MLM model (softplus)
- **G' KG mem**: .248/3.94 — best h@5 and best PPL among all MLM models
- **G' text mem**: .140/5.01 — best text PPL among MLM models, #2 h@5 behind E' (.143)
- **G' KG trn**: .244/3.77 — best transfer among MLM models
- V rotation gives G a large boost: .179→.248 KG mem, .101→.140 text mem
- Per-relation slot angles (G) slightly trail shared slot angles (A) without V rotation, but G' pulls ahead of A' with V rotation
- G' KG PPL (3.94) beats even E' (3.98, which is degraded by forgetting)

### H' (MLM) is marginally better than H
- **H' text mem**: .118/5.09 vs H .105/5.12 — slight improvement with V rotation
- **H' KG mem**: .231/4.03 vs H .174/4.48 — V rotation helps KG more
- Both still in the MLM slow-learner group, far behind causal counterparts (Hc .917, Hc' .969)

### H (MLM) is as slow as A
- **H text mem**: .105/5.12 — comparable to A (.097/5.13)
- **H KG mem**: .174/4.48 — comparable to A (.183/4.41)
- Fixed cumsum with MLM KG shows same slow convergence as slot angles with MLM KG
- The bottleneck is the MLM objective, not the positional encoding

### E' suffers catastrophic forgetting on KG
- **E' KG mem**: .907/1.54 at 10K → **.238/3.98 at 100K** — massive regression
- **E' text mem**: .009/9.95 at 10K → .143/4.96 at 100K — text slowly improving
- The causal text gradient overwhelms the MLM KG gradient over extended training
- KG knowledge learned early is slowly erased as text training continues
- This does NOT happen with causal KG training: Ec KG improved .670→.892, Ec' KG improved .768→.997
- **Conclusion**: MLM KG training is not just slow — it's unstable over long training. The bidirectional KG objective and causal text objective compete destructively. Causal KG avoids this because both objectives use the same attention pattern.

### MLM models are slow learners with forgetting risk
- A text PPL: 10.16 → 5.13 (10K → 100K) — text slowly improving
- A KG PPL: 11.42 → 4.41 — KG also improving (A may not yet show forgetting)
- E text PPL: 11.92 → 5.18 — text improving
- E KG PPL: 65.05 → 4.28 — KG improving (E without V rotation was broken at 10K, now recovering)
- But E' shows the danger: models that learn KG well early (E' had .907 at 10K) can lose it
- **Correct framing**: causal KG training converges faster AND is stable. MLM KG training converges slower AND risks catastrophic forgetting on KG as text training dominates.

### 10x more training transforms causal and kat models
- **B**: .064→.333 text mem (5x), .000→.217 kg_ex_m (from zero)
- **B'**: .365→.672 text mem (2x), .033→.217 kg_ex_m, .267→.550 tx_ex_m
- **C**: .296→.475 text mem, .033→.250 kg_ex_m
- **C'**: .783→1.000 text mem (ceiling), .067→.983 kg_ex_m (from near-zero to near-perfect)
- **Ec**: .249→.813 text mem (3x), .670→.892 KG mem
- **Ec'**: .329→1.000 text mem, .768→.997 KG mem — both modalities improve together
- 10K iterations was severely undertrained for all models

### V rotation doubles performance (B vs B', C vs C')
- B vs B' text mem: .333 vs .672
- C vs C' text mem: .475 vs 1.000
- C vs C' kg_ex_m text: .250 vs .983 — V rotation is the difference between weak and perfect cross-pollination

### K' is the new non-dual MLM KG champion
- **K' KG mem**: .419/3.47 — beats I' (.416/3.61) on both h@5 and PPL
- **K' KG trn**: .422/3.38 — beats I' (.400/3.57) — best transfer among all MLM models
- **K' KG gen**: .322/4.14 — behind I' (.400/4.02) on h@5 but comparable PPL
- **K' kg_ex_m KG**: .450/3.47 — close to I' (.467/3.61), better PPL
- **K' kg_ex_g KG**: .317/3.96 — beats I' (.283/3.90) on h@5
- **K' text mem**: .167/5.01 — **best text among MLM models**, beats G' (.140/5.01) and E' (.143/4.96) on h@5
- K = I (projected angles) + J (slot angles). The combination slightly edges out I' alone on KG and clearly wins on text.

### J' is strong among MLM models
- **J' KG mem**: .341/3.54 — #3 among MLM models (after K' .419 and I' .416)
- **J' KG trn**: .311/3.41 — #3 behind K' (.422) and I' (.400)
- **J' KG gen**: .289/4.17 — comparable to I' on PPL
- **J' text mem**: .114/5.11 — mid-tier among MLM models
- V rotation helps J significantly: J→J' KG mem .197→.341 (1.7x), KG PPL 4.30→3.54

### Cross-pollination leaderboard (kg_ex_m on text)
| Rank | Model | kg_ex_m h@5 | kg_ex_m PPL |
|------|-------|------------|------------|
| 1 | **C'** (kat) | .983 | **1.19** |
| 2 | **Ec'** | .900 | 1.77 |
| 3 | **Hc** | .800 | 1.81 |
| 4 | **Ec** | .733 | 2.34 |
| 5 | Hc' | .717 | 2.90 |
| 6 | C (kat) | .250 | 4.54 |
| 7 | B (kat) | .217 | 4.68 |
| 8 | B' (kat) | .217 | 4.82 |
| 9 | G (MLM) | .117 | 7.46 |
| 9 | I (MLM, smx) | .117 | 7.48 |
| 11 | A (MLM) | .067 | 7.75 |
| 11 | F (MLM) | .067 | 8.46 |
| 13 | A' (MLM) | .050 | 8.42 |
| 13 | H' (MLM) | .050 | 7.45 |
| 15 | E' (MLM) | .033 | 8.43 |
| 15 | E (MLM) | .033 | 9.41 |
| 15 | I' (MLM, smx) | .033 | 7.87 |
| 18 | G' (MLM) | .017 | 7.73 |
| 18 | F' (MLM) | .017 | 7.85 |
| 20 | H (MLM) | .000 | 7.89 |

### Text leaderboard at l2 100K
| Rank | Model | Text mem h@5 | Text mem PPL |
|------|-------|-------------|-------------|
| 1 | **C'** (kat) | **1.000** | **1.01** |
| 1 | **Ec'** | **1.000** | **1.01** |
| 3 | **Hc'** | .969 | 1.32 |
| 4 | **Hc** | .917 | 1.55 |
| 5 | **Ec** | .813 | 1.84 |
| 6 | B' (kat) | .672 | 2.29 |
| 7 | C (kat) | .475 | 3.16 |
| 8 | B (kat) | .333 | 3.89 |
| 9 | **K' (MLM, smx)** | .167 | **5.01** |
| 10 | E' (MLM) | .143 | 4.96 |
| 11 | **G' (MLM)** | .140 | 5.01 |
| 12 | **I' (MLM, smx)** | .128 | 5.09 |
| 13 | A' (MLM) | .123 | 5.05 |
| 14 | H' (MLM) | .118 | 5.09 |
| 15 | F' (MLM) | .116 | 5.09 |
| 16 | J' (MLM, smx) | .114 | 5.11 |
| 17 | E (MLM) | .109 | 5.18 |
| 18 | I (MLM, smx) | .106 | 5.14 |
| 19 | J (MLM, smx) | .105 | 5.16 |
| 20 | H (MLM) | .105 | 5.12 |
| 21 | F (MLM) | .104 | 5.18 |
| 22 | G (MLM) | .101 | 5.13 |
| 23 | K (MLM, smx) | .101 | 5.20 |
| 24 | A (MLM) | .097 | 5.13 |

### KG leaderboard at l2 100K
| Rank | Model | KG mem h@5 | KG mem PPL |
|------|-------|-----------|-----------|
| 1 | **C'** (lin) | **1.000** | **1.01** |
| 2 | **Ec'** | .997 | 1.24 |
| 3 | **Hc'** | .993 | 1.28 |
| 4 | **Hc** | .980 | 1.58 |
| 5 | **Ec** | .892 | 1.80 |
| 6 | B' (lin) | .664 | 2.30 |
| 7 | C (lin) | .497 | 3.05 |
| 8 | **K' (MLM, smx)** | **.419** | **3.47** |
| 9 | **I' (MLM, smx)** | **.416** | **3.61** |
| 10 | **J' (MLM, smx)** | .341 | 3.54 |
| 11 | B (lin) | .339 | 3.86 |
| 12 | G' (MLM) | .248 | 3.94 |
| 13 | E' (MLM) | .238 | 3.98 |
| 14 | H' (MLM) | .231 | 4.03 |
| 15 | F' (MLM) | .209 | 4.07 |
| 16 | A' (MLM) | .206 | 4.06 |
| 17 | E (MLM) | .205 | 4.28 |
| 18 | K (MLM, smx) | .205 | 4.28 |
| 19 | I (MLM, smx) | .199 | 4.31 |
| 20 | J (MLM, smx) | .197 | 4.30 |
| 21 | F (MLM) | .193 | 4.33 |
| 22 | A (MLM) | .183 | 4.41 |
| 23 | G (MLM) | .179 | 4.50 |
| 24 | H (MLM) | .174 | 4.48 |

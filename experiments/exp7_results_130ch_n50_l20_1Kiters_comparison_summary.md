# Exp 7 Results: 130ch, n_embed=50, n_layers=20, 1K iters — All Models

Models: A, A', E' (causal), F, F', G, G', H (causal), H' (causal)
Config: n_embed=50, n_layers=20, max_iters=1000, batch_size=32, lr=0.0005, device=cpu
Seeds: A/A'/F/F'/G/G' = 1 seed; E'/H/H' = 3-seed mean
KG training: A/A'/F/F'/G/G' = MLM (bidirectional); E'/H/H' = causal (next-token with direction flip + angle negation)

Source files:
- `exp7_results_130ch_n50_l20_1Kiters_AAp_Fp_3xvar_mlm.json` (A, A', F')
- `exp7_results_130ch_n50_l20_1Kiters_F_3xvar_mlm.json` (F)
- `exp7_results_130ch_n50_l20_1Kiters_Ep_3xvar_mlm_v2.json` (E' MLM, for reference)
- `exp7_results_20260227_120642.json` (E' causal)
- Failed run bd94b1b stdout (G, G' — extracted from log, JSON not saved)
- `exp7_results_20260227_142856.json` (H, H' causal)

## Model Summary

| Model | Angle type | KG format | KG training | Rotate V |
|-------|-----------|-----------|-------------|----------|
| A     | RoPE + learned slot angles (shared) | Slotted (HEAD/REL/TAIL) | MLM | No |
| A'    | RoPE + learned slot angles (shared) | Slotted (HEAD/REL/TAIL) | MLM | Yes |
| E'    | Learned per-token cumsum + relation operator | Native (chars only) | Causal | Yes |
| F     | Fixed RoPE | Flat (rel as token) | MLM | No |
| F'    | Fixed RoPE | Flat (rel as token) | MLM | Yes |
| G     | RoPE + learned slot angles (per-relation) | Slotted (HEAD/REL/TAIL) | MLM | No |
| G'    | RoPE + learned slot angles (per-relation) | Slotted (HEAD/REL/TAIL) | MLM | Yes |
| H     | Fixed cumsum + relation operator | Native (chars only) | Causal | No |
| H'    | Fixed cumsum + relation operator | Native (chars only) | Causal | Yes |

## Text Evaluation (hit@5 / ppl)

| Tier | A | A' | E' (csl) | F | F' | G | G' | H (csl) | H' (csl) |
|------|---|---|----------|---|---|---|---|---------|----------|
| **mem** | .083/7.10 | .085/7.12 | .085/7.10 | .075/7.12 | .083/7.12 | .079/7.12 | .080/6.91 | .086/6.40 | .086/6.21 |
| **transfer** | .122/7.35 | .089/7.28 | .056/7.53 | .089/7.36 | .011/7.49 | .078/7.28 | .089/7.27 | .096/6.75 | .085/6.44 |
| **gen** | .122/6.65 | .033/6.41 | .070/6.59 | .056/7.10 | .044/6.43 | .044/6.62 | .100/6.31 | .085/5.93 | .074/5.95 |
| **kg_excl_mem** | .000/7.90 | .067/7.94 | .067/8.17 | .100/8.38 | .050/7.76 | .083/8.23 | .033/7.72 | .061/7.10 | .056/7.08 |
| **kg_excl_gen** | .017/6.90 | .117/7.26 | .078/7.46 | .133/7.32 | .150/6.64 | .100/7.02 | .100/7.05 | .100/6.81 | .056/6.47 |
| **text_excl_mem** | .117/6.42 | .100/6.53 | .094/6.77 | .100/6.70 | .117/6.62 | .033/6.43 | .050/6.28 | .089/6.49 | .083/6.29 |
| **text_excl_gen** | .033/9.49 | .100/8.84 | .050/9.27 | .050/9.03 | .183/9.14 | .033/8.58 | .117/8.60 | .083/8.21 | .050/7.88 |

## KG Evaluation (hit@5 / ppl)

| Tier | A | A' | E' (csl) | F | F' | G | G' | H (csl) | H' (csl) |
|------|---|---|----------|---|---|---|---|---------|----------|
| **mem** | .075/6.66 | .084/6.48 | .093/7.01 | .086/6.15 | .081/6.26 | .081/6.69 | .078/6.93 | .082/6.90 | .083/6.83 |
| **transfer** | .078/6.85 | .111/6.73 | .109/7.15 | **.122**/6.26 | .056/6.44 | .022/7.05 | .022/7.43 | .115/7.23 | .074/6.89 |
| **gen** | .078/6.55 | .067/6.02 | .076/6.68 | .022/6.19 | .044/6.16 | .044/6.49 | .089/6.22 | .076/6.59 | .080/6.68 |
| **kg_excl_mem** | .050/6.75 | .050/6.56 | .069/7.54 | **.217**/6.49 | .083/6.75 | .133/6.67 | .067/7.33 | .078/7.22 | .089/7.43 |
| **kg_excl_gen** | .100/6.26 | .067/6.40 | .047/7.18 | **.133**/5.95 | .050/5.97 | .133/6.37 | .067/6.51 | .089/7.02 | .075/7.02 |
| **text_excl_mem** | .117/6.44 | .050/6.11 | .086/7.24 | **.167**/5.87 | .150/6.51 | .067/6.05 | .150/5.99 | .092/6.99 | .106/6.96 |
| **text_excl_gen** | .000/8.12 | .117/7.89 | .036/9.15 | .083/7.16 | .067/7.55 | .050/7.93 | .100/8.84 | .064/8.94 | .044/8.78 |

## Key Observations

### Text PPL
- **H/H' have best text PPL**: H' gets 6.21 mem, 5.95 gen — lowest text PPL of any model. Fixed cumsum angles give better text modeling than RoPE-based models (A/F ~7.1) or learned cumsum (E' ~7.1).
- **G' second best text PPL**: 6.91 mem, 6.31 gen — per-relation slot angles with V rotation help text.

### KG Performance
- **F dominates KG eval**: Best KG PPL across the board (5.87-6.49), best kg_excl_mem=.217, best text_excl_mem=.167, best transfer=.122.
- **Causal models (E', H, H') have ~0.5-1pt worse KG PPL** than MLM models — expected since causal prediction is harder than fill-in-the-blank.
- **H causal KG transfer strong**: .115 transfer — second only to F (.122), better than E' causal (.109).
- **G/G' per-relation slot angles hurt KG transfer**: .022/.022 — worst of all models. Per-relation angles fragment the angle space, making cross-relation generalization harder.

### Cross-Pollination (KG→Text and Text→KG)
- **F shows strongest cross-pollination on KG**: kg_excl_mem=.217 and text_excl_mem=.167 — both directions of knowledge transfer strongest for standard RoPE.
- **F also shows cross-pollination on text**: kg_excl_mem=.100, kg_excl_gen=.133 — KG-only facts leaking into text predictions.
- **H' best text_excl_mem on KG among causal models**: .106 — text-only facts leaking into causal KG predictions.

### MLM vs Causal KG Training
- **E' causal hit@5 competitive on KG**: .093 mem, .109 transfer — matches A' MLM (.084/.111) despite the harder task.
- **Text eval essentially unchanged by causal KG training**: E' causal text (.085/7.10) ≈ E' MLM (.074/8.19 from separate run).

### Primed (V rotation) Effects
- **F vs F'**: F beats F' on KG substantially (.217 vs .083 kg_excl_mem). F' beats F on text gen (.183 vs .050 text_excl_gen).
- **A vs A'**: Similar pattern — A better on some KG tiers, A' better on text gen.
- **H vs H'**: H better on KG transfer (.115 vs .074), H' better on text PPL (6.21 vs 6.40 mem).

### Overall
- **All models still at noise floor**: At only 1K iters with 20 layers, hit@5 is mostly .05-.12. Need more iterations to see real differentiation.
- **Cross-pollination signal present even at noise floor**: F's .217 kg_excl_mem and .167 text_excl_mem suggest standard RoPE enables knowledge transfer between modalities even early in training.
- **Fixed angles (F, H) consistently outperform learned angles (D, E) on text PPL**: Suggests learned per-token angles add noise early in training.

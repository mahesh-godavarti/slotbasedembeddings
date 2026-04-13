# Exp 7a Results: 130ch, n_embed=100, n_layers=20, 10K iters — Partial (G/G'/I/I' still running)

Models completed: A, A', E, E', F, F', H, H'
Models in progress: G (56%), I (26%) — G' and I' pending
Config: n_embed=100, n_layers=20, max_iters=10000, batch_size=32, lr=0.0005, device=cpu
Seeds: 1
KG training: A/A'/F/F'/G/G' = MLM; E/E'/H/H'/I/I' = causal

Source: background tasks b530a1e (A/A'/F/F'/G/G') and b081759 (E/E'/H/H'/I/I')

## Text Evaluation (h@5 / PPL)

| Tier | A | A' | E | E' | F | F' | H | H' |
|------|---|---|---|---|---|---|---|---|
| **mem** | .093/5.62 | .098/5.52 | .157/4.97 | .142/5.05 | .092/5.63 | .100/5.49 | .129/5.05 | .154/5.00 |
| **transfer** | .133/5.74 | .089/5.76 | .144/5.06 | .122/5.30 | .056/5.91 | .044/5.64 | .089/5.14 | .200/4.99 |
| **gen** | .122/5.45 | .100/5.39 | .100/5.06 | .067/5.19 | .056/5.49 | .078/5.55 | .167/5.14 | .089/4.93 |
| **kg_excl_mem** | .033/6.89 | .033/7.08 | .067/5.34 | .100/5.34 | **.183**/6.64 | .133/6.53 | .117/5.21 | .067/5.35 |
| **kg_excl_gen** | .050/6.72 | .033/6.96 | .067/5.69 | .067/5.81 | .050/5.98 | .067/6.44 | .100/5.37 | .033/5.52 |
| **text_excl_mem** | .000/5.95 | .117/5.42 | .050/5.34 | **.150**/5.25 | .083/5.81 | .067/5.58 | .067/5.30 | .083/5.29 |
| **text_excl_gen** | .067/6.57 | **.167**/7.00 | .050/6.05 | .083/5.75 | .000/6.91 | .067/6.53 | .117/5.81 | .050/5.83 |

## KG Evaluation (h@5 / PPL)

| Tier | A | A' | E | E' | F | F' | H | H' |
|------|---|---|---|---|---|---|---|---|
| **mem** | .139/5.01 | .161/4.87 | .323/4.53 | **.809**/2.37 | .148/4.96 | .158/4.84 | .146/5.54 | .172/5.45 |
| **transfer** | .122/5.08 | .189/4.63 | .278/4.50 | **.806**/2.32 | .111/4.97 | .122/5.11 | .172/5.40 | .178/5.28 |
| **gen** | .156/4.94 | .111/4.96 | .272/5.04 | **.633**/4.34 | .144/4.97 | .122/5.09 | .133/5.69 | .072/5.74 |
| **kg_excl_mem** | .067/5.71 | .167/5.06 | .317/4.65 | **.858**/2.17 | .117/5.25 | **.217**/5.44 | .175/5.43 | .117/5.47 |
| **kg_excl_gen** | .100/5.39 | .067/5.08 | .150/5.18 | **.625**/3.69 | .167/4.99 | .117/5.49 | .083/5.76 | .100/5.67 |
| **text_excl_mem** | .033/6.08 | .050/6.08 | .017/7.12 | .050/12.12 | .083/5.81 | .067/5.72 | .067/6.05 | .025/6.07 |
| **text_excl_gen** | .133/5.86 | .150/5.92 | .092/8.10 | .108/15.67 | .100/5.94 | .133/5.70 | .067/6.99 | .050/6.72 |

## Key Observations

### KG Evaluation
- **E' dominates KG**: .809/.806/.633 mem/trn/gen, .858 kg_excl_mem -- far ahead of all others. Learned cumsum + V rotation is the clear KG winner at depth (20 layers).
- **E (no V rotation) much weaker**: .323 mem KG vs E' .809 -- V rotation is critical for KG at 20 layers.
- **F' second best kg_excl_mem on KG**: .217 -- standard RoPE + V rotation enables some KG-exclusive learning.
- **A/A'/F/H/H' all cluster around .12-.19 KG mem**: These models haven't differentiated much on KG at this config.

### Text Evaluation
- **Text PPL best for H'/E/H**: ~4.93-5.05 ppl, vs A/F ~5.5. Fixed/learned cumsum angles give better text modeling at depth.
- **H' best text transfer**: .200 h@5 -- highest of any model.
- **Text h@5 still low overall**: Best is E .157 mem -- models haven't converged on text with this config.

### Cross-Pollination (KG-exclusive on text eval, text-exclusive on KG eval)
- **F strongest KG->Text cross-pollination**: kg_excl_mem on text = .183 -- KG-only facts leaking to text predictions.
- **E' strongest Text->Text cross-pollination**: text_excl_mem on text = .150.
- **text_excl on KG near zero for all**: .017-.083 -- text-only facts barely transfer to KG eval direction.
- **E' KG-exclusive on KG is outstanding**: .858 -- higher than its own memorization (.809), suggesting KG-exclusive facts (fewer, simpler) are easier to learn.
- **E' text_excl on KG is high PPL**: 12.12/15.67 -- E' heavily specializes KG eval, text-only facts get punished.

### Comparison to Previous Runs
- At n50/l20/1K iters: all models were at noise floor (.05-.12 h@5). Now at n100/l20/10K: E' has broken away on KG (.809 mem vs .093 before).
- At n500/l2/10K iters (1000 chains): E' got .806 KG mem with 2 layers. Here E' gets .809 with 20 layers -- similar KG performance but much better text PPL (5.05 vs 9.84).
- Cross-pollination still weak but present: F .183 kg_excl_mem (text), vs .100 at n50/l20/1K.

### V Rotation Effects at Depth
- **E vs E'**: Massive gap on KG (.323 vs .809 mem). V rotation transforms E from mediocre to dominant.
- **H vs H'**: Minimal difference on KG (.146 vs .172 mem). For fixed cumsum, V rotation doesn't help much.
- **A vs A'**: Small gap on KG (.139 vs .161 mem). V rotation helps but not dramatically for slotted models.
- **F vs F'**: Small gap on KG (.148 vs .158 mem). For flat RoPE, V rotation is marginal.
- Conclusion: V rotation is only transformative when combined with learned per-token angles (E/E').

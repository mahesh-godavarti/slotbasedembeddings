# Exp 8 Results & Observations

## Run 4: B/B'/C/C' with --kg_as_text, 200K iters

**Config**: n_embed=500, n_layers=2, iters=200000, seeds=1, vocab_size=16000, kg_as_text=True
**Date**: 2026-03-01
**Chain counts**: 669 total (154 family, 400 synonym, 50 antonym-synonym, 31 capital-language, 34 hypernym)

### Text Evaluation (seed 0)

| Tier                         | B h@5 | B PPL   | B' h@5 | B' PPL  | C h@5 | C PPL   | C' h@5 | C' PPL  |
|------------------------------|-------|---------|--------|---------|-------|---------|--------|---------|
| memorization                 | 0.133 | 70.62   | 0.219  | 55.08   | 0.130 | 82.74   | 0.123  | 79.78   |
| transfer                     | 0.020 | 281.41  | 0.013  | 314.49  | 0.026 | 298.45  | 0.013  | 326.83  |
| generalization               | 0.011 | 408.36  | 0.000  | 752.29  | 0.011 | 595.86  | 0.000  | 544.69  |
| kg_exclusive_memorization    | 0.000 | 1144.93 | 0.000  | 1881.79 | 0.000 | 2577.55 | 0.015  | 1616.23 |
| kg_exclusive_generalization  | 0.000 | 1644.04 | 0.000  | 1694.33 | 0.000 | 1969.57 | 0.000  | 1832.34 |
| text_exclusive_memorization  | 0.058 | 83.35   | 0.130  | 60.37   | 0.014 | 126.46  | 0.130  | 59.35   |
| text_exclusive_generalization| 0.018 | 245.31  | 0.000  | 307.42  | 0.035 | 405.93  | 0.018  | 376.77  |

### Linearized KG Evaluation (seed 0)

| Tier                         | B h@5 | B PPL   | B' h@5 | B' PPL  | C h@5 | C PPL   | C' h@5 | C' PPL  |
|------------------------------|-------|---------|--------|---------|-------|---------|--------|---------|
| memorization                 | 0.107 | 95.77   | 0.110  | 86.24   | 0.067 | 137.63  | 0.091  | 111.85  |
| transfer                     | 0.031 | 242.90  | 0.015  | 202.16  | 0.018 | 263.59  | 0.018  | 250.60  |
| generalization               | 0.014 | 573.72  | 0.014  | 585.42  | 0.000 | 628.70  | 0.014  | 630.86  |
| kg_exclusive_memorization    | 0.000 | 441.53  | 0.019  | 394.29  | 0.026 | 488.37  | 0.032  | 543.90  |
| kg_exclusive_generalization  | 0.030 | 719.80  | 0.020  | 747.12  | 0.020 | 716.42  | 0.010  | 844.16  |
| text_exclusive_memorization  | 0.000 | 191.61  | 0.019  | 191.01  | 0.006 | 239.96  | 0.000  | 218.97  |
| text_exclusive_generalization| 0.031 | 316.90  | 0.015  | 348.23  | 0.015 | 429.50  | 0.015  | 387.94  |

### Observations

1. **B' is the best memorizer.** PPL 55 and h@5 21.9% on text memorization — best by a wide margin. V-rotation with RoPE excels at recalling seen facts.

2. **C' caught up to C** as expected with more training. Memorization PPL 80 vs 83, and C' ties B' on text_exclusive_mem (PPL 59 vs 60, h@5 13% both). The learned-angle + V-rotation combination needs more training but eventually converges.

3. **C' beats C on kg_exclusive** (1616 vs 2578 text). V-rotation significantly helps C-architecture models with KG-related content, consistent with the hypothesis that V-rotation aids cross-format transfer.

4. **B still best on generalization** (PPL 408 vs B' 752, C 596, C' 545). Primed variants overfit memorization at the expense of generalization — a memorization-generalization tradeoff from V-rotation.

5. **C' did not become the best overall.** Despite improvement, C' still trails B/B' on most tiers. The learned cumsum angles underperform fixed RoPE even with V-rotation. C' may need even more training or a different angle initialization to compete.

6. **PPL progression across runs** (B memorization text): 1698 (10K) → 928 (10K wider) → 142 (100K) → 71 (200K). Still improving — not plateaued.

7. **Primed variants excel at memorization but hurt generalization.** This is the clearest pattern: B' memorization 55 vs generalization 752; B memorization 71 vs generalization 408. V-rotation sharpens recall of seen data but doesn't help unseen combinations.

---

## Run 3: B/B'/C/C' with --kg_as_text, 100K iters (STOPPED — seeds 0 full, seed 1 partial)

**Config**: n_embed=500, n_layers=2, iters=100000, seeds=3, vocab_size=16000, kg_as_text=True
**Date**: 2026-03-01
**Chain counts**: 669 total (154 family, 400 synonym, 50 antonym-synonym, 31 capital-language, 34 hypernym)
**Status**: Stopped early to launch 200K run. Seed 0: all models. Seed 1: B, B' only.

### Text Evaluation (seed 0: all models; seed 1: B, B' only)

| Tier                         | B s0 h@5 | B s0 PPL | B s1 h@5 | B s1 PPL | B' s0 h@5 | B' s0 PPL | B' s1 h@5 | B' s1 PPL | C s0 h@5 | C s0 PPL | C' s0 h@5 | C' s0 PPL |
|------------------------------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|----------|----------|-----------|-----------|
| memorization                 | 0.028    | 141.55   | 0.039    | 142.37   | 0.048     | 121.78    | 0.043     | 159.77    | 0.028    | 185.78   | 0.028     | 197.32    |
| transfer                     | 0.013    | 367.57   | 0.007    | 296.16   | 0.007     | 359.44    | 0.007     | 255.08    | 0.007    | 374.51   | 0.000     | 465.78    |
| generalization               | 0.011    | 532.40   | 0.045    | 418.78   | 0.033     | 453.97    | 0.045     | 441.44    | 0.011    | 561.46   | 0.011     | 520.71    |
| kg_exclusive_memorization    | 0.000    | 1491.08  | 0.000    | 1262.29  | 0.000     | 1217.63   | 0.000     | 952.69    | 0.000    | 1807.21  | 0.000     | 1524.73   |
| kg_exclusive_generalization  | 0.000    | 1596.41  | 0.000    | 1447.59  | 0.000     | 1392.12   | 0.021     | 1398.01   | 0.000    | 2026.68  | 0.000     | 1793.00   |
| text_exclusive_memorization  | 0.014    | 144.48   | 0.014    | 141.05   | 0.043     | 110.36    | 0.069     | 126.21    | 0.029    | 145.07   | 0.029     | 157.29    |
| text_exclusive_generalization| 0.000    | 321.16   | 0.000    | 696.73   | 0.000     | 349.41    | 0.037     | 493.97    | 0.000    | 349.04   | 0.000     | 314.68    |

### Linearized KG Evaluation (seed 0: all models; seed 1: B, B' only)

| Tier                         | B s0 h@5 | B s0 PPL | B s1 h@5 | B s1 PPL | B' s0 h@5 | B' s0 PPL | B' s1 h@5 | B' s1 PPL | C s0 h@5 | C s0 PPL | C' s0 h@5 | C' s0 PPL |
|------------------------------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|----------|----------|-----------|-----------|
| memorization                 | 0.037    | 207.18   | 0.029    | 239.46   | 0.031     | 195.83    | 0.054     | 219.17    | 0.012    | 323.80   | 0.019     | 311.46    |
| transfer                     | 0.018    | 353.54   | 0.021    | 315.90   | 0.034     | 326.50    | 0.025     | 278.20    | 0.006    | 415.25   | 0.003     | 436.90    |
| generalization               | 0.010    | 757.39   | 0.005    | 575.66   | 0.005     | 777.26    | 0.005     | 570.36    | 0.005    | 839.61   | 0.005     | 893.26    |
| kg_exclusive_memorization    | 0.013    | 628.83   | 0.019    | 544.20   | 0.019     | 523.09    | 0.032     | 471.06    | 0.006    | 865.06   | 0.000     | 644.71    |
| kg_exclusive_generalization  | 0.010    | 826.70   | 0.010    | 749.93   | 0.010     | 778.45    | 0.019     | 767.41    | 0.000    | 954.71   | 0.000     | 954.32    |
| text_exclusive_memorization  | 0.013    | 370.42   | 0.020    | 287.69   | 0.013     | 324.95    | 0.026     | 261.49    | 0.013    | 358.89   | 0.006     | 478.49    |
| text_exclusive_generalization| 0.000    | 467.10   | 0.008    | 1104.75  | 0.008     | 498.21    | 0.008     | 853.52    | 0.008    | 539.21   | 0.000     | 542.09    |

### Observations

1. **10x more training dramatically improves PPL.** B text memorization: 928 (10K iters) → 142 (100K iters), a ~6.5x reduction.

2. **Hit rates are no longer zero.** Small but nonzero h@5 (1-5%). First sign of any chain reasoning capability. B' seed 1 reaches 6.9% h@5 on text_exclusive_mem.

3. **B' is the best model overall.** Lowest PPL on most tiers, especially kg_exclusive_mem (953-1218 text) and best hit rates. V-rotation helps with sufficient training — a reversal from the 10K-iter runs.

4. **C' outperforms C on key tiers.** C' beats C on generalization (521 vs 561), kg_exclusive_mem (1525 vs 1807), kg_exclusive_gen (1793 vs 2027), and text_exclusive_gen (315 vs 349). C' is only behind C on memorization and transfer — tiers where more training should help.

5. **V-rotation benefits both architectures.** Both B' > B and C' > C on kg_exclusive and generalization tiers. Primed variants need more training to converge but outperform unprimed once sufficiently trained.

6. **B is stable across seeds** (memorization PPL 142 both seeds). Some variance on smaller tiers (text_exclusive_gen: 321 vs 697).

7. **kg_exclusive tiers remain high PPL** (~950-1800 text eval). Linearized KG doesn't transfer to natural language templates. More training may help.

8. **Stopped early to run at 200K iters** — PPL still improving and primed variants appear undertrained.

---

## Run 1: B/B'/C/C' with --kg_as_text (chain-based tiers)

**Config**: n_embed=100, n_layers=20, iters=10000, seeds=3, vocab_size=16000, kg_as_text=True
**Date**: 2026-03-01
**Chain counts**: 669 total (154 family, 400 synonym, 50 antonym-synonym, 31 capital-language, 34 hypernym)
**Results file**: exp8_results_20260301_084134.json

### Text Evaluation (averaged over 3 seeds)

| Tier                         | B PPL   | B' PPL  | C PPL   | C' PPL  |
|------------------------------|---------|---------|---------|---------|
| memorization                 | 1697.72 | 1771.99 | 2871.50 | 3383.52 |
| transfer                     | 2137.14 | 2173.83 | 3491.48 | 3959.95 |
| generalization               | 2832.55 | 2773.59 | 4661.85 | 4706.77 |
| kg_exclusive_memorization    | 2489.63 | 2925.81 | 4404.87 | 5258.06 |
| kg_exclusive_generalization  | 2937.76 | 3322.24 | 4314.37 | 4435.63 |
| text_exclusive_memorization  | 1412.35 | 1521.31 | 2444.91 | 3059.91 |
| text_exclusive_generalization| 2337.79 | 2316.55 | 3766.46 | 4339.72 |

Hit rates (h@1, h@5) were 0.000 across all models and tiers.

### Observations

1. **No model can predict derived facts.** All hit rates are zero. PPL is in the thousands — the models aren't learning chain reasoning even when facts are explicitly in the training text (memorization tier).

2. **B consistently beats C** (~1.5-2x lower PPL). Standard RoPE positional encoding outperforms learned per-token cumsum angles for text-only models.

3. **V-rotation (primed variants) doesn't help.** B' is slightly worse than B, C' is slightly worse than C. The rotate-V mechanism adds no benefit for text-only linearized KG training.

4. **PPL follows expected tier ordering**: memorization < transfer < generalization, and text_exclusive_mem has the lowest PPL (~1412 for B) since those chains have full template text in training.

5. **kg_exclusive tiers have high PPL** even though KG triples are linearized as text — the model sees `"james <father_of> john"` as text but can't transfer that to predict derived facts in natural language template form.

---

## Run 2: B/B'/C/C' with --kg_as_text, wider/shallower (chain-based tiers)

**Config**: n_embed=500, n_layers=2, iters=10000, seeds=3, vocab_size=16000, kg_as_text=True
**Date**: 2026-03-01
**Chain counts**: 669 total (154 family, 400 synonym, 50 antonym-synonym, 31 capital-language, 34 hypernym)
**Results file**: exp8_results_20260301_111916.json

### Text Evaluation (averaged over 3 seeds)

| Tier                         | B PPL   | B' PPL  | C PPL   | C' PPL  |
|------------------------------|---------|---------|---------|---------|
| memorization                 | 927.89  | 909.00  | 1223.69 | 1593.27 |
| transfer                     | 1229.53 | 1104.18 | 1657.07 | 2025.14 |
| generalization               | 1884.90 | 1885.69 | 2414.22 | 3074.17 |
| kg_exclusive_memorization    | 1844.24 | 1760.62 | 2578.53 | 2728.68 |
| kg_exclusive_generalization  | 1935.76 | 2282.88 | 2703.91 | 3098.14 |
| text_exclusive_memorization  | 782.79  | 779.11  | 952.70  | 1357.63 |
| text_exclusive_generalization| 1543.35 | 1635.21 | 2035.81 | 2374.98 |

Hit rates (h@1, h@5) were 0.000 across all models and tiers.

### Observations

1. **Wider embeddings help significantly.** n_embed=500 / n_layers=2 reduces PPL by ~45% vs n_embed=100 / n_layers=20 (B memorization: 928 vs 1698). Width matters more than depth for these text-only models.

2. **Still no chain reasoning.** Despite much lower PPL, hit rates remain zero. The models fit training text better but still can't derive new facts from chains.

3. **Same architectural ranking holds.** B > C (RoPE > learned cumsum), V-rotation still provides no benefit (B' ≈ B, C' worse than C).

4. **B' shows slight improvement over B on some tiers** (e.g., transfer: 1104 vs 1230), unlike Run 1 where B' was consistently worse. With more capacity, V-rotation is at least not harmful for B.

5. **C' is consistently the worst model**, with PPL 30-50% higher than C on most tiers. The combination of learned angles + V-rotation hurts.

6. **Tier ordering preserved**: text_exclusive_mem (783) < memorization (928) < transfer (1230) < generalization (1885).

---

## Run 0 (old tier system): A/A' without kg_as_text

**Config**: n_embed=100, n_layers=20, iters=10000, seeds=3, vocab_size=16000
**Date**: 2026-03-01
**Note**: This used the OLD co-occurrence-based tier system (before chain-based rewrite)
**Results file**: exp8_results_20260301_060107.json

### Text Evaluation (averaged over 3 seeds)

| Tier                         | A PPL   | A' PPL  |
|------------------------------|---------|---------|
| memorization                 | 6788.64 | 5621.07 |
| transfer                     | 6309.67 | 5376.99 |
| generalization               | 7062.40 | 5928.76 |
| kg_exclusive_memorization    | 4662.14 | 4144.27 |
| kg_exclusive_generalization  | 4061.13 | 3666.47 |
| text_exclusive_memorization  | 502.16  | 487.44  |
| text_exclusive_generalization| 290.78  | 304.50  |

### KG Evaluation (averaged over 3 seeds)

| Tier                         | A h@5 | A PPL  | A' h@5 | A' PPL |
|------------------------------|-------|--------|--------|--------|
| memorization                 | 0.297 | 126.38 | 0.295  | 121.83 |
| transfer                     | 0.288 | 157.85 | 0.292  | 145.94 |
| generalization               | 0.293 | 141.77 | 0.292  | 134.85 |
| kg_exclusive_memorization    | 0.145 | 206.13 | 0.127  | 209.93 |
| kg_exclusive_generalization  | 0.150 | 185.09 | 0.147  | 190.62 |
| text_exclusive_memorization  | 0.000 | 14316  | 0.000  | 15034  |
| text_exclusive_generalization| 0.000 | 12106  | 0.000  | 15669  |

### Observations (old tier system)

1. **No cross-modal transfer.** Text-only tiers learn text well (PPL ~300-500), KG tiers learn KG well (h@5 ~30%, PPL ~130), but KG knowledge doesn't help text predictions and vice versa.

2. **A vs A' shows no meaningful difference.** V-rotation doesn't help for slotted KG models either.

3. **KG eval works**: ~30% h@5 on memorization/transfer/generalization shows the KG pathway learns. But text_exclusive KG PPL is ~14K (random), confirming zero text→KG transfer.

4. **Old tier system was noisy**: text_exclusive used arbitrary word co-occurrences like `(the, text_cooccurrence, the)`, making those results less meaningful. The new chain-based system is a much better test.

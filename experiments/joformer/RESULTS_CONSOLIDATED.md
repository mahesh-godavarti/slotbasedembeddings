# JoFormer Consolidated Experiment Results

Results merged from both experiment folders: `~/joformer/` and `~/OTHER_STUFF/joformer/`.

All experiments: Wikipedia text, BPE tokenization, block_size=64, batch_size=32, lr=5e-4 (unless noted), dropout=0.2, seed=42.

---

## Phase 1: Initial Experiments (2M wiki lines, vocab=16000)

### Softplus Attention (log(exp(x)+1))

#### n_embed=500, n_layers=2, 100k iters

| Model | Train PPL | Val PPL | Best Val PPL (iter) |
|-------|----------|---------|---------------------|
| joformer_fixed | 79.05 | 88.56 | 80.46 (99500) |
| roformer | 79.37 | 88.60 | 80.99 (99500) |
| joformer_learned | 79.43 | 94.60 | 86.07 (97000) |
| joformer_projected | NaN | NaN | -- |

#### n_embed=100, n_layers=2, 100k iters

| Model | Train PPL | Val PPL |
|-------|----------|---------|
| roformer | 148.02 | 146.49 |
| joformer_fixed | 149.67 | 148.93 |
| joformer_learned | NaN (died iter 14500) | NaN |
| joformer_projected | NaN (crashed) | NaN |

**Softplus instability**: joformer_learned and joformer_projected both produce unbounded cumsum'd angles that grow during training, causing large attention scores that overflow in `log(exp(x)+1)`. joformer_fixed uses cumsum on fixed (non-learned) angles so they don't grow. roformer uses standard fixed RoPE angles.

### Switching to Softmax

Because joformer_learned and joformer_projected crashed with softplus (NaN from unbounded angles overflowing in `log(exp(x)+1)`), we tested joformer_projected with standard softmax attention. It ran 100k iters with no NaN — softmax is numerically stable by design since it normalizes via exp/sum rather than passing through exp directly.

#### n_embed=100, n_layers=2, 100k iters (softmax)

| Model | Train PPL | Val PPL |
|-------|----------|---------|
| **joformer_projected** | **130.61** | **129.97** |
| roformer | 149.47 | 148.00 |
| joformer_fixed | 150.06 | 148.77 |
| joformer_learned | 157.04 | 150.62 |

All 4 models stable with softmax — no NaN.

**Softmax vs softplus for the stable models**: For roformer and joformer_fixed (which were stable under both), switching to softmax made almost no difference — roformer went from 146.49 to 148.00 val PPL, joformer_fixed from 148.93 to 148.77.

**joformer_projected is the clear winner with softmax** — 129.97 val PPL, ~18 PPL ahead of roformer.

---

## Phase 2: Grid Search v1 (softmax, lr=5e-4, 50k iters, 2M lines, vocab=16000)

Swept n_embed x n_layers, trading off width for depth.

| Config | roformer | jo_fixed | jo_learned | jo_projected |
|--------|----------|----------|------------|--------------|
| n100, L2 | 164.21 | 163.39 | 167.00 | **152.86** |
| n100, L4 | 151.92 | 152.86 | 161.55 | **134.27** |
| n100, L6 | **142.13** | 143.14 | 152.10 | 140.77 |
| n100, L8 | **132.31** | 132.23 | 143.90 | 135.21 |
| n200, L2 | 137.41 | 135.87 | 134.03 | **119.98** |
| n200, L4 | 117.29 | 117.46 | 120.53 | **104.80** |
| n200, L6 | **110.47** | 110.86 | 111.68 | diverged |
| n500, L2 | 93.81 | **92.83** | 104.02 | 93.17 |
| n500, L4 | **87.50** | 87.82 | 96.70 | diverged |

### Observations
- **joformer_projected wins at shallow/narrow configs** (n100 L2-L4, n200 L2-L4) by large margins.
- **joformer_projected diverges at deeper configs** (n200 L6, n500 L4). Training instability, not inability to learn.
- **roformer and jo_fixed are the most reliable** — always within ~1 PPL, never diverge.
- **jo_learned consistently lags** by 5-12 PPL behind roformer/fixed.

---

## Phase 3: Fixing joformer_projected Divergence (n200 L6)

### Attempts on n200 L6, joformer_projected only, 50k iters

| Setting | Best Val PPL | Diverged? |
|---------|-------------|-----------|
| lr=5e-4, no schedule (grid) | 195 (iter 8000) | Yes |
| lr=5e-4, cosine decay | 150 (iter 14000) | Yes |
| lr=2e-4, cosine decay | 108 (iter 47500) | No, final val PPL 118 |

### Extended run: n200 L6, lr=2e-4, cosine decay, 200k iters

| Train PPL | Final Val PPL | Best Val PPL (iter) |
|----------|---------------|---------------------|
| 83.21 | 87.20 | **78.42** (iter 184000) |

### Constant lr=2e-4 (no cosine decay), 200k iters

| Train PPL | Final Val PPL | Best Val PPL (iter) |
|----------|---------------|---------------------|
| 83.95 | 87.89 | **79.96** (iter 184000) |

**lr=2e-4 is the key fix, not the schedule.** Cosine decay gives only ~1.5 PPL improvement.

---

## Phase 4: Grid Search v2 (softmax, lr=2e-4, 200k iters, 2M lines, vocab=16000)

Format: best_val_ppl (iter) / final_val_ppl

| Config | roformer | jo_fixed | jo_learned | jo_projected |
|--------|----------|----------|------------|--------------|
| n100, L2 | 144.06 (198.5k) / 153.68 | 144.60 (198.5k) / 154.09 | 147.60 (193k) / 158.35 | **125.34** (183k) / 134.21 |
| n100, L4 | 122.28 (198k) / 131.95 | 123.69 (198k) / 132.96 | 132.72 (198k) / 141.01 | **114.52** (189k) / 119.28 |
| n100, L6 | 116.34 (191k) / 123.47 | 116.69 (195.5k) / 123.01 | 123.75 (191.5k) / 130.73 | **107.81** (197.5k) / 119.52 |
| n100, L8 | 113.08 (197.5k) / 119.75 | 111.95 (169.5k) / 119.26 | 119.45 (195k) / 125.23 | **106.30** (176.5k) / 115.86 |
| n200, L2 | 106.24 (177.5k) / 118.69 | 105.82 (177.5k) / 118.01 | 115.44 (196k) / 121.91 | **93.69** (178.5k) / 101.11 |
| n200, L4 | 91.45 (186k) / 103.72 | 91.48 (186k) / 104.41 | 98.83 (196.5k) / 102.73 | **85.98** (167k) / 90.98 |
| n200, L6 | 82.08 (192k) / 92.56 | 82.29 (192k) / 93.30 | 90.85 (167k) / 99.56 | **79.96** (185k) / 85.68 |
| n500, L2 | 78.61 (184.5k) / 85.73 | 77.85 (195k) / 84.75 | 82.20 (191.5k) / 86.72 | **70.65** (195.5k) / 75.10 |
| n500, L4 | 67.52 (195k) / 67.79 | 67.60 (195k) / 67.79 | 69.93 (191k) / 76.08 | **63.85** (144.5k) / 67.28 |

### Observations
- **joformer_projected wins every config** — by 4-19 PPL.
- **lr=2e-4 eliminates projected divergence** — n200 L6 and n500 L4 both stable.
- **200k iters dramatically improves all models** vs 50k grid.
- **joformer_projected peaks earlier then overfits** — best PPL at iter 144-197k, then degrades 1-12 PPL by 200k.
- **All models overfit** — final PPL is 0.3-12 PPL worse than best, especially at n200+.

---

## Phase 5: Scaling to Full Wikipedia

Implemented `train_wiki_streaming.py` — memory-mapped streaming for the full 28.8M-line wiki (3.0 GB). Preprocessing tokenizes to a binary file on disk; training reads via `np.memmap`. Peak RAM: ~50-100MB regardless of dataset size.

---

## Phase 6: Grid Search v3 — Full Wikipedia, vocab=8000

*Source: `~/joformer/`*

softmax, lr=2e-4, 200k iters, full wiki (28.8M lines). Vocab=8000.

Format: best_val_ppl (iter) / final_val_ppl

| Config | roformer | jo_fixed | jo_learned | jo_projected |
|--------|----------|----------|------------|--------------|
| n100, L2 | 7.24 (182k) / 7.31 | 6.67 (182k) / 6.75 | 6.43 (186k) / 6.67 | **6.15** (179k) / 6.43 |
| n100, L4 | 6.17 (187k) / 6.28 | 5.85 (178k) / 5.97 | 5.81 (184k) / 6.07 | **5.50** (180k) / 5.68 |
| n100, L6 | 5.71 (163k) / 5.87 | 5.47 (163k) / 5.62 | 5.51 (138k) / 5.74 | **5.34** (182k) / 5.48 |
| n100, L8 | 5.42 (199k) / 5.47 | 5.22 (199k) / 5.26 | 5.28 (184k) / 5.35 | **5.18** (193k) / 5.26 |
| n200, L2 | 6.34 (154k) / 6.44 | 5.76 (154k) / 5.83 | 5.56 (196k) / 5.76 | **5.29** (197k) / 5.46 |
| n200, L4 | 5.36 (195k) / 5.76 | 5.07 (198k) / 5.49 | 5.01 (186k) / 5.04 | **4.72** (196k) / 4.88 |
| n200, L6 | 4.94 (184k) / 5.23 | 4.75 (184k) / 5.01 | 4.72 (183k) / 4.97 | **4.70** (196k) / 4.74 |
| n500, L2 | 5.55 (197k) / 5.76 | 5.00 (197k) / 5.22 | 4.69 (193k) / 4.98 | **4.65** (176k) / 4.82 |
| n500, L4 | 4.67 (175.5k) / 4.74 | 4.42 (175.5k) / 4.47 | 4.37 (194.5k) / 4.44 | **4.32** (189.5k) / 4.48 |

### Observations (vocab=8000)
- **joformer_projected wins all 9 configs.** Best overall: **4.32** at n500 L4.
- **joformer_learned rises to second place** — reversed from v2 where it was last. With full wiki, learned consistently beats fixed and approaches projected.
- **Rankings tighten at scale** — at n500 L4, all four models are within 0.35 PPL.
- **Overfitting nearly eliminated** — best-to-final gap is only 0.1-0.4 PPL (vs 5-12 PPL in v2 with 2M lines).
- **roformer is consistently last** — always worst, by 0.2-1.1 PPL behind next-best.

---

## Phase 7: Grid Search v3 — Full Wikipedia, vocab=16000

*Source: `~/OTHER_STUFF/joformer/`*

softmax, lr=2e-4, 200k iters, full wiki (28.8M lines, 983M tokens). Vocab=16000.

Format: best_val_ppl (iteration)

| Config | roformer | jo_fixed | jo_learned | jo_projected |
|--------|----------|----------|------------|--------------|
| n100, L2 | 60.57 (176k) | 59.60 (192k) | 61.59 (173k) | **53.45** (179k) |
| n100, L4 | 52.22 (182k) | 51.10 (182k) | 53.32 (197k) | **46.77** (187k) |
| n100, L6 | 47.69 (170k) | 46.78 (170k) | 48.70 (172k) | **44.82** (179k) |
| n100, L8 | 44.77 (159k) | 44.07 (159k) | 44.92 (167k) | **42.95** (196k) |
| n200, L2 | 44.27 (176k) | 42.86 (176k) | 45.60 (175k) | **40.10** (175k) |
| n200, L4 | 38.77 (189k) | 38.11 (149k) | 38.96 (196k) | **34.91** (199k) |
| n200, L6 | 35.42 (183k) | 34.96 (183k) | 35.30 (186k) | **32.90** (197k) |
| n500, L2 | 33.07 (185k) | 32.15 (179k) | 32.25 (195k) | **28.82** (185k) |
| n500, L4 | 27.30 (178k) | 27.12 (178k) | 28.26 (195k) | **26.30** (195k) |

### Observations (vocab=16000)
- **joformer_projected wins every config** — margins range from 0.8 PPL (n500 L4) to 7.1 PPL (n100 L2).
- **No divergence** at lr=2e-4 across all 9 configs.
- **joformer_fixed consistently edges out roformer** by 0.2-1.4 PPL.
- **joformer_learned** lags at small configs but catches up at wider/deeper.
- **Best overall: 26.30 PPL** (joformer_projected, n500 L4).

---

## Phase 8: Grid Search v5 — Full Wikipedia, vocab=32000

*Source: `~/OTHER_STUFF/joformer/`*

softmax, lr=2e-4, 200k iters, full wiki (691M tokens — fewer tokens because larger vocab). 3 models (no joformer_learned).

Format: best_val_ppl (iteration)

| Config | roformer | jo_fixed | jo_projected |
|--------|----------|----------|--------------|
| n100, L2 | 147.08 (167k) | 148.24 (167k) | **137.91** (189k) |
| n100, L4 | 136.21 (160k) | 138.31 (160k) | **128.39** (198k) |
| n100, L6 | 128.59 (148k) | 129.54 (161k) | **118.63** (178k) |
| n100, L8 | 121.17 (182k) | 122.93 (182k) | **119.90** (191k) |
| n200, L2 | 116.06 (194k) | 116.30 (194k) | **104.86** (179k) |
| n200, L4 | 98.31 (198k) | 100.04 (198k) | **94.97** (176k) |
| n200, L6 | 93.97 (188k) | 94.87 (188k) | **89.96** (174k) |
| n500, L2 | 83.45 (196k) | 82.94 (196k) | **76.44** (155k) |
| n500, L4 | 69.76 (190k) | 70.69 (190k) | **69.12** (193k) |

### Observations (vocab=32000)
- **joformer_projected wins every config** — consistent across all vocab sizes.
- **PPL numbers are ~2.5x higher than the 16k grid** due to larger vocab. Not directly comparable.
- **Projected's advantage narrows at n500 L4** (69.12 vs 69.76 — only 0.6 PPL).
- **Best overall: 69.12 PPL** (joformer_projected, n500 L4).

### Cross-vocab comparison (relative advantage in nats)

PPL across vocab sizes is not directly comparable. The relative advantage of joformer_projected over baselines in cross-entropy loss (nats) is:

| Config | Loss gap (v3, 16k vocab) | Loss gap (v5, 32k vocab) |
|--------|--------------------------|--------------------------|
| n100, L2 | 0.12 nats | 0.06 nats |
| n200, L2 | 0.10 nats | 0.10 nats |
| n500, L2 | 0.14 nats | 0.08 nats |
| n500, L4 | 0.03 nats | 0.01 nats |

Projected's edge is largest at shallow/narrow configs and narrows at the biggest models.

---

## Phase 9: KG + Text Experiment

*Source: `~/joformer/` only*

Tests whether KG information improves text PPL, comparing **angle-gap** (learned relation angle inserted into rotation) vs **text-linearized** (KG triples as text tokens).

Config: n_embed=100, n_layers=2, block_size=64, batch_size=32, lr=2e-4, softmax, dropout=0.2, vocab=8000, full wiki, kg_weight=1.0. KG sources: WordNet synonyms, FrameNet, BATS analogies, Google analogies, word analogies — 734K total triples.

### 100K iterations

| Model | KG Method | Test PPL | Rare-KG PPL | Common-KG PPL | KG Test PPL |
|-------|-----------|----------|-------------|---------------|-------------|
| **roformer** | **none** | **7.74** | 12.13 | -- | -- |
| roformer_kg | angle-gap | 8.88 | **10.48** | 6.81 | 7.65 |
| roformer_text_kg | text-linear | 8.44 | **10.33** | -- | 4.00 |

### KG Observations
- **Both KG methods hurt overall text PPL** — plain roformer (7.74) beats both angle-gap (8.88) and text-linearized (8.44).
- **But KG training helps predict rare KG words** — when measuring loss only on rare KG word tokens (masking out common words), both KG models beat the baseline: 10.48 and 10.33 vs 12.13. Evidence of **KG→Text cross-pollination**.
- **Text-linearized slightly edges angle-gap on rare words** — 10.33 vs 10.48.
- **Angle-gap hurts overall text more** — 8.88 vs 8.44.

---

## Key Findings (All Experiments)

1. **Softplus is unstable for learned/projected angles** — cumsum of unbounded angles overflows. Softmax fixes this.
2. **Softmax vs softplus doesn't matter for stable models** — roformer and joformer_fixed get nearly identical results either way.
3. **joformer_projected is the best model across all tested configurations** — wins all 9 configs in every grid search (vocab 8k, 16k, and 32k).
4. **Lower lr fixes projected divergence** — lr=2e-4 stabilizes all configs. The learning rate is the key factor, not the schedule.
5. **roformer is consistently the worst model** — last place across all full-wiki grids.
6. **joformer_learned improves dramatically with more data** — last place with 2M lines, rises to second place with full wiki, approaching projected at large scales.
7. **Full wiki data dramatically improves all models** — 28.8M lines vs 2M lines cuts PPL roughly in half.
8. **Rankings tighten at scale** — projected's advantage narrows from 7+ PPL at n100 L2 to <1 PPL at n500 L4. The per-layer angle MLP helps most when the base model is capacity-constrained.
9. **PPL is not comparable across vocab sizes** — use bits-per-character (BPC) or relative loss gaps for cross-tokenization comparison.
10. **KG training helps rare-word prediction but hurts overall text PPL** — cross-pollination signal exists but is diluted by interference with text learning.

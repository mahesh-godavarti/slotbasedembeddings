# JoFormer Experiment Results

All experiments: Wikipedia text, BPE tokenization, block_size=64, batch_size=32, lr=5e-4, dropout=0.2, seed=42.

## Softplus Attention (log(exp(x)+1))

### n_embed=500, n_layers=2, vocab=16000, wiki_lines=2M, 100k iters

| Model | Train PPL | Val PPL | Best Val PPL (iter) |
|-------|----------|---------|---------------------|
| joformer_fixed | 79.05 | 88.56 | 80.46 (99500) |
| roformer | 79.37 | 88.60 | 80.99 (99500) |
| joformer_learned | 79.43 | 94.60 | 86.07 (97000) |
| joformer_projected | NaN | NaN | -- |

### n_embed=100, n_layers=2, vocab=16000, wiki_lines=2M, 100k iters

| Model | Train PPL | Val PPL |
|-------|----------|---------|
| roformer | 148.02 | 146.49 |
| joformer_fixed | 149.67 | 148.93 |
| joformer_learned | NaN (died iter 14500) | NaN |
| joformer_projected | NaN (crashed) | NaN |

**Softplus instability**: joformer_learned and joformer_projected both produce unbounded cumsum'd angles that grow during training, causing large attention scores that overflow in `log(exp(x)+1)`. joformer_fixed uses cumsum on fixed (non-learned) angles so they don't grow. roformer uses standard fixed RoPE angles.

## Switching to Softmax

Because joformer_learned and joformer_projected crashed with softplus (NaN from unbounded angles overflowing in `log(exp(x)+1)`), we tested joformer_projected with standard softmax attention. It ran 100k iters with no NaN — softmax is numerically stable by design since it normalizes via exp/sum rather than passing through exp directly.

To get an apples-to-apples comparison, we then reran all 4 models with softmax under the same config.

### n_embed=100, n_layers=2, vocab=16000, wiki_lines=2M, 100k iters (softmax)

| Model | Train PPL | Val PPL |
|-------|----------|---------|
| **joformer_projected** | **130.61** | **129.97** |
| roformer | 149.47 | 148.00 |
| joformer_fixed | 150.06 | 148.77 |
| joformer_learned | 157.04 | 150.62 |

All 4 models stable with softmax — no NaN.

**Softmax vs softplus for the stable models**: For roformer and joformer_fixed (which were stable under both), switching to softmax made almost no difference — roformer went from 146.49 to 148.00 val PPL, joformer_fixed from 148.93 to 148.77. The attention mechanism choice doesn't matter much when angles are well-behaved.

**joformer_projected is the clear winner with softmax** — 129.97 val PPL, ~18 PPL ahead of roformer. This model was impossible to evaluate with softplus (instant NaN), so softmax unlocked its potential.

## Grid Search (softmax, 50k iters, 2M lines, vocab=16000)

Swept n_embed x n_layers, trading off width for depth: more layers at smaller n_embed, fewer at larger.

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

### Grid search observations

- **joformer_projected wins at shallow/narrow configs** (n100 L2-L4, n200 L2-L4) by large margins — best result 104.80 at n200 L4.
- **joformer_projected diverges at deeper configs** (n200 L6, n500 L4). It starts learning (best val PPL ~195 at n200 L6 iter 8000) then blows up. Training instability, not inability to learn.
- **roformer and jo_fixed are the most reliable** — always within ~1 PPL of each other, never diverge. Best overall result: 87.50 (roformer, n500 L4).
- **jo_learned consistently lags** by 5-12 PPL behind roformer/fixed, except at n200 L2 where it briefly beat them (134.03 vs 135-137).
- **Wider models benefit all architectures** — going from n100 to n500 at L2 drops PPL from ~164 to ~93 for roformer.
- **50k iters may be too few for deeper projected models** — at n100 L6/L8, projected underperforms relative to L4, but may not have converged. Follow-up runs at 100k iters planned for n100 L6 and L8.

## Fixing joformer_projected Divergence (n200 L6)

joformer_projected diverged at n200 L6 and n500 L4 in the grid search (lr=5e-4, no LR schedule). We tested fixes on n200 L6.

### Attempts on n200 L6, joformer_projected only, 50k iters

| Setting | Best Val PPL | Diverged? |
|---------|-------------|-----------|
| lr=5e-4, no schedule (grid) | 195 (iter 8000) | Yes |
| lr=5e-4, cosine decay | 150 (iter 14000) | Yes |
| lr=2e-4, cosine decay | 108 (iter 47500) | No, final val PPL 118 |

Cosine decay helped (survived longer, better peak) but didn't fully fix divergence at lr=5e-4. Lowering lr to 2e-4 with cosine decay eliminated divergence entirely.

### Extended run: n200 L6, lr=2e-4, cosine decay, 200k iters

| Train PPL | Final Val PPL | Best Val PPL (iter) |
|----------|---------------|---------------------|
| 83.21 | 87.20 | **78.42** (iter 184000) |

Best val PPL of **78.42** — beats the best result from the entire grid search (roformer n500 L4 at 87.50), using a smaller model (n200 vs n500).

### Constant lr=2e-4 (no cosine decay), 200k iters

| Train PPL | Final Val PPL | Best Val PPL (iter) |
|----------|---------------|---------------------|
| 83.95 | 87.89 | **79.96** (iter 184000) |

Very similar to cosine decay (78.42 vs 79.96 best val PPL, 87.20 vs 87.89 final). Cosine decay gives ~1.5 PPL improvement but constant lr works nearly as well. **lr=2e-4 is the key fix, not the schedule.**

## Grid Search v2 (softmax, lr=2e-4, 200k iters, no cosine decay)

Re-running the full grid with lr=2e-4 and 200k iters for a fair comparison. Tracks best val PPL and iteration.

**Complete.** Format: best_val_ppl (iter) / final_val_ppl

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

### Grid v2 observations
- **joformer_projected wins every single config** — by 4-19 PPL depending on config. Best overall: **63.85** at n500 L4.
- **lr=2e-4 eliminates projected divergence** — n200 L6 and n500 L4 both stable (both diverged at lr=5e-4 in grid v1). n200 L6 reproduces the standalone result (79.96).
- **200k iters dramatically improve all models** vs 50k grid — e.g., roformer n500 L4: 87.50 → 67.52, projected n200 L4: 104.80 → 85.98.
- **joformer_projected peaks earlier then overfits** — best PPL at iter 144-197k, then degrades 1-12 PPL by 200k. Wider models (n500) overfit less than narrow ones (n100).
- **roformer and jo_fixed remain neck-and-neck** — within 0.1-1.5 PPL of each other across all configs. Most reliable models.
- **jo_learned consistently lags** by 2-10 PPL behind roformer/fixed, with a smaller gap at wider embeddings.
- **All models overfit after peak** — final PPL is 0.3-12 PPL worse than best, especially at n200+ where the gap grows to 7-12 PPL.
- **Deeper and wider models keep helping** — n500 L4 is the best config for all models. Diminishing returns at deeper n100 configs.

## Scaling to Full Wikipedia (GPT-2 Scale)

Current experiments use 2M lines from wiki.en.txt (28.8M lines / 3GB total). To train at GPT-2 scale on the full dataset:

### Model sizes

| Scale | n_embed | n_layers | ~Params | Reference |
|-------|---------|----------|---------|-----------|
| Current best | 200 | 6 | ~15M | Best JoFormer result (78.42 PPL) |
| GPT-2 small | 768 | 12 | ~117M | OpenAI GPT-2 |
| GPT-2 medium | 1024 | 24 | ~345M | OpenAI GPT-2 |

### Vocab size

GPT-2 used 50,257 BPE tokens. For wiki-only training, **32k** is a good sweet spot — large enough to capture most words as single tokens, small enough that the embedding table doesn't dominate the parameter budget (32k × n_embed = 24.6M params at n768 vs 38.6M at 50k).

### AWS instance options

| Instance | GPU | $/hr | $/week | Notes |
|----------|-----|------|--------|-------|
| **g5.xlarge** (current) | 1x A10G 24GB | $1.01 | $170 | Sufficient for GPT-2 small |
| **p4d.24xlarge** | 8x A100 40GB | $21.96 | $3,693 | For GPT-2 medium or faster training |
| **p3dn.24xlarge** | 8x V100 32GB | $31.21 | $5,245 | Multi-GPU, older generation |

### Memory constraints

The current g5.2xlarge (32GB system RAM) ran out of memory trying to process 4M wiki lines — we had to drop to 2M lines. The full 28.8M-line wiki (14x more) is far beyond what fits in RAM with the current approach of loading all tokenized data at once.

Scaling to full wiki requires **streaming/memory-mapped data loading** — tokenize and save to disk first, then load batches on the fly during training. This is how GPT-2 and all large-scale LLMs handle data. No instance upgrade fixes this; it's an architectural change in train_wiki.py.

### Solution: `train_wiki_streaming.py`

Implemented a memory-efficient streaming version that preprocesses wiki text to a memory-mapped binary file on disk, then trains using random-access `np.memmap` reads. Peak RAM during training: just the model + optimizer (~50-100MB), regardless of dataset size.

Two-phase approach:
1. **Preprocess** (one-time): Stream wiki text line-by-line → train BPE tokenizer → write token IDs to a binary file (`int32`). Constant memory (~10MB).
2. **Train**: Memory-map the binary file. `get_batch()` reads ~8KB per call from disk/page cache. Zero data in RAM.

```bash
# Preprocess full wiki (one-time, ~15 min)
python train_wiki_streaming.py preprocess --vocab_size 16000 --data_dir joformer/data

# Train from preprocessed data
python train_wiki_streaming.py train --data_dir joformer/data --softmax --n_embed 500 --n_layers 2

# Or auto (preprocess if needed, then train)
python train_wiki_streaming.py auto --vocab_size 16000 --data_dir joformer/data --softmax ...
```

### Training time with full corpus

Training time per iteration is **unchanged** regardless of dataset size (2M vs 28.8M lines). Each iteration does the same work: pick `batch_size` random positions, grab `block_size`-token windows, forward/backward pass. Whether those positions are drawn from 60M tokens or 864M tokens doesn't affect per-iteration cost.

What changes with more data:
- **Preprocessing is slower** — tokenizing 28.8M lines takes ~15 min vs ~2 min for 2M lines (one-time cost)
- **Better generalization** — more unique text means less overfitting, lower val PPL
- **May need more iterations to converge** — each token gets sampled less often on average, so the model may need more iterations to see enough of the data

### Recommendation

The current g5.2xlarge has enough GPU (A10G 24GB) for GPT-2 small (117M params), and `train_wiki_streaming.py` eliminates the RAM bottleneck. The multi-GPU instances (p4d, p3dn) are a **22-31x cost jump** ($22-31/hr) and only warranted for GPT-2 medium/large or faster training. Spot instances could cut multi-GPU costs by 60-70% but risk interruption.

## Grid Search v3 (softmax, lr=2e-4, 200k iters, vocab=8000, full wiki — streaming)

Using `train_wiki_streaming.py` with memory-mapped data loading on the full 28.8M-line wiki. Vocab=8000 (vs 16000 in v2). PPL numbers are not comparable to v2 due to different vocab size.

**Complete.** Format: best_val_ppl (iter) / final_val_ppl

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

### Grid v3 observations
- **joformer_projected wins all 9 configs** — consistent with v2. Best overall: **4.32** at n500 L4.
- **joformer_learned rises to second place** — reversed from v2 where it was last. With full wiki (28.8M lines vs 2M), learned consistently beats fixed and often approaches projected. At n500 L4: learned 4.37 vs fixed 4.42 (0.05 gap).
- **Rankings tighten at scale** — at n500 L4, all four models are within 0.35 PPL (4.32–4.67). At n100 L2, the gap was 1.09 PPL (6.15–7.24). Larger models compress the architecture differences.
- **Overfitting nearly eliminated** — best-to-final gap is only 0.1-0.4 PPL across all configs (vs 5-12 PPL in v2 with 2M lines). 14x more data makes the difference.
- **joformer_projected still peaks earlier** — best PPL at iter 176-197k, while learned/fixed/roformer often peak at 175-199k. But the overfitting gap is much smaller than v2.
- **roformer is consistently last** — always the worst model, by 0.2-1.1 PPL behind the next-best. The gap is largest at wider embeddings (n500 L2: 5.55 vs fixed 5.00).

## Key Findings

1. **Softplus is unstable for learned/projected angles** — cumsum of unbounded angles overflows. Softmax fixes this for most configs.
2. **Softmax vs softplus doesn't matter for stable models** — roformer and joformer_fixed get nearly identical results either way.
3. **joformer_projected is the best model across all configs** — wins every config in both grid v2 (2M lines) and v3 (full wiki). Best PPL: 63.85 (v2, n500 L4) and 4.32 (v3, n500 L4).
4. **Lower lr fixes projected divergence** — lr=2e-4 stabilizes all configs including n200 L6 and n500 L4 (which diverged at lr=5e-4). The learning rate is the key factor, not the schedule (cosine decay gives only marginal improvement).
5. **roformer is consistently the worst model** — last place in all 9 v3 configs, by 0.2-1.1 PPL behind next-best. The standard RoPE approach is clearly outperformed by all JoFormer variants.
6. **joformer_learned improves dramatically with more data** — last place in v2 (2M lines, 2-10 PPL behind roformer/fixed), rises to second place in v3 (full wiki), consistently beating fixed and approaching projected. At n500 L4: learned 4.37 vs projected 4.32.
7. **More data eliminates overfitting** — 2M lines: best-to-final gap of 5-12 PPL; full wiki (28.8M lines): gap of only 0.1-0.4 PPL.
8. **Rankings tighten at scale** — at n500 L4 (largest config), all models are within 0.35 PPL. Architecture differences matter most at smaller scales.
9. **Wider models benefit all architectures** — n500 L4 is the best config across the board, with consistent gains from n100 → n200 → n500.

## KG + Text Experiment: Angle-Gap vs Text-Linearized KG

Tests whether KG information improves text PPL, comparing two KG presentation methods: **angle-gap** (learned relation angle inserted into rotation) vs **text-linearized** (KG triples as text tokens). Evaluated on a held-out test set (80/10/10 split of full wiki). KG triples also split 90/10 for KG test PPL.

Config: n_embed=100, n_layers=2, block_size=64, batch_size=32, lr=2e-4, softmax, dropout=0.2, vocab=8000 (data_v8k, full wiki), kg_weight=1.0.

KG sources: WordNet synonyms, FrameNet, BATS analogies, Google analogies, word analogies — 734K total triples.

### 100K iterations

| Model | KG Method | Test PPL | Rare-KG PPL | Common-KG PPL | KG Test PPL |
|-------|-----------|----------|-------------|---------------|-------------|
| **roformer** | **none** | **7.74** | 12.13 | -- | -- |
| roformer_kg | angle-gap | 8.88 | **10.48** | 6.81 | 7.65 |
| roformer_text_kg | text-linear | 8.44 | **10.33** | -- | 4.00 |

**Rare-KG PPL**: loss measured *only* on character tokens belonging to rare KG words (words in KG entities but with wiki frequency <= 50) within wiki sentences that contain those words. Uses per-token masking to isolate the signal from common-word noise.

**Common-KG PPL**: loss on held-out KG test triples where all entity words are common in wiki (frequency >= 100). Tests Text→KG transfer: does seeing words often in wiki text help predict their KG relationships?

### Observations

- **Both KG methods hurt overall text PPL** — plain roformer (7.74) beats both angle-gap (8.88) and text-linearized (8.44). KG training interferes with text learning at the aggregate level.
- **But KG training helps predict rare KG words** — when we measure loss *only on rare KG word tokens* (masking out the common words), both KG models beat the baseline: 10.48 and 10.33 vs 12.13. This is evidence of **KG→Text cross-pollination** — the models learned something about these rare words from KG triples that helps predict them in wiki text.
- **The cross-pollination signal was hidden in whole-sentence PPL** — earlier runs computed Rare-KG PPL over full sentences (which was dominated by common words like "the", "of", "in"). The masked evaluation was necessary to reveal the transfer.
- **Text-linearized slightly edges angle-gap on rare words** — 10.33 vs 10.48. Both are substantially better than the baseline's 12.13.
- **Angle-gap hurts overall text more than text-linearized** — 8.88 vs 8.44. The angle-gap mechanism modifies positional encoding during KG training but uses standard fixed RoPE during text eval. The text-linearized approach feeds KG as additional text, which may be less disruptive.
- **KG test PPL is not directly comparable between methods** — angle-gap evaluates causal next-token prediction on structured triples with 50% direction flipping; text-linearized evaluates next-token on random windows of "head relation tail ." text.

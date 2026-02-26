# Exp 7: Experiment Log

## Setup
- **Data**: Synthetic family tree (~30 males, 3-4 generations) + geography (~10 cities, ~5 countries)
- **Tokenization**: Character-level, vocab_size=62
- **Training**: Per-sentence batching, batch_size=32, lr=5e-4
- **Config**: n_embed=24, n_layers=1

## Debugging Phase

### BOS/EOS Tokens Cause Catastrophic Failures

With BOS/EOS tokens, rotation-based models exhibited catastrophic perplexity:
- **Per-sentence batching**: B' catastrophic (ppl=187K), B fine (ppl=143)
- **Concatenated batching**: Pattern REVERSES — B catastrophic (ppl=49K), B' fine (ppl=133)
- C/C' also showed swapped dominance between batching styles

**Removing BOS/EOS fixed ALL models** to reasonable perplexity (10-60 range).

Mechanism unknown. Open research question.

### Pad Mask: No Effect (for text with causal mask)

Tested pad_mask ON vs OFF (after removing BOS/EOS). Results **identical** for text-only models (B/B'/C/C'). Causal mask already prevents attending to later PAD positions, and loss uses `ignore_index=-100`.

Note: pad_mask IS important for KG training which uses bidirectional attention — without it, tokens would attend freely to PAD positions. This is correctly implemented in all KG forward methods.

### Shakespeare Sanity Check (Validated)

Models B, B', C, C' on tiny Shakespeare (n_embed=90, 1 layer, 10K iters):
- B: val ppl=7.18, B': val ppl=6.64, C: val ppl=7.14, C': val ppl=6.56
- Ordering: C' > B' > C > B (correct). Matches joformer paper results.

## Results

### B/B'/C/C' Only (Shakespeare-validated model code, 10K iters)

n_embed=24, n_layers=1, block_size=35, per-sentence batching, no BOS/EOS.

| Tier | Model | hit@1 | hit@5 | ppl | full_acc |
|------|-------|-------|-------|-----|----------|
| **Memorization** | B | 0.333 | 0.667 | 20.93 | 0.333 |
| | B' | 0.278 | 0.667 | 12.84 | 0.278 |
| | C | 0.306 | 0.694 | 12.92 | 0.306 |
| | C' | 0.306 | 0.722 | **10.42** | 0.306 |
| **Transfer** | B | 0.417 | 0.833 | 9.80 | 0.333 |
| | B' | 0.333 | 0.833 | 10.61 | 0.333 |
| | C | 0.250 | 0.667 | 12.90 | 0.250 |
| | C' | 0.250 | 0.833 | **9.03** | 0.250 |
| **Generalization** | B | 0.167 | 0.417 | 60.80 | 0.167 |
| | B' | 0.167 | 0.583 | **22.89** | 0.167 |
| | C | 0.167 | 0.500 | 33.83 | 0.167 |
| | C' | 0.167 | 0.583 | 35.15 | 0.167 |

### All 10 Models (kg_text_experiment.py, 5K iters, exp 7a, seed 0)

n_embed=24, n_layers=1, block_size=48, per-sentence batching, no BOS/EOS.
Text-only models (B/B'/C/C'): causal next-token prediction on linearized text.
KG models (A/A'/D/D'/E/E'): mixed text (causal) + KG (bidirectional MLM).

| Tier | Metric | A | A' | B | B' | C | **C'** | D | D' | E | E' |
|------|--------|---|----|----|----|----|--------|---|----|----|-----|
| **Mem** | hit@1 | .222 | .194 | .139 | .250 | .333 | **.361** | .111 | .167 | .194 | .250 |
| | hit@5 | .500 | .444 | .528 | .556 | .639 | **.722** | .500 | .472 | .528 | .528 |
| | ppl | 27.80 | 27.29 | 18.91 | 17.42 | 14.69 | **12.01** | 40.86 | 50.19 | 55.83 | 33.67 |
| **Trans** | hit@1 | .167 | .167 | .250 | .167 | .250 | .250 | .167 | .083 | .167 | .167 |
| | hit@5 | .333 | .500 | .500 | .667 | .750 | **.833** | .417 | .583 | .417 | .500 |
| | ppl | 71.51 | 37.71 | 14.49 | 17.47 | **11.44** | 13.31 | 38.89 | 28.45 | 102.02 | 51.76 |
| **Gen** | hit@1 | .167 | .167 | .083 | .083 | .250 | **.333** | .167 | .167 | .250 | .167 |
| | hit@5 | .333 | .417 | .500 | .417 | .500 | **.583** | .333 | .333 | .417 | .583 |
| | ppl | 207.10 | 76.92 | 34.34 | 68.95 | **28.65** | 67.75 | 63.72 | 63.41 | 96.80 | 55.06 |

## Current Understanding

### What works
1. **C/C' (journey cumsum, linearized text) dominate** across all tiers
   - C' best on memorization (hit@1=0.361, ppl=12.01) and transfer hit@5 (0.833)
   - C best on generalization ppl (28.65) — unprimed generalizes better than primed
2. **Primed (operator-based) helps memorization** but **hurts generalization** for most models
3. **Linearized models (B/B', C/C') outperform native KG models (A/A', D/D', E/E')** across the board

### The problem: Native KG models underperform

| Model Group | Mem ppl | Trans ppl | Gen ppl |
|-------------|---------|-----------|---------|
| C/C' (linearized cumsum) | 12-15 | 11-13 | 29-68 |
| B/B' (linearized RoPE) | 17-19 | 14-17 | 34-69 |
| A/A' (RoPE+slots) | 27-28 | 38-72 | 77-207 |
| D/D' (flat KG, rel as token) | 41-50 | 28-39 | 63-64 |
| E/E' (native KG, rel as operator) | 34-56 | 52-102 | 55-97 |

**Why are native KG models worse?** Possible reasons to investigate:
1. **Training split**: KG models split iterations between text and KG batches (alternating). They see ~50% fewer text training steps than linearized models. Linearized models see ALL KG facts as text + more text overall.
2. **KG training objective mismatch**: KG uses bidirectional MLM (predict masked tokens). Text evaluation uses causal completion. The KG training signal may not transfer well to the causal evaluation setting.
3. **Data imbalance**: Only 91 KG triples vs thousands of text tokens. KG batches may overtrain on a small set.
4. **Relation angle learning**: E's relation-as-operator and A's slot angles may need more data or different initialization to learn meaningful representations.
5. **Architecture mismatch**: The same transformer weights handle both causal text and bidirectional KG. These may be conflicting objectives.

### Next steps
- Investigate why native KG models underperform
- Consider giving KG models more text training iterations (compensate for mixed training)
- Consider whether bidirectional MLM is the right KG objective for a causal evaluation

## Saved Log Files
- `shakespeare_test.log` — Shakespeare sanity check
- `kg_shakespeare_test_n90.log` — n_embed=90, with BOS/EOS, per-sentence
- `kg_shakespeare_test_n24_no_padmask.log` — n_embed=24, with BOS/EOS, no pad_mask
- `kg_shakespeare_test_n24_padmask.log` — n_embed=24, with BOS/EOS, pad_mask ON
- `kg_shakespeare_test_batching_comparison.log` — per-sentence vs concatenated, with BOS/EOS
- `kg_shakespeare_test_no_bos_eos_padmask_ON_FINAL.log` — no BOS/EOS, pad_mask ON
- `kg_shakespeare_test_no_bos_eos_padmask_OFF_FINAL.log` — no BOS/EOS, pad_mask OFF
- `exp7_all_10_models_7a_seed0.log` — All 10 models, exp 7a, seed 0, 5K iters

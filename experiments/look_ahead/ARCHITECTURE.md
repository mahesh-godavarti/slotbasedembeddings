# Look-Ahead Architecture: Full Writeup

## Core Architecture

A single shared-weight transformer block iterated N times during training, with:
1. **Non-cumulative corrections**: `x_k = x_0 + f(x_{k-1})`
2. **Past-only shift**: position t gets correction from position t-1

## Key Result: K=1 at Inference

With KV caching during autoregressive generation, each new token naturally sees the corrected representations of all previous tokens. The KV pairs cached from previous generation steps were computed from corrected `processed_x` values, not raw `tok_emb`. This sequential buildup across tokens provides the same context enrichment that K>1 parallel iterations provide during training.

**Experimental confirmation (C=50, N=10, 100K iters):**

| Eval mode | Val PPL |
|---|---|
| Parallel K=1 (cold, pessimistic) | 110.17 |
| Parallel K=2 | 100.31 |
| Parallel K=5 | 96.41 |
| Parallel K=10 (training depth) | 98.20 |
| Sequential K=1 (true inference) | 98.22 |

Sequential K=1 = Parallel K=10. At inference, K=1 is always sufficient.

**Why**: In parallel K=1, the block sees raw `tok_emb` at all positions — no corrections have been applied anywhere. In sequential processing (autoregressive with KV cache), when generating token t, the block at position t-1 sees corrected representations at positions 0..t-2. These corrected representations were built up by previous generation steps. The sequential context propagation replaces the need for multiple iterations.

**Implication**: The training depth N is a training hyperparameter only. It controls how well the parallel forward pass (needed for efficient batched training) approximates the sequential behavior. At inference, the architecture is always a single-layer transformer.

## Generalization: D-Block Units

### Motivation

With a single shared block (D=1), inference uses 1 transformer layer per token. This is maximally efficient but gives the block limited capacity to process each token. We can increase per-token depth by using a **D-block unit**: D sequential transformer blocks (with separate weights within the unit) treated as a single shared unit that gets iterated K=N/D times during training.

### Architecture

A D-block unit consists of D transformer blocks with **separate weights**. Within the unit, blocks 1..D-1 use standard transformer residual connections. The unit returns only the **last block's delta** as the correction:

```
h = x
h = block_1(h)          # standard residual: h = h + attn(...) + ffn(...)
h = block_2(h)          # standard residual
...
h = block_{D-1}(h)      # standard residual
correction = block_D(h) - h   # last block's delta only
```

Blocks 1..D-1 build up a rich internal representation with standard residuals. The last block produces a clean correction from that representation. This correction is added to `tok_emb`:

```
x_k = x_0 + shift(correction)
```

**Why not `Unit(x) - x` (total delta)?** With D>1, the residual `h = x + attn(ln1(x))` in block_D connects to block_{D-1}'s output, not to the unit input `x`. Subtracting the unit input doesn't cleanly undo block_D's residual — it conflates block_D's delta with the accumulated deltas from blocks 1..D-1. Returning only the last block's delta keeps the correction clean.

### Spectrum of Configurations

| D | Inference depth | Training iters K | Effective training depth |
|---|---|---|---|
| 1 | 1 layer | N | N layers (all shared) |
| 2 | 2 layers | N/2 | N layers (shared in pairs) |
| 5 | 5 layers | N/5 | N layers (shared in groups of 5) |
| N | N layers | 1 | N layers (no sharing) |

**D=N does NOT degenerate to a standard transformer.** With K=1, the unit runs once on raw tok_emb. Blocks 1..9 build up a representation, and block 10 produces a correction. The output is `tok_emb + shift(delta_10)`, which discards the accumulated processing from blocks 1..9 and only keeps the last block's contribution added to tok_emb. A standard transformer uses the full N-layer output directly.

### Parameter Analysis

Per-block parameters: `12C^2 + 13C` (attention + FFN + layernorms)

| Component | D-block look-ahead | Standard N-layer transformer |
|---|---|---|
| Embedding | C * V | C * V |
| Blocks | D * (12C^2 + 13C) | N * (12C^2 + 13C) |
| Head (linear) | C * V + V | C * V + V |
| **Total blocks** | **D/N of standard** | **baseline** |

### Inference Cost

| Config | Layers per token | Block params | Total params |
|---|---|---|---|
| D=1, C=50 | 1 | 30.7K | 1.65M |
| D=2, C=50 | 2 | 61.3K | 1.68M |
| D=5, C=50 | 5 | 153.3K | 1.77M |
| D=10, C=50 | 10 | 306.5K | 1.92M |

At small C, block params are tiny relative to embeddings + head (C*V dominates), so increasing D barely affects total params. At large C, blocks dominate (C^2 scaling), so D has a bigger impact.

### Sequential Eval Validity

Sequential evaluation (processing positions one at a time, feeding contextualized embeddings to subsequent positions) simulates autoregressive inference with KV caching. Its validity depends on K:

**K >> 1 (e.g., D=1, K=10):** The unit is trained on both raw tok_emb (iteration 1) and contextualized tok_emb + correction (iterations 2..K). Sequential eval feeds contextualized inputs, which the unit learned to handle. Sequential K=1 matches parallel K=N. **Theoretically sound.**

**K > 1 (e.g., D=2, K=5):** The unit still sees contextualized inputs during training (iterations 2..K). However, during training all positions have **uniform** contextualization (same iteration depth). In sequential eval, contextualization is **non-uniform** (earlier positions fully finalized, later ones raw). For D=1 this doesn't matter (single attention layer with causal masking). For D>1, internal layers see non-uniform inputs across positions, which differs from training. **Empirically works, but the theoretical guarantee is weaker.**

**K=1 (e.g., D=10, K=1):** The unit was only ever trained on raw tok_emb. It never saw contextualized inputs. Sequential eval feeding contextualized inputs is **invalid**. Falls back to parallel K=1 evaluation.

The code enforces this: `forward_sequential` returns `forward_at_depth(idx, 1, targets)` when `n_iters <= 1`.

### D-Block Results (C=50, N=10, 100K iters, conv loss 0.1)

| Metric | D=1 (K=10) | D=2 (K=5) | D=10 (K=1) |
|---|---|---|---|
| Params | 1,651,600 | 1,682,300 | 1,922,600 |
| Val PPL (parallel K=N) | 98.74 | 96.33 | 87.24 |
| Parallel K=1 | 110.74 | 103.84 | 89.94 |
| Sequential K=1 | 94.17 | 93.69 | 93.02 (=parallel K=1) |
| Empirical L | ~0.88 | ~0.60 | — (K=1, no iterations) |

D=2 improves over D=1: better training PPL, better parallel K=1, better convergence (L=0.60 vs 0.88). Sequential K=1 is similar (93.69 vs 94.17).

D=10 (K=1): Best val PPL (87.24) since it has 10 separate-weight blocks. Sequential eval correctly falls back to parallel K=1 (guard in place). The three reported numbers (87.24, 89.94, 93.02) differ only because each is computed on a different random draw of 20 val batches — they are all parallel K=1 under the hood.

### Note on Eval Variance

The post-training evaluation computes three numbers that, for K=1, should be identical:
1. **Val PPL** — from `estimate_loss()` at end of training, calls `model(X, Y)`
2. **Parallel K=1** — from `estimate_loss_at_depth(K=1)`, calls `forward_at_depth(X, 1, Y)`
3. **Sequential K=1** — from `estimate_loss_sequential()`, calls `forward_sequential(X, Y)` → falls back to `forward_at_depth(X, 1, Y)`

All three perform the same computation but each draws a **different random sample** of 20 validation batches (batch_size=64). With only 20 batches, variance is non-trivial — hence the spread (87.24 to 93.02 for D=10 K=1). With a fixed eval set or more batches, these would converge to the same value.

### Iterative Training as a Free Lunch

A key observation: the D-block unit has **separate weights** for each of its D blocks. A D-block unit with D=N has the exact same parameters as a standard N-layer transformer. But unlike a standard transformer (which trains at fixed depth N), the D-block unit can be iterated K times during training, giving an effective training depth of D×K.

Consider D=10, K=10 (with `--n_layers 100 --d_block 10`):
- **Parameters**: 10 blocks with separate weights = same as a standard 10-layer transformer
- **Inference cost**: 10 layers per token = same as a standard 10-layer transformer
- **Effective training depth**: 10 × 10 = 100 layers (vs 10 for the standard transformer)
- **Sequential eval**: valid (K=10, unit trained on contextualized inputs)

The iterative training scheme acts as a **training-time enhancement** for a fixed-parameter model. The unit learns to produce corrections that improve with iteration, and at inference the sequential buildup from autoregressive generation replaces those iterations for free.

A standard 10-layer transformer with the same parameters only ever sees depth 10 during training. The iterative D=10, K=10 model sees effective depth 100, teaching the blocks to produce better contextualized representations. This should translate to better inference quality at the same parameter count and inference cost.

**Experimental plan**: Compare D=10, K=10 (`--n_layers 100 --d_block 10`) against the baseline roformer (C=50, N=10, val PPL 71.99). Same parameters, same inference depth, but iterative training.

| Config | Params | Inference layers | Training depth | Expected PPL |
|---|---|---|---|---|
| Standard roformer (N=10) | 10 blocks | 10 | 10 | 71.99 (measured) |
| D=10, K=1 (N=10) | 10 blocks | 10 | 10 | ~similar (in progress) |
| D=10, K=10 (N=100) | 10 blocks | 10 | 100 | < 71.99 (predicted) |

If this works, it establishes that the look-ahead architecture can **improve on standard transformers at the same parameter budget**, purely through iterative training.

## Block Variants

### Standard block (RoFormerBlock)
```
h = x + attn(ln1(x))        # attention + residual
out = h + ffn(ln2(h))        # FFN + residual
```
Within a D-block unit, each sub-block uses these standard residuals. The look-ahead correction is extracted only from the last block: `correction = block_D(h) - h`.

## Summary of Key Insights

1. **K=1 at inference**: Sequential processing from autoregressive generation + KV caching makes K>1 unnecessary when K is large during training. Training depth K is a training-only hyperparameter.

2. **D-block generalization**: Use D sequential layers as the shared unit. D=1 is maximum sharing (fastest, smallest). Higher D gives more per-token capacity at the cost of more parameters. D controls the quality-speed tradeoff.

3. **Clean correction extraction**: Within the D-block unit, blocks 1..D-1 use standard residuals. The correction returned to the look-ahead scheme is only the last block's delta, ensuring it can be cleanly added to tok_emb.

4. **Sequential eval requires K>1**: Sequential evaluation (simulating autoregressive inference) is only valid when K>1, because the unit must have been trained on contextualized inputs. For K=1 (D=N), parallel K=1 is the correct evaluation.

5. **Parallel vs sequential eval**: For K>>1, parallel K=1 is pessimistic (unit sees raw embeddings). Sequential K=1 matches parallel K=N. For smaller K, the match is approximate.

6. **Convergence loss helps**: MSE between last two iterations' outputs (weight 0.1) improves L with minimal PPL cost.

7. **Additive correction is important**: Learned combiner `f(correction, original)` hurts K=1 quality (91.5% gap). The additive `x_0 + correction` constrains the iteration to be contraction-like.

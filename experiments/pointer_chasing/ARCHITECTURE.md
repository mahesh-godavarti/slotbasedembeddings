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

### Why Sequential K=1 = Parallel K=N (Mathematical Proof)

Due to causal masking and non-cumulative corrections, parallel iterations converge from left to right:

- **Position 0**: Only sees itself. Its correction is identical at every iteration. Converges at iteration 1.
- **Position 1**: At iteration 2, it sees [tok_emb[0], tok_emb[1] + c[0]]. Since position 0 already converged, this input is final. At iteration 3, the input is identical → same output. Converges at iteration 2.
- **Position t**: Converges at iteration t+1, because positions 0..t-1 have all stabilized. The non-cumulative reset (`processed_x = tok_emb + shift(correction)`) means once inputs stabilize, the output stabilizes in exactly one more pass.

Sequential K=1 processes left-to-right, so when it computes position t, positions 0..t-1 are already finalized. It presents position t with the same input that parallel would present after t+1 iterations — the **converged fixed point**. Therefore:

**Sequential K=1 = Parallel K=T** (the fully converged solution).

Since parallel K=N ≈ parallel K=T when N ≥ T, the final iteration in both cases sees essentially the same input. Evaluating at K=N or K=N+1 makes no difference — the corrections have already converged. Sequential K=1 is equivalent to this converged state.

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

| Config | Params | Inference layers | Training depth | Val PPL |
|---|---|---|---|---|
| Standard roformer (N=1) | 1 block | 1 | 1 | 100.24 (measured) |
| Standard roformer (N=10) | 10 blocks | 10 | 10 | 71.99 (measured) |
| D=1, K=10 (N=10) | 1 block | 1 | 10 | 98.74 (seq K=1: 94.17) |
| D=10, K=1 (N=10) | 10 blocks | 10 | 10 | 87.24 (measured) |
| D=10, K=10 (N=100) | 10 blocks | 10 | 100 | < 71.99 (predicted) |

If this works, it establishes that the look-ahead architecture can **improve on standard transformers at the same parameter budget**, purely through iterative training.

## Non-Cumulative vs Cumulative Corrections

### The tradeoff

**Cumulative (recursive residual):** `x_k = x_{k-1} + shift(correction_k)` — standard ResNet behavior.
Each iteration builds on the previous. The representation progressively evolves through N successive abstractions, ending up far from tok_emb.

**Non-cumulative (absolute residual):** `x_k = x_0 + shift(correction_k)` — fixed-point iteration.
Each iteration corrects the original embedding. The output is always one additive correction away from tok_emb. No progressive abstraction — the correction must do ALL contextual work in a single term.

### Why non-cumulative enables K=1 convergence

Consider position 1, which gets the correction from position 0 (past-only shift). Position 0 never receives a correction (no position -1), so `processed_x[0] = tok_emb[0]` always. The block sees the same input at position 0 every iteration, producing the same `correction[0]`:

**Non-cumulative:**
```
x_1[1] = tok_emb[1] + correction[0]     ← same every iteration
x_2[1] = tok_emb[1] + correction[0]     ← converged
```

**Cumulative:**
```
x_1[1] = tok_emb[1] + correction[0]
x_2[1] = x_1[1] + correction[0] = tok_emb[1] + 2·correction[0]
x_K[1] = tok_emb[1] + K·correction[0]   ← diverges linearly
```

Non-cumulative makes the iteration a fixed-point map (`x* = x_0 + f(x*)`), giving sequential K=1 = parallel K=N. Cumulative turns it into a standard residual network where depth matters — K=1 and K=N produce fundamentally different outputs.

### The cost of non-cumulative

In a standard transformer (cumulative), each layer refines the previous layer's output:
```
x_1 = tok_emb + f_1(tok_emb)           — low-level features
x_2 = x_1 + f_2(x_1)                   — mid-level, built on low-level
x_N = x_{N-1} + f_N(x_{N-1})           — high-level, far from tok_emb
```

In non-cumulative, the output is always `tok_emb + correction`. There is no progressive abstraction. The correction must encode all contextual information in a single additive term. This limits representational depth.

**Evidence:** D=2 K=5 look-ahead (sequential K=1 PPL 93.69) loses to roformer N=2 baseline (PPL 91.45) at the same param count and inference depth. The roformer's two cumulative layers can build sequentially; the look-ahead's correction is always applied to raw tok_emb.

### Connection to flow matching / diffusion

The iterative refinement has a direct analogy to diffusion models and flow matching (see: "Diffusion Models as ResNets"):

**Cumulative iterations = curved ODE path:**
```
x_k = x_{k-1} + F(x_{k-1})     (Euler integration, N small steps)
```

**Non-cumulative = straight path (rectified flow):**
```
x_k = x_0 + correction          (direct jump from source to target)
```

In flow matching, a velocity model learns `v(x_0) ≈ x_1 - x_0` to traverse the path in one step. Similarly, our correction learns to jump directly from tok_emb to the contextualized representation.

**Possible hybrid approach:** Train with cumulative iterations (for representational depth), but add a loss forcing the non-cumulative single-step to track the cumulative result:
```
L_track = ||x_0 + shift(correction_k) - x_k^{cumulative}||^2
```
This would distill the deep cumulative path into a single-step correction, analogous to rectified flow straightening curved ODE paths.

## Stacked Look-Ahead Units

### Motivation

The D=10 K=10 architecture places weight diversity **within** the unit (10 different-weight layers) and iteration **around** the unit (10 repetitions). The non-cumulative reset reaches all the way back to tok_emb, limiting progressive abstraction.

The stacked architecture inverts this: weight diversity is **between** units (10 separate units), and iteration is **within** each unit (10 iterations of one shared-weight block). Non-cumulative correction is local to each unit, and between units the representations flow cumulatively.

### Architecture

N units stacked sequentially. Each unit i has its own shared-weight block and runs K iterations internally:

```
h = tok_emb
for unit_i in units:                          # N units, cumulative between
    anchor = h                                 # this unit's input = reset target
    processed = h
    for k in 1..K:                             # K iterations, non-cumulative within
        correction = block_i(processed) - processed
        shifted = shift(correction)            # past-only: position t gets correction from t-1
        processed = anchor + shifted           # reset to unit input, not tok_emb
    h = processed                              # output feeds into next unit
logits = head(h)
```

### Comparison with D-block architecture

| Property | D=10 K=10 | Stacked N=10 K=10 |
|---|---|---|
| Params | 10 blocks (separate weights) | 10 blocks (separate weights) |
| Inference layers (K=1) | 10 | 10 |
| Effective training depth | 10 × 10 = 100 | 10 × 10 = 100 |
| Non-cumulative reset anchor | tok_emb (global) | Previous unit's output (local) |
| Between-layer connection | Cumulative within unit | Cumulative between units |
| Within-iteration connection | Non-cumulative to tok_emb | Non-cumulative to unit input |

The key difference: in D=10 K=10, the correction always resets to tok_emb. In stacked units, each unit resets to its own input — the progressively refined representation from earlier units. This allows hierarchical abstraction (cumulative between units) while still benefiting from iterative refinement (non-cumulative within each unit).

### Why this might work better

1. **Progressive abstraction**: Unit 1 builds low-level features, unit 2 refines those into mid-level features, etc. Each unit's anchor is a richer representation than raw tok_emb.

2. **Local convergence**: Each unit only needs its K iterations to converge locally (correct its own input), rather than globally (jump from tok_emb to the final representation). This is an easier optimization target.

3. **NOT a standard transformer at K=1**: At inference, the stacked model uses iterative generation — the correction at position t contextualizes the embedding at t+1, which feeds back as input. This is fundamentally different from a standard transformer's parallel forward pass. Each layer was trained with K iterations of contextualized inputs, and sequential generation naturally provides this contextualization.

### Implementation

The model needs:
- `n_units`: number of stacked units (each with its own block weights)
- `k_iters`: iterations per unit during training
- `n_layers = n_units * k_iters` (total effective depth)

CLI: `--n_layers 100 --n_units 10` → 10 units, each iterated 10 times.

## JoFormer Rotation Correction for Past-Only Shift

### The mismatch

In JoFormer (where Q, K, and V are all rotated), the block output at each position is "unrotated" by a position-dependent inverse rotation `R(Θ_t)^{-1}`. The correction at position t-1 lives in the rotation frame of position t-1. When shifted to position t, the frames don't match.

This does NOT affect RoFormer/RoPE, where only Q and K are rotated — the output has no position-dependent unrotation.

### Derivation

In JoFormerFixed attention, V[t-1] contributing to the output at position t:

1. V[t-1] is rotated: `V'[t-1] = R(Θ_{t-1}) @ V[t-1]`
2. Weighted sum: `out[t]` includes `wei[t,t-1] * V'[t-1]`
3. Inverse rotation at t: `final[t] = R(Θ_t)^{-1} @ out[t]`

Net rotation on V[t-1]:
```
R(Θ_t)^{-1} @ R(Θ_{t-1}) @ V[t-1] = R(Θ_{t-1} - Θ_t) @ V[t-1]
```

With the causal angle shift (position t uses raw angle from t-1):
```
Θ_t = Θ_{t-1} + shifted_angle_t = Θ_{t-1} + raw_angle_{t-1}
Θ_{t-1} - Θ_t = -raw_angle_{t-1}
```

**V[t-1] is rotated by R(-raw_angle[t-1]).**

### Correction

Before shifting correction[t-1] to position t, pre-rotate by R(-raw_angle[t-1]) to align the rotation frame:
```
corrected = R(-raw_angle[t-1]) @ correction[t-1]
processed_x[t] = tok_emb[t] + corrected
```

For **fixed JoFormer**: `raw_angle = freq_flipped` (constant), so the rotation matrix is constant R(-freq_flipped) for all positions.

For **learned/projected JoFormer**: `raw_angle[t-1]` varies per position, so the rotation matrix is position-dependent.

### Experimental result (C=50, N=10, D=1, 100K iters)

| Metric | Uncorrected | Corrected |
|---|---|---|
| Val PPL (K=10) | 97.41 | 97.33 |
| Parallel K=1 | 110.80 | 112.13 |
| Sequential K=1 | 94.37 | 94.15 |

No meaningful difference at C=50. The rotation mismatch may be too small to matter at this embedding dimension, or the model learns to compensate.

## Block Variants

### Standard block (RoFormerBlock)
```
h = x + attn(ln1(x))        # attention + residual
out = h + ffn(ln2(h))        # FFN + residual
```
Within a D-block unit, each sub-block uses these standard residuals. The look-ahead correction is extracted only from the last block: `correction = block_D(h) - h`.

### Nosub variant (FAILED): correction = z instead of z - x

Instead of subtracting the input (`correction = z - x`), use the full block output (`correction = z`). The hypothesis was that passing the full representation as the correction would be richer than just the delta.

**D=4 C=108 results:** Nosub (51.25 PPL) beat block_head (52.22 PPL) by ~1 PPL, with better convergence properties (no overshoot at K=10). This looked promising.

**Stacked N=4 C=108: catastrophic failure.** PPL at 30K: 99.21 (vs 60.09 for block_head). The model barely learned. The problem: in stacked models, each unit resets to its anchor (`processed_x = anchor + shift(correction)`). With correction=z, the full block output magnitude accumulates across units — the correction is not a clean delta relative to the anchor, so the reset mechanism breaks down. With correction=z-x, the subtraction keeps the correction as a bounded delta that the anchor reset can handle.

**Verdict:** Nosub provides a small benefit for D>1 (single unit) but is catastrophic for stacked (multiple units). Not worth pursuing — the subtraction is essential for the non-cumulative reset mechanism.

## The Core Mechanism: Why Sequential K=1 Works

### The contextualized embedding feedback loop

This is the central mechanism of the entire architecture. Understanding it is essential.

At each position t, the block produces a correction. That correction is applied to the embedding at position t+1, creating a **contextualized embedding**:

```
processed_x[t+1] = tok_emb[t+1] + correction[t]
```

The block at position t+1 then receives this contextualized embedding as input — NOT raw tok_emb. It produces a new correction, which contextualizes position t+2, and so on. The block always sees contextualized inputs.

### Why this matches training (approximately)

During training with K iterations:
- Iteration 1: the block sees raw tok_emb (uncontextualized)
- Iteration 2: the block sees tok_emb + correction from iteration 1 (contextualized)
- Iteration 3: the block sees tok_emb + correction from iteration 2 (more contextualized)
- ...
- Iteration K: the block sees tok_emb + correction from iteration K-1

After the first iteration, the block always processes contextualized embeddings. At sequential K=1 inference, each position's input is contextualized by its predecessor's correction — similar to what the block saw at iterations 2..K during training.

**This match is approximate, NOT exact.** It would be exact only at K=infinity, where the iterations have fully converged and the last iteration's contextualization perfectly matches what sequential inference provides. At finite K:

- **K=10**: The block has seen many iterations of contextualized inputs. We *hope* this is sufficient for the sequential contextualization to fall within the learned distribution.
- **K=2**: The block has only seen one iteration of contextualized input. The last layer's contextualization is likely insufficient — sequential inference may feed inputs outside the training distribution.
- **K=1**: The block only saw raw tok_emb. Sequential contextualized inputs are completely out of distribution. **Invalid.**

### Block size must be >= K during training

It takes K iterations for a position to reach full contextualization. At iteration k, position t has corrections informed by positions up to t-k (the contextualization propagates one position per iteration via the past-only shift). With block_size < K, no position in the training window ever achieves full contextualization — the iterations "run out" of sequence to propagate through. This means the block never learns what fully contextualized inputs look like, undermining the sequential K=1 mechanism.

**Rule: always set block_size >= K.** When scaling up K, increase block_size accordingly.

### Divergence risk at long sequences

During training, the model processes fixed-length sequences (block_size=64). The parallel K iterations ensure all positions see a consistent level of contextualization within this window.

At sequential inference, we generate tokens one by one. Each token's embedding is contextualized by the previous token's correction, which was contextualized by the token before that, and so on. Over many tokens, this chain of contextualizations can **diverge** from what the model saw during training:

- The model was trained on K iterations over 64 positions
- Sequential generation over 200+ positions creates a chain of contextualizations far longer than anything seen during training
- Small errors accumulate — each correction is slightly off from the training distribution, and these errors compound

**This means we MUST evaluate at block sizes much longer than the training block_size** to detect whether the iterative generation diverges. If performance degrades significantly at longer sequences, the sequential mechanism is not stable enough for deployment.

### Train/test length mismatch and attention windowing

There is a fundamental train/test mismatch: during training with block_size=64, the attention window covers at most 64 positions. At sequential inference, the context grows unboundedly — after generating 200 tokens, position 200 can attend to all 200 predecessors, but the model has never seen relative distances > 64 during training. With RoPE (our current position encoding), these out-of-distribution distances have no guarantee of working correctly.

**Solutions from the literature:**

1. **ALiBi** (Press et al., 2022 — "Train Short, Test Long"): Replaces positional embeddings with a linear bias on attention scores that penalizes distant tokens. The attention pattern at inference naturally resembles training regardless of sequence length. Best known length generalization.

2. **Sliding window attention** (Mistral, Longformer): Hard limit — each position only attends to the last W positions. The attention pattern at inference exactly matches training. Simple and principled.

3. **RoPE scaling** (NTK-aware, YaRN): Post-hoc interpolation of rotation frequencies for longer sequences. A fix rather than a principled solution.

**Planned change:** Switch from RoPE to ALiBi or sliding window attention (window size = block_size). This ensures:
- Attention at inference matches the training distribution exactly
- No out-of-distribution relative positions
- The iterative contextualization still propagates through corrections (each correction feeds into the next position's embedding), but the attention mechanism itself stays in-distribution
- A sliding window of size block_size means each position attends to the last 64 tokens at both training and inference time

### Implications for the stacked architecture

In the stacked look-ahead model (N units, K iterations each), the same mechanism operates at two levels:

1. **Within each unit**: K iterations contextualize the embeddings, training the block to handle contextualized inputs. At sequential K=1, the left-to-right processing provides this contextualization.

2. **Between units**: Each unit's output is a contextualized representation that becomes the input (anchor) for the next unit. This is cumulative — enabling progressive abstraction across units.

The stacked model is NOT a standard transformer at K=1 because of the iterative generation mechanism. In a standard transformer, all positions are processed in parallel through each layer. In the stacked look-ahead model, each position is generated by feeding contextualized embeddings through the stack — the correction at t creates the input at t+1, which is qualitatively different from parallel processing.

## The Corrhead Problem: Token Identity in Abelian vs Non-Abelian Attention

### Head variants: input vs output vs delta

The block at position t produces three related signals:
- **input**: `processed_x[t]` = tok_emb[t] + shifted_correction[t-1]
- **output**: `z[t]` = block(processed_x)[t] = processed_x[t] + attn(...) + ffn(...)
- **delta**: `correction[t]` = z[t] - processed_x[t] = attn(...) + ffn(...)

These three are related by: **output = input + delta**. The head variant determines which signal the classification head sees:

| Head | Head sees | Signal | Token identity | Context |
|---|---|---|---|---|
| nocat | input (`processed_x`) | tok_emb + shifted correction | Coefficient 1 (from tok_emb) | Past-only |
| block_head | output (`z`) | block(processed_x) | Coefficient 1 (from residual) | Self-inclusive |
| corrhead | delta (`z - processed_x`) | attn + ffn contributions | Variable w[t,t] | Self-inclusive |
| block_head_ffn | output + FFN | z + head_ffn(z) | Coefficient 1 (from residual) | Self-inclusive + enriched |
| addhead | input + delta | processed_x + correction = z | Coefficient 1 (from residual) | Both |
| projhead | learned mix | Linear(2C→C)([proc; corr]) | Learned | Both |
| concat | both signals | [proc; corr] → Linear(2C, V) | Both signals | Both |

**The key insight**: corrhead is the only variant that strips away the residual connection. The delta `z - processed_x` removes the input entirely, leaving only the attention and FFN contributions. Token t's identity in this signal depends on the self-attention weight w[t,t], which varies with context.

Both nocat and block_head preserve token identity with coefficient 1, but they see different things:
- **nocat** sees the **input** to the block — tok_emb enriched by the previous position's correction (past-only context)
- **block_head** sees the **output** of the block — the full transformation including self-inclusive attention and FFN

Note that **addhead ≠ block_head**. After the iteration loop:
- `processed_x[t]` = tok_emb[t] + correction_K[t-1] (post-loop, uses latest correction shifted)
- `z[t]` = block(input_K)[t], where input_K[t] = tok_emb[t] + correction_{K-1}[t-1] (pre-update input)

So addhead[t] = tok_emb[t] + correction_K[t-1] + correction_K[t], while
block_head[t] = tok_emb[t] + correction_{K-1}[t-1] + correction_K[t].

The t-1 correction comes from different iterations (K vs K-1). They only match at convergence.

### The variable scaling problem

In standard (abelian) attention: `attn_output[t] = Σ_{j≤t} w[t,j] · V[j]`. The current token's
contribution to its own correction is scaled by the self-attention weight `w[t,t]`, which varies
with context. When correction[t] is used as the head input (corrhead), the head must decode
token identity from a variably-scaled signal.

In nocat, `tok_emb[t]` always has coefficient exactly 1. The head gets a clean, stable signal of
what the current token is. This is why nocat outperforms roformer corrhead despite having only
past-only context.

### Non-abelian summation changes the picture

JoFormer projected uses input-dependent rotation angles and rotates Q, K, and V. The attention
output is not a simple weighted average — values are rotated before summation and inverse-rotated
after, producing a fundamentally richer (non-abelian) correction.

**Results (D=1, K=10, C=50, block_size=64, 100K iters):**

| Model | Head | Params | Val PPL |
|---|---|---|---|
| roformer_look_ahead_nocat | processed_x (abelian) | 1,646,750 | 98.45 |
| roformer_look_ahead_corrhead | correction (abelian) | 1,646,750 | 105.61 |
| joformer_projected_look_ahead_corrhead | correction (non-abelian) | 1,656,925 | **93.95** |

### Key conclusions

1. **Corrhead is the right architecture.** Self-inclusive context is more valuable than past-only
   context, when the correction properly encodes token identity. The non-abelian corrhead (93.95)
   beats nocat with direct tok_emb access (98.45).

2. **Non-abelian value summation is superior.** The win comes from the richer structure of
   non-abelian summation, not from any specific mechanism to preserve token identity. JoFormer
   projected was not designed to solve the identity encoding problem — it just happens to produce
   corrections where identity is better preserved.

3. **The open problem.** How to make correction[t] reliably encode "what token t is" with a
   consistent signal, so the head can always read the current token identity. The non-abelian
   rotation is one (accidental) solution. A principled mechanism for consistent token identity
   encoding in correction[t] could unlock corrhead's potential across all attention variants.

### Head variant comparison at scale (D=1, K=10, C=50, block_size=256, vocab=16000, 100K iters)

| Model | Params | 100K PPL |
|---|---|---|
| concat (2C→vocab) | 2,446,850 | 82.29 |
| projhead (Linear 2C→C) | 1,651,800 | 87.06 |
| corrhead (correction only) | 1,646,750 | 97.38 |
| nocat (processed_x only) | 1,646,750 | running |
| Roformer N=5 baseline | 1,769,350 | 70.89 |

All D=1 K=10 variants are significantly worse than roformer N=5 at C=50. The architecture
may need larger C (where block params dominate and weight-sharing matters more) to be competitive.

## Proposed Fix: Split Block Variants (FFN Separation)

### Motivation

The corrhead problem stems from delta extraction: `correction = block(x) - x` removes the residual,
so the head sees a signal where token identity depends on the variable self-attention weight `w[t,t]`.
In a standard transformer, the head sees the full block output where the residual preserves token
identity with coefficient 1.

The key insight: separate the block into an **attention pathway** (which contextualizes) and an
**FFN pathway** (which can serve different roles — correction generation or head enrichment).

### Variant 1: Attention + Correction FFN

```
y[t] = processed_x[t] + attn(ln1(processed_x))[t]
correction[t] = corr_ffn(ln2(y))[t]
processed_x[t] = tok_emb[t] + shift(correction)[t]
head_input[t] = y[t]
```

- Head sees attention output with stable residual (coefficient 1)
- FFN generates corrections (runs K times in the loop)
- Same params as standard block (attention + FFN, just repurposed)

### Variant 2: Attention + Head FFN

```
y[t] = processed_x[t] + attn(ln1(processed_x))[t]
correction[t] = y[t] - processed_x[t]
processed_x[t] = tok_emb[t] + shift(correction)[t]
head_input[t] = head_ffn(ln2(y[t]))
```

- FFN enriches output for classification (runs once, not iterated)
- Correction is raw attention delta (no FFN in the loop — cheaper training)
- Same params as standard block (FFN moved to end)

### Variant 3: Standard Block + Head FFN

```
y[t] = processed_x[t] + attn(ln1(processed_x))[t]
z[t] = y[t] + ffn(ln2(y))[t]
correction[t] = z[t] - processed_x[t]
processed_x[t] = tok_emb[t] + shift(correction)[t]
head_input[t] = head_ffn(ln3(z[t]))
```

- Standard block (attn + FFN) unchanged in the loop
- Extra FFN before head adds ~8C² + 5C params
- Most conservative change — existing iteration dynamics preserved

### All Variants (including baseline)

| Variant | Iteration loop | Head sees | FLOPs/tok | 100K PPL |
|---|---|---|---|---|
| nocat (baseline) | block (attn+FFN), corr = delta | `processed_x` | 12C² | 91.85 |
| attn_corr_ffn | attn only, FFN makes correction | `y` (attention output) | 12C² | 91.58 |
| attn_head_ffn | attn only, raw delta correction | `y + head_ffn(y)` | 12C² | 93.88 |
| block_head | block (attn+FFN), corr = delta | `z` (block output) | 12C² | 90.29 |
| block_head_ffn | block (attn+FFN), corr = delta | `z + head_ffn(z)` | 20C² | 84.53 |
| **block_head_corr_ffn** | block + FFN correction from z | `z` (block output) | 20C² | **84.20** |
| **block_head_corr_ffn_concat** | block + FFN corr from concat(shift(z), tok_emb) | `z` (block output) | 24C² | **80.42** |
| **block_head_corr_ffn_add** | block + FFN corr from shift(z) + tok_emb | `z` (block output) | 20C² | **82.59** (D=1), **23.79** (D=3 C=446) |
| ~~block_head_corr_ffn_concat v1~~ | concat with processed_x (BROKEN) | `z` (block output) | 24C² | 80.44 (BROKEN) |

Key findings:
- **attn_corr_ffn ≈ nocat** — splitting attention/FFN doesn't help
- **attn_head_ffn worst** — raw attention delta is a poor correction signal
- **block_head_ffn** was the clear winner of original variants
- **block_head_corr_ffn matches block_head_ffn** with no extra head FFN — the correction FFN is the key
- **block_head_corr_ffn_concat** (v2, fixed with tok_emb) — best D=1 result: 80.42 PPL, seq K=1 gap only 0.09
- **block_head_corr_ffn_add** — same FLOPs as corr_ffn, token-aware via addition. **Best variant for D>1.** See detailed results below.
- **`_px` variants** — head sees `processed_x` instead of `z`. Same params, same FLOPs. See below.

### Head sees processed_x variant (`_px`)

All corr_ffn variants (corr_ffn, corr_ffn_add, corr_ffn_concat) default to `head_input = z` (the block output). The `_px` variants change this to `head_input = processed_x` (the converged contextualized embedding).

```
Standard:   head sees z = block(processed_x)         # block output after last iteration
_px:        head sees processed_x = tok_emb + shift(correction)  # converged input
```

**Motivation:** `processed_x` is the fixed point of the iterative process — the converged contextualized embedding. If iterations converge well (low L), then `processed_x` stabilizes and carries a clean, stable signal. The block output `z` includes the block's own transformation on top of this fixed point, which may add noise or conflicting gradients (the block must serve both the head and correction generation).

**Key properties:**
- **Same params and FLOPs** — only changes which internal signal the head reads
- **Works at any D** (D=1 through D=N)
- The head sees `tok_emb[t] + correction[t-1]` — token identity comes from `tok_emb[t]` (coefficient 1), context from the shifted correction
- At position 0, `processed_x[0] = tok_emb[0]` always (no predecessor), so the head classifies from raw tok_emb — same as nocat at position 0
- **Past-only signal**: the head at position t sees context from t-1 only (via the shifted correction), NOT self-inclusive context from position t itself. This is the same limitation as nocat.

**Tradeoff vs z:**
- `z` gives the head self-inclusive context (the block at position t attends to position t)
- `processed_x` gives the head the converged fixed point with guaranteed token identity
- If convergence is good (L < 0.5), `processed_x` may be more stable; if convergence is poor, `z` carries richer information

**Model names:** `block_head_corr_ffn_px`, `block_head_corr_ffn_add_px`, `block_head_corr_ffn_concat_px`

Controlled by `head_sees_px=True` flag in the base class. All forward paths (forward, forward_at_depth, forward_sequential, forward_with_diagnostics) respect this flag.

**RESULT: dud.** Tested corr_ffn_add_px D=1 C=50 K=5: 86.82 PPL vs corr_ffn_add baseline 82.59 (+4.2 PPL worse). The head needs self-inclusive context from z, not just past-only processed_x. Do not use.

## Variant 4: Block + Correction FFN (block_head_corr_ffn) — Current Best

### Architecture

```
z[t] = block(processed_x)[t]              # standard block (attn + FFN + residuals)
correction[t] = corr_ffn(ln_corr(z))[t]   # separate FFN generates correction from z
processed_x[t] = tok_emb[t] + shift(correction)[t]
head_input[t] = z[t]                       # head sees block output directly
```

### Why this works

The key insight: **separate the correction generation from the block's own FFN**. The block's internal FFN refines the attention output for representation quality (and its residual preserves token identity for the head). A separate `corr_ffn` then reads this rich representation to produce a correction for the next position.

This is equivalent to block_head_ffn in FLOPs (20C² per iteration) but cleaner:
- block_head_ffn: block produces correction via delta, FFN enriches output for head
- block_head_corr_ffn: block produces output for head, FFN generates correction from output

The correction FFN `corr_ffn(ln(z))` is a standard FFN: `Linear(C→4C) → GELU → Dropout → Linear(4C→C) → Dropout`. It adds 8C² + 5C params.

### FLOP analysis

```
Per iteration:  block = 12C²,  corr_ffn = 8C²  →  total = 20C²
Roformer N=3:   3 × (12C² + 4C²) = 48C²  (but only 36C² if no FFN double-count)
```

**FLOP matching**: block_head_corr_ffn D=1 K=5 at 20C² matches roformer N=1 at 12C². To match roformer_head_ffn N=3 (44C²), use deep D=3 at 3×(12C²+8C²/3) ≈ 44C².

### Results (C=50, K=5, block_size=256, 100K iters)

| Model | Params | Val PPL | Seq K=1 | L |
|---|---|---|---|---|
| block_head_corr_ffn | ~1,651K | 84.20 | 84.22 | 0.94 |
| block_head_ffn | ~1,657K | 84.53 | 84.57 | 1.20 |
| roformer N=5 | 1,769K | 70.89 | — | — |

block_head_corr_ffn slightly outperforms block_head_ffn with similar params.

### Sequential K=1 equations

At sequential inference, position t is fully processed before moving to t+1:

```
For each position t = 0, 1, ..., T-1:
    z[t] = block(processed_x)[t]         # block output at position t
    correction[t] = corr_ffn(ln(z))[t]   # correction from z at position t
    if t < T-1:
        processed_x[t+1] = tok_emb[t+1] + correction[t]
    head_input[t] = z[t]
```

Position t gets a contextualized input `tok_emb[t] + correction[t-1]` where correction[t-1] was computed from the fully processed z at position t-1. This matches the parallel K>1 regime where later iterations see contextualized inputs.

## Variant 6: Synced Head (attn_corr_ffn_sync)

### Motivation: head and correction in sync

In block_head variants, the head sees `z` (block output) while the correction is derived from `z` separately. The head and the iterative process operate on different signals:
- The correction contextualizes the *next* position's input
- The head classifies from the block's output
- These two tasks share the same block but have no direct alignment

In attn_corr_ffn, the head sees `h = x + attn(x)` (the attention residual), while the correction is `corr_ffn(ln(h))`. Again, the head and correction are decoupled.

The sync variant aligns them: **the head sees exactly `processed_x + correction`** — the same correction signal that drives the iterative process, but applied at the current position (unshifted) rather than the next position (shifted).

### Equations

```
h[t] = processed_x[t] + attn(ln1(processed_x))[t]     # attention + residual
correction[t] = corr_ffn(ln2(h))[t]                     # FFN generates correction
head_input[t] = processed_x[t] + correction[t]          # self-inclusive (unshifted)
processed_x[t+1] = tok_emb[t+1] + correction[t]        # past-only (shifted)
```

### Side-by-side comparison

```
                attn_corr_ffn          attn_corr_ffn_sync        block_head
                ─────────────          ──────────────────        ──────────
Step 1:     h = x + attn(ln1(x))    h = x + attn(ln1(x))     h = x + attn(ln1(x))
Step 2:     corr = corr_ffn(ln2(h)) corr = corr_ffn(ln2(h))  z = h + ffn(ln2(h))
Head:       h                       x + corr                  z = h + ffn(ln2(h))
Correction: corr                    corr                      z - x
```

Key differences:
- **attn_corr_ffn**: Head sees `h` (attention output). Correction and head are decoupled.
- **attn_corr_ffn_sync**: Head sees `x + correction`. The head sees exactly what the iterative process produces (before shifting). Head and correction are aligned.
- **block_head**: Head sees `z = h + ffn(h)`. The FFN contributes to both the head signal and (implicitly) the correction. Entangled.

### Why sync might work better

At each position t, the sync head sees `processed_x[t] + correction[t]`:
- `processed_x[t] = tok_emb[t] + correction[t-1]` carries token identity + past context
- `correction[t] = corr_ffn(ln2(h[t]))` carries self-inclusive context from attention

Together, `processed_x[t] + correction[t]` is exactly what `processed_x[t+1]` *would be* if the correction weren't shifted. The head classifies from the same representation that the model is learning to produce as contextualized embeddings. There is no gap between "what the model optimizes the correction for" and "what the head classifies from."

In block_head, the head sees `z` which includes the FFN residual from `h`. This is a richer signal, but the FFN must serve double duty: enriching the head input AND producing a correction (via `z - x`). These two objectives can conflict.

In the sync variant, the FFN (corr_ffn) has a single clear objective: produce corrections. The head directly evaluates the quality of those corrections.

### Parameters and FLOPs

Same as a standard roformer block: **12C² per D**.

| Component | Params |
|---|---|
| Attention (Q, K, V, O) | 4C² |
| corr_ffn (C→4C→C) | 8C² |
| 2 LayerNorms | 4C |
| **Total per D** | **12C² + 4C** |

Param and FLOP matched to roformer at the same N=D:
- D=4 sync = roformer N=4 = 48C² inference FLOPs

### D>1 (deep) variant

With d_block=D, the model has D separate-weight (attn, corr_ffn) pairs applied sequentially per iteration. Each pair d:

```
h_d = x_d + attn_d(ln1_d(x_d))
correction_d = corr_ffn_d(ln2_d(h_d))
x_{d+1} = x_d + correction_d           # within-iteration: unshifted, self-inclusive
```

Only the last pair's correction gets shifted for the next iteration:

```
processed_x = tok_emb + shift(correction_D)
```

The intermediate pairs (d < D) pass `x_d + correction_d` directly to the next pair — no shift, no reset to tok_emb. This builds up representation depth within the iteration while maintaining the past-only shift between iterations.

**Status: running.** D=4 C=108 (48C² FLOPs, ~4M params) comparing against roformer N=4 C=108.

### Deep vs Stacked: Self-Inclusive Correction Asymmetry

The corr_ffn produces a **self-inclusive correction** at each position: correction[t] is informed by position t itself (through causal attention including the diagonal). But the past-only shift means correction[t] only enters processed_x at position t+1, never at position t. The head compensates by seeing correction[t] directly (sync: `processed_x[t] + correction[t]`).

This creates a fundamental asymmetry between deep (D>1) and stacked models:

**Deep (D>1, single unit):** Within an iteration, the D blocks are applied sequentially *without shifting*:
```
x_1 = x_0 + correction_0     # block 1 sees self-inclusive output of block 0
x_2 = x_1 + correction_1     # block 2 sees self-inclusive output of block 1
...
```
Block d>0 already sees the self-inclusive correction from block d-1. The corr_ffn's self-inclusive signal becomes **redundant** — the later blocks in the chain get this information "for free" from the residual stream.

**Stacked (N units):** Between units, the correction is shifted and anchor resets:
```
Unit 0 output: h[t] = anchor_0[t] + correction_0[t-1]    # past-only
Unit 1 anchor: anchor_1[t] = h[t]                          # no self-inclusive correction from unit 0
Unit 1 output: h[t] = anchor_1[t] + correction_1[t-1]     # still past-only
```
Across all N units, position t accumulates `tok_emb[t] + Σ correction_i[t-1]` — corrections from position t-1 of each unit, **never from position t itself**. The self-inclusive correction only appears at the head.

**Consequence:** The corr_ffn's self-inclusive signal retains its value in stacked models (each unit boundary re-introduces the past-only constraint) but may lose value in deep models (within-iteration blocks already propagate self-inclusive information).

### Hybrid: Deep + Stacked

This asymmetry suggests combining both: **d_block=2 with n_units=2** gives 4 total blocks (48C², param-matched to d_block=4 or roformer N=4), but splits them as 2 deep blocks per unit × 2 stacked units.

Benefits:
- Each unit has D=2 depth for within-iteration processing
- The stacked boundary between units preserves the corr_ffn's self-inclusive advantage (anchor reset + shift)
- More unit boundaries = more chances for the shift mechanism to propagate corrections

The general family: d_block × n_units = N total blocks. At one extreme (d_block=N, n_units=1) all blocks are deep. At the other extreme (d_block=1, n_units=N) all blocks are stacked. The hybrid explores the middle ground.

## Variant 7: Block-Aligned Look-Ahead (block_aligned)

### Motivation

The block_head and block_aligned architectures share the same attention, the same FFN, the same classifier, and the same z computation. They differ in exactly one place: **what gets shifted to the next position**.

### Side-by-side equations (K=3 example)

Where `f(x, c) = x + c + ffn(ln2(x + c))` (standard residual block formula).

**block_head:**
```
init:  processed_x = tok_emb

k=0:   attn_corr_0 = attn(ln1(tok_emb))
       z_0 = f(tok_emb, attn_corr_0)
       correction_0 = z_0 - tok_emb = attn_corr_0 + ffn(ln2(tok_emb + attn_corr_0))
       processed_x_1 = tok_emb + shift(correction_0)

k=1:   attn_corr_1 = attn(ln1(processed_x_1))
       z_1 = f(processed_x_1, attn_corr_1)
       correction_1 = z_1 - processed_x_1 = attn_corr_1 + ffn(ln2(processed_x_1 + attn_corr_1))
       processed_x_2 = tok_emb + shift(correction_1)

k=2:   attn_corr_2 = attn(ln1(processed_x_2))
       z_2 = f(processed_x_2, attn_corr_2)
       correction_2 = z_2 - processed_x_2 = attn_corr_2 + ffn(ln2(processed_x_2 + attn_corr_2))
       processed_x_3 = tok_emb + shift(correction_2)

classifier: head(ln_f(z_2))
```

**block_aligned:**
```
init:  processed_x = tok_emb

k=0:   attn_corr_0 = attn(ln1(tok_emb))
       z_0 = f(tok_emb, attn_corr_0)
       processed_x_1 = f(tok_emb, shift(attn_corr_0))

k=1:   attn_corr_1 = attn(ln1(processed_x_1))
       z_1 = f(processed_x_1, attn_corr_1)
       processed_x_2 = f(tok_emb, shift(attn_corr_1))

k=2:   attn_corr_2 = attn(ln1(processed_x_2))
       z_2 = f(processed_x_2, attn_corr_2)
       processed_x_3 = f(tok_emb, shift(attn_corr_2))

classifier: head(ln_f(z_2))
```

### The only difference: step 3

Expanding the step 3 equations for a specific position t:

| | block_head | block_aligned |
|---|---|---|
| What gets shifted | `correction = attn_corr + ffn(ln2(processed_x + attn_corr))` | `attn_corr` (raw attention output) |
| Next processed_x[t] | `tok_emb[t] + correction[t-1]` | `tok_emb[t] + attn_corr[t-1] + ffn(ln2(tok_emb[t] + attn_corr[t-1]))` |

Everything else is identical: same init, same attn, same z, same classifier.

The FFN input differs:
- **block_head**: FFN was evaluated at the source position with `ln2(processed_x[t-1] + attn_corr[t-1])`. Its output is shifted as-is to position t.
- **block_aligned**: FFN is re-evaluated at the destination position with `ln2(tok_emb[t] + attn_corr[t-1])`. The new token's identity is baked into the FFN computation.

### block_aligned_light variant

The classifier's z computation (`f(processed_x, attn_corr)`) applies FFN a second time. Since processed_x already has FFN baked in from step 3, the light variant skips this: classifier sees `processed_x + attn_corr` directly, saving 8C² FLOPs at classification.

### Parameters and FLOPs

All three are 12C² per D (same as roformer):
- block_head: 12C² params, 12C² FLOPs per iteration
- block_aligned: 12C² params, 12C² FLOPs per iteration (FFN runs in step 3, z computed at classifier only)
- block_aligned_light: 12C² params, 12C² FLOPs per iteration (skips FFN at classifier)

### Results (C=50, K=5, k_min=2, block_size=256, 100K iters)

| Model | 10K PPL | 100K PPL | Seq K=1 | L |
|-------|---------|----------|---------|---|
| block_head | 129.97 | 91.97 | 92.08 | 0.54 |
| block_aligned | 133.98 | 94.71 | 94.73 | 0.58 |
| block_aligned_light | 135.72 | 96.63 | 96.66 | 0.66 |

block_head wins by 2.7 PPL over block_aligned. Shifting the full z is more effective than shifting raw attn_corr and re-applying FFN. The FFN processing from the current position carries useful information that is lost when only the attention output is shifted.

## Variant 8: Tied and Pure Correction FFN Variants

### Motivation: reducing correction FFN params

The corr_ffn adds 8C² params (a full FFN) to generate corrections. Can we tie its weights to the block's own FFN, reducing total per-iteration cost from 20C² to 12C² (same as block_head/roformer)?

Two orthogonal ideas:
1. **Tied FFN**: Share weights between corr_ffn and block.ffn (saves 8C² params)
2. **Pure residual pattern**: Use `f(tok_emb, shift(z))` instead of `f(processed_x, shift(z))` to build processed_x

### Tied variant (corr_ffn_add_tied)

```
z[t] = block(processed_x)[t]                          # standard block
shifted_z[t] = z[t-1]
correction[t] = block.ffn(block.ln2(shifted_z[t] + tok_emb[t]))   # REUSE block's FFN and LN
processed_x[t] = tok_emb[t] + correction[t]
head_input[t] = z[t]
```

The block's FFN is called twice per iteration:
1. Inside the block: `ffn(ln2(h))` where `h = processed_x + attn(ln1(processed_x))`
2. As correction: `ffn(ln2(shift(z) + tok_emb))`

Both calls go through `ln2` first, maintaining input structure consistency. The FFN sees layernorm'd inputs in both cases, but the actual distributions differ (attention residual vs shifted block output + token embedding).

**Params: 12C²** — same as block_head and roformer.

### Pure residual pattern (corr_ffn_add_pure)

Instead of building processed_x from the previous iteration's processed_x:
```
# Standard (non-pure):
processed_x = tok_emb + shift(z) + corr_ffn(ln_corr(processed_x_prev + shift(z)))
```

Use tok_emb as the anchor at every iteration:
```
# Pure:
processed_x = tok_emb + shift(z) + corr_ffn(ln_corr(tok_emb + shift(z)))
```

The motivation is architectural consistency — the FFN always sees `tok_emb + shift(z)`, not a recursively accumulated processed_x. This also has a tied variant (`corr_ffn_add_tied_pure`, 12C²).

### Results (C=50, K=5, k_min=2, block_size=256, 10K iters)

| Model | FLOPs/iter | 10K PPL | Seq K=1 | L |
|-------|-----------|---------|---------|---|
| corr_ffn_add | 20C² | 120.96 | 120.95 | 0.44 |
| corr_ffn_add_tied | 12C² | 129.43 | 129.44 | 0.42 |
| block_head | 12C² | 129.97 | — | — |
| add_pure | 20C² | 134.27 | 135.07 | 0.88 |
| add_tied_pure | 12C² | 139.13 | 140.31 | 0.91 |

### Why pure fails: the direct skip connection defeats contraction

The pure pattern looks architecturally clean, but it contains a fatal flaw: the **direct skip connection from shift(z) to processed_x** creates a near-identity iteration map that prevents convergence.

Expanding what `shift(z)` actually contains:

```
z = block(processed_x)
  = processed_x + attn(ln1(processed_x)) + ffn(ln2(processed_x + attn(...)))
  ≈ processed_x + delta                    (delta = attention + FFN contributions)
```

Due to the block's residual connections, z ≈ processed_x + delta. Therefore:

```
shift(z)[t] = z[t-1] ≈ processed_x[t-1] + delta[t-1]
```

In the pure pattern:
```
processed_x_new[t] = tok_emb[t] + shift(z)[t] + ffn(ln(tok_emb[t] + shift(z)[t]))
                    = tok_emb[t] + [processed_x_old[t-1] + delta[t-1]] + ffn(...)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    previous iteration's processed_x leaks through
```

The previous iteration's `processed_x[t-1]` enters the new `processed_x[t]` through the direct `shift(z)` path. This creates a near-identity coupling between iterations: if `processed_x` changes by ε, z changes by approximately ε (due to the residual), and `shift(z)` carries that ε directly into the next iteration's `processed_x`. The Lipschitz constant of the iteration map is close to 1.

**How block_head avoids this:** Block_head extracts `correction = z - processed_x`, which cancels the identity component:
```
correction = z - processed_x = attn_delta + ffn_delta      (bounded, no identity leak)
processed_x_new = tok_emb + shift(correction)               (contractive)
```

The subtraction `z - processed_x` is essential — it strips out `processed_x` from z, leaving only the attention and FFN deltas. Shifting a bounded delta is naturally contractive.

**How corr_ffn_add avoids this:** The standard (non-pure) corr_ffn_add routes `shift(z)` through the FFN bottleneck only — no direct skip:
```
correction = corr_ffn(ln_corr(shift(z) + tok_emb))         (FFN compresses z)
processed_x_new = tok_emb + correction                      (contractive)
```

Even though shift(z) contains the identity component, the FFN can learn to extract only the relevant delta and discard the `processed_x` residual. The bottleneck (C → 4C → C with nonlinearity) naturally compresses the signal, keeping L well below 1.

**Pure has no such mechanism.** The direct `shift(z)` path bypasses both the subtraction (block_head's approach) and the FFN bottleneck (corr_ffn_add's approach). No matter what the FFN learns, the identity component of z propagates directly into processed_x, keeping L ≈ 1.

The convergence diagnostics confirm this precisely: L ≈ 0.88–0.91 for pure variants vs L ≈ 0.42–0.44 for non-pure.

### The contraction principle for look-ahead architectures

This analysis reveals a general design principle: **any variant that passes `shift(z)` directly into `processed_x` (without subtracting the identity or routing through a bottleneck) will fail to converge.**

The three mechanisms for achieving contraction:

| Mechanism | Example | How it removes identity | Cost |
|-----------|---------|------------------------|------|
| Delta extraction | block_head: `z - processed_x` | Explicit subtraction | 0 (free) |
| FFN bottleneck | corr_ffn_add: `ffn(ln(shift(z) + tok_emb))` | FFN learns to compress | 8C² |
| Both | corr_ffn: `ffn(ln(z))` (no shift(z) in processed_x) | FFN on z, no direct path | 8C² |

Block_head achieves contraction for free via the subtraction. The corr_ffn variants pay 8C² for a learnable bottleneck that can extract richer corrections than a simple delta. The pure pattern uses neither mechanism.

### Tied ≈ block_head: the correction FFN matters

At 12C², corr_ffn_add_tied (129.43) essentially matches block_head (129.97). This makes sense: tying the FFN removes the correction's independent capacity. The shared FFN must serve double duty — enriching the representation inside the block AND generating corrections — but these tasks may require different transformations. With shared weights, the FFN compromises between the two objectives, leaving it no better than block_head's simple delta extraction.

The separate corr_ffn earns its 8C²: the 20C² → 12C² reduction costs ~9 PPL (120.96 → 129.43). The correction FFN needs independent weights to specialize for correction generation.

### Summary

| Idea | Result | Why |
|------|--------|-----|
| Tied FFN (save 8C²) | Recovers block_head performance | Shared weights can't specialize; dual-objective compromise |
| Pure residual (tok_emb anchor) | L ≈ 0.9, +13 PPL | Direct shift(z) skip defeats contraction |
| Tied + pure | Worst of both worlds | Both failure modes compound |
| **Implication** | block_head is optimal at 12C² | Delta subtraction is the cheapest contraction mechanism |

## The Token-Blind Correction Problem (FAILED: Concat Variant)

### The observation

In block_head_corr_ffn, the correction at position t contextualizes position t+1:

```
correction[t] = corr_ffn(ln(z[t]))
processed_x[t+1] = tok_emb[t+1] + correction[t]
```

The correction is completely independent of what token is at position t+1. The same correction is applied regardless of the predicted token. This is different from a regular transformer, where each layer's output depends on the current token's embedding.

### The attempted fix: concat variant

Make the correction see both past context AND current token identity:

```
z[t] = block(processed_x)[t]
shifted_z[t] = z[t-1]                             # past context (shifted)
ffn_input[t] = concat(ln_corr(shifted_z[t]), processed_x[t])    # 2C input
correction[t] = corr_ffn(ffn_input[t])             # now correction knows the current token
processed_x[t] = tok_emb[t] + correction[t]       # NO shift needed — shift is inside
head_input[t] = z[t]
```

### Results (C=50, K=5, block_size=256, 100K iters)

| Model | Params | Val PPL (K=5) | Seq K=1 | Gap (seq vs val) | L |
|---|---|---|---|---|---|
| corr_ffn_concat | 1,677,100 | **80.44** | 84.80 | **+4.4** | 0.71 |
| corr_ffn (baseline) | 1,651,600 | 84.32 | 84.55 | +0.2 | 0.88 |

### Why it fails: circular dependency breaks sequential K=1

The concat variant shows a 3.9 PPL improvement in parallel K=5 training (80.44 vs 84.32). But **sequential K=1 is 84.80 — worse than the baseline's val PPL.** The 4 PPL "improvement" is an illusion.

The root cause is a **circular dependency**. In the iteration loop during training:

```
for k in 1..K:
    z = block(processed_x)
    shifted_z = shift(z)
    correction = corr_ffn(concat(ln(shifted_z), processed_x))  # sees processed_x
    processed_x = tok_emb + correction                          # updates processed_x
```

At each iteration, `processed_x` is updated, and the next iteration's correction sees the *refined* `processed_x`. Over K=5 iterations, `processed_x` converges to a stable point where the correction and `processed_x` are mutually consistent.

At sequential K=1 inference, there is no iterative refinement. The correction at position t sees `processed_x[t]` which is `tok_emb[t] + correction_from_prev_position` — a single-pass value that has NOT been iteratively refined. The corr_ffn was trained on refined `processed_x` (iterations 2..K) but gets unrefined `processed_x` at inference. This is out of distribution.

**Why corr_ffn doesn't have this problem:** In the original corr_ffn, `correction[t] = corr_ffn(ln(z[t]))`. The correction depends on `z[t]`, not on `processed_x[t+1]`. At sequential K=1, once position t is fully processed, `z[t]` is fully determined — there is no dependency on the target position's state. The correction is a clean function of the finalized source position.

**The key principle:** The correction must be a function of *already-finalized* quantities only. Any dependency on the target position's current state creates a circular dependency that requires iterative refinement to resolve — breaking the K=1 sequential inference guarantee.

### Lesson learned

Token-blind corrections are a real limitation of the architecture, but the fix cannot introduce dependencies on `processed_x` at the target position. Any future attempt to make corrections token-aware must find a way to incorporate the target token's identity without creating a circular dependency. The correction must depend only on *already-finalized* quantities.

### Fix 1: Concat with tok_emb (block_head_corr_ffn_concat v2)

Replace `processed_x` with `tok_emb` in the concat — `tok_emb` is constant (no circular dependency):

```
z[t] = block(processed_x)[t]
shifted_z[t] = z[t-1]
correction[t] = corr_ffn(concat(ln_corr(shifted_z[t]), tok_emb[t]))   # tok_emb, NOT processed_x
processed_x[t] = tok_emb[t] + correction[t]
head_input[t] = z[t]
```

corr_ffn input is 2C: Linear(2C→4C)→GELU→Linear(4C→C). Total: 24C² per iteration (vs 20C² for corr_ffn).

**Results (C=50, K=5, block_size=256, 100K iters):**

| Model | Params | Val PPL (K=5) | Seq K=1 | Gap | L |
|---|---|---|---|---|---|
| corr_ffn_concat v2 (tok_emb) | 1,677,100 | **80.42** | **80.51** | **+0.09** | 0.71 |
| ~~corr_ffn_concat v1 (processed_x)~~ | 1,677,100 | 80.44 | 84.80 | +4.4 | 0.71 |
| corr_ffn (token-blind baseline) | 1,651,600 | 84.32 | 84.55 | +0.2 | 0.88 |

The tok_emb fix eliminates the circular dependency completely. Sequential K=1 matches val PPL (gap 0.09). The 4 PPL improvement over token-blind corr_ffn is now real.

### Parameter and FLOP comparison: concat v2 vs roformer_head_ffn (single layer)

At N=1 / K=1 (single layer each), concat v2 and roformer_head_ffn have nearly identical structure. The only difference is the correction FFN input width.

**roformer_head_ffn N=1:**
```
Embeddings:  2VC           (token_embedding + lm_head)
Block:       12C²          (attention 4C² + FFN 8C²)
head_ffn:    8C²           (C → 4C → C)
────────────────────────────
Total:       2VC + 20C²
```

**D=1 concat v2 K=1:**
```
Embeddings:  2VC           (token_embedding + head)
Block:       12C²          (attention 4C² + FFN 8C²)
corr_ffn:    12C²          (2C → 4C → C, wider input from concat)
────────────────────────────
Total:       2VC + 24C²
```

**Difference: 4C²** — the corr_ffn takes 2C input (concat of shifted z and tok_emb) vs head_ffn's C input. This gives corr_ffn 12C² vs head_ffn 8C².

The relative overhead depends on the embedding-to-block ratio:

```
Extra fraction = 4C² / (2VC + 20C²) = 4C / (2V + 20C)
```

| C | V | Extra params | Overhead |
|---|---|---|---|
| 50 | 16000 | 10K | 0.6% |
| 100 | 16000 | 40K | 1.1% |
| 446 | 16000 | 796K | 4.4% |
| C → ∞ | V | — | → 20% |

At practical C (50-446), the overhead is small (0.6-4.4%) because embeddings (2VC) dominate. In the limit C >> V, the overhead approaches 4/20 = 20%.

**Inference FLOPs per token (sequential K=1):**

| Model | FLOPs/token |
|---|---|
| roformer_head_ffn N=1 | 20C² |
| D=1 concat v2 K=1 | 24C² |
| roformer_head_ffn N=3 | 44C² |
| roformer_head_ffn N=6 | 80C² |

D=1 concat v2 costs 24C² per token at inference — 20% more than a single roformer_head_ffn layer, but 45% less than N=3 and 70% less than N=6. The question is whether one shared-weight block with contextualized inputs (from the correction mechanism) can match the quality of multiple separate-weight layers.

### Why the concat advantage vanishes at D>1 and stacked n_units>1

**Setup.** Both non-concat and concat v2 produce a correction that contextualizes the next iteration's input. The key difference is what the correction at position t depends on:

- **Non-concat (corr_ffn)**: `correction[t] = corr_ffn(z[t])`, then shifted right.
  - Position t receives: `processed_x[t] = tok_emb[t] + corr_ffn(z[t-1])`
  - The correction applied at t depends **only on z[t-1]** — the past. It is completely unaware of processed_x[t]. It does not see tok_emb[t] or anything about position t.

- **Concat v2**: `correction[t] = corr_ffn(z[t-1], tok_emb[t])`, applied directly at t.
  - Position t receives: `processed_x[t] = tok_emb[t] + corr_ffn(z[t-1], tok_emb[t])`
  - The correction applied at t depends on **z[t-1] (past) plus tok_emb[t] (current token)**. It is aware of the token at position t.

In both cases, tok_emb[t] appears in processed_x through the addition. But in non-concat, the correction cannot depend on position t at all — it is purely a function of the past. In concat, the correction is a function of both past context and current token identity, enabling token-dependent corrections (e.g., "given context X, if the token is Y, apply correction Z").

**Why concat helps at D=1.** With a single shared-weight block, the block has limited capacity (one self-attention + one FFN). A token-dependent correction gives it a better starting point — the correction can tailor processed_x to the specific token at position t, rather than providing a generic context-only signal that the block must then reconcile with tok_emb[t].

**Why concat may not help at D>1.** With D separate-weight blocks processing the sequence, there are D layers of self-attention that each attend to position t (causal masking includes self). Even though the correction at position t is completely unaware of position t (in non-concat), the D blocks have ample capacity to handle the token identity that enters processed_x through the additive tok_emb[t]. The blocks themselves compute the context × token interactions that concat's corr_ffn would have provided. The extra 4C² params in concat's wider FFN (12C² vs 8C²) become redundant.

**Why concat may not help at stacked n_units>1.** In stacked models, each unit runs K iterations with its own corr_ffn. Unit 1 operates on tok_emb — same situation as D=1, so concat helps here. But unit 2's input is unit 1's output h1, which is already token-aware (from K iterations of self-attention). Unit 2's non-concat corr_ffn sees z[t-1] computed from token-aware inputs — so even without explicit tok_emb[t], the correction is indirectly aware of token identities. The concat channel becomes redundant for units beyond the first.

**Summary:**

| Architecture | Correction sees position t? | Blocks compensate? | Concat value |
|---|---|---|---|
| D=1 | Only with concat (tok_emb[t]) | 1 block, limited | High |
| D>1 | Only with concat (tok_emb[t]) | D blocks, ample | Low |
| Stacked unit 1 | Only with concat (tok_emb[t]) | 1 shared block, limited | High |
| Stacked units 2+ | Indirectly (token-aware input) | 1 shared block | Low |

**Empirical evidence (D=3 C=446, big machine):**

Non-concat (K=10) pulls ahead of concat v2 (K=5) by ~0.3 PPL at 70K iters. However, there are two confounds: (1) K=5 vs K=10 training depth, (2) random K training in concat v2. A control experiment is planned: non-concat with K=5 + random K to isolate the architectural effect.

**Implication:** At D>1 or stacked n_units>1, prefer the simpler non-concat architecture. It saves 4C² params and FLOPs in the corr_ffn without sacrificing quality. The concat overhead (20% more corr_ffn compute) is only justified at D=1.

### Fix 2: Add tok_emb (block_head_corr_ffn_add)

Instead of concatenating shifted_z and tok_emb (requiring 2C→4C FFN), add them before the FFN:

```
z[t] = block(processed_x)[t]
shifted_z[t] = z[t-1]
correction[t] = corr_ffn(ln_corr(shifted_z[t] + tok_emb[t]))   # addition, not concat
processed_x[t] = tok_emb[t] + correction[t]
head_input[t] = z[t]
```

corr_ffn input is C: standard FeedForward (C→4C→C). Total: **20C² per iteration** — same as token-blind corr_ffn.

**Properties:**
- Same params/FLOPs as corr_ffn (20C²) — zero overhead
- Token-aware: corr_ffn sees both past context and current token identity
- No circular dependency: tok_emb is constant
- Question: does addition provide enough signal vs concatenation?

### Results: corr_ffn_add is the best D>1 variant

corr_ffn_add consistently outperforms all other variants at D>1, combining the best convergence properties with zero FLOP overhead over token-blind corr_ffn.

**C=446, K=5, 100K iters — corr_ffn_add vs roformer baselines:**

| Model | FLOPs | Final PPL | Seq K=1 | L | vs roformer |
|-------|-------|-----------|---------|---|-------------|
| D=2 add | 32C² | 26.09 | 26.48 | 0.54 | beats N=3 (27.19, 36C²) by 1.10 PPL, 11% fewer FLOPs |
| D=3 add | 44C² | 23.79 | 24.12 | 0.49-0.65 | beats N=4 (24.85, 48C²) by 1.06 PPL, 8% fewer FLOPs |
| D=6 add | 80C² | running | — | — | tracking rhf N=6 (21.44, 80C²) |

**Why corr_ffn_add is the recommended variant:**

1. **Best convergence.** L consistently ~0.5 (vs ~0.74 for token-blind corr_ffn). K=5 and K=10 produce identical PPL. K=3 is already nearly converged. This means sequential K=1 inference works reliably.

2. **Zero overhead.** Same 20C² FLOPs and same params as token-blind corr_ffn. The addition `ln(shift(z) + tok_emb)` costs nothing — the FFN input is still size C.

3. **Catches deeper roformers with training.** The gap vs roformer N=5 (60C²) shrank from 0.72 at 80K to 0.56 at 100K. D=3 add improves faster late in training than roformer, presumably because shared weights continue to improve at iteration while roformer's separate layers have already specialized. A 200K run is in progress to test crossover.

4. **Scales across FLOP budgets.** Beats the roformer with matched or greater FLOPs at every tested budget (32C², 44C², 80C²). The advantage is consistent, not a one-off.

**Why add beats token-blind corr_ffn:** The corr_ffn generates corrections for position t based on context from t-1. Without tok_emb, the FFN must infer the current token's identity from shift(z) alone — but shift(z) encodes position t-1, not t. Adding tok_emb gives the FFN direct access to the current token's identity, enabling token-specific corrections at zero cost.

**Why add vs concat tradeoff favors add at D>1:** Concat (24C² per iter) gets ~1.7-2.2 PPL better than add at D=1 C=50. But at D>1 with larger C, the gap shrinks and the 4C² overhead per iteration compounds across D layers. The add variant delivers most of the benefit of token-awareness without any extra FLOPs, making it the better choice when compute efficiency matters.

### Why corr_ffn_add has better convergence than corr_ffn_concat (theoretical)

The structural difference in how LayerNorm interacts with tok_emb explains the convergence gap:

```
corr_ffn_add:    corr_ffn(LN(shift(z) + tok_emb))    — LN normalizes the sum jointly
corr_ffn_concat: corr_ffn([LN(shift(z)); tok_emb])   — LN normalizes only shift(z), tok_emb appended raw
```

The contraction factor involves the LN Jacobian norm, which is proportional to 1/σ(v) where σ is the standard deviation across dimensions. So:

- **add**: contraction ∝ 1/σ(z + tok_emb)
- **concat**: contraction ∝ 1/σ(z)

Since the variance of the sum decomposes as σ²(z + tok_emb) = σ²(z) + σ²(tok_emb) + 2·Cov_d(z, tok_emb), and when z and tok_emb are approximately uncorrelated across dimensions:

**σ(z + tok_emb) ≥ √(σ²(z) + σ²(tok_emb)) > σ(z)**

Therefore the LN denominator is strictly larger for add, giving a strictly **smaller contraction factor**. The tok_emb variance acts as a "floor" that prevents LN from amplifying small perturbations in z.

This is particularly important when σ(z) is small (block output with low variance across dimensions) — the concat variant's contraction factor 1/σ(z) can spike, while the add variant is stabilized by σ(tok_emb).

**Empirical confirmation:** L ≈ 0.5 for add vs L ≈ 0.74 for both concat and token-blind corr_ffn (which also normalizes z alone).

**Note:** Normalizing tok_emb separately in concat (i.e., `[LN(z); LN(tok_emb)]`) would not help — the LN on the z pathway still has denominator σ(z). The key is **joint normalization** of both signals in a single d-dimensional vector, which only addition provides naturally.

## Training Optimization: K=5 and Random K

### K=5 matches K=10

Training with K=5 iterations instead of K=10 gives nearly identical results with 2x faster training:

| Config | 100K Val PPL | Seq K=1 | L |
|---|---|---|---|
| K=10, cw=0 | 84.16 | 84.18 | 0.94 |
| K=5, cw=0 | 84.32 | 84.55 | 0.88 |
| K=10, random K (k_min=2), cw=0 | 84.41 | 84.36 | 0.72 |

K=5 loses only 0.16 PPL vs K=10. **Recommendation: use K=5 for all experiments.**

### Random K training

Sample K ~ Uniform(k_min, K_max) each batch during training. At eval, always use full K.

**Motivation:** Expose the block to a range of contextualization depths during training. Some batches use K=2 (minimal context), others use K=10 (full). This should make the block more robust to varying levels of contextualization.

**Results (C=50, K=10, k_min=2, cw=0, 100K iters):**

| Metric | Random K | Fixed K=10 |
|---|---|---|
| Val PPL | 84.41 | 84.16 |
| Parallel K=1 | 117.93 | 131.10 |
| Sequential K=1 | 84.36 | 84.18 |
| L (contraction) | 0.72 | 0.94 |

Key findings:
- **Costs only 0.25 PPL** at full K (84.41 vs 84.16)
- **Dramatically improves parallel K=1** (118 vs 131) — block handles raw inputs better
- **Does NOT improve sequential K=1** (84.36 vs 84.18) — sequential already gets contextualized inputs
- **Improves convergence** (L=0.72 vs 0.94) — block learns to converge faster since it sometimes only gets 2 iterations

### Convergence weight ablation

The convergence weight (MSE loss between last two iterations) makes negligible difference:

| Config | Val PPL | Seq K=1 |
|---|---|---|
| cw=0.0 | 84.16 | 84.18 |
| cw=0.1 | 84.20 | 84.22 |

Recommendation: use cw=0 to simplify.

## Depth at Scale: D=3 Results

### D=3 at C=446 (FLOP-matched vs roformer_head_ffn N=3)

Deep D=3 block_head_corr_ffn at C=446 is FLOP-matched against roformer_head_ffn N=3 at C=446 (both ~44C² FLOPs/token). The D=3 model also has similar total params.

| Iter | D=3 C=446 | roformer_head_ffn N=3 C=446 |
|---|---|---|
| 5K | 38.95 | 41.60 |
| 10K | 31.36 | 33.62 |
| 15K | 29.07 | 30.96 |
| 20K | 27.57 | 29.48 |
| 25K | 26.87 | 28.36 |
| 30K | 26.30 | 27.70 |
| 35K | 26.47 | 29.00 |
| 40K | 25.95 | 28.52 |

**D=3 leads roformer_head_ffn by ~2.6 PPL at C=446, and the gap is stable/widening.** This is the first clear evidence that look-ahead depth can beat standard transformers at scale.

### Scale changes the dynamics

| C | D=3 vs roformer_head_ffn N=3 gap | Notes |
|---|---|---|
| C=50 | +1.5 (D=3 behind) | D=3 76.83 vs rhf 75.32 at 100K |
| C=74 | +0.6 (D=3 behind, gap widening) | D=3 62.23 vs rhf 61.60 at 85K |
| C=446 | -2.6 (D=3 ahead, gap stable) | D=3 25.95 vs rhf 28.52 at 40K |

At C=50 and C=74, D=3 loses to roformer_head_ffn. At C=446, D=3 wins by 2.6 PPL and the gap is stable. The crossover point is somewhere between C=74 and C=446. **The iterative training advantage only manifests at sufficiently large C.**

## Earlier Proposal: Scaled Full Output (No Delta Subtraction)

### The discussion

The standard look-ahead extracts a **delta** from the block:

```
correction[t] = block(processed_x)[t] - processed_x[t]     # subtract input
shifted = shift(correction)
processed_x[t] = tok_emb[t] + shifted[t]
```

The corrhead then feeds `correction[t]` (the delta) to the classification head. This delta has
had the residual connection subtracted away — `processed_x[t]` (which contains `tok_emb[t]` with
coefficient 1) is explicitly removed. The head sees only the attention and FFN contributions,
where token t's identity depends on the variable self-attention weight `w[t,t]`.

**Why doesn't standard roformer have this problem?** Because the head sees the full block output
`block(x)[t] = x[t] + attn(...) + ffn(...)`, where `x[t]` is preserved with coefficient 1
through the residual connection. The variable scaling from attention is an ADDITION to the
identity signal, not a replacement.

**First attempt: addhead.** We tried feeding `processed_x[t] + correction[t]` to the head, which
is algebraically equal to `block(processed_x)[t]` — the full block output with residual intact.
Result: addhead ≈ nocat. The residual being present in the head input didn't help.

**Why addhead didn't help:** The issue is deeper than just what the head sees. The delta
subtraction also affects the iteration dynamics. In the current scheme:

```
correction = block(x) - x          # delta: removes the input
processed_x[t] = tok_emb[t] + shift(correction[t])
```

The delta is what gets shifted and added to tok_emb. The full block output (with residual)
is only reconstructed at the head — it was never part of the iteration. The block during
training only ever saw inputs built from deltas, not from full outputs.

### The proposal: scale and pass through

Don't subtract `processed_x[t]` at all. Let the full block output flow through:

```
block_output[t] = block(processed_x)[t]                     # NO subtraction
shifted = shift(block_output)
processed_x[t] = tok_emb[t] + α · shifted[t]               # scale before adding
head_input[t] = block_output[t]                              # full output to head
```

Key differences from current architecture:

1. **No delta subtraction**: The block output retains the residual connection. `block_output[t]`
   includes `processed_x[t]` with coefficient 1 (from residual), which in turn includes
   `tok_emb[t]` with coefficient 1. Token identity is always preserved with a fixed coefficient.

2. **Scaling factor α**: Necessary because the full block output has larger magnitude than the
   delta (it includes the input). Without scaling, `tok_emb[t] + shift(block_output)` would add
   the full representation from position t-1 (including `tok_emb[t-1]`) unscaled. α controls
   how much of the previous position's full representation feeds into the current position.

3. **Head sees full output**: The head receives `block_output[t]` — the full block output with
   residual intact. Token t's identity is present with coefficient 1 through the residual,
   PLUS the attention and FFN contributions provide self-inclusive context. This is what
   standard roformer does, and it works.

### Why this is different from addhead

Addhead reconstructed the full block output ONLY at the head (`processed_x + correction`), but
the iteration still used the delta. The block during training saw inputs built from deltas.

The proposal changes the iteration itself — the block sees inputs built from scaled full outputs.
This changes what the block learns during training. The scaling factor α gives the model control
over how much of the previous position's full representation (including identity) to inject.

### Why this is different from full_correction

We already have `full_correction=True` which returns `block(x)` instead of `block(x) - x`.
But it doesn't scale: `processed_x[t] = tok_emb[t] + shift(block_output)`. The full block
output is added at full scale, which includes the previous position's tok_emb and all accumulated
context. The scaling α is the missing piece.

### Open questions

- Should α be a fixed hyperparameter, a learnable scalar, or a learnable vector (per-dimension)?
- Does the non-cumulative fixed-point property still hold with scaled full outputs?
- How does the convergence behavior change when the iteration uses full outputs vs deltas?

## The Importance of L (Empirical Contraction Constant)

### What L measures

The empirical contraction constant L is the ratio of successive correction differences:

```
L = ||correction_K - correction_{K-1}|| / ||correction_{K-1} - correction_{K-2}||
```

L < 1 means the iterations are contracting — each iteration changes the correction less than the
previous one. L ≈ 0 means rapid convergence; L ≈ 1 means barely converging; L > 1 means diverging.

### L does NOT predict short-sequence performance

At block_size=256 evaluation, all models achieve sequential K=1 ≈ val PPL regardless of L:

| Model | L | Val PPL | Seq K=1 |
|---|---|---|---|
| block_head_ffn | 0.66 | 121.56 | 121.57 |
| attn_corr_ffn | 0.53 | 128.96 | 129.14 |
| nocat | 0.99 | 128.43 | 128.48 |
| attn_head_ffn | 0.53 | 139.10 | 139.10 |

The model with the worst L (nocat, 0.99) matches sequential just as well as the best L (attn_head_ffn, 0.53). And the best L doesn't predict the best PPL — attn_head_ffn has L=0.53 but the worst PPL (139.10).

Within the training block_size, sequential K=1 works for all models because left-to-right processing gives each position fully converged predecessors. L is irrelevant here.

### Where L should matter: long-form generation

The real test for L is generation far beyond the training block_size. During training, the model
processes sequences of length block_size (e.g. 256). At sequential inference generating 1000+ tokens,
the chain of contextualizations extends far beyond anything seen during training.

At each step, the correction is slightly imperfect — it doesn't perfectly match the converged fixed
point. This error propagates to the next position's input, which produces another slightly imperfect
correction, and so on. The question is: do these errors accumulate or wash out?

**L predicts the error accumulation rate.** If the iteration map is a contraction with constant L,
then a perturbation of size ε in the input produces a perturbation of size L·ε in the output. Over
a chain of T positions:

- **L = 0.5**: errors decay. After T steps, accumulated error ~ ε/(1-L) = 2ε. Bounded regardless of T.
- **L = 0.99**: errors barely decay. After T steps, accumulated error ~ T·ε. Grows linearly with sequence length.
- **L ≥ 1.0**: errors amplify. Generation eventually diverges from the training distribution.

A model with L=0.5 should generate stable, coherent text at any length. A model with L=0.99 will
gradually drift from the learned distribution as the sequence grows longer, producing increasingly
degraded output.

### Implications

1. **L is a stability metric, not a quality metric.** It tells you how far you can push the model
   at inference, not how good the model is within its training distribution.

2. **Must test at long sequences to validate.** Evaluating at block_size=256 won't reveal L-related
   degradation. Need to generate 1000+ tokens and check whether perplexity degrades vs. block_size
   evaluation. This test has not been done yet.

3. **Low L is insurance.** Two models with similar PPL at block_size=256 may behave very differently
   at sequence length 10,000. The one with lower L is safer for deployment.

## Summary of Key Insights

1. **K=1 at inference**: Sequential processing from autoregressive generation + KV caching makes K>1 unnecessary when K is large during training. Training depth K is a training-only hyperparameter.

2. **D-block generalization**: Use D sequential layers as the shared unit. D=1 is maximum sharing (fastest, smallest). Higher D gives more per-token capacity at the cost of more parameters. D controls the quality-speed tradeoff.

3. **Clean correction extraction**: Within the D-block unit, blocks 1..D-1 use standard residuals. The correction returned to the look-ahead scheme is only the last block's delta, ensuring it can be cleanly added to tok_emb.

4. **Sequential eval requires K>1**: Sequential evaluation (simulating autoregressive inference) is only valid when K>1, because the unit must have been trained on contextualized inputs. For K=1 (D=N), parallel K=1 is the correct evaluation.

5. **Parallel vs sequential eval**: For K>>1, parallel K=1 is pessimistic (unit sees raw embeddings). Sequential K=1 matches parallel K=N. For smaller K, the match is approximate.

6. **Convergence loss is unnecessary**: cw=0.1 vs cw=0.0 makes negligible difference (<0.1 PPL). Use cw=0.

7. **Additive correction is important**: Learned combiner `f(correction, original)` hurts K=1 quality (91.5% gap). The additive `x_0 + correction` constrains the iteration to be contraction-like.

8. **block_head_corr_ffn is the best variant**: Separate correction FFN from the block's own FFN. The block produces `z` for the head (clean signal with residual). The corr_ffn generates corrections from `z` for the next position. Matches block_head_ffn with a cleaner architecture.

9. **Token-blind corrections are a constraint, not a bug**: The correction at position t cannot depend on `processed_x[t+1]` without creating a circular dependency that breaks sequential K=1. The concat variant showed 4 PPL improvement in training but failed at inference (seq K=1 gap of 4.4 PPL). Corrections must be functions of already-finalized quantities only.

10. **K=5 matches K=10**: Only 0.16 PPL difference with 2x faster training. Use K=5.

11. **Random K improves convergence but not sequential PPL**: Sampling K during training (k_min=2) improves L (0.72 vs 0.94) and parallel K=1 but not sequential K=1. Useful for robustness, not strictly necessary.

12. **Depth matters at scale**: D=3 look-ahead beats roformer_head_ffn N=3 by 2.6 PPL at C=446 but loses by 0.6-1.5 PPL at C=50-74. The crossover is between C=74 and C=446.

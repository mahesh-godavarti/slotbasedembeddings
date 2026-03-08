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

### Head variants

The look-ahead architecture produces two signals at each position t:
- `processed_x[t] = tok_emb[t] + shifted_correction[t-1]` — past-only context, clean token identity
- `correction[t]` — self-inclusive context (attention sees positions 0..t), noisy token identity

Different head variants choose what to feed the classification head:

| Head | Input | Params (C=50, V=16000) | Token identity | Context |
|---|---|---|---|---|
| nocat | `processed_x[t]` | 1,646,750 | Clean (coefficient 1) | Past-only |
| corrhead | `correction[t]` | 1,646,750 | Variable scaling | Self-inclusive |
| addhead | `processed_x[t] + correction[t]` | 1,646,750 | Clean (coefficient 1) | Both |
| projhead | `Linear(2C→C)([proc; corr])` | 1,651,800 | Learned | Both |
| concat | `[proc; corr]` → `Linear(2C, V)` | 2,446,850 | Both signals | Both |

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

### Comparison

| Variant | Iteration block | Correction | Head input | Extra params | FFN in loop? |
|---|---|---|---|---|---|
| 1: Attn + Corr FFN | attention only | ffn(y) | y (attn output) | none | yes (corr FFN) |
| 2: Attn + Head FFN | attention only | attn delta | ffn(y) | none | no |
| 3: Block + Head FFN | attn + FFN | block delta | extra_ffn(block_out) | ~8C² + 5C | standard FFN yes |

All three give the head a signal with stable token identity (coefficient 1 from residual).

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

## Summary of Key Insights

1. **K=1 at inference**: Sequential processing from autoregressive generation + KV caching makes K>1 unnecessary when K is large during training. Training depth K is a training-only hyperparameter.

2. **D-block generalization**: Use D sequential layers as the shared unit. D=1 is maximum sharing (fastest, smallest). Higher D gives more per-token capacity at the cost of more parameters. D controls the quality-speed tradeoff.

3. **Clean correction extraction**: Within the D-block unit, blocks 1..D-1 use standard residuals. The correction returned to the look-ahead scheme is only the last block's delta, ensuring it can be cleanly added to tok_emb.

4. **Sequential eval requires K>1**: Sequential evaluation (simulating autoregressive inference) is only valid when K>1, because the unit must have been trained on contextualized inputs. For K=1 (D=N), parallel K=1 is the correct evaluation.

5. **Parallel vs sequential eval**: For K>>1, parallel K=1 is pessimistic (unit sees raw embeddings). Sequential K=1 matches parallel K=N. For smaller K, the match is approximate.

6. **Convergence loss helps**: MSE between last two iterations' outputs (weight 0.1) improves L with minimal PPL cost.

7. **Additive correction is important**: Learned combiner `f(correction, original)` hurts K=1 quality (91.5% gap). The additive `x_0 + correction` constrains the iteration to be contraction-like.

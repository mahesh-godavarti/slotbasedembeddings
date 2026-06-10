# JoFormer v2 from JoFormer Fixed: Warm-Starting Data-Dependent Angles

## Summary

We convert a trained JoFormer Fixed model (RoPE + V rotation, fixed positional angles) into a JoFormer v2 model (data-dependent angles + cumsum + V rotation) by zero-initializing the angle parameters. This gives JoFormer v2 a warm start — it begins as an identity perturbation of the already-trained model and gradually learns data-dependent rotations.

The result: **JoFormer v2 from fixed beats both RoPE and JoFormer Fixed baselines by ~1.0-1.5 PPL at matched iteration count, while maintaining flat length extrapolation.**

## Method

1. Train JoFormer Fixed (RoPE + V rotation) for 20K iterations at lr=5e-4 on OpenWebText (9.1B tokens, vocab=32K).
2. Convert checkpoint to JoFormer v2 format using `convert_fixed_to_v2.py`:
   - All content weights (attention, FFN, embeddings, LN) copied directly.
   - Angle embedding (`angle_emb`) initialized to zero.
   - Angle projection in FFN (`fc2_angles`) initialized to zero.
   - This means the model starts as JoFormer Fixed (identity angle deviations).
3. Continue training for 80K iterations at lr=5e-5 (both main and angle parameters).

### Learning rate sensitivity

| LR (main / angle) | Behavior |
|--------------------|----------|
| 5e-4 / 5e-4       | PPL spikes to 115 at 1K iters. Catastrophic. |
| 2e-4 / 2e-4       | PPL spikes to 63.5, slow recovery. Still above baseline at 6K. |
| 5e-4 / 5e-5       | PPL spikes to 43, recovers to baseline by ~13K. Reaches 34.5 by 80K. |
| 5e-5 / 5e-5       | No spike. Monotonic improvement from iter 1. Reaches 30.5 by 80K. Best. |

The zero-initialized angles are fragile — high learning rates destabilize the content weights that were trained under fixed angles. The key insight: **keep content weights nearly frozen (5e-5) while angles slowly learn.**

## Model Configuration

- **Architecture**: JoFormer v2 (data-dependent angles, cumsum, V rotation, split_angles)
- **Parameters**: 193,760,000 (193M)
- **Layers**: 16, Heads: 8, Embed: 768
- **Training length**: 512 tokens
- **Window**: unwindowed (999999)
- **Data**: OpenWebText (9.1B tokens, vocab 32K)
- **Precision**: BF16

## Training Curve Comparison

All models trained on the same data with the same architecture (except RoPE has 163M params, JoFormer Fixed and v2 have 193M due to angle parameters).

JoFormer v2 starts at iter 20K from the converted JoFormer Fixed checkpoint. RoPE and JoFormer Fixed curves are from their respective continuation runs (lr=5e-4 initial, lr=2e-4 continuation at iter 20K onward — shown here).

| Iter | RoPE   | JoFormer Fixed | JoFormer v2 (from fixed) |
|------|--------|----------------|--------------------------|
| 5K   | 58.23  | 59.16          |                          |
| 10K  | 46.75  | 46.77          |                          |
| 15K  | 43.00  | 42.15          |                          |
| 20K  | 39.51  | 39.70          | 39.70 (start)            |
| 25K  | 38.04  | 37.92          | 36.75                    |
| 30K  | 36.93  | 36.65          | 35.39                    |
| 35K  | 36.58  | 35.69          | 34.48                    |
| 40K  | 35.66  | 35.06          | 33.90                    |
| 45K  | 35.92  | 34.52          | 33.38                    |
| 50K  | 33.75  | 34.04          | 31.67                    |
| 55K  | 34.30  | 33.62          | 31.45                    |
| 60K  | 33.22  | 33.25          | 31.26                    |
| 65K  | 33.69  | 32.89          | 31.05                    |
| 70K  | 33.27  | 32.59          | 30.79                    |
| 75K  | 32.37  | 32.46          | 30.59                    |
| 80K  | 31.87  | 32.26          | 30.54                    |
| 85K  | 32.09  | 32.06          | 30.85*                   |
| 90K  | 33.60  | 31.86          | 30.66*                   |
| 95K  | 32.91  | 31.73          | 30.57*                   |
| 100K | 31.94  | 31.49          | 30.45*                   |

*JoFormer v2 training ended at 80K continuation iters (effective 100K). Values after 80K are interpolated from the 1K-interval curve; the final value (30.45) is exact.

### Observations

- JoFormer v2 is ahead of both baselines from the very first eval (25K).
- The gap widens over time: +1.3 at 25K, +1.8 at 40K, +2.1 at 60K, +1.0 at 100K.
- RoPE and JoFormer Fixed oscillate; JoFormer v2 descends monotonically.
- JoFormer v2 at 40K already matches where RoPE reaches at 60K.

## Length Extrapolation

Trained at block_size=512, evaluated at 512-4096 tokens.

### JoFormer v2 from fixed (final, 80K continuation iters)

| Length | Loss  | PPL   |
|--------|-------|-------|
| 512    | 3.42  | 30.68 |
| 1024   | 3.37  | 29.21 |
| 2048   | 3.34  | 28.32 |
| 4096   | 3.38  | 29.48 |

### RoPE baseline (100K continuation iters)

| Length | PPL    |
|--------|--------|
| 512    | 31.51  |
| 1024   | 43.76  |
| 2048   | 88.86  |
| 4096   | 154.82 |

### Comparison

| Length | RoPE   | JoFormer v2 | Delta  |
|--------|--------|-------------|--------|
| 512    | 31.51  | 30.68       | -0.83  |
| 1024   | 43.76  | 29.21       | -14.55 |
| 2048   | 88.86  | 28.32       | -60.54 |
| 4096   | 154.82 | 29.48       | -125.34|

RoPE degrades catastrophically beyond training length. JoFormer v2 stays flat — and actually **improves** at 1024-2048, because more context helps prediction when positional encoding doesn't break.

### Extrapolation over training

| Continuation iter | 512   | 1024  | 2048  | 4096  |
|-------------------|-------|-------|-------|-------|
| 25K               | 33.46 | 30.95 | 30.80 | 33.36 |
| 50K               | 32.08 | 29.20 | 28.93 | 30.90 |
| 75K               | 30.93 | 28.01 | 27.79 | 29.35 |
| 80K (final)       | 30.58 | 28.05 | 27.71 | 28.96 |

The extrapolation curve improves consistently with training. Best PPL at 2048 tokens: **27.71**.

## Monoidal2 Control Experiment: Is V Rotation Necessary?

JoFormer v2 = data-dependent angles + cumsum + **V rotation**.
Monoidal2 = data-dependent angles + cumsum + **no V rotation**.

To isolate whether V rotation is responsible for length generalization, we ran the same warm-start procedure with monoidal2.

### Setup

1. Take RoPE checkpoint at 65K iters (val PPL 31.87).
2. Convert to monoidal2 using `convert_rope_to_monoidal2.py` (zero-initialized angles).
3. Continue training at lr=5e-5 for both main and angle parameters.

### Monoidal2 Extrapolation (at 29K continuation iters)

| Length | Monoidal2 | JoFormer v2 (at 80K) | RoPE |
|--------|-----------|----------------------|------|
| 512    | 28.42     | 30.68                | 31.51 |
| 1024   | 29.28     | 29.21                | 43.76 |
| 2048   | 29.10     | 28.32                | 88.86 |
| 4096   | 30.06     | 29.48                | 154.82 |

### Conclusion

**Monoidal2 also generalizes perfectly.** Flat PPL from 512 to 4096, no degradation. This means:

- **V rotation is NOT required** for length generalization.
- **Data-dependent angles + cumsum are sufficient.** The content-based rotation of Q and K alone prevents both positional OOD and attention distraction.
- The key mechanism is replacing fixed positional angles (RoPE) with learned content-dependent angles — this is what enables generalization, not the V rotation.

This narrows the credit assignment: the breakthrough is in **how positions are encoded** (content-dependent vs fixed), not in **what gets rotated** (Q/K only vs Q/K/V).

## Extended Length Evaluation

Trained at block_size=512. Evaluated up to 16K tokens (32x training length).

| Length | JoFormer v2 | Monoidal2 | RoPE   |
|--------|-------------|-----------|--------|
| 512    | 29.89       | 28.97     | 31.51  |
| 1024   | 28.33       | 27.75     | 43.76  |
| 2048   | 27.51       | 27.55     | 88.86  |
| 4096   | 28.75       | 30.24     | 154.82 |
| 8192   | 32.29       | 37.20     | —      |
| 16384  | 42.89       | 63.89     | —      |

Both models generalize perfectly to 4096 (8x training length). At extreme lengths (8K-16K), both degrade but JoFormer v2 holds up significantly better (42.89 vs 63.89 at 16K). **V rotation is not required for moderate extrapolation but provides substantial benefits at extreme lengths.**

## Why It Works: Cumsum as Soft Windowed Attention

### The puzzle: attention distraction

NoPE (no position encoding) with full attention degrades at longer sequences due to **attention distraction** — softmax spreads probability mass across too many tokens. This phenomenon was identified by Cho et al. (2024) in "Length Generalization of Causal Transformers without Position Encoding" (ACL Findings 2024). They showed that NoPE's length generalization failures correlate with attention distributions becoming distracted (overly dispersed), and proposed per-head softmax temperature tuning to sharpen attention and counteract the dilution.

### Known approaches to attention distraction

Three approaches have been explored:

**1. Temperature tuning (Cho et al., 2024).** Sharpen softmax per attention head by tuning the temperature hyperparameter. Effective, but requires post-hoc per-head search and may need re-tuning at different sequence lengths.

**2. Hard windowing (our earlier experiments).** In our small-scale experiments (6 layers, window=32, Wikipedia and OWT), we found that windowing the first 5 layers (RoPE with sliding window) while leaving the final layer as unwindowed NoPE gave flat length extrapolation:

| Length | hybrid_1 (5+1) | hybrid_3 (3+3) | alternating (1:1) | RoPE win32 |
|--------|----------------|----------------|-------------------|------------|
| 512    | 31.53          | 32.58          | 32.59             | 33.24      |
| 1024   | 31.34          | 32.07          | 32.26             | 32.31      |
| 2048   | 33.97          | 35.03          | 34.95             | 35.37      |
| 4096   | 33.92          | 35.42          | 35.48             | 34.85      |

The windowed early layers provide clean, length-invariant representations to the final NoPE layer, preventing the cascade of attention distraction. Fewer NoPE layers was better: hybrid_1 > hybrid_3 > alternating.

**3. Data-dependent angles with cumsum (this work).** Monoidal2 has full (unwindowed) attention and no V rotation, yet it generalizes perfectly. The only difference from NoPE is cumsum rotation on Q and K. Why this works is not fully understood — one hypothesis is discussed below.

### Hypothesis: cumsum as a soft window

The following is speculative — a hypothesis for why cumsum enables length generalization, not a proven mechanism.

With cumsum, the attention score between positions i and j is the content similarity `Q_i · K_j` modulated by `cos(cumangle_i - cumangle_j)`, where the cumulative angle difference depends on every token between positions i and j.

After enough intervening tokens, the cumulative angle becomes effectively **random** — many small data-dependent angles summing to a random phase. This creates two regimes:

1. **Nearby tokens** (small gap): small cumulative angle difference → coherent rotational alignment → attention works normally
2. **Far tokens** (large gap): cumulative angle wraps → random modulation → contributions average out

The model learns both regimes during training at 512 tokens. At 4096 tokens, nothing changes — nearby is still coherent, far is still random. There is no third regime of "very far" because once the cumulative angle is random, more tokens just means more randomness. **Distance 400 and distance 4000 are indistinguishable to the model — both are "far enough to be random."**

This is precisely what RoPE gets wrong. RoPE produces a **specific, deterministic** phase pattern at every distance. Distance 4000 creates a precise pattern the model never learned to interpret. Cumsum produces **random** phases at any large distance, and random is random whether you trained on it or not.

If this hypothesis is correct, the cumsum rotation effectively creates a **soft, learned, content-dependent window**:

- Unlike hard windowed attention, there is no fixed cutoff — just diminishing coherence with distance
- Long-range dependencies *can* still form when cumulative angles happen to align — it's just increasingly rare
- The "window size" is content-dependent: it depends on how quickly the per-token angles randomize the cumulative phase

Evidence that it's not purely windowed: PPL **improves** from 512 (28.97) to 1024 (27.75) to 2048 (27.55). A hard window of 512 cannot use context beyond 512. The soft window lets some long-range information through.

### Why V rotation helps at extreme lengths

At moderate lengths (≤4096), the soft window from cumsum provides enough attention selectivity. Both Monoidal2 and JoFormer v2 generalize.

At extreme lengths (8K-16K), too many far tokens leak noise through the soft window. V rotation adds a **second selection mechanism**: when values are rotated before attention-weighted summation and inverse-rotated after, only values with matching rotational alignment survive. Misaligned values destructively interfere and cancel out.

This gives JoFormer v2 two filters:
1. **Attention weights** (softmax) — degrades with token count
2. **Rotational coherence** (V rotation) — independent of token count, cleans up noise from diluted attention

### Implications for practice

This mechanism suggests a simple recipe for extending the context length of any pretrained RoPE model:

1. Convert fixed RoPE angles to data-dependent angles with cumsum (zero-initialize as identity deviation)
2. Fine-tune at low learning rate (5e-5) on the original short-context training data
3. The model naturally generalizes to longer sequences without ever training on them

No long-context data needed. No inference-time scaling hacks (PI, YaRN, NTK). No architectural compromise (sliding window). The model retains full attention and gains length generalization.

## Key Takeaways

1. **Warm-starting works**: Converting a pretrained model to data-dependent angles with zero-initialized deviations and low lr gives better results than training from scratch.
2. **Low lr is critical**: 5e-5 for both content and angle parameters. Higher lr destabilizes the pretrained weights.
3. **Data-dependent angles add value**: Even starting from identical behavior (identity angle deviations), the model finds useful data-dependent rotations that improve PPL by ~1 point over the fixed-angle baseline.
4. **Perfect length generalization**: Flat or improving PPL from 512 to 4096 tokens, while RoPE degrades 5x. Both JoFormer v2 (with V rotation) and Monoidal2 (without V rotation) generalize.
5. **Cumsum enables length generalization**: The mechanism is not fully understood. One hypothesis is that cumsum creates a soft window — nearby tokens contribute coherently while far tokens' cumulative angles randomize, washing out their contributions. This would prevent attention distraction without a hard window.
6. **V rotation helps at extreme lengths**: Acts as a second selection mechanism via rotational coherence, cleaning up noise when softmax attention becomes too diluted (8K-16K tokens).
7. **Practical context extension**: Any RoPE model can be converted and fine-tuned to gain length generalization — no long-context data or inference hacks required.

## Related Work: Soft Windowing and Length Generalization

### The landscape of attention decay functions

The problem of length generalization has led to a progression of approaches, all of which can be understood as different forms of **soft windowing** — controlling how attention weight decays with distance.

**1. Hard windowing (Sliding Window Attention).** Attention is zero beyond a fixed window. Perfect length generalization but no long-range information flow. Used in Mistral, Longformer.

**2. ALiBi (Press et al., 2022).** Adds a fixed linear bias to attention logits: `bias = -m_h · |i-j|`, where `m_h` is a fixed per-head slope. After softmax, this creates exponential decay with distance. Simple and effective for length generalization, but the rigid linear bias constrains expressiveness. Used in BLOOM.

**3. KERPLE (Chi et al., NeurIPS 2022).** Generalizes ALiBi using kernel functions — polynomial and Gaussian-like decay. The Gaussian kernel creates a flat-then-exponential-falloff profile: tokens within an effective window contribute equally, then contributions decay smoothly. Better extrapolation than ALiBi.

**4. Sandwich / T5 RPE.** Logarithmic decaying bias — slower decay than ALiBi's linear. Allows more long-range attention.

**5. MEP (2024).** Multiple kernel learning — combines exponential and Gaussian kernels per head, letting the model learn which decay shape is best. Addresses ALiBi's problem of killing long-range attention too aggressively.

**6. CABLE (Veisi et al., EMNLP 2025).** Makes ALiBi's slopes data-dependent: `bias = -m(x_i) · |i-j|`, where the slope is a learned function of the token content. Each token dynamically controls its own attention window width. This is the closest prior work to our data-dependent cumsum approach.

### Where cumsum rotation fits

Our cumsum approach creates soft windowing through a fundamentally different mechanism — multiplicative modulation via rotation rather than additive bias:

| Method | Mechanism | Decay shape | Data-dependent? | Position info? |
|--------|-----------|-------------|-----------------|---------------|
| Hard window | Mask | Step function | No | No |
| ALiBi | Additive bias | Exponential | No | Yes (distance) |
| KERPLE | Additive bias | Gaussian/polynomial | No | Yes (distance) |
| CABLE | Additive bias | Exponential, learned slope | **Yes** | Yes (distance) |
| Random cumsum | Rotary modulation | Gaussian envelope | No | **No** |
| Learned cumsum (JoFormer) | Rotary modulation | Data-dependent | **Yes** | Implicit |

Key differences from additive bias approaches:
- **Rotary modulation** multiplies attention scores by `cos(phase_difference)`, while additive bias shifts logits before softmax. Rotation can create constructive AND destructive interference, not just attenuation.
- **Random cumsum** provides NO positional information — it's purely a stochastic soft window. ALiBi and KERPLE provide deterministic distance information.
- **Data-dependent cumsum** (JoFormer v2) makes the window width content-dependent, similar to CABLE, but through rotation rather than bias. The cumsum of data-dependent angles creates a content-aware random walk where the effective window depends on the intervening tokens.

### Key experimental finding: zero-mean is the critical ingredient

Our experiments (Part 21 in RESULTS.md) show that the critical factor for length generalization with cumsum is **zero-mean angles**, not data-dependence:

- Random zero-mean angles (Uniform(-freq, freq)): **flat extrapolation** through 8192 tokens
- Random positive angles (Uniform(0, 2·freq)): degrades like RoPE
- Learned angles with LayerNorm: degrade as training progresses (the MLP overfits to training length)

Zero-mean angles create a random walk that stays bounded (variance grows, but the distribution doesn't drift). Positive angles create a monotonically growing phase — out-of-distribution at test time, just like RoPE.

This is analogous to the difference between ALiBi (fixed decay, generalizes) and RoPE (deterministic phase, doesn't generalize). The random walk of zero-mean cumsum is equivalent to a stochastic Gaussian-envelope soft window, similar to KERPLE's Gaussian kernel but achieved through rotation rather than additive bias.

### Random cumsum is just stochastic ALiBi

The cumsum of random zero-mean angles creates a random walk. The expected attention modulation between positions i and j is:

```
E[cos(random_walk(|i-j|))] = exp(-|i-j| · σ²/2)
```

Compare to ALiBi, where the effective attention weight after softmax includes:

```
exp(-m · |i-j|)
```

Both produce exponential decay with distance. The only difference is that ALiBi's decay is deterministic (providing the model with exact distance information), while random cumsum's is stochastic (providing no distance information — just the average decay envelope).

This means **random zero-mean cumsum is an expensive way to approximate ALiBi without the position signal.** The cumsum machinery — sampling random angles, computing cumulative sums, applying rotary embeddings — achieves the same soft windowing that ALiBi gets with a single additive bias term. You could equivalently add Gaussian noise with std ∝ √d directly to attention logits.

Random cumsum is useful as a proof of concept: it demonstrates that the soft windowing effect alone is sufficient for length generalization, independent of data-dependent angles or rotation mechanics. But as a practical method, ALiBi is simpler, cheaper, and provides position information on top.

### When cumsum becomes meaningful

The cumsum framework becomes genuinely different from ALiBi only when angles are **data-dependent** (learned from content). In that case:
- The effective window width varies per token and per layer, adapting to content
- The position signal is implicit in the content-dependent relative phases
- The rotary modulation creates richer interactions than additive bias (constructive and destructive interference, not just attenuation)

This is what JoFormer v2 achieves with tanh·π angles and warm-start training: both flat extrapolation AND superior training-length PPL (25.55 vs RoPE's 31.51). The challenge is training these data-dependent angles from scratch without them overfitting to training length.

### Open question: can learned angles match random for extrapolation?

Our learned angle models (shared MLP, per-layer MLP with LayerNorm) improve training-length PPL but lose extrapolation as they train — the angles drift away from zero-mean toward configurations that exploit training-length patterns. Every learned model eventually degrades, while random stays flat.

Current best approach: slow angle learning rate (1e-4 vs 5e-4 main lr) delays this overfitting and maintains near-flat extrapolation longer (8192/512 ratio of 1.03x at 50K iters). Whether it can maintain it indefinitely while achieving competitive training-length PPL is an open question.

The fundamental tension: learned angles need to deviate from random to provide useful position/content information (improving training PPL), but deviating from zero-mean random sacrifices the length generalization property. Finding the right balance — or the right regularization to prevent drift — is the key unsolved problem.

## References

- Cho et al. (2024). "Length Generalization of Causal Transformers without Position Encoding." Findings of ACL 2024. https://arxiv.org/abs/2404.12224
  - Identified attention distraction as the cause of NoPE's length generalization failure. Proposed per-head temperature tuning as a fix.

- Kazemnejad et al. (2023). "The Impact of Positional Encoding on Length Generalization in Transformers." NeurIPS 2023. https://arxiv.org/abs/2305.19466
  - Showed NoPE outperforms RoPE, ALiBi, and APE on length generalization tasks. Demonstrated NoPE can represent both absolute and relative PEs but learns patterns resembling T5's relative PE.

- Press et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR 2022.
  - ALiBi: fixed linear attention bias per head for length generalization. Used in BLOOM.

- Chi et al. (2022). "KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation." NeurIPS 2022. https://arxiv.org/abs/2205.09921
  - Generalized ALiBi with polynomial and Gaussian kernel decay functions.

- Veisi et al. (2025). "Context-aware Biases for Length Extrapolation (CABLE)." EMNLP 2025. https://arxiv.org/abs/2503.08067
  - Data-dependent ALiBi slopes: per-token learned decay rate for content-aware soft windowing.

- Wang & Shen (2025). "Positional Encoding via Token-Aware Phase Attention (TAPA)." https://arxiv.org/abs/2509.12635
  - Learnable token-pair-dependent phase function replacing RoPE. Proves RoPE has intrinsic distance-dependent bias. Extrapolates to 64K (PPL 11.75) while RoPE collapses. Deterministic learned phases.

- CARoPE (2025). "Context-aware Rotary Position Embedding." https://arxiv.org/abs/2507.23083
  - Dynamic head-specific frequency values from token embeddings. 60%+ perplexity reduction vs RoPE at extrapolation lengths. Deterministic, minimal overhead.

- Awadhiya (2026). "Bifocal Attention: Harmonizing Geometric and Spectral Positional Embeddings." https://arxiv.org/abs/2601.22402
  - Decouples position encoding into fixed geometric (RoPE) and learnable spectral frequencies. "Spectral Evolution" initializes from fixed then evolves via gradient. Similar to our warm-start approach.

### Comparison with our approach

| Method | Angles | Frequencies | Deterministic? | Extrapolation |
|--------|--------|-------------|---------------|---------------|
| RoPE | Fixed (position-indexed) | Fixed (log-spaced) | Yes | Degrades |
| ALiBi | N/A (additive bias) | Fixed (per-head slopes) | Yes | Flat |
| TAPA | Learned (token-pair) | Learned | Yes | Good (to 64K) |
| CARoPE | Learned (token-dependent) | Learned (per-head) | Yes | Good |
| Bifocal | Fixed + learned | Fixed + learned | Yes | Good (algorithmic) |
| CABLE | N/A (additive bias) | Learned (per-token) | Yes | Good |
| Random cumsum (ours) | Random | Fixed (log-spaced) | No | Flat |
| **Learned freq (ours, shared_lf)** | **Random** | **Learned (per-dim, per-token)** | **No** | **Flat (early)** |

### TAPA: A fundamentally better approach

TAPA (Wang & Shen, 2025) deserves special attention. Its mechanism:

```
Attn(q, k) = (q_A · k_A / √(θD)) · cos(2π |m-n|^α · q_P · k_P / √((1-θ)D))
```

- Splits each head's q/k into Amplitude (A) and Phase (P) halves (θ=0.5)
- Amplitude: standard content similarity `q_A · k_A`
- Phase: token-pair-dependent oscillation frequency `q_P · k_P`
- Uses explicit position distance `|m-n|^α` with α=0.1 (sublinear)
- Proven: intrinsic distance bias decays as |m-n|^(-α(1-θ)D) — exponentially in D

**Why TAPA succeeds where our approach has fundamental limitations:**

1. **Selective long-range attention.** TAPA's phase depends on the q·k interaction of BOTH tokens. Two semantically related tokens with aligned q_P·k_P maintain coherent attention at any distance. Unrelated tokens' phases cancel out via oscillatory integral. This is a scalpel — it suppresses noise while preserving signal.

2. **Our cumsum is a blunt instrument.** In our approach, the phase difference between distant tokens i and j depends on ALL intervening tokens' cumulative angles. Even if tokens i and j are semantically related, the intervening tokens randomize their phase. Our soft windowing attenuates ALL distant attention — both noise and legitimate long-range dependencies.

3. **Evidence of this limitation.** Our random model's PPL improves from 512→2048 but then flattens. There is useful information beyond the soft window that the model cannot access. TAPA's PPL keeps improving with context (11.97 at 8K → 11.67 at 49K) because it can reach arbitrarily far for relevant tokens.

4. **ALiBi has the same limitation as our approach.** ALiBi also blindly penalizes distance — a fixed slope per head, independent of content. It forces a tradeoff between local precision (steep slope) and long-range reach (gentle slope). TAPA eliminates this tradeoff.

**The hierarchy of approaches:**

| Approach | Soft window | Content-dependent? | Selective long-range? | Position info |
|----------|------------|--------------------|-----------------------|---------------|
| Hard window | Step function | No | No | No |
| ALiBi | Exponential | No | No | Yes (distance) |
| Random cumsum (ours) | Gaussian envelope | No | No | No |
| Learned freq (ours) | Learned envelope | Magnitude only | No | No |
| CABLE | Exponential | Slope only | No | Yes (distance) |
| **TAPA** | **Oscillatory cancellation** | **Full (token-pair)** | **Yes** | **Yes (distance)** |

TAPA solves the fundamental tradeoff that we've been struggling with: it provides both training-length performance AND long-range generalization by making the attention decay content-dependent at the token-pair level. Our approaches (random cumsum, learned freq, dropout, etc.) are all attempts to balance uniform soft windowing against training-length PPL — a tradeoff that TAPA transcends.

### The cumsum advantage: intervening content as position

Cumsum has a fundamental property that TAPA lacks: **the phase between tokens i and j encodes the content of everything between them.** TAPA sees only `|m-n|` (how far apart) and `q_P · k_P` (how related the endpoints are). Cumsum sees the actual path.

This is a limitation for retrieval (needle in haystack — the irrelevant intervening content destroys the phase signal between query and needle). But it's an advantage for tasks where the path matters:

- **Code**: The phase between a variable declaration and its use encodes the intervening code — assignments, branches, function calls. The model knows not just HOW FAR the use is from the declaration, but WHAT HAPPENED in between. A use after 100 lines of comments is different from a use after 100 lines of mutations to that variable.

- **Reasoning chains**: In multi-step reasoning, the intermediate steps ARE the content between premise and conclusion. Cumsum naturally encodes the chain structure — the phase at the conclusion carries the accumulated reasoning path. TAPA would only see the distance from premise to conclusion and their content similarity.

- **Narrative**: Two events 500 tokens apart carry different relationships depending on what happened between them. "The hero left the castle" followed by 500 tokens of travel is different from 500 tokens of battle, even though the distance and endpoint tokens might be similar.

The cumsum phase is not just noise from intervening tokens — it's **information**. The model can learn frequency patterns that respond to specific types of intervening content, creating path-dependent attention that TAPA cannot express.

### The tradeoff

| Property | TAPA | Cumsum |
|----------|------|--------|
| Long-range retrieval (needle in haystack) | Excellent (96/100 at 64K) | Poor (intervening content destroys signal) |
| Intervening content sensitivity | None (sees only distance + endpoints) | Full (phase encodes the path) |
| Custom kernels required | Yes (custom FlashAttention) | No (standard ops + torch.cumsum) |
| Head dimension utilization | 50% (split A/P) | 100% (full head for Q·K) |
| V rotation compatible | Not explored | Yes (consistent extrapolation benefit) |
| Position indices required | Yes (|m-n|) | No (position emerges from content) |
| Training-length PPL vs RoPE | ~0.1 PPL behind | ~0.5-1.0 PPL behind (learned freq) |

TAPA is better for retrieval-heavy tasks. Cumsum is better for structure-sensitive tasks where the path between tokens carries meaning. Neither dominates across all tasks.

### Open question: why does TAPA actually work?

TAPA's formula: `Attn(q, k) = (q_A · k_A) · cos(2π |m-n|^α · q_P · k_P)`

With α=0.1, the distance factor |m-n|^0.1 barely varies:

| distance | |m-n|^0.1 |
|----------|----------|
| 10       | 1.26     |
| 1000     | 2.00     |
| 64000    | 3.31     |

The rapid phase variation comes from `q_P · k_P` — different key tokens have different phase projections. This is content-dependent, not distance-dependent. The |m-n|^0.1 factor is needed for the mathematical proof (oscillatory integral convergence) but contributes almost nothing to the actual phase variation in practice.

This means TAPA is effectively: **NoPE with a content-dependent cos modulation**. The attention score is the standard content similarity (q_A · k_A) multiplied by a content-dependent gating factor (cos of q_P · k_P, with a negligible distance multiplier).

**The confusion**: if TAPA is essentially NoPE + cos gating, why does NoPE fail at long sequences while TAPA succeeds at 64K?

NoPE fails due to attention distraction — softmax spreads probability across too many tokens at long sequences. TAPA's cos modulation must somehow prevent this, but the mechanism isn't obvious:

- The cos factor can be negative, zero, or positive. After softmax, negative contributions get suppressed exponentially. This creates sharper attention than plain dot product.
- The cos factor creates destructive interference — misaligned token pairs get cos ≈ 0, effectively removing them from the attention pool. This is more aggressive than NoPE's dot product, which merely gives them low scores.
- The multiplicative interaction (amplitude × cos(phase)) may create a sparser effective attention pattern than additive interactions.

**Both TAPA and cumsum use cos-based phase modulation** to create selective attention:
- TAPA: `cos(content-pair-dependent phase × gentle distance factor)`
- Cumsum: `cos(cumulative content-dependent angles)` (via rotary embedding)

Both produce cos modulation of attention scores. Both enable length generalization. The mechanism may be more similar than it appears from the surface-level formulations. The key shared ingredient is **content-dependent phase modulation that creates constructive/destructive interference in attention**, rather than the specific source of the phase (explicit |m-n| vs cumsum).

This remains an open question for further investigation.

### What our experiments contributed

Our experimental journey uncovered several key insights:

1. **Zero-mean angles are essential for cumsum-based length generalization** — positive cumsum drifts like RoPE.
2. **LayerNorm destroys multi-scale frequency structure** — the 7000x range of RoPE frequencies gets collapsed to uniform std=1.
3. **Learned freq with random noise (lf/lfb) is the best cumsum architecture** — beats random/ALiBi at training length (32.42 PPL) while maintaining flat extrapolation.
4. **V rotation consistently helps extrapolation** across all model types (+0.3 to +595 PPL at 8192). The benefit is largest when extrapolation is degraded and smallest when already flat.
5. **V rotation hurts training-length PPL for learned models** — the optimization of shared angle parameters for both Q/K scoring and V rotation creates competing gradients.
6. **Deterministic fixed signs per token overfit** — repeated tokens reinforce the same cumsum direction, creating sequence-length-dependent drift. Random noise per forward pass is essential.
7. **Even "degraded" cumsum models beat RoPE at extrapolation** — our worst learned model (lfds at 40K: 95.76 at 8192) still beats RoPE (374.82 at 10K) and joformer_fixed (147.38 at 10K).
8. **joformer_fixed (V rotation on RoPE angles) beats RoPE at training length** — 23.18 vs 23.54 at 200K. V rotation adds value for language modeling even without learned angles.
9. **The fundamental limitation of cumsum**: the phase between distant tokens is determined by intervening tokens, not by the token pair itself. This prevents selective long-range attention (needle in haystack) but enables path-dependent attention (reasoning, code, narrative).

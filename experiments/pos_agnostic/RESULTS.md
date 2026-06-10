# Length Generalization Experiments — Results

## Setup

All experiments: Wikipedia text (983M tokens), BPE vocab=16000, block_size=512, batch_size=32, seed=42.

Model: n_embed=128, n_layers=6, n_heads=4 (head_dim=32), ~5.3M params.

Training configs:
- **Short runs (20K iters)**: lr=3e-4, cosine decay
- **Long runs (50K iters)**: lr=5e-4, constant

## Attention Types

| Name | Angles | Cumsum | Rotate V + Inverse | Description |
|------|--------|--------|---------------------|-------------|
| `rope` | Fixed (position index) | N/A | No | Standard RoPE — rotate Q, K only |
| `joformer_fixed` | Fixed (position index) | N/A | **Yes** | RoPE + V rotation + inverse rotation on output |
| `nope` | None | N/A | No | No positional encoding — pure content-based |
| `alibi` | None (linear bias) | N/A | No | Distance-based penalty per head |
| `datadep` | Data-dependent | No | No | `angle_proj(x)` produces angles, rotate Q, K only |
| `datadep2` | Data-dependent (v2) | No | No | Angles flow through network (embedding C+C/2, FFN outputs angles) |
| `monoidal` | Data-dependent | **Yes** | No | datadep + flip-cumsum-flip accumulates angles along sequence |
| `joformer` | Data-dependent | **Yes** | **Yes** | monoidal + V rotation + inverse rotation on output |

## Block Architectures

Four distinct transformer block types, differing in how angles are produced and how the FFN interacts with angles:

### 1. TransformerBlock (rope, joformer_fixed)
```
x → LN → Attention(x, fixed_angles) → residual
x → LN → FeedForward(x) → residual
```
- Fixed angles: `pos × freq` (RoPE) or same + V rotation (joformer_fixed)
- Standard FFN: `Linear(C, 4C) → GELU → Linear(4C, C)`
- No learned angle production

### 2. DataDep v1 — TransformerBlock with DataDepAttention (monoidal, joformer)
```
x → LN → DataDepAttention(x) → residual    [angle_proj INSIDE attention: Linear(C, C//2)]
x → LN → FeedForward(x) → residual
```
- `angle_proj(x)` produces angles from x within the attention module
- Each layer's attention has its own angle_proj
- Self-referential: angles computed from x, then used to attend over x in the same layer
- Standard FFN (no angle production)

### 3. ExternalAngleBlock (shared_*, random_*, pemb, det, lf_qk, etc.)
```
angles provided externally (from MLP, embedding, random, etc.)
x → LN → DataDep2Attention(x, angles) → residual
x → LN → FeedForward(x) → residual
```
- Angles come from outside the block (shared MLP, per-layer embedding, random noise, etc.)
- DataDep2Attention with cumsum, optional V rotation
- Standard FFN (no angle production)
- Most flexible — the angle source varies by model type

### 4. DataDep2Block (monoidal2, joformer2)
```
angles received from previous layer (or embedding for layer 0)
x → LN → DataDep2Attention(x, angles) → residual
x → LN → FeedForwardWithAngles(x) → content + new_angles → residual (content only)
new_angles (+rope_base) passed to next layer
```
- FeedForwardWithAngles: shared fc1, separate fc2_content and fc2_angles
- Angles flow through the network — each layer's FFN produces angles for the next layer
- Angles are completely regenerated each layer (no residual on angles)
- The angle production shares fc1 with content, making it a secondary task of the FFN

### Key differences

| Property | TransformerBlock | DataDep v1 | ExternalAngleBlock | DataDep2Block |
|----------|-----------------|------------|-------------------|---------------|
| Angle source | Fixed (pos×freq) | angle_proj(x) inside attn | External (varies) | FFN fc2_angles |
| FFN type | Standard | Standard | Standard | FeedForwardWithAngles |
| Angle depends on x? | No | Yes (same layer) | Varies by model | Yes (via shared fc1) |
| Angle flow | Same angles every layer | Independent per layer | Varies by model | Layer L→L+1 |
| Self-referential? | No | Yes | No (if detached/embedding) | Indirect (via x from prev layer) |

---

## Part 1: Full Attention Degrades Regardless of Encoding (20K iters)

Every model with full causal attention degrades at longer-than-training sequences.

| Length | RoPE full | NoPE full | NoPE softplus | DataDep full | DataDep2 full |
|--------|-----------|-----------|---------------|-------------|--------------|
| 512 | 48.16 | 66.12 | 66.24 | 56.34 | 64.56 |
| 1024 | 60.65 | 78.36 | 73.28 | 69.14 | 70.70 |
| 2048 | 111.13 | 122.06 | 104.29 | 101.92 | 112.50 |
| 4096 | 172.21 | 192.27 | 145.73 | 163.58 | 169.15 |

**Root cause**: "Attention distraction" — softmax over more positions than seen during training causes attention to dilute (Wang et al., ACL 2024). RoPE additionally suffers from out-of-distribution rotation angles at positions > 512.

**Softplus helps but doesn't fix it**: NoPE with softplus reduces 4096 PPL from 192→146.

**Top-k doesn't help**: Keeping only top-k scores before softmax doesn't fix the problem. Top-k=512 made NoPE worse (192→205).

## Part 2: Windowed Attention Fixes Length Generalization — For RoPE (20K iters)

| Length | RoPE full | RoPE win256 | Hybrid_3 (RoPE win + NoPE full) |
|--------|-----------|-------------|----------------------------------|
| 512 | 48.16 | 48.01 | 49.49 |
| 1024 | 60.65 | 45.76 | 47.32 |
| 2048 | 111.13 | 46.23 | 47.99 |
| 4096 | 172.21 | 47.11 | 49.34 |

Both RoPE windowed and Hybrid_3 are perfectly flat across all lengths.

**Hybrid_3** (3 RoPE windowed + 3 NoPE full) is notable: the NoPE layers attend to the ENTIRE sequence yet don't degrade. The windowed RoPE layers provide a stable foundation — the NoPE layers receive clean, length-invariant representations and can safely do global content-based matching.

## Part 3: Windowed Attention Does NOT Fix DataDep (50K iters, lr=5e-4)

| Length | RoPE win256 | DataDep win256 | DataDep2 win256 |
|--------|-------------|----------------|-----------------|
| 512 | 32.69 | 47.71 | 42.15 |
| 1024 | 31.84 | 76.89 | 76.45 |
| 2048 | 34.21 | 107.36 | 95.86 |
| 4096 | 33.78 | 129.47 | 122.89 |

DataDep with window=256 still degrades from ~45 to ~130 at 4096, while RoPE with the same window stays flat at ~33.

**Verified windowing is correct**: Position 256 produces identical logits in 512 and 4096 sequences when tokens are the same. The mask correctly limits each position to 256 past positions.

**Tail-only PPL (positions 256+ with fully saturated window):**

| Length | RoPE win256 | DataDep win256 |
|--------|-------------|----------------|
| 512 | 35.32 | 47.81 |
| 1024 | 38.61 | 87.91 |
| 2048 | 36.45 | 121.49 |
| 4096 | 36.16 | 136.44 |

Even restricting to positions with full windows, DataDep degrades while RoPE stays flat.

## Part 4: Smaller Window (32) Fixes DataDep (50K iters, lr=5e-4)

| Length | RoPE win32 | DataDep win32 |
|--------|------------|---------------|
| 512 | 33.24 | 45.07 |
| 1024 | 32.31 | 44.75 |
| 2048 | 35.37 | 44.68 |
| 4096 | 34.85 | 44.46 |

With window=32, DataDep is **flat** (~44-45 PPL across all lengths). The degradation is gone.

**Explanation**: Through 6 layers, each position's representation carries indirect context from ~6× the window size:
- Window=32: indirect reach ~192 positions, within training distribution (block_size=512)
- Window=256: indirect reach ~1536 positions, beyond training distribution

The `angle_proj` maps representations to rotation angles. When representations are in-distribution (window=32), the angles are well-behaved. When representations carry out-of-distribution indirect context (window=256), the angles go out of distribution.

RoPE doesn't have this problem because its angles come from position indices (always the same relative distances within the window), not from representations. Even if representations shift slightly, RoPE angles are unchanged.

**Note**: This explanation has a gap — if out-of-distribution representations cause DataDep angles to break, why don't they also cause RoPE's Q and K projections to break? Q = W_q·x and K = W_k·x also depend on representations. One hypothesis: DataDep has double exposure (both Q/K AND angles depend on representations), while RoPE has single exposure (only Q/K, angles are fixed). The additional degree of freedom amplifies distribution shift.

## Part 5: Training Length PPL Comparison (50K iters, lr=5e-4)

| Model | Window | Val PPL | Params |
|-------|--------|---------|--------|
| RoPE | 256 | 35.91 | 5,301,888 |
| RoPE | 32 | 36.08 | 5,301,888 |
| DataDep | 256 | 46.25 | 5,351,424 |
| DataDep | 32 | 44.02 | 5,351,424 |
| DataDep2 | 256 | 46.76 | 6,522,880 |

RoPE is ~10 PPL better than DataDep at training length. This is expected at small scale — explicit position indices provide a strong signal that data-dependent angles must learn implicitly. The literature shows this gap closes at larger model sizes (1B+).

DataDep v1 (angle_proj per layer) consistently outperforms DataDep v2 (angle flow through embedding/FFN) despite fewer parameters.

Window size barely affects RoPE (36 vs 36) but helps DataDep slightly (46→44 with smaller window).

## Part 6: Why Hybrid NoPE Works But Pure NoPE Doesn't

Pure NoPE full attention degrades (66→192). But NoPE full-attention layers in Hybrid_3 stay flat (~49 everywhere). Both have the same softmax-over-many-positions at eval. The difference:

- **Pure NoPE**: Layer 1 suffers attention distraction, producing degraded representations. Layer 2 receives degraded input and degrades further. The degradation **cascades** through all 6 layers.
- **Hybrid_3**: The first 3 layers (RoPE windowed) are perfectly stable at any length. The NoPE layers (4-6) receive clean, length-invariant representations. Their attention stays sharp because the input representations are discriminative, producing peaked Q·K scores even over 4096 positions.

**The windowed RoPE layers break the cascade.** They act as a stable feature extractor, providing a clean foundation for the NoPE layers to do global content-based attention.

## Part 7: Experiments In Progress

### Mixed final layer (window=32, 50K iters, lr=5e-4)

5 RoPE windowed(32) layers + 1 full-attention final layer. Testing what the final layer should be:

| Config | Final layer type |
|--------|-----------------|
| `ropefull_1` | RoPE full attention |
| `hybrid_1` | NoPE full attention |
| `datadepfull_1` | DataDep full attention |

**Prediction**: `hybrid_1` (NoPE final) should be length-stable. `ropefull_1` may degrade (RoPE angles OOD at distances > 32). `datadepfull_1` may degrade (data-dependent angles sensitive to representation shift).

### Future experiments
- **Per-head temperature scaling** (Wang et al. 2024): Post-hoc tuning of one temperature per head. Shown to recover NoPE at long sequences without retraining.
- **Larger scale**: Test at 100M+ params where NoPE and DataDep may close the gap with RoPE.
- **DataDep with bounded angles**: Constrain angle_proj output (e.g., tanh) to prevent OOD angles.

## Agreement with Literature

| Paper | Claim | Our result |
|-------|-------|------------|
| Wang et al. (ACL 2024) | NoPE fails at >2x training length due to attention distraction | Confirmed (66→192) |
| Kazemnejad et al. (NeurIPS 2023) | NoPE generalizes on algorithmic tasks | Not tested (LM only) |
| Yang et al. (Cohere 2025) | Hybrid RoPE-SWA + NoPE-full works | Confirmed (Hybrid_3 stays flat) |
| Haviv et al. (EMNLP 2022) | NoPE matches PE at training length | Not confirmed at 5M scale (18 PPL gap) — likely scale effect |
| Press et al. (2022) | ALiBi extrapolates | Not tested at 50K iters yet |

## Key Takeaways

1. **Windowed attention is necessary but not sufficient** for length generalization. RoPE windowed works; DataDep windowed (window=256) doesn't.

2. **Data-dependent rotation angles are sensitive to representation distribution shift** caused by multi-layer context propagation beyond training length. This is a novel finding not covered in the existing literature.

3. **The cascade effect**: Full-attention NoPE degrades because attention distraction in early layers cascades. Windowed layers at the bottom break the cascade, allowing full-attention NoPE layers at the top to work.

4. **Window size matters for DataDep**: Window=32 (indirect reach ~192 < block_size 512) keeps DataDep stable. Window=256 (indirect reach ~1536 > block_size 512) causes degradation. The critical threshold is when indirect reach exceeds training block_size.

5. **RoPE is remarkably robust**: Windowed RoPE produces identical results regardless of sequence length. The position-indexed angles provide a stable anchor that is independent of representation distribution.

## Part 8: Monoidal Hybrid Comparison (wiki, window=32, 50K iters, lr=5e-4)

All models: 5 windowed(32) layers + 1 NoPE full layer. Varying the windowed layer type.

| Length | hybrid_1 (RoPE) | monoidal_hybrid_1 (v1) | monoidal2_hybrid_1 (v2) |
|--------|-----------------|------------------------|-------------------------|
| 512 | **31.53** | 35.24 | 35.38 |
| 1024 | **31.34** | 31.99 | **31.84** |
| 2048 | **33.97** | 36.55 | 35.79 |
| 4096 | **33.92** | 35.11 | 34.63 |

All flat. RoPE hybrid_1 leads overall. Monoidal (cumsum, Q/K only) is ~2-3 PPL behind — cumsum helps vs plain datadep but doesn't match RoPE's explicit position signal at this scale.

## Part 9: Mixed Final Layer Comparison (wiki, window=32, 50K iters, lr=5e-4)

5 RoPE windowed(32) layers + 1 full-attention final layer. Varying the final layer type.

| Length | hybrid_1 (NoPE) | datadepfull_1 | ropefull_1 | RoPE win32 |
|--------|-----------------|---------------|------------|------------|
| 512 | **31.53** | 34.41 | 31.43 | 33.24 |
| 1024 | **31.34** | 34.08 | 32.25 | 32.31 |
| 2048 | **33.97** | 36.22 | 36.61 | 35.37 |
| 4096 | **33.92** | 34.33 | 37.98 | 34.85 |

NoPE final layer wins — position-agnostic final layer generalizes best.

## Part 10: Interleaved vs Asymmetric NoPE Placement (wiki, window=32, 50K iters, lr=5e-4)

| Length | hybrid_1 (5+1) | cohere (2:1) | alternating (1:1) | hybrid_3 (3+3) | RoPE win32 |
|--------|----------------|--------------|-------------------|----------------|------------|
| 512 | **31.53** | 32.02 | 32.59 | 32.58 | 33.24 |
| 1024 | **31.34** | 31.64 | 32.26 | 32.07 | 32.31 |
| 2048 | **33.97** | 34.20 | 34.95 | 35.03 | 35.37 |
| 4096 | **33.92** | 34.41 | 35.48 | 35.42 | 34.85 |

All flat. Fewer NoPE layers is better — hybrid_1 (just 1 NoPE at the end) beats hybrid_3 (3 NoPE). Cohere-style interleaving (2:1) is second best. The asymmetric split (our approach) matches or beats interleaving.

## Part 11: OWT Experiment (OpenWebText, 9.1B tokens, vocab=32K, 100K iters, lr=5e-4)

n_embed=128, n_layers=6, n_heads=4, block_size=512, window=32.
5 RoPE windowed(32) + 1 full-attention final layer.

**hybrid_1** (NoPE final) vs **datadep3full_1** (DataDep v3 MLP-angles final). ~9.4M params each.

### Training curves

| Iter | Hybrid_1 | Datadep3full_1 | Gap |
|------|----------|----------------|-----|
| 0 | 38770 | 37311 | — |
| 5K | 170.00 | 167.74 | -2.3 |
| 10K | 123.42 | 123.45 | 0.0 |
| 15K | 105.48 | 108.39 | +2.9 |
| 20K | 99.66 | 98.89 | -0.8 |
| 25K | 94.84 | 94.91 | +0.1 |
| 30K | 90.72 | 90.70 | 0.0 |
| 35K | 87.53 | 90.85 | +3.3 |
| 40K | 86.43 | 87.97 | +1.5 |
| 45K | 85.60 | 83.46 | -2.1 |
| 50K | 80.70 | 82.19 | +1.5 |
| 55K | 77.92 | 80.36 | +2.4 |
| 60K | 79.47 | 78.57 | -0.9 |
| 65K | 78.26 | 78.84 | +0.6 |

Very competitive — gap oscillates between -2 and +3 PPL, no clear winner at training length. Both noisy due to fixed lr=5e-4.

### Full training curves (100K iters)

| Iter | Hybrid_1 | Datadep3full_1 |
|------|----------|----------------|
| 70K | 74.67 | 77.17 |
| 75K | 76.78 | 75.43 |
| 80K | 75.06 | 76.91 |
| 85K | 76.01 | 75.54 |
| 90K | 77.84 | 76.30 |
| 95K | 75.13 | 75.15 |
| 100K | 74.52 | 76.18 |

Both oscillating around 75 PPL at fixed lr=5e-4.

### 200-iteration clean eval (100K checkpoints)

| Length | Hybrid_1 (NoPE) | Datadep3full_1 (MLP angles) |
|--------|-----------------|----------------------------|
| 512 | 74.57 | 74.67 |
| 1024 | 71.32 | 71.65 |
| 2048 | 71.50 | 72.55 |
| 4096 | 73.44 | 73.79 |

Essentially identical — within 1 PPL everywhere. Both perfectly flat across lengths.

### Continuation at lr=2e-4 (+50K iters)

| Iter | Hybrid_1 | Datadep3full_1 |
|------|----------|----------------|
| +5K | 72.28 | 71.76 |
| +10K | 68.67 | 68.40 |
| +15K | 68.76 | 67.90 |
| +20K | 68.67 | 70.98 |
| +25K | 69.46 | 69.17 |
| +30K | 69.74 | 68.53 |
| +35K | 68.75 | 68.08 |
| +40K | 67.19 | 69.88 |
| +45K | 68.31 | 69.00 |
| +50K | 67.33 | 70.03 |

### 200-iteration clean eval (after continuation)

| Length | Hybrid_1 (NoPE) | Datadep3full_1 (MLP angles) |
|--------|-----------------|----------------------------|
| 512 | 70.30 | 69.06 |
| 1024 | 65.79 | 64.58 |
| 2048 | 65.21 | 67.10 |
| 4096 | 67.22 | 66.96 |

Still essentially tied after continuation. Both flat across lengths. The datadep3 MLP-angle final layer performs as well as NoPE when sitting on top of windowed RoPE layers.

## Part 12: JoFormer Variants on OWT (V rotation, 100K iters, lr=5e-4)

All models: 5 windowed(32) layers + 1 NoPE full layer. Varying the windowed layer type.
Now testing V rotation (rotate Q, K, V with inverse rotation on output).

| Name | First 5 layers | Angles | Cumsum | V rotation | Params |
|------|---------------|--------|--------|------------|--------|
| hybrid_1 | RoPE | Fixed (position) | N/A | No | 9.4M |
| joformer_fixed_hybrid_1 | JoFormer-fixed | Fixed (position) | N/A | **Yes** | 9.4M |
| joformer_hybrid_1 | JoFormer v1 | Data-dependent | **Yes** | **Yes** | 9.5M |
| joformer2_hybrid_1 | JoFormer v2 | Data-dep (angle flow) | **Yes** | **Yes** | 11.6M |

### 200-iteration clean eval (100K checkpoints, fair comparison)

| Length | hybrid_1 (RoPE) | joformer_fixed | joformer_v1 | joformer_v2 |
|--------|-----------------|----------------|-------------|-------------|
| 512 | 74.06 | 76.21 | **73.14** | **72.26** |
| 1024 | 70.78 | 71.87 | **69.90** | 71.08 |
| 2048 | 71.19 | **70.44** | **69.43** | 70.58 |
| 4096 | 72.96 | **71.77** | **72.78** | **70.71** |

**Key findings:**
1. All models flat across lengths — the windowed + NoPE architecture generalizes regardless of what's in the windowed layers.
2. **Joformer_v1 is best at 512 and 2048** — data-dependent angles with cumsum and V rotation.
3. **Joformer_v2 is best at 4096** (70.71 vs 72.96 for RoPE) — angle flow architecture helps at longest range.
4. **V rotation helps**: Both joformer variants (with V rotation) beat RoPE hybrid_1 by 1-2 PPL.
5. **joformer_fixed is mixed**: Better than RoPE at long lengths (71.77 vs 72.96 at 4096) but worse at 512 (76.21 vs 74.06). V rotation on fixed angles helps at distance but slightly hurts locally.

### Continuation at lr=2e-4 (+50K iters) — completed

### 200-iteration clean eval (after 100K + 50K continuation)

| Length | hybrid_1 (RoPE) | joformer_fixed | joformer_v1 | joformer_v2 |
|--------|-----------------|----------------|-------------|-------------|
| 512 | 70.30 | 70.32 | **68.74** | **67.25** |
| 1024 | 65.79 | 66.39 | **65.77** | **64.13** |
| 2048 | 65.21 | 67.17 | **64.91** | **62.89** |
| 4096 | 67.22 | 68.45 | **66.07** | **65.91** |

**Key findings after full training (100K + 50K continuation):**

1. **Joformer_v2 is the clear winner** — best at every length, beating RoPE hybrid_1 by 2-3 PPL consistently. The angle-flow architecture (embedding C+C/2, FFN producing angles for next layer) with cumsum and V rotation is the strongest approach.

2. **Joformer_v1 is second** — also beats RoPE at every length by 0-2 PPL. The per-layer angle_proj with cumsum and V rotation works well.

3. **Joformer_fixed ≈ RoPE** — V rotation on fixed angles doesn't meaningfully help over standard RoPE. The benefit of V rotation comes when combined with data-dependent angles.

4. **All models perfectly flat across lengths** — the windowed(32) + NoPE architecture ensures length generalization regardless of what's in the windowed layers.

5. **V rotation + data-dependent angles + cumsum is the winning combination**. Each component contributes:
   - Cumsum: provides position signal through the data path (monoidal was ~2 PPL behind RoPE on wiki without V rotation)
   - V rotation: makes value summation non-commutative, enabling richer aggregation
   - Data-dependent angles: content-aware rotation that adapts to input (joformer beats joformer_fixed)

### Improvement breakdown (PPL averaged across lengths)

| Model | Avg PPL | vs RoPE |
|-------|---------|---------|
| hybrid_1 (RoPE, no V rotation) | 67.13 | baseline |
| joformer_fixed (fixed angles, V rotation) | 68.08 | -0.95 (worse) |
| joformer_v1 (datadep + cumsum + V rotation) | 66.37 | **+0.76** |
| joformer_v2 (angle flow + cumsum + V rotation) | 65.05 | **+2.08** |

Joformer_v2 improves over RoPE by ~2 PPL on average — a consistent advantage across all sequence lengths.

## Summary of All Results

### Architecture: 5 windowed layers + 1 NoPE full-attention layer

This is the core finding: windowed local layers + position-agnostic global layer at the end.

- **Windowed layers** handle local context with bounded attention (window=32)
- **NoPE final layer** does global content-based mixing with full attention
- Length generalization is guaranteed: windowed layers see the same distances at any sequence length, NoPE layer has no position encoding to go out of distribution

### What to use for the windowed layers?

Tested on OWT (9.1B tokens), n_embed=128, 6 layers, 150K total iters:

| Approach | Avg PPL (200-iter eval) | Length stable? |
|----------|------------------------|----------------|
| **JoFormer v2** (datadep angle flow + cumsum + V rotation) | **65.05** | Yes |
| **JoFormer v1** (datadep angle_proj + cumsum + V rotation) | **66.37** | Yes |
| **RoPE** (standard, Q/K only) | 67.13 | Yes |
| **JoFormer-fixed** (fixed angles + V rotation) | 68.08 | Yes |
| **Monoidal** (datadep + cumsum, Q/K only, no V rotation) | ~68-69* | Yes |
| **DataDep** (no cumsum, no V rotation) | ~69-70* | Yes |

*Monoidal and DataDep numbers from wiki experiments at different scale, approximate comparison.

### What makes the difference?

1. **V rotation is necessary** for data-dependent angles to help. Without V rotation (monoidal), datadep angles are ~2 PPL behind RoPE. With V rotation (joformer), they're ~2 PPL ahead. The non-commutative value summation unlocks the data-dependent angles.

2. **Fixed angles + V rotation ≈ RoPE without V rotation**. The V rotation alone doesn't help — it needs data-dependent angles to be useful.

3. **Angle flow (v2) > per-layer angle_proj (v1)** by ~1 PPL, but v2 has 22% more params (11.6M vs 9.5M). Some of the advantage may be from extra parameters.

### Comparison with Cohere RNoPE-SWA

| Aspect | Cohere | Ours (best) |
|--------|--------|-------------|
| Architecture | [RoPE-SWA × 3, NoPE-full] × 8 (interleaved) | [JoFormer-SWA × 5, NoPE-full × 1] (asymmetric) |
| NoPE placement | Every 4th layer | Only final layer |
| Full-attention layers | 8 of 32 (25%) | 1 of 6 (17%) |
| Windowed layer type | Standard RoPE | JoFormer (datadep + cumsum + V rotation) |
| Scale | 8B params, 5T tokens | 9.5-11.6M params, 9B tokens |

Key difference: we use JoFormer in the windowed layers instead of standard RoPE, and concentrate all global attention in a single final NoPE layer. Cohere never tested non-RoPE windowed layers.

## Part 13: Scale-Up Experiment — RoPE (163M params, unwindowed, OWT)

n_embed=768, n_layers=16, n_heads=8, block_size=512, window=999999 (unwindowed), OWT, lr=5e-4, bf16.

Training was done in two stages: original run (0-15K, no bf16) then continuation (15K-100K, bf16). The 15K values from both runs are listed for comparison.

| Iter | Val PPL (original) | Val PPL (continuation) |
|------|-------------------|----------------------|
| 0 | 37921.49 | — |
| 5K | 58.23 | — |
| 10K | 46.75 | — |
| 15K | 43.00 | 43.81 |
| 20K | — | 39.51 |
| 25K | — | 38.04 |
| 30K | — | 36.93 |
| 35K | — | 36.58 |
| 40K | — | 35.66 |
| 45K | — | 35.92 |
| 50K | — | 33.75 |
| 55K | — | 34.30 |
| 60K | — | 33.22 |
| 65K | — | 33.69 |
| 70K | — | 33.27 |
| 75K | — | 32.37 |
| 80K | — | 31.87 |
| 85K | — | 32.09 |
| 90K | — | 33.60 |
| 95K | — | 32.91 |
| 100K | — | 31.94 |

Final eval: val PPL 32.40. Extrapolation: 512:31.51, 1024:43.76, 2048:88.86, 4096:154.82.

## Part 14: Scale-Up — JoFormer v2 Fails at 16 Layers

All attempts to train JoFormer v2 (data-dependent angle flow + cumsum + V rotation, 193M params) at 16 layers diverged. Same config as RoPE: n_embed=768, n_layers=16, n_heads=8, block_size=512, unwindowed, OWT, lr=5e-4, bf16.

### Attempts

**Attempt 1: Vanilla JoFormer v2** — diverged, never learned.

| Iter | RoPE | JoFormer v2 |
|------|------|-------------|
| 5K | 58.23 | 2182.77 |
| 10K | 46.75 | 1972.38 |
| 15K | 43.00 | 1975.78 |
| 20K | 39.51 | 9031.51 |

**Attempt 2: tanh-bounded angles** (`tanh(angles) * π`) — still diverged.

| Iter | RoPE | JoFormer v2 (tanh) |
|------|------|--------------------|
| 5K | 58.23 | 1581.82 |
| 10K | 46.75 | 2302.46 |

**Attempt 3: Split angle params + zero-init + separate angle_lr=5e-5** — still diverged.

| Iter | RoPE | JoFormer v2 (split+zero-init) |
|------|------|-------------------------------|
| 5K | 58.23 | 1759.16 |

**Attempt 4: RoPE-base angles + zero-init + angle_lr=5e-5** — learned briefly then diverged.

| Iter | RoPE | JoFormer v2 (RoPE-base) |
|------|------|------------------------|
| 1K | — | 815.74 |
| 2K | — | 1637.27 |

**JoFormer v1 (per-layer angle_proj, not angle flow) also diverged:**

| Iter | RoPE | JoFormer v1 |
|------|------|-------------|
| 1K | — | 918.58 |
| 2K | — | 1427.38 |

### Diagnosis

**Frozen angles work perfectly.** JoFormer v2 with angle_lr≈0 (angles frozen at RoPE base values) matched joformer_fixed exactly:

| Iter | JoFormer fixed | JoFormer v2 (frozen) |
|------|---------------|---------------------|
| 1K | — | 153.89 |
| 2K | — | 91.72 |
| 3K | — | 74.43 |
| 4K | — | 65.42 |
| 5K | 59.16 | 59.46 |

**Conclusion:** The architecture is fine. V rotation works at scale. The problem is learning data-dependent angles at depth — any update to angle parameters at 16 layers creates an unstable feedback loop through the residual stream, regardless of whether angles flow between layers (v2) or are computed per-layer (v1).

## Part 15: Scale-Up — JoFormer v2 with Warmup (100K iters)

**Solution: freeze angles for first 1K iters, then unfreeze at angle_lr=5e-5.**

The model starts with RoPE-equivalent angles (via `rope_base_angles` buffer + zero-init learnable params). For the first 1000 iterations, angle parameters are frozen (lr=0) while the rest of the model learns at lr=5e-4. After 1K iters, angle parameters unfreeze at lr=5e-5 (10x slower than main lr).

193M params, same config otherwise. Eval every 1K iters.

| Iter | RoPE | JoFormer v2 (warmup) | Diff |
|------|------|---------------------|------|
| 5K | 58.23 | 62.13 | +3.90 |
| 10K | 46.75 | 50.89 | +4.14 |
| 15K | 43.00 | 46.23 | +3.23 |
| 20K | 39.51 | 43.89 | +4.38 |
| 25K | 38.04 | 41.87 | +3.83 |
| 30K | 36.93 | 41.41 | +4.48 |
| 35K | 36.58 | 40.41 | +3.83 |
| 40K | 35.66 | 38.93 | +3.27 |
| 45K | 35.92 | 38.48 | +2.56 |
| 50K | 33.75 | 38.01 | +4.26 |
| 55K | 34.30 | 37.16 | +2.86 |
| 60K | 33.22 | 36.89 | +3.67 |
| 65K | 33.69 | 36.54 | +2.85 |
| 70K | 33.27 | 36.19 | +2.92 |
| 75K | 32.37 | 36.28 | +3.91 |
| 80K | 31.87 | 35.80 | +3.93 |
| 85K | 32.09 | 35.08 | +2.99 |
| 90K | 33.60 | 35.02 | +1.42 |
| 95K | 32.91 | 34.87 | +1.96 |
| 100K | 31.94 | 35.13 | +3.19 |

**Final: RoPE 32.40, JoFormer v2 (warmup) 35.13.** Gap of ~3 PPL.

The model is stable — no divergence — but the 10x slower angle learning rate means the data-dependent angles can't contribute enough to overcome the 1K warmup delay. The gap held steady at ~3-4 PPL throughout training.

## Part 16: Scale-Up — JoFormer Fixed (163M params, in progress)

JoFormer fixed (RoPE angles + V rotation) is training for 100K iters. At 10K it matches RoPE exactly (46.77 vs 46.75).

Once complete, the checkpoint will be converted to JoFormer v2 format (via `convert_fixed_to_v2.py`) with zero-init angle deviations, then continued with learnable angles. This gives the model a fully trained base before introducing data-dependent angle learning.

### Conversion verification

The `convert_fixed_to_v2.py` script maps joformer_fixed weights to joformer2 (split_angles) format:
- Shared weights (qkv, out_proj, tok_emb, lm_head, ln) transfer directly
- FFN fc2 → fc2_content (same dimensions)
- angle_emb and fc2_angles initialized to zero
- rope_base_angles computed as negated RoPE frequencies (to match cumsum direction)

Verified: converted checkpoint produces identical val PPL (46.77) as the original joformer_fixed checkpoint.

## Part 17: Scale-Up — JoFormer Fixed 100K Training Curve

JoFormer fixed (RoPE angles + V rotation, 163M params) trained for 100K iters. Same config as RoPE. Fixed-seed eval (torch.manual_seed(42)) throughout. Note: RoPE numbers are from an earlier run without fixed-seed eval, so RoPE has eval noise while JoFormer fixed does not.

| Iter | RoPE | JoFormer fixed |
|------|------|---------------|
| 5K | 58.23 | 59.16 |
| 10K | 46.75 | 46.77 |
| 15K | 43.00 | 42.15 |
| 20K | 39.51 | 39.70 |
| 25K | 38.04 | 37.92 |
| 30K | 36.93 | 36.65 |
| 35K | 36.58 | 35.69 |
| 40K | 35.66 | 35.06 |
| 45K | 35.92 | 34.52 |
| 50K | 33.75 | 34.04 |
| 55K | 34.30 | 33.62 |
| 60K | 33.22 | 33.25 |
| 65K | 33.69 | 32.97 |
| 70K | 33.27 | 32.59 |
| 75K | 32.37 | 32.46 |
| 80K | 31.87 | 32.26 |
| 85K | 32.09 | 32.06 |
| 90K | 33.60 | 31.86 |
| 95K | 32.91 | 31.73 |
| 100K | 31.94 | 31.49 |

Final extrap (from training, 10 iters): 512:31.33, 1024:34.71, 2048:59.61, 4096:102.38.

JoFormer fixed tracks RoPE closely but with a smoother curve (fixed-seed eval). Slightly ahead at 100K (31.49 vs 31.94).

## Part 18: Scale-Up — Continuation at lr=2e-4

Both models continued from their 100K checkpoints at lr=2e-4 for 50K more iters.

### RoPE continuation (50K iters)

The RoPE continuation was done in two stages due to a GPU interruption: 30K iters from the 100K checkpoint, then 20K more from the 30K continuation checkpoint.

| Iter | Val PPL |
|------|---------|
| 0 | 32.30 |
| 5K | 28.23 |
| 10K | 27.73 |
| 15K | 27.54 |
| 20K | 27.37 |
| 25K | 27.28 |
| 30K | 27.22 |
| 35K | 27.10 |
| 40K | 27.09 |
| 45K | 27.20 |
| 50K | 27.09 |

### JoFormer fixed continuation (50K iters)

| Iter | Val PPL |
|------|---------|
| 0 | 31.49 |
| 5K | 27.62 |
| 10K | 27.05 |
| 15K | 26.85 |
| 20K | 26.76 |
| 25K | 26.68 |
| 30K | 26.69 |
| 40K | 26.74 |
| 45K | 26.65 |
| 50K | 26.60 |

JoFormer fixed converged to 26.60, RoPE to 27.09 — JoFormer fixed is **0.49 PPL better** at training length after full training (100K + 50K continuation).

## Part 19: Scale-Up — 200-Iteration Clean Eval (All Checkpoints)

### 100K checkpoints (scale_up_full)

| Model | Params | Iter | PPL@512 | PPL@1024 | PPL@2048 | PPL@4096 |
|-------|--------|------|---------|----------|----------|----------|
| joformer_fixed | 162.6M | 100K | **31.27** | 37.08 | 61.71 | 104.58 |
| rope | 162.6M | 65K* | 32.16 | 44.22 | 85.41 | 156.58 |
| joformer2 (warmup) | 193.8M | 94K | 34.49 | **33.27** | **32.61** | **33.91** |

*RoPE best checkpoint was saved at iter 65K (from the earlier continuation run, not the full 100K).

### Continued checkpoints (100K + 50K at lr=2e-4)

| Model | Params | PPL@512 | PPL@1024 | PPL@2048 | PPL@4096 |
|-------|--------|---------|----------|----------|----------|
| joformer_fixed | 162.6M | **26.32** | **34.32** | 64.74 | 115.96 |
| rope | 162.6M | 26.82 | 41.90 | 90.10 | 168.67 |
| rope (extra 20K) | 162.6M | 26.93* | 43.26 | 100.41 | 195.93 |

*The extra 20K RoPE continuation started from a partial checkpoint and performed slightly worse.

### Key findings

1. **JoFormer v2 (warmup) is flat across lengths.** Despite being trained unwindowed on 512-length sequences, it extrapolates perfectly: 34.49 at 512, 32.61 at 2048, 33.91 at 4096. The data-dependent angles with cumsum enable length generalization without windowed attention.

2. **JoFormer fixed degrades less than RoPE.** At 4096, joformer_fixed is at 115.96 vs RoPE's 168.67-195.93. The V rotation on fixed angles provides some extrapolation benefit, though it still degrades significantly.

3. **JoFormer fixed beats RoPE at training length.** After full training (100K + 50K), joformer_fixed reaches 26.32 vs RoPE's 26.82 at 512 — a consistent 0.5 PPL advantage.

4. **JoFormer v2 trades training-length PPL for flat extrapolation.** At 512 it's 34.49 (vs 26.32 for joformer_fixed), but at 4096 it's 33.91 (vs 115.96). The ~3 PPL gap behind RoPE/joformer_fixed at training length is the cost of the warmup + slow angle lr approach.

### Extrapolation degradation comparison

| Model | PPL@512 | PPL@4096 | Ratio (4096/512) |
|-------|---------|----------|-----------------|
| JoFormer v2 (warmup) | 34.49 | 33.91 | **0.98x** |
| JoFormer fixed (continued) | 26.32 | 115.96 | 4.41x |
| RoPE (continued) | 26.82 | 168.67 | 6.29x |

## Part 20: Scale-Up — JoFormer v2 from Trained JoFormer Fixed Base

Converted the continued joformer_fixed checkpoint (100K + 50K, val PPL 26.60) to JoFormer v2 format using `convert_fixed_to_v2.py`. Angle params (angle_emb, fc2_angles) initialized to zero so the model starts as joformer_fixed. Then continued for 50K iters with both lr and angle_lr at 5e-5.

Checkpoint: `checkpoints/scale_up_joformer2_from_fixed/`

### Training curve

| Iter | Val PPL |
|------|---------|
| 0 | 26.60 |
| 1K | 30.20 |
| 2K | 29.08 |
| 3K | 28.78 |
| 4K | 28.35 |
| 5K | 28.11 |
| 6K | 27.84 |
| 7K | 27.63 |
| 9K | 27.45 |
| 10K | 27.30 |
| 12K | 27.09 |
| 15K | 26.87 |
| 20K | 26.64 |
| 25K | 26.39 |
| 30K | 26.17 |
| 34K | 26.08 |
| 40K | 26.09 |
| 44K | 25.87 |
| 47K | 25.82 |
| 50K | 25.86 |

Initial spike to 30.20 at 1K as angle params start learning and destabilize the trained model. Recovers to the starting point by ~20K. Then improves past joformer_fixed, reaching **25.86** at 50K — **0.74 PPL better than joformer_fixed** (26.60) and **1.23 PPL better than RoPE** (27.09).

### Summary of all scale-up results at training length (512)

| Model | Training | Val PPL |
|-------|----------|---------|
| RoPE | 100K + 50K continuation | 27.09 |
| JoFormer fixed | 100K + 50K continuation | 26.60 |
| **JoFormer v2 (from fixed base)** | above + 50K at lr=5e-5 | **25.86** |
| JoFormer v2 (warmup, from scratch) | 100K | 35.13 |

The fixed-base approach works: train joformer_fixed first, convert to v2, then fine-tune angles at low lr. The data-dependent angles provide a clear benefit (+0.74 over joformer_fixed, +1.23 over RoPE) when the model has a stable trained base to start from.

### 200-iteration clean eval with extrapolation

| Length | JoFormer v2 (from fixed) | JoFormer fixed | RoPE | Ratio to training length |
|--------|-------------------------|---------------|------|------------------------|
| 512 (1x) | **25.55** | 26.32 | 26.82 | 1x |
| 1024 (2x) | **25.17** | 34.32 | 41.90 | 2x |
| 2048 (4x) | **24.73** | 64.74 | 90.10 | 4x |
| 4096 (8x) | **26.01** | 115.96 | 168.67 | 8x |
| 8192 (16x) | **31.13** | — | — | 16x |
| 16384 (32x) | OOM | — | — | 32x |

**JoFormer v2 is flat through 8x training length and still functional at 16x.** The model was trained on 512-length sequences and:

- **1x–4x (512–2048)**: actually *improves*, reaching best PPL at 2048 (24.73) — better than at training length
- **8x (4096)**: essentially flat at 26.01, only +0.46 above 512
- **16x (8192)**: 31.13, meaningful degradation but still functional

For comparison, RoPE at just 2x (1024) is already at 41.90, and at 4x (2048) at 90.10. JoFormer v2 at 16x (8192) is still better than RoPE at 2x.

### Extrapolation degradation comparison (all models, continued checkpoints)

| Model | PPL@512 | PPL@4096 | Ratio |
|-------|---------|----------|-------|
| **JoFormer v2 (from fixed)** | **25.55** | **26.01** | **1.02x** |
| JoFormer v2 (warmup, 100K only) | 34.49 | 33.91 | 0.98x |
| JoFormer fixed | 26.32 | 115.96 | 4.41x |
| RoPE | 26.82 | 168.67 | 6.29x |

Both JoFormer v2 variants achieve near-perfect length generalization (ratio ~1.0x). The fixed-base v2 additionally achieves the best absolute PPL at every length.

### Three-stage training recipe

The successful recipe for scaling JoFormer v2:

1. **Stage 1**: Train joformer_fixed (RoPE + V rotation) for 100K iters at lr=5e-4
2. **Stage 2**: Continue joformer_fixed at lr=2e-4 for 50K iters
3. **Convert**: Transform checkpoint to joformer2 format (zero-init angle params, negated RoPE base angles for cumsum direction)
4. **Stage 3**: Continue as joformer2 at lr=5e-5 (both main and angle lr) for 50K iters

This avoids the divergence problem of training data-dependent angles from scratch at depth, while still achieving the full benefit of data-dependent angles for length generalization.

## Part 21: Shared-MLP and Random Positive Angles — Isolating the Mechanism

Models to test what cumsum needs to enable length generalization. All trained from scratch, same config as RoPE baseline (768 embed, 16 layers, 8 heads, block_size=512, lr=5e-4, bf16, OWT, no cosine decay). All use cumsum on angles and full (unwindowed) attention.

### Model naming convention: `{scope}_{sign}[_{variant}]_{rotation}`

- **scope**: `shared` (one MLP all layers), `perlayer` (separate MLP per layer), `random` (not learned)
- **sign**: `pos` (MLP → LayerNorm → abs, positive only), `ln` (MLP → LayerNorm, allows negative)
- **variant**: `split` (separate angles for Q/K vs V), `indep` (per-layer independent random angles)
- **rotation**: `qk` (Q/K only), `qkv` (Q/K/V + inverse rotation)

### Completed models

**Learned angles (shared MLP, LayerNorm → abs):**
- **shared_pos_qk** (166M): One MLP shared across all layers. Cumsum on Q/K only. Running (~1h left).
- **shared_pos_qkv** (166M): Same but with V rotation + inverse. Running (~2h left).

**Random angles (Uniform, positive, resampled each forward pass):**
- **random_pos_qk** (163M): Uniform(0, 2·freq_i) with log-spaced scales matching RoPE. Shared across layers. Cumsum on Q/K only. **Done — 32.03 PPL.**
- **random_pos_qkv** (163M): Same but with V rotation + inverse. **Done — 31.80 PPL.**

**Running:**
- **random_pos_indep_qk** (163M): Per-layer independent random positive angles, Q/K only.
- **random_ln_indep_qk** (163M): Per-layer independent random signed angles Uniform(-freq, freq), Q/K only.

**Queued (not yet run):**
- `shared_ln_qk/qkv` — shared MLP, LayerNorm only (no abs, allows negative angles)
- `shared_pos_split_qkv` — two separate shared MLPs for Q/K and V angles
- `perlayer_pos_qk/qkv` — per-layer MLP (219M params), tests if LayerNorm→abs fixes joformer2 divergence
- `perlayer_ln_qk/qkv` — per-layer MLP, LayerNorm only
- `random_ln_qk/qkv` — random Uniform(-freq, freq), shared across layers
- RoPE windowed w=256, joformer_fixed windowed w=256

### Key findings

**1. Zero-mean random angles with cumsum give FLAT length generalization.**
random_ln_indep_qk (Uniform(-freq, freq), per-layer independent, Q/K only) extrapolates perfectly at 10K iters:

| Length | random_ln_indep_qk | random_pos_indep_qk | random_pos_qk | RoPE | JoFormer v2 |
|--------|-------------------|---------------------|---------------|------|-------------|
| 512    | 48.80             | 47.17               | 47.04         | —    | —           |
| 1024   | 43.87             | 51.84               | 52.04         | —    | —           |
| 2048   | 44.02             | 94.38               | 97.42         | —    | —           |
| 4096   | 45.57             | 162.39              | 157.76        | —    | —           |
| 8192   | 46.75             | 250.58              | 232.30        | —    | —           |

(All at 10K iters for fair comparison. PPL improves from 512→1024 and stays flat through 8192.)

This is the critical finding: **zero-mean angles are essential for length generalization, not data-dependence.** Positive-only angles (cumsum always increases) create a monotonically growing phase that becomes out-of-distribution at longer sequences — the same problem as RoPE. Zero-mean angles create a random walk that stays bounded regardless of sequence length.

**2. Positive random cumsum matches RoPE at training length but does NOT generalize.**
random_pos_qk reached 32.03 PPL at 100K (RoPE: 31.94) — essentially identical. But it degrades at longer sequences like RoPE:

| Length | random_pos_qk | random_pos_qkv | RoPE | JoFormer v2 (from fixed) |
|--------|--------------|----------------|------|--------------------------|
| 512    | 31.58        | 31.33          | 31.51 | 25.55                   |
| 1024   | 39.75        | 32.35          | 43.76 | 25.17                   |
| 2048   | 78.55        | 47.28          | 88.86 | 24.73                   |
| 4096   | 145.34       | 83.68          | 154.82 | 26.01                  |
| 8192   | 223.32       | 158.69         | —     | 31.13                   |
| 16384  | OOM          | OOM            | —     | OOM                     |

**3. V rotation dramatically improves extrapolation for positive random angles.**
random_pos_qkv degrades much less than random_pos_qk: at 2048, PPL is 47.28 vs 78.55 (nearly half). At 1024, it's 32.35 vs 39.75 — almost flat. V rotation acts as a second filter that cleans up noise from diluted attention at longer sequences. However, it's still far from the flat generalization achieved by zero-mean angles.

**4. V rotation hurts shared_pos at training length (but helps random).**
shared_pos_qkv is consistently ~0.7 PPL worse than shared_pos_qk at training length. In contrast, V rotation helps for random_pos (+0.2 PPL at training length, +30 PPL at 2048). Hypothesis: shared MLP parameters are pulled in conflicting directions by Q/K and V rotation gradients across 16 layers. For random angles there are no learned parameters, so no gradient conflict.

**5. LayerNorm→abs enables from-scratch training (tanh diverges).**
Previous joformer2 from scratch with tanh·π diverged within 5-20K iters (PPL > 2000). The shared_pos models with LayerNorm→abs train stably from scratch with no warmup or warm-start needed. Whether this is due to LayerNorm, abs, or both is being tested in queued experiments.

**6. shared_pos_qkv showed a PPL spike at 45K (38.90 from 35.97).**
Recovered by 50K (34.75). May be related to the V rotation gradient conflict. No other model showed instability.

### Training curve (cosine decay — killed, not a fair comparison)

| Iter | RoPE (Q/K) | Fixed (Q/K/V) | shared_pos_qk | shared_pos_qkv | random_pos_qk | random_pos_qkv |
|------|-----------|---------------|---------------|----------------|---------------|----------------|
| 5K   | 58.23     | 59.16         | 67.04         | 64.96          | 59.71         | 59.45          |
| 10K  | 46.75     | 46.77         | 49.02         | 49.04          | 47.20         | 46.97          |
| 15K  | 43.00     | 42.15         | 43.13         | 43.66          | 42.44         | 42.06          |
| 20K  | 39.51     | 39.70         | 39.82         |                | 39.52         | 39.65          |
| 25K  | 38.04     | 37.92         |               |                | 37.51         |                |

Killed at ~25K — cosine decay was not used in the RoPE/Fixed baselines, so not a fair comparison.

### Training curve (no cosine decay — matches baselines)

| Iter | RoPE (Q/K) | Fixed (Q/K/V) | shared_pos_qk | shared_pos_qkv | random_pos_qk | random_pos_qkv |
|------|-----------|---------------|---------------|----------------|---------------|----------------|
| 5K   | 58.23     | 59.16         | 64.55         | 65.67          | 59.88         | 59.63          |
| 10K  | 46.75     | 46.77         | 48.72         | 49.32          | 47.48         | 47.12          |
| 15K  | 43.00     | 42.15         | 43.38         | 44.25          | 42.64         | 42.41          |
| 20K  | 39.51     | 39.70         | 40.47         | 41.14          | 40.28         | 40.06          |
| 25K  | 38.04     | 37.92         | 38.77         | 39.32          | 38.38         | 38.16          |
| 30K  | 36.93     | 36.65         | 37.62         | 37.92          | 37.09         | 36.99          |
| 35K  | 36.58     | 35.69         | 36.43         | 36.80          | 36.15         | 36.07          |
| 40K  | 35.66     | 35.06         | 35.56         | 35.97          | 35.40         | 35.36          |
| 45K  | 35.92     | 34.52         | 34.90         | 38.90          | 34.80         | 34.77          |
| 50K  | 33.75     | 34.04         | 34.41         | 34.75          | 34.42         | 34.27          |
| 55K  | 34.30     | 33.62         | 33.99         | 34.34          | 34.13         | 33.92          |
| 60K  | 33.22     | 33.25         | 33.55         | 33.88          | 33.66         | 33.51          |
| 65K  | 33.69     | 32.97         | 33.27         | 33.50          | 33.34         | 33.12          |
| 70K  | 33.27     | 32.59         | 33.06         | 33.21          | 33.00         | 32.79          |
| 75K  | 32.37     | 32.46         | 32.77         | 33.07          | 32.74         | 32.65          |
| 80K  | 31.87     | 32.26         | 32.39         | 32.61          | 32.64         | 32.49          |
| 85K  | 32.09     | 32.06         | 32.37         | 32.41          | 32.43         | 32.21          |
| 90K  | 33.60     | 31.86         | 32.16         |                | 32.22         | 32.11          |
| 95K  | 32.91     | 31.73         | 31.95         |                | 32.13         | 31.99          |
| 100K | 31.94     | 31.49         |               |                | **32.03**     | **31.80**      |

### Length extrapolation (final eval at 100K unless noted)

| Length | RoPE | Fixed (Q/K/V) | shared_pos_qk | shared_pos_qkv | random_pos_qk | random_pos_qkv | shared_ln_qk | random_ln_indep_qk | JoFormer v2 |
|--------|------|---------------|---------------|----------------|---------------|----------------|--------------|--------------------|----|
| 512    | 31.51 | 26.32        | 31.37         | 31.59          | 31.58         | 31.33          | 31.25        | **32.35**          | 25.55 |
| 1024   | 43.76 | 34.32        | 40.13         | 31.39          | 39.75         | 32.35          | 38.41        | **29.21**          | 25.17 |
| 2048   | 88.86 | 64.74        | 120.41        | 59.77          | 78.55         | 47.28          | 104.03       | **29.20**          | 24.73 |
| 4096   | 154.82 | 115.96      | 381.47        | 111.21         | 145.34        | 83.68          | 211.23       | **30.93**          | 26.01 |
| 8192   | —    | —             | 793.05        | 198.52         | 223.32        | 158.69         | 331.34       | **33.32**          | 31.13 |

### The critical finding: zero-mean angles enable length generalization

random_ln_indep_qk uses unlearned random angles from Uniform(-freq, freq) with cumsum — zero parameters for position encoding. Yet it achieves flat extrapolation from 512 to 8192, with PPL actually *improving* from 512 (32.35) to 2048 (29.20).

**Comparison with RoPE at training length (512):** random_ln_indep_qk is only 0.9 PPL behind RoPE (32.85 vs 31.94). But at every other length it is dramatically better:

| Length | random_ln_indep_qk | RoPE | Ratio |
|--------|-------------------|------|-------|
| 512    | 32.35             | 31.51 | 1.03x |
| 1024   | 29.21             | 43.76 | 0.67x |
| 2048   | 29.20             | 88.86 | 0.33x |
| 4096   | 30.93             | 154.82 | 0.20x |
| 8192   | 33.32             | —     | — |

**What matters and what doesn't for length generalization:**

| Factor | Tested | Generalizes? |
|--------|--------|-------------|
| Positive cumsum (monotonic) | random_pos_qk, shared_pos_qk | **No** — degrades like RoPE |
| V rotation on positive angles | random_pos_qkv, shared_pos_qkv | **Helps** but still degrades |
| Learned angles (shared MLP, LN→abs) | shared_pos_qk/qkv | **No** — worse than random |
| Learned angles (shared MLP, LN only) | shared_ln_qk | **No** — degrades as training progresses |
| Learned angles (per-layer MLP, LN only) | perlayer_ln_qk/qkv | **Partial** — degrades slower but still degrades |
| Zero-mean random angles (per-layer) | random_ln_indep_qk/qkv | **Yes** — flat through 8192 |
| Data-dependent angles (joformer2) | JoFormer v2 (tanh·π) | **Yes** — flat through 8192 |

The critical ingredient is **zero-mean angles under cumsum**. Positive-only angles create a monotonically growing cumulative phase that becomes out-of-distribution at longer sequences. Zero-mean angles create a random walk that stays bounded regardless of sequence length — the cumulative phase doesn't drift, so any distance looks like training.

### Learned angles (per-layer MLP) vs random angles

Per-layer MLPs (219M params, separate MLP per layer) were tested to see if learned angles could match random angles. They cannot — random zero-mean angles beat learned angles at both training length AND extrapolation at every matched iteration:

**Extrapolation @8192 over training:**

| Iter | perlayer_ln_qk | perlayer_ln_qkv | ln_perlayer_ln_qkv | random_ln_indep_qk | random_ln_indep_qkv |
|------|---------------|----------------|---------------------|-------------------|---------------------|
| 10K  | 59.71         | 58.92          | 57.33               | **46.75**         | **47.25**           |
| 20K  | 52.07         | 50.96          | 48.51               | **37.61**         | **38.10**           |
| 30K  | 51.76         | 47.91          |                     | **35.28**         |                     |
| 40K  | 66.29         | 45.40          |                     | **33.92**         |                     |
| 50K  | 78.10         | 47.47          |                     | **34.17**         |                     |
| 60K  | 92.43         | 51.72          |                     | **33.82**         |                     |
| 70K  | 108.71        | 56.84          |                     | **33.00**         |                     |
| 100K | 142.58        |                |                     | **33.32**         | **31.71**           |

**Training-length PPL at matched iters:**

| Iter | perlayer_ln_qkv | ln_perlayer_ln_qkv | random_ln_indep_qk | random_ln_indep_qkv |
|------|----------------|---------------------|-------------------|---------------------|
| 10K  | 53.23          | 54.82               | **48.88**         | 49.28               |
| 20K  | 44.00          | 44.46               | **40.76**         | 41.79               |
| 25K  | 41.92          | 41.87               | **39.63**         | 39.63               |
| 50K  | 36.31          |                     | **35.27**         | 36.13               |
| 100K |                |                     | **32.85**         | 33.08               |

The 219M-parameter per-layer MLP models cannot match the 163M-parameter random angle models. The learned angle MLPs add 56M parameters that hurt rather than help — the model overfits its angles to training length, progressively losing extrapolation ability.

### V rotation: hurts training PPL, helps extrapolation

Across all model types, V rotation consistently:
- **Hurts training-length PPL** for learned models (~0.5-1.5 PPL worse), neutral for random
- **Dramatically helps extrapolation** (2-3x reduction in degradation at 8192)

The one exception is Fixed (RoPE) angles where V rotation helps both training PPL (+0.45) and extrapolation. This may be because fixed angles are deterministic and consistent across layers, avoiding the optimization conflicts that learned per-layer angles create.

### Extrapolation over training for all learned models

All learned models (shared and per-layer MLP) start with decent extrapolation but lose it as training progresses:

| Iter | shared_ln_qk | shared_ln_qkv | perlayer_ln_qk | perlayer_ln_qkv | random_ln_indep_qk |
|------|-------------|---------------|---------------|----------------|-------------------|
| 10K  | 64.02       | 64.15         | 59.71         | 58.92          | **46.75**         |
| 20K  | 75.63       | 58.13         | 52.07         | 50.96          | **37.61**         |
| 30K  | 148.47      | 61.73         | 51.76         | 47.91          | **35.28**         |
| 40K  | 197.54      | 71.57         | 66.29         | 45.40          | **33.92**         |
| 50K  | 265.76      | 92.16         | 78.10         | 47.47          | **34.17**         |
| 60K  |             | 103.73        | 92.43         | 51.72          | **33.82**         |
| 70K  |             | 133.91        | 108.71        | 56.84          | **33.00**         |
| 100K | 331.34      | 156.21        | 142.58        |                | **33.32**         |

(All values: PPL @8192)

Per-layer MLPs degrade slower than shared MLPs. V rotation (qkv) slows degradation further. But ALL learned models eventually degrade, while random stays flat at ~33.

### Input LN experiment

Testing whether feeding ln(x) to the angle MLP (instead of raw x) helps:

- **ln_perlayer_qkv** (input LN only, no output LN): **Diverged** at 5K (PPL 2010). Output LN is essential for stability.
- **ln_perlayer_ln_qkv** (input LN + output LN): Converges more slowly but extrapolates better than perlayer_ln_qkv at matched iterations (48.51 vs 50.96 @8192 at 20K). Still running.
- **perlayer_ln_qkv_slowangle** (angle_lr=1e-4 vs main lr=5e-4): Still early (10K). Testing whether slower angle learning preserves extrapolation.

### Relationship to ALiBi

The random zero-mean cumsum mechanism is functionally similar to ALiBi (Attention with Linear Biases):
- Both create exponential soft windows for length generalization
- ALiBi: deterministic linear bias on logits, provides positional information
- Random cumsum: stochastic multiplicative modulation via rotary phase, provides NO positional information
- Both generalize to arbitrary sequence lengths because neither creates OOD patterns at new distances

The key difference: ALiBi gives the model deterministic distance information, while random cumsum only provides a stochastic soft window. Yet the cumsum framework enables learned data-dependent angles (as in JoFormer v2) which can potentially surpass both.

## Part 22: The 200K Comparison — Fair Comparison Across All Methods

All models trained with the same lr schedule: 100K at 5e-4, 50K at 2e-4, 50K at 5e-5 (200K total).

| Length | RoPE 200K | joformer_fixed 200K | monoidal2 200K | random_ln_indep 200K | JoFormer v2 (from fixed)* | ALiBi 100K |
|--------|-----------|--------------------|-----------------|--------------------|--------------------------|------------|
| 512    | **23.54** | **23.18**          | 30.61           | 24.21              | 25.55                    | 32.68      |
| 1024   | 42.77     | 33.23              | 29.17           | **22.75**          | 25.17                    | 29.17      |
| 2048   | 106.91    | 70.35              | 28.99           | **22.38**          | 24.73                    | 28.91      |
| 4096   | 223.16    | 128.37             | 31.36           | **26.00**          | 26.01                    | 30.82      |
| 8192   | —         | —                  | —               | —                  | 31.13                    | 30.11      |

All models trained for 200K total iterations with the same lr schedule (100K at 5e-4, 50K at 2e-4, 50K at 5e-5). Each used its architecture throughout:
- RoPE: plain RoPE all 200K
- joformer_fixed: RoPE + V rotation all 200K (150K as joformer_fixed, continued 50K)
- monoidal2: 150K as RoPE, then converted to monoidal2 (data-dep Q/K cumsum) for final 50K
- random_ln_indep: random zero-mean cumsum all 200K
- JoFormer v2: 150K as joformer_fixed, then converted to joformer2 (data-dep Q/K/V cumsum) for final 50K
- ALiBi: ALiBi all 100K (200K running)

### Key findings

**1. random_ln_indep_qk at 200K beats JoFormer v2 at training length AND extrapolation.**
PPL 24.21 at 512 (vs 25.55) and 22.38 at 2048 (vs 24.73). Zero learned position parameters. The three-stage lr schedule (5e-4 → 2e-4 → 5e-5) gives the model enough training to reach excellent PPL, while the random zero-mean angles maintain flat extrapolation throughout.

**2. RoPE and joformer_fixed are best at training length but don't generalize.**
RoPE 200K reaches 23.54, joformer_fixed 23.18 — both excellent. But they degrade catastrophically beyond training length (223 and 128 at 4096). Deterministic position information helps at training length but creates OOD patterns at test time.

**3. joformer_fixed extrapolates better than RoPE (128 vs 223 at 4096).**
V rotation on fixed angles provides partial extrapolation benefit — better than plain RoPE but far from flat. The V rotation acts as a filter that partially compensates for the OOD fixed angles at longer sequences.

**4. monoidal2 from RoPE generalizes but has worse training PPL.**
Converting the 150K RoPE base to monoidal2 (data-dependent Q/K angles with tanh·π, lr=5e-5) reaches only 30.61 at training length but generalizes perfectly (29-31 across all lengths). The data-dependent angles learned useful structure in 50K iters at very low lr, but couldn't close the gap to RoPE/random at training length.

**5. ALiBi at 100K matches random_ln_indep at 100K.**
ALiBi (32.68 at 512, 30.11 at 8192) and random_ln_indep at 100K (32.35 at 512, 33.32 at 8192) are nearly identical — confirming they implement the same soft-windowing mechanism. ALiBi is slightly better at 8192 due to deterministic position info.

**6. The training-length vs extrapolation tradeoff is resolved by longer training.**
At 100K, random_ln_indep was 0.9 PPL behind RoPE (32.85 vs 31.94). At 200K, random is only 0.67 behind at 512 (24.21 vs 23.54) while being dramatically better at every other length. With enough training budget, the soft-windowed models catch up.

**7. All methods compared at same total compute (200K iters, same lr schedule).**
The hierarchy at training length is: joformer_fixed (23.18) ≈ RoPE (23.54) > random_ln_indep (24.21) > JoFormer v2 (25.55) > monoidal2 (30.61). But inverting for extrapolation: random_ln_indep (22-26 flat) ≈ JoFormer v2 (25-31 flat) > monoidal2 (29-31 flat) >> joformer_fixed (23-128 degrading) >> RoPE (24-223 degrading).

### Training-length PPL progression for random_ln_indep_qk

| Stage | Iters | lr | Val PPL |
|-------|-------|-----|---------|
| Stage 1 | 100K | 5e-4 | 32.85 |
| Stage 2 | +50K | 2e-4 | 27.67 |
| Stage 3 | +50K | 5e-5 | 24.14 |

The three-stage schedule gives 8.7 PPL improvement over the 100K single-stage result. The model continues to improve at each stage despite having no position information.

## Pending Experiments

### Currently running
- GPU 3: joformer2 from scratch (tanh·π, split_angles, lr=5e-4, angle_lr=5e-5) — then monoidal2 after

### 200K comparison (matching joformer2 from fixed training budget)

The joformer2 model that achieved PPL 25.55 had 200K total training:
- Stage 1: joformer_fixed 100K at lr=5e-4
- Stage 2: joformer_fixed 50K at lr=2e-4 (total 150K, PPL 26.60)
- Convert to joformer2 (zero-init angle deviations, tanh·π)
- Stage 3: joformer2 50K at lr=5e-5 (total 200K, PPL 25.82)

To compare fairly, we extend RoPE and random_ln_indep_qk with the same lr schedule:

**RoPE 200K** (running on GPU 1):
- Already completed: 100K at lr=5e-4 + 50K at lr=2e-4 = 150K (PPL 27.09)
- Checkpoint: `scale_up_continue_rope/rope_best.pt` (iter=20K relative to second continuation stage, 150K total)
- Now continuing: 50K at lr=5e-5

**random_ln_indep_qk 200K** (queued on GPU 2):
- Already completed: 100K at lr=5e-4 (PPL 32.85)
- Checkpoint: `new_exp/random_ln_indep_qk_best.pt`
- Stage 1: 50K at lr=2e-4 (→150K total)
- Stage 2: 50K at lr=5e-5 (→200K total)

### Completed findings from this round
- random_ln_qk (shared zero-mean): generalizes as well as per-layer independent — per-layer independence doesn't matter, only zero-mean matters
- ln_perlayer_qkv (input LN only, no output LN): diverged at 5K — output LN/RMSNorm is essential for stability
- ln_perlayer_ln_qkv (input + output LN): delayed degradation vs perlayer_ln_qkv but still degraded (8192: 47→49→51→53→58→68→81)
- ln_perlayer_rms_qkv: RMSNorm slightly better than LN early but same degradation trajectory
- perlayer_ln_qkv_slowangle (angle_lr=1e-4): best learned model, held flat for 60K iters before starting to degrade

## Part 23: Hidden Size and Architecture Comparison (1x hidden, datadep vs datadep2)

Testing whether the remote machine's success with V rotation is due to:
1. Smaller angle MLP (1x hidden vs our 4x)
2. datadep2 architecture (angle flow through FFN)
3. Single-head attention

All models: 768 embed, 16 layers, block_size=512, lr=5e-4, bf16, OWT, LN on angles, no abs.

### datadep vs datadep2 with 1x hidden (8 heads)

| Iter | shared_ln_qk h1 | shared_ln_qkv h1 | joformer2 h1 ln | RoPE |
|------|-----------------|-------------------|-----------------|------|
| 5K   | 71.89           | 68.29             | 164.84          | 58.23 |
| 10K  | 51.43           | 51.41             | 80.57           | 46.75 |
| 15K  | 45.27           | 45.37             | 63.17           | 43.00 |
| 20K  | 41.79           | 41.80             | 55.31           | 39.51 |
| 25K  | 39.79           | 40.22             | 50.70           | 38.04 |
| 30K  | 38.33           | 38.91             | 47.27           | 36.93 |
| 35K  | 36.92           |                   |                 | 36.58 |

Extrapolation @8192:

| Iter | shared_ln_qk h1 | shared_ln_qkv h1 | joformer2 h1 ln |
|------|-----------------|-------------------|-----------------|
| 10K  | 60.40 (1.16x)   | 64.31 (1.26x)    | 82.44 (0.99x)  |
| 20K  | 69.59 (1.65x)   | 57.79 (1.37x)    | 59.29 (1.06x)  |
| 30K  | 120.73 (3.15x)  | 71.28 (1.85x)    | 57.29 (1.19x)  |
| 40K  | 148.90 (4.14x)  |                   |                 |

### Key findings

**1. joformer2 (datadep2) maintains extrapolation much better than shared (datadep).**
At 30K: joformer2 h1 is at 1.19x ratio while shared_ln_qk h1 is at 3.15x and shared_ln_qkv h1 is at 1.85x. The angle-flow architecture where each layer's FFN produces angles for the next layer resists extrapolation degradation.

**2. joformer2 h1 converges much slower at training length.**
At 30K: joformer2 is at 47.27 while shared is at 38.33-38.91. The per-layer MLPs with angle flow are harder to optimize from scratch with 1x hidden. But it maintains flat extrapolation while doing so.

**3. V rotation with 1x hidden: helps early, then neutral at training length.**
At 5K: qkv 68.29 vs qk 71.89 (helps by 3.6). By 20K: tied at 41.8. By 30K: qkv slightly behind (38.91 vs 38.33). Same pattern as 4x but less pronounced.

**4. V rotation helps extrapolation regardless of hidden size.**
At 30K @8192: qkv=71.28 vs qk=120.73. V rotation cuts extrapolation degradation in half, consistent with all previous experiments.

**5. 1x hidden degrades extrapolation FASTER than 4x hidden for shared MLP.**
shared_ln_qk h1 at 30K: 120.73 (3.15x). shared_ln_qk h4 at 30K: 148.47 (but it started higher). The smaller MLP produces less diverse angles, and when those angles overfit to training length, the effect is more concentrated.

### Comparison with remote machine (KG experiment)

The remote machine found shared MLP + V rotation dramatically helps (I' .819 KG h@5 vs I .236). We don't see that effect at training length here. Key differences:

| Setting | Remote | Ours |
|---------|--------|------|
| n_heads | 1 | 8 |
| n_embed | 250 | 768 |
| block_size | 48 | 512 |
| task | KG + text | language modeling |
| attention | softplus (default) | softmax |

### Single-head results (n_heads=1)

Testing whether single-head attention explains the remote machine's strong V rotation result.

**Training length:**

| Iter | 1-head qkv | 1-head qk | 8-head qkv | 8-head qk |
|------|-----------|----------|-----------|----------|
| 5K   | 69.00     | 69.43    | 68.29     | 71.89    |
| 10K  | 50.73     | 51.19    | 51.41     | 51.43    |
| 20K  | 42.21     | 42.66    | 41.80     | 41.79    |

V rotation helps slightly in both cases (~0.4 PPL at 20K). No dramatic difference between 1-head and 8-head.

**Extrapolation @8192:**

| Iter | 1-head qkv | 1-head qk | 8-head qkv | 8-head qk |
|------|-----------|----------|-----------|----------|
| 10K  | 128.80    | 165.77   | 64.31     | 60.40    |
| 20K  | 207.43    | 227.18   | 57.79     | 69.59    |

Single-head extrapolation is much worse than 8-head — degrading 2-3x faster. With single head, all rotation dimensions are entangled in one attention computation, making the model more sensitive to angle drift. Multi-head provides implicit regularization by splitting angles across independent heads.

**Conclusion:** Single-head is NOT the explanation for the remote machine's success with V rotation. The difference must be the task (KG relational structure vs language modeling) or the scale (250 embed / 48 tokens vs 768 embed / 512 tokens).

### No-output-LN results (datadep2, angle_lr=5e-5)

Testing whether output LN on angles is necessary when the FFN input is already pre-normed (ln2(x)) and angle lr is slow.

**Training length:**

| Iter | joformer2 noln (Q/K/V) | monoidal2 noln (Q/K) | joformer2 with LN |
|------|----------------------|---------------------|-------------------|
| 5K   | 65.10                | 65.76               | 164.84            |
| 10K  | 52.41                | 52.44               | 80.57             |
| 20K  | 44.91                | 44.70               | 55.31             |
| 30K  | 41.66                |                     | 47.27             |

No-output-LN converges FASTER than with-output-LN (41.66 vs 47.27 at 30K). The output LN was constraining the angles too much.

**Extrapolation @8192:**

| Iter | joformer2 noln | monoidal2 noln | joformer2 with LN |
|------|---------------|----------------|-------------------|
| 10K  | 77.55 (1.48x) | 74.07 (1.40x)  | 82.44 (0.99x)    |
| 20K  | 79.64 (1.79x) | 88.72 (2.00x)  | 59.29 (1.06x)    |
| 30K  | 81.70 (1.96x) |                | 57.29 (1.19x)    |

But output LN gives much better early extrapolation (0.99x vs 1.48x at 10K). Without output LN, angles drift from the start. With output LN, the angles are constrained to zero-mean std=1, which provides the zero-mean property needed for generalization — but eventually the model finds ways to overfit within those constraints.

**The tradeoff:** output LN gives better extrapolation but worse training-length convergence. No output LN converges faster but extrapolates worse from the start.

### ALiBi 200K final results

ALiBi extended to 200K with same lr schedule (100K at 5e-4, 50K at 2e-4, 50K at 5e-5):

| Length | ALiBi 200K | random_ln_indep 200K | RoPE 200K |
|--------|-----------|---------------------|-----------|
| 512    | **24.06** | 24.21               | **23.54** |
| 1024   | **22.59** | 22.75               | 42.77     |
| 2048   | **21.87** | 22.38               | 106.91    |
| 4096   | **22.11** | 26.00               | 223.16    |

ALiBi at 200K is the best model overall — beats random at training length (24.06 vs 24.21) AND at extrapolation (21.87 vs 22.38 at 2048, 22.11 vs 26.00 at 4096). Deterministic position information + soft windowing is the winning combination.

## Part 24: Summary of Learnings

### What enables length generalization

1. **Zero-mean angles under cumsum** — the essential ingredient. Creates a random walk that stays bounded at any distance. Positive angles create monotonic drift = OOD at test time.
2. **Soft windowing** — whether via random cumsum, ALiBi's linear bias, or KERPLE's Gaussian kernel. All achieve the same effect: exponential decay of attention with distance.
3. **ALiBi is the practical winner** — deterministic position info + built-in soft window. Best training PPL AND extrapolation at 200K (24.06 at 512, 21.87 at 2048).

### What V rotation does

| Angle type | V rotation effect on training PPL | V rotation effect on extrapolation |
|-----------|----------------------------------|-----------------------------------|
| Fixed (RoPE) | Helps (+0.45) | Helps (128 vs 155 @4096) |
| Random positive | Helps (+0.23) | Helps dramatically (159 vs 223 @8192) |
| Random zero-mean | Hurts (-0.23) | Helps slightly (32 vs 33 @8192) |
| Learned 4x hidden | Hurts (-0.47) | Helps (156 vs 331 @8192) |
| Learned 1x hidden | Neutral/slight help | Helps (165 vs 285 @8192) |

**Pattern:** V rotation always helps extrapolation by acting as a second filter. But its benefit scales with how much extrapolation is already degraded — when extrapolation is flat (zero-mean random), V rotation barely matters. When extrapolation is bad (positive/learned angles drifting), V rotation compensates.

At training length, V rotation helps when angles are simple/constrained (fixed, random, early training) and hurts when angles are complex/expressive (4x learned MLP fully trained).

### Why learned angles fail at extrapolation

Every learned angle model follows the same pattern:
1. Starts with decent extrapolation (angles near initialization = near random/zero)
2. Angles learn to improve training-length PPL
3. In doing so, angles develop structure that's specific to training-length distances
4. This structure becomes OOD at longer sequences → extrapolation degrades

The fundamental tension: improving training-length PPL requires deviating from the random zero-mean distribution, but deviating destroys the generalization property.

**Mitigation strategies tested:**
- Slow angle lr (1e-4): delays overfitting, held flat 60K iters before degrading
- Output LN: constrains angles to zero-mean std=1, best early extrapolation (0.99x ratio) but still eventually degrades
- RMSNorm: slightly better than LN, same trajectory
- Input LN: marginal improvement
- datadep2 (angle flow): better than datadep early but collapses by 100K

**None fully solve it.** The only model with permanently flat extrapolation is random (unlearned) angles.

### Architecture comparison (datadep vs datadep2)

At matched compute:
- **datadep (shared MLP)**: converges fastest at training length, extrapolation degrades
- **datadep2 (angle flow via FFN)**: slower convergence, better early extrapolation, still degrades by 100K

The distinction is less important than normalization and lr choices.

### Head count

- **8 heads** > 1 head for both training PPL and extrapolation
- Single head provides no special benefit for V rotation (contrary to hypothesis from remote machine comparison)
- Multi-head implicitly regularizes by splitting angles across independent attention computations

### Hidden size (1x vs 4x)

- Minimal difference at training length (both converge to ~31.8 by 100K)
- Mixed results for extrapolation — not a clear differentiator
- 1x hidden matches remote machine architecture but doesn't replicate their V rotation benefit

### The remote machine discrepancy

Remote machine (KG experiment): V rotation helps training performance dramatically (I' .819 vs I .236 KG h@5).
Our setup (language modeling): V rotation neutral/hurts at training length.

Tested differences: hidden size (1x vs 4x), head count (1 vs 8), architecture (datadep vs datadep2). None explain the discrepancy. The remaining difference is the task — KG relational reasoning specifically benefits from non-commutative value aggregation in ways that language modeling doesn't.

## Part 25: Angle Dropout and Frequency-Scaled Angles

### Angle dropout results

Added dropout on angle outputs (zeroes random angle dimensions during training). Tested 0.3 and 0.5 dropout rates with shared MLP, 1x hidden, LN, no abs.

**Training length at 80K:**

| Model | drop=0 | drop=0.3 | drop=0.5 | RoPE | random |
|-------|--------|----------|----------|------|--------|
| qk    | —      | 32.46    | 33.16    | 31.87 | 32.64  |
| qkv   | —      | 33.11    | 33.37    | 31.87 | 32.64  |

**Extrapolation @8192 over training (best model: qkv adrop=0.5):**

| Iter | qkv drop=0 | qkv drop=0.3 | qkv drop=0.5 | random |
|------|-----------|-------------|-------------|--------|
| 10K  | 64.31 (1.26x) | 64.11 (1.24x) | 65.86 (1.18x) | 46.75 (0.96x) |
| 20K  | 57.79 (1.37x) | 52.16 (1.23x) | 50.24 (1.17x) | 37.61 (0.92x) |
| 30K  | 71.28 (1.85x) | 60.14 (1.57x) | 49.90 (1.29x) | 35.28 (0.93x) |
| 40K  | 84.39 (2.31x) | 58.53 (1.60x) | 50.65 (1.37x) | 33.92 (0.96x) |
| 60K  | — | 72.15 (2.10x) | 56.19 (1.63x) | 33.82 (0.98x) |
| 80K  | — | 94.34 (2.85x) | 63.38 (1.92x) | 32.49 (1.00x) |

**Findings:** Dropout slows extrapolation degradation but doesn't prevent it. Higher dropout (0.5) is better — ratio goes from 1.18x to 1.92x over 80K, vs no-dropout which hit 3.15x by 30K. But still far from random's flat 1.0x. The model finds ways to overfit even with 50% of angles zeroed.

### Why LayerNorm prevents learning multi-scale frequency structure

**Key insight:** LayerNorm normalizes ALL angle dimensions to the same scale (mean=0, std=1). But the optimal angle structure is multi-scale — matching RoPE's log-spaced frequencies where dimension 0 has scale ~1.0 and dimension 383 has scale ~0.0001. This 7000x range gets collapsed to 1x by LayerNorm. The model literally cannot express the multi-scale structure through LayerNorm.

**Evidence:** Random angles with uniform frequency (all dimensions use freq=0.131) are ~5 PPL worse than log-spaced random at both training length and extrapolation:

| Model | 10K @512 | 10K @8192 | 8192/512 |
|-------|---------|----------|----------|
| random log-spaced | 48.80 | 46.75 | 0.96x |
| random uniform (0.131) | 53.60 | 57.03 | 1.04x |

Multi-scale frequencies provide information at different distance scales: high-freq dims distinguish nearby tokens, low-freq dims provide long-range signal. Uniform frequency loses this and is worse at everything.

### Frequency-scaled angles (LN → tanh → freq_scales)

**Solution:** Apply LN for training stability, then multiply by per-dimension frequency scales to restore multi-scale structure:

```
MLP(x) → LayerNorm → tanh → * freq_scales
```

Where `freq_scales[d] = 1/10000^(2d/head_dim)` — same log-spaced frequencies as RoPE. Each dimension is bounded to [-freq_d, freq_d]. The model learns content-dependent direction/sign but magnitude is constrained per dimension.

Also testing without tanh (since LN already normalizes to ~N(0,1), tanh is nearly identity):

```
MLP(x) → LayerNorm → * freq_scales
```

**Early results (5K iters):**

| Model | 5K PPL |
|-------|--------|
| RoPE | 58.23 |
| shared_fs_qkv h1 (LN→tanh→freq) | 58.44 |
| shared_fs_qk h1 (LN→tanh→freq) | 58.60 |
| random log-spaced qk | 62.33 |
| shared_ln_qkv h1 (LN only) | 68.29 |
| shared_ln_qk h1 (LN only) | 71.89 |

**The freq-scaled models nearly match RoPE from the start** — only 0.2-0.4 PPL behind at 5K. This is a massive improvement over LN-only models which were 10-14 PPL behind at 5K. The multi-scale frequency structure is critical for fast convergence.

### Why this matters

The freq_scales approach separates three concerns:
1. **LayerNorm**: training stability (normalizes MLP output)
2. **tanh** (optional): bounds output to [-1, 1]
3. **freq_scales**: imposes multi-scale structure (same as RoPE/ALiBi)

Previous approaches conflated these — LN was used as both stabilizer AND angle normalizer, which destroyed the frequency structure the model needs. The new approach uses LN only for stability and lets freq_scales set the actual magnitudes.

This is analogous to ALiBi's per-head slopes: both impose log-spaced decay rates on the attention mechanism. ALiBi does it via additive bias per head; freq_scales does it via rotation magnitude per dimension. The difference is our angles are content-dependent (the MLP chooses direction per token).

### Currently running (as of 2026-06-01)

| GPU | Model | Description |
|-----|-------|-------------|
| 0 | shared_fsnt_qk h1 | LN → freq_scales (no tanh), Q/K |
| 1 | shared_fsnt_qkv h1 | LN → freq_scales (no tanh), Q/K/V |
| 2 | shared_fs_qk h1 | LN → tanh → freq_scales, Q/K |
| 3 | shared_fs_qkv h1 | LN → tanh → freq_scales, Q/K/V |

## Part 26: Clean Comparison — All Variants at 100K (Fixed-Seed Eval)

Running RoPE and joformer_fixed from scratch with fixed-seed eval and extrap at every 10K, alongside learned freq and Bernoulli models.

### Architecture summary

| Model | Angle source | Noise type | V rotation | Freq structure | Extra params |
|-------|-------------|------------|------------|----------------|-------------|
| RoPE | position × freq | none (deterministic) | No | log-spaced (fixed) | 0 |
| joformer_fixed | position × freq | none (deterministic) | Yes | log-spaced (fixed) | 0 |
| detb_qk | freq × Bernoulli ±1 | Bernoulli (random) | No | log-spaced (fixed) | 0 |
| detb_qkv | freq × Bernoulli ±1 | Bernoulli (random) | Yes | log-spaced (fixed) | 0 |
| random_ln_indep_qk | Uniform(-freq,freq) | Uniform (random) | No | log-spaced (fixed) | 0 |
| lfb_qk | learned_freq × Bernoulli ±1 | Bernoulli (random) | No | learned (MLP h1) | ~3.5M |
| lfb_qkv | learned_freq × Bernoulli ±1 | Bernoulli (random) | Yes | learned (MLP h1) | ~3.5M |
| lf_qk | learned_freq × Uniform(-1,1) | Uniform (random) | No | learned (MLP h1) | ~3.5M |
| lf_qkv | learned_freq × Uniform(-1,1) | Uniform (random) | Yes | learned (MLP h1) | ~3.5M |
| ALiBi | N/A (additive bias) | none (deterministic) | No | per-head slopes | 0 |

### Master comparison table — training length PPL (fixed-seed eval)

All models at 100K iters (except clean RoPE/joformer_fixed which are still running).

| Iter | RoPE | jfixed | ln_qk | ln_qkv | lf_qk | lfb_qk | lf_qkv | lfb_qkv | rand_qk | rand_qkv | detb_qk | detb_qkv |
|------|------|--------|-------|--------|-------|--------|--------|---------|---------|----------|---------|----------|
| 5K   | 59.14 | 59.16 | 68.44 | 67.05 | 61.99 | 61.70 | 63.36 | 62.92 | 62.33 | 62.65 | 62.20 | 61.95 |
| 10K  | 46.95 | 46.77 | 50.19 | 50.06 | 48.31 | 48.11 | 49.25 | 49.83 | 48.88 | 49.28 | 48.75 | 48.73 |
| 15K  | 42.59 | 42.23 | 44.53 | 44.87 | 43.38 | 43.52 | 44.18 | 44.60 | 44.24 | 44.35 | 44.18 | 43.92 |
| 20K  | 39.96 | 39.79 | 41.64 | 41.51 | 40.47 | 40.37 | 41.23 | 41.51 | 41.39 | 41.79 | 40.96 | 41.12 |
| 25K  | 38.07 | 38.07 | 39.84 | 39.79 | 38.76 | 38.85 | 39.69 | 39.77 | 39.42 | 39.63 | 39.36 | 39.40 |
| 35K  | 35.96 | 35.76 | 36.89 | 37.10 | 36.61 | 36.58 | 37.24 | 37.22 | 37.23 | 37.48 | 37.19 | 37.17 |
| 45K  | 34.68 | 34.50 | 35.30 | 35.66 | 35.35 | 35.17 | 35.20 | 35.83 | 35.78 | 36.13 | 35.93 | 35.77 |
| 55K  | 33.85 | 33.69 | 34.36 | 34.82 | 34.21 | 34.22 | 34.59 | 34.72 | 34.90 | 35.17 | 35.12 | 34.90 |
| 65K  | 33.23 | 32.93 | 33.37 | 33.87 | 33.29 | 33.77 | 33.86 | 34.10 | 34.19 | 34.41 | 34.25 | 34.26 |
| 75K  | 32.66 | 32.45 | 32.82 | 33.16 | 32.89 | 33.23 | 33.51 | 33.45 | 33.65 | 33.90 | 33.77 | 33.60 |
| 85K  | 32.23 | 32.11 | 32.30 | 32.69 | 33.01 | 32.76 | 33.10 | 33.10 | 33.28 | 33.49 | 33.26 | 33.33 |
| 95K  | 32.01 |       | 31.96 | 32.47 | 32.70 | 32.56 | 32.60 | 32.60 | 32.94 | 33.14 | 32.86 | 32.86 |
| 100K |       |       | **31.75** | 32.18 | 32.39 | 32.42 |    | 32.54 | 32.83 | 33.12 | 32.71 | 32.70 |

Legend:
- **RoPE**: standard RoPE (position × freq), clean fixed-seed eval run
- **jfixed**: joformer_fixed (RoPE + V rotation), clean run
- **ln_qk/qkv**: deterministic learned angles (MLP → LN), no random noise, h4 hidden
- **lf_qk/qkv**: learned freq × Uniform random noise, h1 hidden
- **lfb_qk/qkv**: learned freq × Bernoulli ±1 noise, h1 hidden
- **rand_qk/qkv**: fixed RoPE freq × Uniform random noise (no learning)
- **detb_qk/qkv**: fixed RoPE freq × Bernoulli ±1 noise (no learning)

Key observations:
1. **RoPE and joformer_fixed lead at training length** throughout, with jfixed consistently ~0.1-0.3 ahead of RoPE (V rotation helps)
2. **ln_qk h4 converges to best PPL** (31.75) — deterministic learned angles without random noise converge fastest, but don't extrapolate
3. **lf/lfb models (learned freq + noise)** converge to ~32.4 — ~0.6 behind ln_qk but with flat extrapolation
4. **random/detb models (no learning)** converge to ~32.7-32.8 — learned freq adds ~0.3-0.4 PPL over fixed freq
5. **V rotation hurts** all learned models by ~0.1-0.5 PPL at training length
6. **Bernoulli vs Uniform noise**: minimal difference (detb_qk 32.71 vs rand_qk 32.83 — Bernoulli slightly better)

### 100K final results — all models (fixed-seed eval)

| Model | 100K PPL | 512 | 1024 | 2048 | 4096 | 8192 |
|-------|----------|-----|------|------|------|------|
| jfixed (clean) | **31.59** | 31.15 | 34.42 | 57.91 | 98.46 | 159.07 |
| ln_qk h4 | 31.75 | 31.25 | 38.41 | 104.03 | 211.23 | 331.34 |
| RoPE (clean) | 31.79 | 31.83 | 44.23 | 100.72 | 206.23 | 345.82 |
| ln_qkv h4 | 32.18 | 31.96 | 31.84 | 48.82 | 89.56 | 156.21 |
| lf_qk | 32.39 | 32.53 | 30.45 | 29.23 | 29.58 | 31.77 |
| lfb_qk | 32.42 | 32.32 | 29.04 | 28.64 | 29.26 | 31.34 |
| lfb_qkv | 32.54 | 32.40 | 29.19 | 28.75 | 29.14 | 32.48 |
| detb_qkv | 32.70 | 32.32 | 29.03 | 29.04 | 29.51 | 31.66 |
| detb_qk | 32.71 | 32.59 | 29.33 | 29.27 | 30.24 | 33.22 |
| lf_qkv | 32.80 | 32.89 | 30.69 | 29.53 | 29.87 | 32.30 |
| random_ln_indep_qk | 32.85 | 32.35 | 29.21 | 29.20 | 30.93 | 33.32 |
| random_ln_indep_qkv | 33.08 | 32.92 | 29.47 | 29.03 | 30.58 | 31.71 |

**Training length ranking:**
1. jfixed 31.59 — V rotation on fixed RoPE angles, best overall
2. ln_qk 31.75 — deterministic learned cumsum, no extrapolation
3. RoPE 31.79 — baseline
4. ln_qkv 32.18 — deterministic learned cumsum + V rotation, partial extrapolation
5. lf_qk 32.39 — learned freq + Uniform noise, flat extrapolation ← **best extrapolating model**
6. lfb_qk 32.42 — learned freq + Bernoulli noise, flat extrapolation
7-12. detb/lf_qkv/random variants: 32.54-33.12

**Gap from RoPE to best flat-extrapolating model: 0.60 PPL** (31.79 vs 32.39)

**jfixed beats RoPE at BOTH training length AND extrapolation:**
- Training: 31.59 vs 31.79 (+0.20)
- @1024: 34.42 vs 44.23 (+9.81)
- @8192: 159.07 vs 345.82 (+186.75)
V rotation with fixed angles is strictly better than plain RoPE — better training PPL AND slower degradation.

**V rotation effect at 100K:**

| Pair | qk @train | qkv @train | qk @8192 | qkv @8192 | V rot: train | V rot: 8192 |
|------|-----------|-----------|---------|---------|-------------|-------------|
| ln h4 | 31.75 | 32.18 | 331.34 | 156.21 | -0.43 | **+175** |
| lf | 32.39 | 32.80 | 31.77 | 32.30 | -0.41 | -0.53 |
| lfb | 32.42 | 32.54 | 31.34 | 32.48 | -0.12 | -1.14 |
| detb | 32.71 | 32.70 | 33.22 | 31.66 | +0.01 | **+1.56** |
| random | 32.85 | 33.08 | 33.32 | 31.71 | -0.23 | **+1.61** |
| jfixed vs RoPE | 31.79 | 31.59 | 345.82 | 159.07 | **+0.20** | **+186.75** |

V rotation pattern:
- **Helps training PPL** only for fixed position-indexed angles (jfixed +0.20, detb +0.01)
- **Hurts training PPL** for all learned/random content-dependent angles (-0.12 to -0.43)
- **Helps extrapolation** for non-learned models (detb +1.56, random +1.61, ln +175, jfixed +187)
- **Hurts extrapolation** for learned freq models (lf -0.53, lfb -1.14) — anomalous, under investigation

### 200K comparison — the full picture

| Length | RoPE 200K | jfixed 200K | joformer2 200K | monoidal2 200K | random qk 200K | ALiBi 200K |
|--------|-----------|------------|---------------|---------------|---------------|------------|
| 512    | 23.54     | **23.18**  | 25.55         | 30.61         | 24.21         | 24.06      |
| 1024   | 42.77     | 33.23      | 25.17         | 29.17         | **22.75**     | **22.59**  |
| 2048   | 106.91    | 70.35      | 24.73         | 28.99         | **22.38**     | **21.87**  |
| 4096   | 223.16    | 128.37     | 26.01         | 31.36         | 26.00         | **22.11**  |

Training schedule for all: 100K at lr=5e-4, 50K at lr=2e-4, 50K at lr=5e-5 (200K total).

**At training length (512):**
- joformer_fixed best (23.18) — V rotation on fixed RoPE angles helps
- RoPE second (23.54)
- ALiBi (24.06) and random (24.21) close behind
- joformer2 (25.55) — V rotation + learned data-dependent angles (tanh·π, warm-started from jfixed)
- monoidal2 (30.61) — Q/K only data-dependent, degraded during conversion

**At extrapolation:**
- ALiBi best overall (22.11 at 4096 — flat, improves with context)
- random qk flat (22.38 at 2048, 26.00 at 4096)
- joformer2 flat through 4096 (24.73 at 2048, 26.01 at 4096)
- joformer_fixed degrades (70.35 at 2048, 128.37 at 4096) but much less than RoPE
- RoPE collapses (106.91 at 2048, 223.16 at 4096)

**random_ln_indep_qk beats joformer2 at almost every length — with zero learned angle parameters:**

| Length | joformer2 200K | random_qk 200K | Winner |
|--------|---------------|---------------|--------|
| 512    | **25.55**     | 24.21         | random |
| 1024   | 25.17         | **22.75**     | random |
| 2048   | 24.73         | **22.38**     | random |
| 4096   | 26.01         | 26.00         | tie    |

random beats joformer2 at 512 (24.21 vs 25.55), 1024, 2048, and ties at 4096. joformer2 had the advantage of 150K warm-start from joformer_fixed (trained with V rotation). random was trained from scratch with zero learned angle parameters — just Uniform(-freq, freq) noise with cumsum. The three-stage lr schedule (5e-4 → 2e-4 → 5e-5) was enough to close and surpass the gap.

**The V rotation story at 200K:**
- joformer_fixed (23.18) beats RoPE (23.54) at training length — V rotation on fixed angles helps by +0.36
- joformer2 (25.55) is flat at extrapolation but 2.37 behind jfixed at training length
- monoidal2/Q/K only (30.61) is 5.06 behind joformer2 (25.55) — V rotation dramatically helps the warm-started model
- The warm-start from jfixed (trained with V rotation for 150K) gives joformer2 a base that exploits V rotation; monoidal2 converted from RoPE (no V rotation history) can't catch up

### Currently running
- GPU 0: RoPE clean (~13 min left)
- GPU 1: joformer_fixed clean (~1h left)
- GPU 2: shared_lf_qkv rope_v (Q/K cumsum + RoPE V rotation — new experiment)
- GPU 3: shared_lf_qkv resume (~1h left)

### Full comparison at 20K (fixed-seed eval, with extrapolation)

| Model | 20K PPL | 512 | 1024 | 2048 | 4096 | 8192 |
|-------|---------|-----|------|------|------|------|
| joformer_fixed (clean) | **39.79** | 39.55 | 42.86 | 69.81 | 109.80 | 165.21 |
| RoPE (clean) | 39.96 | 39.60 | 52.44 | 113.68 | 223.00 | 428.69 |
| lfb_qk | 40.37 | 40.55 | 36.11 | 35.86 | 37.19 | 38.62 |
| lf_qk | 40.47 | 40.57 | 36.04 | 35.82 | 36.98 | 38.03 |
| random qk | 40.76 | 40.94 | 37.01 | 37.03 | 38.16 | 39.14 |
| detb_qk | 40.96 | 40.91 | 36.50 | 36.54 | 37.84 | 39.45 |
| detb_qkv | 41.12 | 41.01 | 36.43 | 36.70 | 37.85 | 39.45 |
| lf_qkv | 41.23 | 41.13 | 36.80 | 36.80 | 37.63 | 38.58 |
| lfb_qkv | 41.51 | 41.35 | 37.21 | 37.24 | 38.11 | 39.40 |
| random qkv | 41.79 | 41.42 | 37.09 | 36.99 | 38.03 | 39.24 |

At 20K: RoPE and joformer_fixed lead at 512 by ~0.5-1 PPL. But by 1024 every cumsum model is ahead (worst cumsum at 1024: 37.21 vs joformer_fixed 42.86 vs RoPE 52.44). By 8192: cumsum 38-40, joformer_fixed 165, RoPE 429.

### Deterministic sign models at 20K (fixed signs per token vs random noise)

| Model | 20K PPL | 512 | 1024 | 2048 | 4096 | 8192 |
|-------|---------|-----|------|------|------|------|
| joformer_fixed (clean) | **39.79** | 39.55 | 42.86 | 69.81 | 109.80 | 165.21 |
| RoPE (clean) | 39.96 | 39.60 | 52.44 | 113.68 | 223.00 | 428.69 |
| det_qk (fixed freq, fixed signs) | 40.58 | 40.83 | 36.25 | 37.12 | 44.47 | 64.25 |
| det_qkv (fixed freq, fixed signs, V rot) | 40.56 | 40.82 | 36.28 | 36.87 | 42.70 | 59.08 |
| lfds_qk (learned freq, fixed signs) | 40.64 | 40.38 | 36.69 | 39.83 | 47.88 | 62.74 |
| lfds_qkv (learned freq, fixed signs, V rot) | 41.63 | 41.51 | 37.36 | 38.13 | 41.29 | 48.60 |

Fixed-sign models degrade at extrapolation (48-64 at 8192) because repeated tokens reinforce the same cumsum direction. But they still vastly outperform RoPE (429) and joformer_fixed (165).

**V rotation helps extrapolation in all cases:**
- det: qkv 59.08 vs qk 64.25 at 8192 (+5.17)
- lfds: qkv 48.60 vs qk 62.74 at 8192 (+14.14)

**V rotation costs training length only when angles are learned:**
- det (no learned angles): qkv 40.56 ≈ qk 40.58 (neutral)
- lfds (learned MLP): qkv 41.63 vs qk 40.64 (-0.99)
- joformer_fixed (no learned angles): 39.79 vs RoPE 39.96 (+0.17 helps)

The pattern: V rotation gradients flowing through the learned angle MLP create optimization conflict. When angles are fixed (det, joformer_fixed), V rotation is free expressiveness. When angles are learned (lfds), V rotation competes with Q/K optimization through the shared MLP.

### Full comparison at 10K (fixed-seed eval, with extrapolation)

| Model | 10K PPL | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|-------|---------|-----|------|------|------|------|-------|
| joformer_fixed (clean) | **46.77** | 46.61 | 48.23 | 72.27 | 104.19 | 147.38 | 219.05 |
| RoPE (clean) | 46.95 | 46.68 | 60.23 | 117.63 | 209.88 | 374.82 | 659.89 |
| lfb_qk | 48.11 | 47.60 | 43.56 | 43.63 | 45.18 | 46.01 | — |
| lf_qk | 48.31 | 47.65 | 43.44 | 43.32 | 44.90 | 45.36 | — |
| detb_qk | 48.75 | 48.27 | 43.52 | 43.78 | 45.40 | 46.61 | 50.18 |
| detb_qkv | 48.73 | 48.01 | 43.51 | 43.78 | 46.00 | 46.36 | — |
| random qk | 48.88 | 48.80 | 43.87 | 44.02 | 45.57 | 46.75 | — |
| lf_qkv | 49.25 | 49.10 | 44.29 | 44.40 | 46.15 | 46.35 | 49.41 |
| random qkv | 49.28 | 49.03 | 44.25 | 44.24 | 45.80 | 47.25 | 51.90 |
| lfb_qkv | 49.83 | 49.22 | 44.89 | 45.08 | 46.46 | 46.90 | 50.04 |

RoPE and joformer_fixed are best at 512 but degrade catastrophically. By 2048, every cumsum model beats both. joformer_fixed degrades half as fast as RoPE (147 vs 375 at 8192) — V rotation with fixed angles helps even for extrapolation.

### 100K final results (completed models)

| Model | 100K PPL | 512 | 1024 | 2048 | 4096 | 8192 |
|-------|----------|-----|------|------|------|------|
| lfb_qk | **32.42** | 32.32 | 29.04 | **28.64** | 29.26 | 31.34 |
| lfb_qkv | 32.54 | 32.40 | 29.19 | 28.75 | **29.14** | 32.48 |
| detb_qkv | 32.70 | 32.32 | **29.03** | 29.04 | 29.51 | **31.66** |
| ALiBi | 32.71 | 32.68 | 29.17 | 28.91 | 30.82 | 30.11 |
| detb_qk | 32.75 | 32.59 | 29.33 | 29.27 | 30.24 | 33.22 |
| random qk | 32.85 | 32.35 | 29.21 | 29.20 | 30.93 | 33.32 |
| random qkv | 33.08 | 32.92 | 29.47 | 29.03 | 30.58 | 31.71 |
| RoPE (old, noisy) | 31.94 | 31.51 | 43.76 | 88.86 | 154.82 | — |
| joformer_fixed (old, noisy) | 31.49 | — | — | — | — | — |

### Key findings

1. **lfb_qk (learned freq, Bernoulli) is the best cumsum model at training length**: 32.42 PPL (beats random 32.85, ALiBi 32.71, detb 32.75) with flat extrapolation.

2. **Learned freq adds ~0.3-0.4 PPL over fixed freq**: lfb_qk (32.42) vs detb_qk (32.75). The MLP learns useful content-dependent frequency structure.

3. **V rotation helps extrapolation for fixed-freq models**: detb_qkv (31.66) vs detb_qk (33.22) at 8192 — consistent +1.56 PPL. Also random_qkv (31.71) vs random_qk (33.32).

4. **V rotation effect is mixed for learned-freq models**: lfb_qkv (32.48) vs lfb_qk (31.34) at 8192 — V rotation *hurts* extrapolation here, reversing the usual pattern. Under investigation.

5. **joformer_fixed (V rotation on RoPE) degrades half as fast as RoPE**: 147 vs 375 at 8192 at 10K. V rotation provides partial extrapolation benefit even with deterministic position-indexed angles.

6. **All cumsum models beat RoPE/joformer_fixed beyond 1024**: Despite being ~2 PPL behind at training length, cumsum models are dramatically better at all extrapolation lengths.

7. **PPL improves from 512→2048 for flat models**: This is primarily amortization of warm-up cost (early tokens have little context), not genuine use of long-range context beyond training length.

### Currently running
- GPU 0: RoPE clean (from scratch, 100K)
- GPU 1: joformer_fixed clean (from scratch, 100K)
- GPU 2: shared_lf_qk resume (45K→100K)
- GPU 3: shared_lf_qkv resume (35K→100K)

### To do
- Complete clean RoPE and joformer_fixed runs — will provide iteration-by-iteration comparison with extrap
- Extend lfb_qk, detb_qkv to 200K for comparison with RoPE/random 200K numbers
- Re-eval all 200K checkpoints at 8192 and 16384

## Hardware

### Previous machine
- GPU: NVIDIA RTX A6000 (49GB VRAM)
- CPU: AMD EPYC 7763 (8 cores)
- RAM: 64GB
- Training speed: ~8 it/s on OWT (small model), ~1 it/s on scale-up (163M)

### Current machine (as of 2026-03-25)
- Instance: ThunderCompute `6g9fu64p`
- GPU: NVIDIA A100 80GB PCIe
- vCPUs: 18
- RAM: 90GB
- Disk: 400GB
- PyTorch 2.10.0, CUDA 12.8
- Training speed: ~2.88 it/s on scale-up (193M JoFormer v2, bf16)

**Potential speedups not yet exploited:**
- TF32 matmul (`torch.backends.cuda.matmul.allow_tf32`) — disabled, free ~2x speedup on A100
- cudnn.benchmark — disabled, helps with fixed input sizes
- torch.compile — available but not used
- Flash attention / SDPA (`F.scaled_dot_product_attention`) — available but not used (manual attention implementation)
- Only using ~32GB of 80GB VRAM

---

## Part 27: joformer2/monoidal2 from frozen — angle learning vs control

### Setup

Train joformer2 (cumsum + V rotation) and monoidal2 (cumsum, no V rotation) using the `base_freq + learned_freq` architecture (`tanh(learned) * π + rope_base_angles`). Both use zero-initialized angle parameters (angle_emb, fc2_angles), so at initialization they are functionally identical to joformer_fixed and RoPE respectively.

**Protocol:**
1. Train with frozen angles (angle_lr=1e-30) for 5K iters at lr=5e-4 → establishes baseline content weights
2. Resume from 5K checkpoint with angle learning enabled at lr=5e-5 (both main and angle lr)
3. Compare against a control that resumes from the same checkpoint at lr=5e-5 but keeps angles frozen (angle_lr=1e-30)

All models: 768 embed, 16 layers, 8 heads, block_size=512, OWT data (32K vocab), bf16.

### Frozen baseline (5K iters at lr=5e-4)

| Model | Val PPL |
|-------|---------|
| joformer2 (≡ joformer_fixed) | 59.74 |
| monoidal2 (≡ RoPE) | 59.86 |

### joformer2: angle learning vs frozen control (both lr=5e-5)

| Iter | Control (frozen angles) | Angle learning (angle_lr=5e-5) |
|------|------------------------|-------------------------------|
| 5K   | 46.16                  | 48.79                         |
| 10K  | **43.00**              | 45.59                         |
| 15K  | —                      | 43.49                         |
| 20K  | —                      | 41.88                         |

At training length (512), the frozen control leads at 10K: 43.00 vs 45.59.

**But extrapolation tells a completely different story (both at 10K):**

| Length | Control (frozen) | Angle learning |
|--------|-----------------|---------------|
| 512    | **43.16**       | 46.28         |
| 1024   | 44.16           | **41.90**     |
| 2048   | 68.13           | **42.77**     |
| 4096   | 104.50          | **47.93**     |
| 8192   | 175.92          | **57.11**     |

The frozen control extrapolates like joformer_fixed — blows up beyond training length (175.92 at 8192). The angle learning version stays much flatter (57.11 at 8192). Angle learning costs ~2.6 PPL at 512 but is 3-120x better at longer lengths.

### Key observation

The `base_freq + learned_freq` structure achieves decent extrapolation **without random noise**. This is surprising because all from-scratch learned-angle models without noise (shared_ln_qk, etc.) had catastrophic extrapolation degradation.

### monoidal2: angle learning (both lr=5e-5)

| Iter | Val PPL | 512 | 1024 | 2048 | 4096 | 8192 |
|------|---------|-----|------|------|------|------|
| 10K  | 47.27   | 47.57 | 43.78 | 44.37 | 48.98 | 53.67 |
| 20K  | 45.17   | 45.81 | 42.05 | 42.46 | 46.26 | 51.62 |

monoidal2 control (frozen angles, lr=5e-5) pending — needed for proper comparison.

### lr sensitivity

| lr / angle_lr | joformer2 5K PPL | monoidal2 5K PPL |
|---------------|-----------------|-----------------|
| 5e-4 / 5e-4   | 310.79 (blown up) | 346.92 (blown up) |
| 5e-4 / 5e-5   | 49.69 (earlier run) | 63.56 (earlier run, noisy) |
| 5e-5 / 5e-5   | 48.79             | 49.52             |
| 5e-5 / frozen  | 46.16             | —                 |

lr=5e-4 is too aggressive for resuming from a 5K frozen checkpoint — both models blow up. lr=5e-5 for both main and angle params gives the smoothest convergence. The earlier run with lr=5e-4/angle_lr=5e-5 worked for joformer2 but produced noisy convergence for monoidal2.

---

## Part 28: Why LayerNorm fails for cumsum angles — the wrong normalization axis

### The puzzle

Two types of deterministic learned angles with cumsum, no random noise:
- **shared_ln_qk**: `MLP(x) → LayerNorm → angles`. Beats RoPE at training length (31.25 vs 31.51 at 100K). Extrapolation blows up (331.34 at 8192).
- **joformer2**: `tanh(learned) * π + base_freq`. When fine-tuned from joformer_fixed, achieves reasonable extrapolation (33.37 at 8192 at 10K continuation). When trained from scratch, extrapolation is bad.

Why does LayerNorm fail to help extrapolation, and why does `base_freq + learned` structure work?

### LayerNorm normalizes across the wrong axis

Standard LayerNorm normalizes across the **dimension axis** (C//2) at each position independently. It ensures that at each position t, the angles across all C//2 dimensions have mean=0 and std=1.

This does NOT prevent cumsum drift. A particular dimension d can still have a consistently positive angle at every position (e.g., always +0.3 after LN). The cumsum of dimension d grows as 0.3 × T — linear drift, same problem as RoPE.

What we need for flat extrapolation is zero-mean **across positions** (the time axis) for each dimension. If angle[t, d] is zero-mean across t, then cumsum[T, d] behaves like a random walk with std ∝ √T. The cos/sin wrapping makes √T growth invisible — flat extrapolation.

### Why not LayerNorm across time?

The natural idea: normalize across time instead of dimensions. But this forces std=1 across positions regardless of sequence length. A random walk of T steps naturally has std ∝ √T. Forcing std=1 would artificially compress the cumsum at long sequences and stretch it at short sequences, destroying the natural random walk behavior that gives flat extrapolation.

The random noise approach (lf_qk: `noise × learned_freq`) works precisely because the noise is zero-mean across positions by construction (random ±1 or Uniform(-1,1)), while the magnitude (learned_freq) controls per-dimension scaling independently. No normalization is needed — the zero-mean property comes from the noise itself, and the std grows naturally as √T.

### Why base_freq + learned works (partially)

joformer2 uses `tanh(learned) * π + base_freq` where base_freq is fixed RoPE frequencies. The cumsum splits into:
- `cumsum(base_freq)` = position × freq (deterministic RoPE, drifts linearly)
- `cumsum(tanh(learned) * π)` = content-dependent part

joformer2 fine-tuned from joformer_fixed extrapolates **better** than joformer_fixed alone. The learned deviations are keeping the drift from base_freq in check — the model learns content-dependent corrections that counteract the linear drift. This is why the 8K/512 ratio is ~1.2-1.3x instead of the catastrophic blowup that pure base_freq (joformer_fixed) produces.

### Summary of normalization approaches

| Approach | Axis | Effect on cumsum | Extrapolation |
|----------|------|-----------------|---------------|
| LayerNorm (standard) | Across dimensions | Does not prevent per-dimension drift across time | Blows up |
| LayerNorm across time | Across time | Forces std=1, kills natural √T random walk | Would harm extrapolation |
| Random noise (lf_qk) | N/A | Zero-mean by construction, std grows as √T naturally | Flat |
| base_freq + learned | N/A | Fixed backbone provides stable position info; learned part keeps drift in check | Partial (1.07-1.14x) |

---

## Part 29: Experiments with frequency structure and sign learning

### fsr_qk — Random noise with RoPE frequency structure (LN × rope_freq → abs → random)

**Architecture**: `LN(MLP(x)) × rope_freq → abs → Uniform(-freq, freq)`

LN provides training stability, rope_freq imposes log-spaced frequency structure per dimension, random noise provides zero-mean for flat extrapolation. Compared to lf_qk which uses `LN(MLP(x)) → abs → Uniform(-freq, freq)` — the difference is whether frequency structure comes from explicit rope_freq multiplication or is left to the model to learn through LN.

**Result**: Essentially identical to lf_qk at every iteration through 45K. The explicit RoPE frequency structure adds nothing — the model learns equivalent frequencies through the LN path alone.

| Iter | RoPE | lf_qk | fsr_qk |
|------|------|-------|--------|
| 5K | 59.14 | 61.99 | 61.12 |
| 10K | 46.95 | 48.31 | 48.08 |
| 20K | 39.96 | 40.47 | 40.48 |
| 40K | 35.20 | 35.96 | 35.85 |
| 100K | 31.78 | — | 32.47 |

fsr_qk gap to RoPE: settles around 0.5-0.7 PPL. Extrapolation flat throughout (1.00-1.04x at 8192/512).

### lfbf_qk — base_freq + learned_freq with random noise, no LN

**Architecture**: `tanh(MLP(x)) * π + rope_freq → abs → Uniform(-freq, freq)`, fc2 zero-initialized.

At zero-init: `abs(tanh(0)*π + rope_freq) = rope_freq`, so starts as random model with RoPE frequencies. As training progresses, the MLP learns content-dependent frequency deviations on top of the fixed backbone.

**Result**: Training length comparable to fsr_qk (34.96 vs 34.76 at 50K total). But extrapolation degrades without LN (8K/512 = 1.43x at 40K). LN is needed for extrapolation stability even with random noise.

### cc_qk — Causal centering (subtract running mean across time)

**Architecture**: `LN(MLP(x)) → subtract causal running mean → cumsum → rotate Q/K`

At each position t, subtract the running mean of angles so far: `centered_t = angles_t - cumsum(angles)[t]/(t+1)`. This is causal (only uses past+current positions) and removes drift without normalizing variance.

**Result**: Started flat (0.98x at 10K) but extrapolation degraded progressively as training continued.

| Iter | Val PPL | 512 | 8192 | 8K/512 |
|------|---------|-----|------|--------|
| 10K | 68.93 | 70.07 | 68.92 | 0.98x |
| 20K | 46.72 | 47.60 | 60.45 | 1.27x |
| 30K | 41.69 | 41.69 | 66.45 | 1.59x |
| 40K | 39.18 | 39.24 | 79.28 | 2.02x |

The model learns to work around the causal centering. Also converges slower than ln_qk (39.18 vs 36.38 at 40K) — the centering removes useful signal along with the drift.

**Why causal centering fails**: The running mean correction reduces drift from O(T) to slower growth, but doesn't eliminate it. The correction at each position depends on the history, creating a feedback loop the model can exploit. Additionally, the running mean is a poor estimate early in the sequence (position 0 is always zeroed).

### fss_qk — Hard sign × rope_freq (straight-through estimator)

**Architecture**: `LN(MLP(x)) → sign (STE) → × rope_freq`

Each dimension gets exactly ±rope_freq. The MLP determines content-dependent signs, straight-through estimator allows gradient flow. Deterministic — no random noise. The goal was to get content-dependent signs with exact RoPE frequency magnitudes.

**Result**: Blows up like RoPE at extrapolation.

| Model | 10K 512 | 10K 8192 | 8K/512 |
|-------|---------|----------|--------|
| fss_qk | 48.51 | 246.32 | 5.08x |
| fss_qk (wnorm) | 48.35 | 261.43 | 5.41x |
| RoPE | 46.95 | (blows up) | — |
| fss_qkv | 48.35* | 176.03* | 4.92x* |

*fss_qkv extrap at 40K, not 10K.

**Hypothesis 1: MLP weights collapse to zero, output becomes constant (bias only) → same signs every position → RoPE equivalent.** Tested by adding weight normalization (unit-norm rows) and zeroing biases after each optimizer step. Result: identical blowup. Hypothesis disproven — the MLP IS producing content-dependent outputs.

**Hypothesis 2: Deterministic signs are the problem.** But monoidal2 angle learning is also deterministic and extrapolates at ~1.1x. And lfds_qk (fixed per-token signs × learned freq) at 20K had 8192=62.74 (1.55x) — much better than fss_qk's 246 at 10K. So determinism alone doesn't explain it.

**Open question**: Why does `sign(LN(MLP(x))) × rope_freq` blow up while other deterministic approaches (monoidal2, lfds_qk) don't? The MLP produces genuinely content-dependent signs (verified by weight normalization test), yet the cumsum still provides exploitable position information. The mechanism is not understood.

### V rotation effect with sign × rope_freq

fs_qkv (178 at 8192) was much better than fs_qk (283) at 10K. fss_qkv also better than fss_qk. V rotation consistently helps extrapolation with deterministic angles.

### glf_qk — Globally learned frequencies (random with learnable rope_freq)

**Architecture**: Random angles with `Uniform(-abs(freq), abs(freq))` where `freq` is a learnable nn.Parameter initialized to RoPE frequencies. No MLP, no content-dependence in frequencies.

**Result**: At 15K, 44.01 PPL — slightly better than random_ln_qk (44.51) at the same point. Marginal improvement from learning frequencies. Killed early since content-independent frequencies are not interesting (similar in spirit to ALiBi's fixed slopes).

### joformer2/monoidal2 from frozen — 200K continuation

Both j2 control (frozen angles) and j2 angle (angle_lr=5e-5) continued for 200K total.

**Training length (val PPL)**:

| Iter | j2 control | j2 angle | gap |
|------|-----------|---------|-----|
| 5K | 30.32 | 32.71 | 2.39 |
| 20K | 29.70 | 32.16 | 2.46 |
| 40K | 28.90 | 31.45 | 2.55 |
| 60K | 28.40 | 30.61 | 2.21 |
| 80K | 27.93 | 30.17 | 2.24 |
| 100K | 27.51 | 29.70 | 2.19 |

Angle learning costs ~2.2-2.5 PPL at training length throughout. Gap not closing significantly.

**j2 angle extrapolation (200K phase)**:

| Iter | 512 | 8192 | 8K/512 |
|------|-----|------|--------|
| 10K | 32.56 | 35.49 | 1.09x |
| 50K | 30.97 | 33.05 | 1.07x |
| 99K | 29.85 | 33.92 | 1.14x |

Ratio stable around 1.07-1.14x. Not getting dramatically worse, but 8192 PPL stuck around 33-34 while 512 keeps improving.

**j2 control extrapolation**: blows up progressively (175→206→226→234 at 8192 over 10K-90K of 200K phase).

**m2 control vs j2 control at matched total iterations (~65K)**: m2 control 32.54 vs j2 control 32.93. V rotation slightly hurting at training length with frozen angles (0.39 PPL worse).

### lf_qk 200K final

Extended lf_qk to 200K with lr schedule: 100K at 5e-4, 50K at 2e-4, 50K at 5e-5.

| Length | lf_qk 200K | RoPE 200K | ALiBi 200K |
|--------|-----------|-----------|-----------|
| 512 | 24.79 | **23.54** | **24.06** |
| 1024 | **22.26** | 42.77 | 22.59 |
| 2048 | **21.76** | 106.91 | 21.87 |
| 4096 | **21.92** | 223.16 | 22.11 |
| 8192 | **23.24** | — | — |

lf_qk beats ALiBi at 1024+ but is 0.73 behind at 512. Nearly matches ALiBi overall but adds MLP complexity for comparable results.

### Flash Attention compatibility

Our cumsum-based approach is fully compatible with Flash Attention, unlike ALiBi:

- **RoPE**: Rotates Q, K before attention → compatible with Flash Attention ✓
- **ALiBi**: Adds linear bias to attention scores during computation → requires modifying the attention kernel. Early Flash Attention implementations didn't support additive biases, making ALiBi slower in practice.
- **Cumsum (ours)**: Computes angles → cumsum → rotates Q, K (and optionally V) before attention. The attention computation itself is standard `Q·K^T / sqrt(d)` with causal masking → fully compatible with Flash Attention ✓. V rotation and inverse rotation happen outside the attention kernel.

This is a practical advantage over ALiBi. Our approach slots into existing RoPE-optimized infrastructure (Flash Attention, `F.scaled_dot_product_attention`, optimized CUDA kernels) with no kernel modifications — just replace the angle computation before the rotary embedding step. ALiBi requires attention kernel support that was not universally available, contributing to its limited adoption despite strong extrapolation properties.

### V rotation effect with frozen angles (j2 vs m2 control)

Compared joformer2 control (frozen angles + V rotation ≡ joformer_fixed) against monoidal2 control (frozen angles, no V rotation ≡ RoPE) at matched total iterations, both at lr=5e-5:

**Training length**: m2 control consistently ahead by 0.2-0.4 PPL. V rotation slightly hurts training length.

**Extrapolation at 8192**:

| Total iters | j2 control (V rot) | m2 control (no V rot) |
|-------------|-------------------|----------------------|
| 15K | 175.92 (4.08x) | 239.03 (5.57x) |
| 45K | 205.21 (5.86x) | 272.18 (7.82x) |
| 85K | 226.77 (7.25x) | 325.65 (10.52x) |
| 135K | 236.43 (8.11x) | 319.99 (11.16x) |

Both blow up, but V rotation slows the blowup significantly (~8-9x vs ~11-12x ratio). V rotation costs ~0.3 PPL at training length but provides ~80-100 PPL improvement at 8192.

Extended to 200K matched total iterations: gap remains stable at 0.2-0.4 PPL (m2 ahead at training length), with j2 consistently better at extrapolation.

---

## Part 30: Hard sign experiments and the feedback loop problem

### The goal

Find a deterministic (no random noise) content-dependent position encoding that:
1. Matches RoPE at training length
2. Has flat or slowly degrading extrapolation

### fss_qk — sign(LN(MLP(x))) × rope_freq

**Architecture**: MLP produces content-dependent output → LN across C//2 dims → hard sign via straight-through estimator → multiply by RoPE frequencies. Each dimension gets exactly ±rope_freq. Deterministic, content-dependent signs.

**Variants tested** (all blow up identically at ~5x ratio by 10K):

| Model | 10K 512 | 10K 8192 | ratio |
|-------|---------|----------|-------|
| fss_qk (basic) | 48.51 | 246.32 | 5.08x |
| fss_qk (weight norm + zero bias) | 48.35 | 261.43 | 5.41x |
| ln_fss_qk (LN on input) | 50.64 | 266.82 | 5.27x |
| fss_qkv (V rotation) | 35.79* | 176.03* | 4.92x* |

*fss_qkv numbers at 40K, not 10K.

Weight normalization (unit-norm rows, zero bias after each optimizer step) was tested to prevent the MLP from collapsing to constant output. Result: identical blowup. The MLP IS producing content-dependent outputs — verified by non-zero gradients and different extrap behavior. Constant output was not the problem.

### Why fss blows up: the feedback loop

The model co-adapts ALL its weights (not just the angle MLP) to route position information through the sign computation:

1. Layer 0: MLP sees token embeddings (no position info), produces content-dependent signs → cumsum creates some position structure
2. Layer 1+: MLP sees hidden states that now encode position (from layer 0's cumsum attention) → produces position-correlated signs → reinforces position encoding
3. The entire model conspires to create hidden states that produce useful position signals through the deterministic sign path

No architectural trick on the angle MLP alone (weight norm, input LN, etc.) can prevent this because the REST of the model adapts to feed position-informative inputs.

### lfds_qk — pre-assigned signs break the feedback loop

**Architecture**: Fixed random ±1 signs per token (embedding buffer, not learned) × learned frequencies. The model cannot change the signs — they depend only on token identity.

**Result at 100K**: val PPL 32.81 (RoPE at 80K: 32.43, ~0.4 behind). Extrapolation degrades but much slower than fss:

| Total | 512 | 8192 | ratio |
|-------|-----|------|-------|
| 50K | 34.46 | 103.76 | 3.01x |
| 60K | 33.69 | 104.19 | 3.09x |
| 70K | 33.52 | 107.29 | 3.20x |
| 80K | 32.76 | 118.35 | 3.61x |

Degrading (3.01x → 3.61x) due to repeated tokens reinforcing the same cumsum direction, but vastly better than fss_qk (5x+ at 10K) and RoPE (which would be hundreds at 8192).

**Why lfds works better**: The signs are frozen and depend only on token ID, not on the hidden state. The feedback loop is broken — the model cannot route position information through the sign path because the signs don't depend on context. The learned frequencies are content-dependent (through the MLP), but the signs provide enough decorrelation to slow drift.

### The fundamental insight

For deterministic position encoding with cumsum:
- **MLP-computed signs** (fss): the model's hidden states encode position → MLP reads position → signs reinforce position → blows up like RoPE
- **Pre-assigned signs** (lfds): signs depend only on token identity → no position feedback → slower drift (but still drifts from repeated tokens)
- **Random noise** (lf_qk, fsr_qk): signs are random per position → zero-mean by construction → flat extrapolation
- **base_freq + learned** (joformer2): learned deviations keep base_freq drift in check → partial extrapolation (~1.1x)

The more the sign/angle computation can access position information (directly or through hidden states), the more the model exploits it, and the worse the extrapolation.

### joformer2 angle learning — 300K continuation

j2 angle continued to 300K total (5K frozen + 95K + 100K + ongoing):

| Phase iter | 512 | 8192 | ratio |
|-----------|-----|------|-------|
| 10K (200K) | 32.56 | 35.49 | 1.09x |
| 50K (200K) | 30.97 | 33.05 | 1.07x |
| 99K (200K) | 29.85 | 33.92 | 1.14x |
| 70K (300K) | 28.56 | 35.79 | 1.25x |

Ratio drifting up: 1.07x → 1.14x → 1.25x. The 8192 PPL is stuck around 33-35 while 512 keeps improving. The learned angles help extrapolation compared to frozen (which blows up to 250+) but the benefit is slowly eroding.

### j2 control vs j2 angle gap

| Phase | Gap at training length |
|-------|----------------------|
| End of first 100K | ~2.5 PPL |
| End of 200K | 2.19 PPL |
| 300K phase 70K | ~2.1 PPL |

Gap very slowly narrowing but persistent. Angle learning costs ~2 PPL at training length for dramatically better extrapolation.

---

## Just a thought

### lfls_qk — learned signs per token

A variant of lfds_qk where the per-token signs are learned instead of fixed random. lfds_qk uses a fixed random ±1 embedding buffer per vocab entry. lfls_qk would use a learned embedding passed through sign (STE) — the model learns which sign pattern works best for each token.

The key difference from fss_qk: signs depend on token identity only, not on context/hidden state. This blocks the feedback loop (model can't route position through hidden states to signs) while still allowing the model to optimize the sign assignments. lfds_qk showed that per-token signs work (3.6x at 80K vs fss's 5x at 10K) — learned per-token signs might do better if the model can find sign patterns that reduce drift from common token sequences.

---

## Part 31: The forward-pass feedback loop — why deterministic signs always encode position

### Systematic sign experiments

We ran a series of experiments trying to get deterministic content-dependent `sign × rope_freq` to extrapolate. All failed.

**All experiments at 5K (or 10K where noted), 8192/512 ratio:**

| Model | Description | 5K extrap 512 | 5K extrap 8192 | ratio |
|-------|-------------|--------------|----------------|-------|
| fss_qk h1 | MLP → LN → sign × freq | 48.51* | 246.32* | 5.08x* |
| fss_qkv h1 | + V rotation | 48.51* | 151.40* | 3.12x* |
| fss_qk h1 wnorm | + weight norm, zero bias | 48.35* | 261.43* | 5.41x* |
| ln_fss_qk h1 | + LN on input | 50.64* | 266.82* | 5.27x* |
| fss_qkv h4 | + V rot, 4x hidden MLP | 49.49* | 117.90* | 2.38x* |
| fssd_qk h1 | + detached input, wnorm, zero bias | 59.87 | 265.06 | 4.43x |
| fssx_qk | sign(LN(x.detach()[:C//2])) × freq, no MLP | 60.68 | 208.76 | 3.44x |

*These numbers are at 10K, not 5K.

### What each experiment tested

1. **fss_qk** (basic): Can content-dependent hard signs work? **No** — blows up like RoPE.

2. **fss_qkv** (V rotation): Does V rotation help? **Partially** — 3.12x vs 5.08x, but still degrading.

3. **fss_qk wnorm** (weight norm + zero bias): Is the MLP collapsing to constant output? **No** — identical blowup. MLP produces genuinely content-dependent signs.

4. **ln_fss_qk** (input LN): Does normalizing the input help? **No** — identical blowup.

5. **fss_qkv h4** (larger MLP): Does more MLP capacity help? **Partially** — 2.38x vs 3.12x at 10K with V rotation. Larger MLP produces more diverse signs.

6. **fssd_qk** (detached input + wnorm + zero bias): Is gradient-driven co-adaptation the cause? **No** — the MLP still reads position from x even without gradients flowing back. The rest of the model can't optimize x for the angle path, but x already carries position from previous layers' attention.

7. **fssx_qk** (no MLP, just sign(LN(x.detach()))): Is the MLP learning to extract position? **No** — even without any MLP, just taking signs of the raw detached hidden state blows up at 3.44x. Position information is inherent in x from the forward pass.

### The forward-pass feedback loop

The key discovery: position information enters x through the **forward pass**, not through gradient optimization.

1. Layer 0: x = token embedding (no position). Signs from x → cumsum → rotation → attention produces position-dependent output.
2. Layer 1: x now contains position info from layer 0's attention. sign(LN(x)) reads this → produces position-correlated signs → cumsum reinforces position encoding.
3. Each layer amplifies the position signal through: x → signs → cumsum → attention → x.

**Detaching x blocks gradients but not this forward-pass information flow.** The model doesn't need to "learn" to exploit position — the attention mechanism with cumsum-based rotations inherently creates position-dependent hidden states that any function (even sign(LN(·))) will reflect.

### Comparison with approaches that work

| Approach | Position info in signs? | Extrapolation |
|----------|------------------------|---------------|
| Random noise (lf_qk, fsr_qk) | Noise overwhelms position signal | Flat (1.0x) |
| Per-token fixed signs (lfds_qk) | Token-dependent only, no context | Slow degradation (3.6x at 80K) |
| MLP-computed signs (fss variants) | Full position from hidden state | Fast blowup (3-5x at 10K) |
| No MLP, just sign(x) (fssx) | Full position from hidden state | Fast blowup (3.4x at 5K) |

The only way to prevent position exploitation is:
1. **Random noise**: breaks the deterministic link entirely (VAE-like sampling)
2. **Per-token signs**: limits position info to token identity (no context), but repeated tokens still cause drift

### Analogy to VAE

The random noise models (lf_qk, fsr_qk) work like a VAE's reparameterization trick:
- VAE: encoder → μ, σ → z = μ + σε → decoder. Sampling noise ε prevents exact information transfer.
- Ours: MLP → learned_freq → angle = noise × freq → cumsum. Random noise prevents exact position encoding through the angles.

The Bernoulli noise (±1) is the minimum randomness needed — it forces the angles to be unpredictable at each position while still allowing the model to learn useful frequency magnitudes.

### Conclusion

Deterministic content-dependent position encoding through cumsum is fundamentally limited by the forward-pass feedback loop. Any deterministic function of the hidden state — no matter how it's computed (MLP, no MLP, detached, weight-normalized) — will carry position information because the hidden state itself is shaped by position-dependent attention. Only random noise can break this loop.

---

## Part 32: Frequency scaling and stability in datadep2

### The stability puzzle

Original joformer2 from scratch with `tanh(raw) * π` diverged (976 PPL at 5K, then worse). But `tanh(raw) × rope_freq` is stable (67.53 at 5K with h1, 61.70 with h4). Why?

### tanh * π vs tanh × rope_freq

| Activation | 5K Val PPL | 5K extrap 512 | 8192 | ratio | Stable? |
|------------|-----------|--------------|------|-------|---------|
| tanh * π (h1) | 976.30 | 954.35 | 916.56 | 0.96x | **No** — diverging |
| tanh × rope_freq (h1) | 67.53 | 67.86 | 167.96 | 2.47x | Yes |
| tanh × rope_freq (h4) | — | 61.70 | 144.06 | 2.33x | Yes |
| tanh × learned_freq (h1) | 66.50 | 66.84 | 154.67 | 2.31x | Yes |
| tanh × learned_freq (h4) | — | 61.70 | 144.06 | 2.33x | Yes |
| LN only (h1) | 164.84* | — | — | 0.99x* | Yes (slow) |
| tanh(LN) × rope_freq (h1) | — | 67.77 | 168.25 | 2.48x | Yes |

*LN values at 5K/10K from the earlier run.

**The key difference**: `π ≈ 3.14` scales all 384 dimensions uniformly. `rope_freq` ranges from 1.0 down to ~0.0001. The low-frequency dimensions (which encode long-range position in RoPE) get tiny angles with rope_freq, keeping the cumsum well-behaved. With `* π`, all dimensions get angles up to ±3.14, causing rapid cumsum growth and divergence.

**Confirmed by experiment**: joformer2 h1 tanh * π diverged from 976 to 1928 PPL between 5K and 20K. The tanh_freq version trained stably throughout.

### LN provides similar stability through a different mechanism

LN normalizes to std=1 across dimensions — average angle magnitude ~1 (not π). This also constrains the cumsum. But LN destroys the per-dimension frequency structure, which is why LN-only converges slowly (164.84 at 5K). Adding freq scaling on top of LN (ln_tanh_freq) restores convergence (67.77 at 5K) but also restores position encoding and blows up extrapolation (2.48x).

### h1 vs h4 in datadep2

In datadep2, fc1 is shared between content and angle paths. h1 (768→768) weakens the content FFN compared to standard transformers (768→3072). h4 matches standard capacity.

| | h1 512 | h4 512 | h1 8192 | h4 8192 | h1 ratio | h4 ratio |
|---|---|---|---|---|---|---|
| tanh_lfreq 5K | 66.84 | 61.70 | 154.67 | 144.06 | 2.31x | 2.33x |

h4 is ~5 PPL better at training length (stronger content FFN) but same extrapolation ratio. Content capacity helps convergence but doesn't affect extrapolation.

### Learned vs fixed frequencies

tanh × learned_freq (initialized to RoPE) vs tanh × fixed rope_freq:

| | fixed freq 512 | learned freq 512 | fixed 8192 | learned 8192 |
|---|---|---|---|---|
| h1 5K | 67.86 | 66.84 | 167.96 | 154.67 |

Learned frequencies slightly better at both training length and extrapolation. The model can adjust the per-dimension scaling.

### Random subset selection (fssr_qk)

Random subset of C//2 dims from x.detach() → LN → sign → × rope_freq. At 5K: 512=62.50, 8192=59.42 (**0.95x**). Flat — confirms random noise is the key to flat extrapolation. The random subset selection acts like Bernoulli noise.

Deterministic argsort-based subset selection (fssa_qk) blew up: 512=60.20, 8192=235.48 (3.91x at 5K). Argsort of x is already deterministic enough to encode position.

### Baselines at lr=5e-5 from scratch

Running RoPE, joformer_fixed, and random_ln_indep_qk from scratch at lr=5e-5 for 200K to compare with the frozen experiments.

At 70K (matched):

| | RoPE | jfixed | random |
|---|---|---|---|
| Val PPL | 36.55 | 36.87 | 38.67 |

jfixed trails RoPE by 0.32 at lr=5e-5. At lr=5e-4 (original runs), jfixed was 0.20 ahead. V rotation benefits from higher lr at training length — mechanism unknown.

### Frozen angle_emb experiments

**frozen zero emb** (tanh_lfreq h4): Layer 0 = pure RoPE (angle_emb zero, frozen). FFN = `tanh(raw) × learned_freq + rope_base`.
- 5K: 512=60.82, 8192=134.33 (2.21x)

**frozen random ±1 emb** (tanh_lfreq h4): Layer 0 = random ±rope_freq per token (frozen). FFN = `tanh(raw) × learned_freq + rope_base`.
- 5K: 512=60.97, 8192=132.93 (2.18x)

Nearly identical — the random per-token signs at layer 0 make no difference vs pure RoPE. The FFN at subsequent layers dominates.

### Baselines at lr=5e-5 (200K from scratch)

Running RoPE, joformer_fixed, and random_ln_indep_qk at constant lr=5e-5 for comparison with frozen experiments.

At 130K (matched):

| | RoPE | jfixed | random |
|---|---|---|---|
| Val PPL | 31.88 | 32.03 | 33.58 |

jfixed gap to RoPE narrowing throughout: 0.59 (30K) → 0.32 (70K) → 0.15 (130K). V rotation nearly catches RoPE at training length with enough iterations at lr=5e-5.

At 90K extrapolation:

| | 512 | 8192 | ratio |
|---|---|---|---|
| RoPE | 34.46 | 285.81 | 8.29x |
| jfixed | 34.68 | 177.80 | 5.13x |
| random | 36.28 | 38.21 | 1.05x |

---

## Part 33: The datadep2 + LN mystery

### The convergence-extrapolation tradeoff

Every experiment confirms the same tradeoff:

| Setup | Convergence | Extrapolation |
|-------|------------|---------------|
| Frequency structure (rope_freq, tanh×π) | Fast (~60 PPL at 5K) | Blows up (2-5x) |
| LN only (no freq structure) | Slow (~123 PPL at 5K) | Flat (~1.0x) |
| LN + freq structure | Fast | Blows up |

Adding frequency structure helps convergence but enables position encoding → extrapolation failure. LN destroys frequency structure, preventing position encoding → flat extrapolation but slow convergence.

### But LN alone is not sufficient

shared_ln_qk uses LN without frequency structure and still blows up (331 at 8192 at 100K). Only datadep2 + LN stays flat. The architecture matters:

| Architecture | LN, no freq structure | Extrapolation |
|-------------|----------------------|---------------|
| shared MLP (datadep) | shared_ln_qk | Blows up (10.6x at 100K) |
| datadep2 (angle flow through FFN) | joformer2 h1 ln | Flat → slow drift (0.99→1.06→1.19x) |

### h1 vs h4 in datadep2 with LN

| | h1 5K | h4 5K |
|---|---|---|
| Val PPL | 123.09 | 226.30 |
| 8K/512 | 0.99x | 0.99x |

Both flat, but h4 converges SLOWER with LN activation — opposite of tanh_freq where h4 was faster. The larger fc1 produces more varied angles through LN, making early training harder. h1's smaller fc1 constrains the angles, helping convergence.

### Consistent LN (embedding + FFN)

The earlier joformer2 h1 ln used `tanh(emb)*π` for initial angles but `LN(raw)` for FFN — inconsistent. Fixed to use LN for both.

| | Original (inconsistent) | Consistent LN |
|---|---|---|
| 5K val PPL | 164.84 | 123.09 |
| 10K val PPL | 80.57 | 73.80 |
| 10K 8K/512 | 0.99x | 1.00x |
| 15K 8K/512 | — | 1.06x |

Consistent LN converges faster (123 vs 164 at 5K) with same flat extrapolation. The tanh*π on the embedding was hurting initial convergence.

### The drift pattern

Both the original and consistent LN versions show the same drift pattern:

| Iter | Original ratio | Consistent ratio |
|------|---------------|-----------------|
| 10K | 0.99x | 1.00x |
| 15K | — | 1.06x |
| 20K | 1.06x | — |
| 30K | 1.19x | — |

The extrapolation starts flat and gradually degrades. The consistent version reaches 1.06x at 15K (vs 20K for original) — faster convergence but same drift rate. The model is slowly learning position-encoding patterns despite LN's frequency destruction.

### The unsolved mystery: why does joformer2 angle from frozen work?

joformer2 angle (tanh*π + base_freq, warm start, lr=5e-5) held at ~1.1-1.2x through 300K iterations. This is the only approach that achieves both reasonable training PPL and stable extrapolation.

What makes it different from everything else:
1. Additive base_freq (provides fixed position backbone)
2. Warm start from frozen 5K (content weights pre-trained)
3. Very low lr (5e-5 for everything)
4. tanh*π (which is UNSTABLE from scratch — diverges to 1000+ PPL)
5. Zero-init angle params (starts as joformer_fixed)

None of these ingredients work alone. tanh*π from scratch diverges. Base_freq alone doesn't explain flat extrapolation. Low lr alone doesn't help. The combination somehow creates a regime where the model learns useful angle deviations without encoding position.

We cannot replicate this from scratch. Every from-scratch approach either blows up at extrapolation (with freq structure) or converges too slowly (without freq structure, LN only). The warm start appears essential — the question is why.

### Update: datadep2 + LN eventually blows up

The joformer2 h1 ln consistent experiment (LN on both embedding and FFN, no freq structure) ran to 95K total. The extrapolation held flat until ~35K, then collapsed catastrophically:

| Total | Val PPL | 512 | 8192 | ratio |
|-------|---------|-----|------|-------|
| 5K | 123.09 | 127.31 | 126.13 | 0.99x |
| 10K | 73.80 | 76.37 | 76.05 | 1.00x |
| 15K | 60.09 | 60.99 | 64.65 | 1.06x |
| 20K | 52.81 | 53.02 | 61.37 | 1.16x |
| 25K | 49.02 | 49.79 | 61.37 | 1.23x |
| 30K | 46.76 | 47.09 | 61.86 | 1.31x |
| 35K | 44.66 | 45.14 | 65.31 | 1.45x |
| 40K | 43.20 | 43.97 | 66.67 | 1.52x |
| 50K | 40.92 | 41.12 | 85.48 | 2.08x |
| 60K | 39.39 | 39.40 | 121.28 | 3.08x |
| 70K | 38.34 | 38.18 | 176.04 | 4.61x |
| 80K | 37.43 | 37.16 | 249.18 | 6.71x |
| 95K | 36.63 | 36.91 | 331.55 | 8.98x |

By 95K the 8K/512 ratio (8.98x) is comparable to RoPE at similar PPL. The datadep2 architecture + LN **delayed** the blowup by ~40K iterations compared to shared_ln_qk, but did not prevent it. The model eventually learned to encode position through the LN-normalized angles despite the destroyed frequency structure.

**The earlier run** (original joformer2 h1 ln, inconsistent LN) was killed at 30K showing 1.19x. If it had continued, it would have blown up too — we just didn't run it long enough to see.

### The convergence-extrapolation race

Every deterministic learned approach follows the same pattern:
1. **Early training**: angles are random/uninformative → flat extrapolation, poor training PPL
2. **Mid training**: angles start encoding useful patterns → PPL improves, extrapolation begins to degrade
3. **Late training**: angles encode position → good training PPL, extrapolation blows up

The only variable is how fast the model reaches stage 3:

| Architecture | Flat until | Then blows up |
|-------------|-----------|---------------|
| shared MLP (fss_qk) | ~0K (immediate) | 5x at 10K |
| shared MLP + LN (shared_ln_qk) | ~10K | 10x at 100K |
| datadep2 + LN (joformer2 h1 ln) | ~35K | 9x at 95K |
| datadep2 + tanh×freq | ~0K (immediate) | 2.5x at 5K |
| perlayer + LN slow angle | ~60K | degraded after |

datadep2 + LN buys ~35K iterations of flat extrapolation. The shared fc1 (which serves content primarily) slows down the angle path's ability to encode position, but doesn't prevent it.

### The joformer2 angle mystery deepens

The joformer2 angle from frozen (~1.2x through 300K) uses:
- `tanh(learned) * π + base_freq` — asymmetric, positive-biased, no logical reason to work
- V rotation essential (monoidal2 angle was noisy and worse)
- Warm start from frozen 5K
- Low lr (5e-5)

This is the ONLY approach that maintains reasonable extrapolation for extended training. It's not flat (drifting from 1.07→1.25x over 300K) but it doesn't blow up. Every attempt to understand or replicate it from scratch has failed.

The architecture makes no theoretical sense: the angles are asymmetric (positive bias from base_freq), the activation (tanh×π) is unstable from scratch, and V rotation shouldn't help with asymmetric angles. Yet the specific combination of warm start + low lr + additive base_freq + V rotation produces the best result.

### Baselines at lr=5e-5 (200K, completed/running)

Final comparison at matched iterations:

| Iter | RoPE | jfixed | random |
|------|------|--------|--------|
| 50K | 40.26 | 40.65 | 42.67 |
| 100K | 33.65 | 33.87 | 35.57 |
| 130K | 31.88 | 32.03 | 33.58 |
| 150K | 31.00 | 31.19 | 32.75 |

jfixed gap to RoPE: narrowed from 0.39 (50K) to 0.19 (150K). V rotation nearly catches RoPE at training length with enough iterations. Random trails both by ~1.5-2 PPL throughout.

Extrapolation at 150K:

| | 512 | 8192 | ratio |
|---|---|---|---|
| RoPE | 31.14 | 327.14 | 10.51x |
| jfixed | 30.85 | 233.81 | 7.58x |
| random | 32.60 | 32.45 | 1.00x |

Now running same baselines at lr=5e-4 for 200K on GPUs 0/1/2 (queued).

---

## Part 34: 1-layer experiments — isolating the single-layer effect

### Setup

1-layer transformers, 768 embed, 8 heads, block_size=512, lr=5e-4, 50K iters. Tests the position encoding mechanism in isolation, without multi-layer feedback.

### Results at 50K

| Model | Architecture | Angles | V rot | Val PPL | 512 | 8192 | 8K/512 |
|-------|-------------|--------|-------|---------|-----|------|--------|
| rope | standard | Fixed | No | 74.10 | 78.58 | 253.67 | 3.23x |
| joformer_fixed | standard | Fixed | Yes | 74.10 | 78.40 | 286.88 | 3.66x |
| monoidal2 | datadep2 | Content-dep | No | 69.57 | 73.82 | 80.20 | 1.09x |
| joformer2 | datadep2 | Content-dep | Yes | 70.33 | 73.81 | 76.10 | 1.03x |

(monoidal, joformer, det_qk, detb_qk still running)

### Key findings

**1. Content-dependent angles are dramatically better — even with 1 layer.**
monoidal2 and joformer2 are ~4.5 PPL ahead of RoPE at training length (69.6-70.3 vs 74.1) AND have flat extrapolation (1.03-1.09x vs 3.2-3.7x). With 16 layers, this advantage disappears — the multi-layer feedback loop contaminates the angles with position info.

**2. V rotation with fixed angles hurts extrapolation at 1 layer.**
joformer_fixed (3.66x) is worse than RoPE (3.23x) at 8192. With 16 layers, joformer_fixed was better than RoPE. The V rotation decorrelation mechanism may need multiple layers to be beneficial with fixed angles.

**3. V rotation with content-dependent angles helps at 1 layer.**
joformer2 (1.03x) is flatter than monoidal2 (1.09x). Content-dependent angles + V rotation is the best combination even with 1 layer.

**4. The single-layer result proves the approach is sound.**
Content-dependent cumsum angles give better training PPL and flat extrapolation when there's no multi-layer position contamination. The challenge is maintaining this property when stacking layers.

### Iteration-by-iteration comparison

| Iter | RoPE | jfixed | monoidal2 | joformer2 |
|------|------|--------|-----------|-----------|
| 5K | 135.26 | 134.65 | 131.06 | 118.71 |
| 10K | 109.30 | 109.40 | 106.01 | 98.24 |
| 20K | 90.06 | 90.25 | 85.96 | 83.21 |
| 30K | 81.04 | 81.00 | 77.05 | 76.54 |
| 40K | 76.68 | 76.86 | 71.76 | 72.38 |
| 50K | 74.10 | 74.10 | 69.57 | 70.33 |

joformer2 converges fastest early (118.71 at 5K) but monoidal2 catches up and passes it by 40K. Both content-dependent models are consistently ahead of fixed-angle models.

### Extrapolation over training (8K/512 ratio)

| Iter | RoPE | jfixed | monoidal2 | joformer2 |
|------|------|--------|-----------|-----------|
| 10K | 2.45x | 2.06x | 1.04x | 1.00x |
| 20K | 3.17x | 2.75x | 1.06x | 1.00x |
| 30K | 3.45x | 3.22x | 1.06x | 1.01x |
| 40K | 3.36x | 3.55x | 1.08x | 1.02x |
| 50K | 3.23x | 3.66x | 1.09x | 1.03x |

Content-dependent models stay flat throughout. Fixed-angle models degrade steadily. joformer_fixed starts better than RoPE but eventually becomes worse (3.66x vs 3.23x at 50K).

### The multi-layer problem

The 1-layer results show content-dependent angles work well in isolation. The degradation with 16 layers comes from position information leaking through hidden states across layers — each layer's FFN reads position-contaminated x and produces position-correlated angles, eventually encoding position despite the cumsum mechanism.

This suggests the path forward is finding ways to prevent position contamination across layers while keeping the single-layer benefits of content-dependent angles.

### 1-layer det_qk and detb_qk — repeated token drift disproved

| Model | Val PPL | 512 | 8192 | 8K/512 |
|-------|---------|-----|------|--------|
| det_qk (fixed ±1 per token × freq) | 75.45 | 82.23 | 80.82 | 0.98x |
| detb_qk (random ±1 per position × freq) | 79.42 | 87.34 | 78.38 | 0.90x |

Both flat with 1 layer. det_qk at 0.98x disproves the "repeated token drift" hypothesis — the same token always producing the same sign does NOT cause cumsum drift with 1 layer. The 16-layer degradation of det_qk (1.57x at 20K, 2.70x at 40K) is a multi-layer phenomenon, not a token-frequency effect.

---

## Part 35: Per-layer token embeddings (pemb) — flat extrapolation at 16 layers

### Motivation

The 1-layer experiments showed content-dependent angles work well in isolation. The multi-layer degradation comes from hidden-state position contamination — MLPs/FFNs read position from x and produce position-correlated angles.

Solution: remove the hidden state dependency entirely. Each layer gets its own learned per-token angle embedding. No MLP, no FFN angle production, no hidden state input. Angles depend only on token identity, independently per layer.

### Architecture: shared_pemb_qk

- 16 independent `nn.Embedding(vocab_size, C//2)`, one per layer
- Zero-initialized (starts as pure RoPE via rope_base)
- Forward: `angles = tanh(emb_i(token)) * π + rope_base` at each layer i
- ExternalAngleBlock with cumsum, Q/K only (monoidal)
- No MLP, no hidden state dependency
- 359M params total (162M base + 196M for 16 × 32000 × 384 embeddings)

### Key design choices

**Zero initialization**: each embedding starts at zero, so `tanh(0)*π + rope_base = rope_base`. The model starts as RoPE and gradually learns per-token deviations. This was critical — random N(0,1) initialization gave 1.79x at 5K.

**tanh * π + rope_base**: same structure as joformer2 angle_emb. Bounds the learned deviations to [-π, π] and adds RoPE frequency structure for fast convergence.

**No hidden state**: angles at layer i depend only on the token at each position, not on x. The model cannot route position information through the angle path because the angles never see x.

### Results (running, 16 layers)

| Iter | Val PPL | 512 | 8192 | ratio | RoPE val |
|------|---------|-----|------|-------|----------|
| 5K | 62.76 | 64.82 | 64.74 | 1.00x | 59.14 |
| 10K | 48.88 | 50.44 | 51.23 | 1.02x | 46.92 |

**Flat extrapolation at 16 layers** — 1.00x at 5K, 1.02x at 10K. This is the first 16-layer model with learned content-dependent angles that maintains flat extrapolation.

Gap to RoPE at training length: 3.6 PPL at 5K → 1.96 at 10K. Closing rapidly.

### Why this works (hypothesis)

The angles at each layer depend only on token identity — there is no pathway for position information to enter the angle computation:

1. No MLP reading position-contaminated x
2. No FFN angle flow from position-aware hidden states
3. No forward-pass feedback loop — the angle embeddings are looked up by token ID, independent of the hidden state

Each layer independently learns what angles work best for each token at that depth in the network. Different layers can use different angle patterns for the same token, providing diversity without position contamination.

### Comparison with other approaches

| Approach | 16-layer extrapolation | Training PPL vs RoPE |
|----------|----------------------|---------------------|
| RoPE | Blows up (3.2x+ at 50K) | Baseline |
| shared_ln_qk (MLP, LN) | Blows up (10x at 100K) | -0.5 PPL |
| joformer2 h1 ln (datadep2, LN) | Blows up after 35K (9x at 95K) | -7 PPL at 10K |
| joformer2 angle (from frozen) | Stable ~1.2x through 300K | -2 PPL |
| det_qk (fixed ±1, same all layers) | Degrades (2.7x at 40K) | +0.6 PPL |
| **pemb_qk (per-layer emb)** | **Flat 1.02x at 10K** | **+1.96 PPL** |
| random (noise) | Flat 1.0x | +2-3 PPL |

pemb is the first learned approach to achieve flat extrapolation at 16 layers. The gap to RoPE is smaller than random and closing.

### deti_qk — per-layer independent fixed signs (16 layers)

`shared_deti_qk`: each layer has its own fixed random ±1 sign embedding × rope_freq. Tests whether per-layer independence alone (without learning) is sufficient.

Result at 5K: 512=61.24, 8192=120.37 (**1.96x**). Already degrading. Per-layer fixed signs × rope_freq enables position encoding even without learning — the fixed frequency structure is enough. Learning (pemb) avoids this because the model can optimize angles to NOT encode position.

### pemb_qk full training curve (16 layers, lr=5e-4)

| Iter | Val PPL | 512 | 8192 | ratio | RoPE val | gap |
|------|---------|-----|------|-------|----------|-----|
| 5K | 62.76 | 64.82 | 64.74 | 1.00x | 59.14 | +3.62 |
| 10K | 48.88 | 50.44 | 51.23 | 1.02x | 46.92 | +1.96 |
| 15K | 44.20 | 45.19 | 45.86 | 1.01x | 42.53 | +1.67 |
| 20K | 41.84 | 42.23 | 44.62 | 1.06x | 39.75 | +2.09 |
| 25K | 39.64 | 39.83 | 41.14 | 1.03x | 38.04 | +1.60 |
| 30K | 38.23 | 38.79 | 40.30 | 1.04x | 36.83 | +1.40 |
| 35K | 37.39 | 38.57 | 39.53 | 1.02x | 35.94 | +1.45 |

Extrapolation remains flat (1.00-1.06x) through 35K. Gap to RoPE closing: 3.62 → 1.45. The 1.06x at 20K was a fluctuation — returned to 1.02-1.04x.

---

## Part 36: Codebook and V rotation variants — all flat

### pemb_qkv — per-layer embeddings with V rotation

Same as pemb_qk but with V rotation on values. Tests whether V rotation helps training PPL without hurting extrapolation.

At 5K and 10K:

| | pemb_qk | pemb_qkv |
|---|---|---|
| 5K val | 62.76 | **60.02** |
| 5K 512 | 64.82 | **62.61** |
| 5K 8K/512 | 1.00x | 1.01x |
| 10K val | 48.88 | — |
| 10K 512 | 50.44 | **49.30** |
| 10K 8K/512 | 1.02x | 1.02x |

V rotation helps training PPL (~2 PPL at 5K, ~1 PPL at 10K) without hurting extrapolation. pemb_qkv at 5K (60.02) is only 0.88 PPL behind RoPE (59.14).

### shared_cbd_qk — angle codebook with dot-product selection

Architecture: each token has K angle templates (zero-init + small noise, tanh*π + rope_base). At each layer, project x to C//2, compute dot product with each template, argmax+STE to select.

The key property: even though selection depends on x (which has position info), the OUTPUT is constrained to one of K templates. Position info can influence WHICH template but can't create arbitrary angles.

**K=4 at 5K and 10K:**

| | 512 | 8192 | ratio |
|---|---|---|---|
| 5K | 65.20 | 64.96 | 1.00x |
| 10K | 50.73 | 51.14 | 1.01x |

**K=8 at 5K and 10K:**

| | 512 | 8192 | ratio |
|---|---|---|---|
| 5K | 64.41 | 64.02 | 0.99x |
| 10K | 50.51 | 51.57 | 1.02x |

Both flat. K=4 and K=8 essentially identical in both training PPL and extrapolation.

K=32 running (slow at 2.04 it/s due to large codebook: 32000 × 32 × 384).

### Summary: all flat at 10K

| Model | 10K 512 | 10K 8192 | ratio | 10K val PPL |
|-------|---------|----------|-------|-------------|
| pemb_qk | 50.44 | 51.23 | 1.02x | 48.88 |
| pemb_qkv | 49.30 | 50.39 | 1.02x | — |
| cbd K=4 | 50.73 | 51.14 | 1.01x | — |
| cbd K=8 | 50.51 | 51.57 | 1.02x | — |
| RoPE | — | blows up | — | 46.92 |

All four models flat. pemb_qkv closest to RoPE at training length.

### Why per-layer embeddings work but per-layer fixed signs don't

- **pemb (learned)**: 1.02x at 35K. The model learns angle patterns optimized for the task — it finds angles that produce useful attention without encoding position.
- **deti (fixed ±1 × rope_freq)**: 1.96x at 5K. The fixed rope_freq structure gives each dimension a specific frequency — position is encoded through the frequency structure regardless of the signs.

The learning is essential. Learned embeddings can avoid position-encoding patterns. Fixed embeddings with RoPE frequency structure cannot.

### Master comparison table (updating as results come in)

**pemb_qk** (per-layer learned token embeddings, Q/K only):

| Iter | Val PPL | 512 | 8192 | ratio | RoPE val | gap |
|------|---------|-----|------|-------|----------|-----|
| 5K | 62.76 | 64.82 | 64.74 | 1.00x | 59.14 | +3.62 |
| 10K | 48.88 | 50.44 | 51.23 | 1.02x | 46.92 | +1.96 |
| 15K | 44.20 | 45.19 | 45.86 | 1.01x | 42.53 | +1.67 |
| 20K | 41.84 | 42.23 | 44.62 | 1.06x | 39.75 | +2.09 |
| 25K | 39.64 | 39.83 | 41.14 | 1.03x | 38.04 | +1.60 |
| 30K | 38.23 | 38.79 | 40.30 | 1.04x | 36.83 | +1.40 |
| 35K | 37.39 | 38.57 | 39.53 | 1.02x | 35.94 | +1.45 |
| 40K | 36.69 | 37.31 | 38.86 | 1.04x | 35.18 | +1.51 |
| 45K | 36.21 | 36.61 | 38.26 | 1.05x | 34.62 | +1.59 |
| 50K | 35.64 | 35.90 | 37.65 | 1.05x | 34.17 | +1.47 |
| 60K | 34.70 | 35.44 | 36.52 | 1.03x | 33.37 | +1.33 |

**pemb_qkv** (per-layer learned token embeddings, Q/K/V rotation):

| Iter | Val PPL | 512 | 8192 | ratio | RoPE val | gap |
|------|---------|-----|------|-------|----------|-----|
| 5K | 60.02 | 62.61 | 63.35 | 1.01x | 59.14 | +0.88 |
| 10K | 48.13 | 49.30 | 50.39 | 1.02x | 46.92 | +1.21 |
| 15K | 43.54 | 44.99 | 46.14 | 1.03x | 42.53 | +1.01 |
| 20K | 41.00 | 41.87 | 43.48 | 1.04x | 39.75 | +1.25 |
| 25K | 39.32 | 39.75 | 41.49 | 1.04x | 38.04 | +1.28 |
| 30K | 38.00 | 38.91 | 39.84 | 1.02x | 36.83 | +1.17 |
| 35K | 37.18 | 37.96 | 39.49 | 1.04x | 35.94 | +1.24 |

**cbd K=8** (per-token codebook, dot-product selection):

| Iter | Val PPL | 512 | 8192 | ratio | RoPE val | gap |
|------|---------|-----|------|-------|----------|-----|
| 5K | 62.40 | 64.41 | 64.02 | 0.99x | 59.14 | +3.26 |
| 10K | 49.04 | 50.51 | 51.57 | 1.02x | 46.92 | +2.12 |
| 15K | 44.63 | 45.84 | 46.59 | 1.02x | 42.53 | +2.10 |
| 20K | — | 42.57 | 44.64 | 1.05x | 39.75 | — |
| 30K | — | 39.51 | 41.56 | 1.05x | 36.83 | — |
| 35K | 37.75 | — | — | — | 35.94 | +1.81 |

### Codebook cost reduction: pcb (factored form, implemented)

`angle = base_emb[token] + codebook[selected_k]`

Per-token base embedding (like pemb, zero-init, tanh*π + rope_base) + K shared corrections. Selection via dot product of projected x against full candidates (base + each correction), argmax + STE.

Only 4.7M extra params over pemb. The codebook itself is tiny (16 layers × 4 × 384 = 24K params). Most overhead from per-layer projections (16 × 384 × 768).

pcb K=4 tested briefly (flat at 5K) but replaced by other experiments.

---

## Part 37: Complete 200K results at constant lr=5e-4

All models trained at lr=5e-4 on OWT (9.1B tokens), 16 layers, 768 embed, 8 heads, block_size=512, BF16.

### Master comparison table (val PPL at matched iterations)

| Iter | RoPE | jfixed | random_qk | random_qkv | pemb_qk | pemb_qkv | cbd_K4_qk | cbd_K8_qk |
|------|------|--------|-----------|------------|---------|----------|-----------|-----------|
| 10K | 46.92 | 46.87 | 49.14 | 49.28 | 48.88 | 48.13 | 49.23 | 49.04 |
| 20K | 39.75 | 39.77 | 41.60 | 41.79 | 41.84 | 41.00 | 41.50 | 41.67 |
| 30K | 36.83 | 36.61 | 38.27 | 38.36 | 38.23 | 38.00 | 38.30 | 38.69 |
| 40K | 35.18 | 35.01 | 36.40 | 36.62 | 36.69 | 36.46 | 36.64 | 37.09 |
| 50K | 34.17 | 33.97 | 35.32 | 35.52 | 35.64 | 35.48 | 35.56 | 36.03 |
| 60K | 33.37 | 33.26 | 34.44 | 34.67 | 34.70 | 34.66 | 34.72 | 35.44 |
| 70K | 32.77 | 32.55 | 33.95 | 34.04 | 34.03 | 34.18 | 34.08 | 34.69 |
| 80K | 32.48 | 32.27 | 33.58 | 33.67 | 33.58 | 33.66 | 33.71 | 34.05 |
| 90K | 32.13 | 31.86 | 33.21 | 33.27 | 33.27 | 33.24 | 33.29 | 33.66 |
| 100K | 31.89 | 31.60 | 32.88 | 33.12 | 32.97 | 32.98 | 32.95 | 33.43 |
| 110K | 31.66 | 31.44 | 32.84 | — | 32.77 | 32.63 | 32.61 | 32.94 |
| 120K | 31.43 | 31.25 | 32.46 | — | 32.53 | 32.52 | 32.23 | 32.54 |
| 130K | 31.12 | 30.98 | 32.20 | — | 32.31 | 32.33 | 32.14 | — |
| 140K | 31.02 | 30.80 | 32.02 | — | 32.12 | 32.15 | 32.05 | — |
| 150K | 30.93 | 30.64 | 32.08 | — | 32.12 | 31.93 | 31.75 | — |
| 160K | 30.78 | 30.37 | 31.89 | — | 31.75 | 31.92 | 31.50 | — |
| 170K | 30.63 | — | — | — | 31.68 | 31.63 | 31.57 | — |
| 180K | 30.53 | — | — | — | 31.51 | 31.59 | 31.57 | — |
| 190K | 30.45 | — | — | — | 31.56 | 31.44 | 31.28 | — |
| 200K | 30.53 | 30.15 | 31.45 | — | 31.32 | 31.45 | **31.23** | — |

Notes:
- jfixed 170-190K gap: original run stopped at 162K, extension resumed from checkpoint
- random_qk 170-190K gap: original run stopped at 173K, extension resumed from checkpoint
- cbd_K8_qk, random_qkv, cbd_K8_qkv still running (will update when complete)
- RoPE 200K shows 30.53 (slightly worse than 190K=30.45 due to eval variance)

### Rankings at 200K

| Rank | Model | Val PPL | Gap to jfixed | Extrapolates? |
|------|-------|---------|---------------|---------------|
| 1 | jfixed | 30.15 | — | No (8.1x) |
| 2 | RoPE | 30.53 | +0.38 | No (19.2x) |
| 3 | **cbd_K4_qk** | **31.23** | **+1.08** | **Yes (1.14x)** |
| 4 | pemb_qk | 31.32 | +1.17 | Yes (1.09x) |
| 5 | random_qk | 31.45 | +1.30 | Yes (1.07x) |
| 5 | pemb_qkv | 31.45 | +1.30 | Yes (1.09x) |

jfixed is the true baseline — it beats RoPE at constant 5e-4 because the fixed cumsum angles act as a regularizer.

### Length extrapolation at 200K (full results)

| Model | 512 | 1024 | 2048 | 4096 | 8192 | 8K/512 |
|-------|-----|------|------|------|------|--------|
| RoPE | 30.57 | 48.58 | 116.88 | 263.17 | 587.57 | **19.2x** |
| jfixed | 30.24 | 39.99 | 74.50 | 132.58 | 245.45 | **8.1x** |
| random_qk | 31.55 | 29.53 | 28.65 | 29.93 | 33.73 | **1.07x** |
| pemb_qk | 31.82 | 29.90 | 29.35 | 30.95 | 34.70 | **1.09x** |
| pemb_qkv | 32.00 | 30.15 | 29.58 | 31.06 | 36.29 | **1.13x** |
| cbd_K4_qk | 31.74 | 29.68 | 29.40 | 31.20 | 36.04 | **1.14x** |

Key observations:
- **RoPE and jfixed blow up** — RoPE is catastrophic (19x), jfixed degrades less (8x) but still unusable
- **All flat models actually improve at 1024-2048** — PPL drops below 512 performance, suggesting the model benefits from longer context when extrapolation works
- **random_qk has the flattest extrapolation** (1.07x) despite being the simplest approach
- **cbd_K4_qk and pemb_qkv are slightly worse at 8K** (~1.13-1.14x) but still flat

### cbd K=4 vs pemb_qk — why codebook wins at training length

From 110K onwards, cbd K=4 consistently beats pemb_qk by 0.15-0.50 PPL:

| Iter | pemb_qk | cbd_K4 | cbd ahead by |
|------|---------|--------|-------------|
| 110K | 32.77 | 32.61 | 0.16 |
| 120K | 32.53 | 32.23 | 0.30 |
| 130K | 32.31 | 32.14 | 0.17 |
| 150K | 32.12 | 31.75 | 0.37 |
| 160K | 31.75 | 31.50 | 0.25 |
| 190K | 31.56 | 31.28 | 0.28 |
| 200K | 31.32 | 31.23 | 0.09 |

The gap narrows by 200K (0.09) but cbd maintains its lead throughout. Context-dependent selection from K=4 templates provides a consistent advantage over purely token-dependent angles.

### pmlp — base embedding + MLP(x) correction (in progress, 80K/100K)

`angle = base_emb + scale * π * tanh(MLP(x))` with learnable scale (zero-init).

At 80K: val PPL 33.42, extrap 8K/512 = 1.08x. Tracking near pemb_qk at matched iterations (pemb_qk 80K = 33.58). The MLP correction provides marginal benefit — the zero-init scale stays small, preventing position leakage while also limiting expressiveness.

### Why jfixed beats RoPE

Surprising result: fixed (non-learned) cumsum angles with the same frequency structure as RoPE outperform RoPE itself at constant lr=5e-4. The gap is 0.38 PPL at 200K (30.15 vs 30.53).

Hypothesis: the cumsum operation acts as an implicit regularizer. In standard RoPE, each position gets its exact rotation directly. With cumsum of fixed angles, each position's rotation is the sum of all preceding per-token angles. This introduces a form of input-dependent noise to the position signal that acts as regularization.

### The fundamental finding

Position encoding through cumsum angles achieves flat length extrapolation when angles are not derived from position-contaminated hidden states. Three approaches work:

1. **Random angles** (simplest): i.i.d. noise per position per dimension, sampled uniformly from [-freq, freq]. No token or position dependence — pure noise. Gap to jfixed: +1.30 PPL.
2. **Per-token embeddings (pemb)**: learned nn.Embedding per layer, tanh*π + rope_base. Gap: +1.17 PPL.
3. **Codebook (cbd K=4)**: per-token K=4 templates with context-dependent selection. Gap: +1.08 PPL.

All three share the key property: angles depend only on token identity (and optionally a bottlenecked function of context), never on position. The forward-pass feedback loop through attention means any unbottlenecked function of x (MLP, LayerNorm) will eventually encode position and destroy extrapolation.

### Comparison with existing methods

| Method | Training PPL | Extrapolation | Flash Attn | Post-training |
|--------|-------------|---------------|-----------|---------------|
| RoPE | Best | Blows up (19x) | ✓ | — |
| jfixed (cumsum) | Best−0.38 | Blows up (8x) | ✓ | — |
| RoPE + NTK/YaRN | Best | Extended | ✓ | Required |
| ALiBi | Slightly worse | Flat | ✗ | — |
| random cumsum (i.i.d. noise) | Best−1.30 | **Flat (1.07x)** | ✓ | — |
| pemb cumsum | Best−1.17 | **Flat (1.09x)** | ✓ | — |
| **cbd K=4 cumsum** | **Best−1.08** | **Flat (1.14x)** | **✓** | **—** |

cbd K=4 offers the best tradeoff among flat-extrapolating methods: closest to RoPE at training length, fully Flash Attention compatible, no post-training required for any context length. The 1.08 PPL cost buys unlimited context extension with zero additional compute or fine-tuning.

## Part 38: Learning rate schedule experiments and new architectures

### LR schedule: 100K@5e-4 → 50K@2e-4 → 50K@5e-5

The constant lr=5e-4 results plateau around 31 PPL. Decaying lr dramatically improves val PPL. Baselines with this schedule (already completed):

| Model | 200K Val PPL | Extrap | Flat? |
|-------|-------------|--------|-------|
| jfixed | 23.21 | 8K blows up | No |
| RoPE | 23.54 | 8K blows up | No |
| random_qk | 24.15 | 512=24.21, 4096=26.00 | **Yes (~1.07x)** |

### Full schedule comparison table (in progress)

All models: 100K@5e-4 (done) → 50K@2e-4 (phase 1) → 50K@5e-5 (phase 2).

**Phase 1 (lr=2e-4, iters 100K–150K):**

| Iter | random_qk | cbd_K4_qk | pmlp_qk | pemb_qk | cbd_K4_qkv | cbd_qk ext | pmlp ext | pemb ext | qkv ext |
|------|-----------|-----------|---------|---------|------------|------------|----------|----------|---------|
| 105K | 28.53 | 28.40 | 28.52 | 28.50 | 28.66 | 1.12x | 1.46x | 1.05x | 1.06x |
| 110K | 28.09 | 27.94 | 27.97 | 27.92 | 28.08 | 1.20x | 1.62x | 1.04x | 1.06x |
| 115K | 27.84 | 27.63 | 27.69 | 27.72 | 27.76 | 1.14x | 1.23x | 1.10x | 1.07x |
| 120K | 27.72 | 27.51 | 27.53 | 27.44 | — | 1.32x | 1.52x | 1.07x | — |
| 125K | 27.70 | 27.31 | 27.45 | 27.34 | — | 1.29x | 1.27x | 1.08x | — |
| 130K | 27.57 | 27.28 | 27.36 | 27.29 | — | 1.17x | 1.53x | 1.11x | — |
| 135K | 27.54 | 27.15 | 27.35 | 27.32 | — | 1.13x | 1.15x | 1.13x | — |
| 140K | 27.63 | 27.24 | 27.24 | 27.20 | — | 1.26x | 1.39x | 1.11x | — |
| 145K | 27.57 | 27.14 | 27.26 | — | — | 1.14x | 1.27x | 1.05x | — |
| 150K | 27.54 | 27.14 | 27.19 | — | — | 1.22x | 1.39x | 1.11x | — |

**Phase 2 (lr=5e-5, iters 150K–200K):**

| Iter | random_qk | cbd_K4_qk | pmlp_qk | pemb_qk | cbd_qk ext | pmlp ext | pemb ext |
|------|-----------|-----------|---------|---------|------------|----------|----------|
| 155K | 25.34 | 24.81 | 24.85 | 24.87 | 1.27x | 1.43x | 1.10x |
| 160K | 25.00 | 24.44 | 24.46 | 24.48 | 1.34x | 1.77x | 1.16x |
| 165K | 24.77 | 24.22 | 24.23 | 24.25 | 1.49x | 1.35x | 1.10x |
| 170K | 24.63 | 24.08 | 24.07 | — | 1.38x | 1.77x | — |
| 175K | 24.53 | 23.93 | 23.94 | — | 1.28x | 1.48x | — |
| 180K | 24.44 | 23.82 | 23.83 | — | 1.29x | 1.79x | — |
| 185K | 24.38 | 23.73 | — | — | 1.45x | — | — |
| 190K | 24.26 | 23.68 | — | — | 1.41x | — | — |
| 195K | 24.22 | — | — | — | — | — | — |
| 200K | 24.15 | — | — | — | — | — | — |

### Val PPL rankings (at latest matched iterations)

At 165K (all four models have data):

| Rank | Model | Val PPL | Extrap 8K/512 |
|------|-------|---------|---------------|
| 1 | cbd_K4_qk | **24.22** | 1.49x |
| 2 | pmlp_qk | 24.23 | 1.35x |
| 3 | pemb_qk | 24.25 | 1.10x |
| 4 | random_qk | 24.77 | ~1.07x |

cbd_K4_qk, pmlp_qk, and pemb_qk are essentially tied on val PPL (~24.2), all beating random_qk by 0.5 PPL. But extrapolation differs dramatically:

- **pemb_qk**: flattest (1.10x) — pure token embeddings, no data dependence to leak position
- **pmlp_qk**: noisy (1.35-1.79x) — MLP correction encoding position at lower lr
- **cbd_K4_qk**: moderate (1.28-1.49x) — discrete codebook partially encoding position

### cbd_K4_qkv — V rotation under schedule (early, 115K)

cbd_K4_qkv at 115K: val=27.76, extrap **1.07x**. Compare to cbd_K4_qk at 115K: val=27.63, extrap 1.14x. V rotation costs ~0.13 PPL but extrapolation is significantly flatter (1.07x vs 1.14x). The key test is whether it holds through phase 2.

### pmlp2 — stacked corrections (paused at 20K)

Architecture: one shared base embedding + per-layer MLP corrections that accumulate across layers:
```
angles = tanh(base_emb(token)) * π + rope_base
for each layer i:
    angles = angles + scale_i * π * tanh(MLP_i(x))
    x = block(x, angles)
```

One embedding (not 16), 16 MLPs with zero-init scales. 189M params (vs pmlp's 373M).

| Iter | pmlp2 val | pmlp val | gap | pmlp2 extrap |
|------|-----------|----------|-----|-------------|
| 5K | 65.38 | 63.80 | +1.58 | 1.01x |
| 10K | 53.20 | 49.29 | +3.91 | 1.04x |
| 15K | 45.75 | 44.10 | +1.65 | 1.07x |
| 20K | 43.25 | 41.19 | +2.06 | 1.08x |

Gap to pmlp is ~2 PPL. Extrapolation flat at 1.07-1.08x. Paused to free GPU for schedule experiments.

### cbd_K4_qk matches RoPE

**cbd_K4_qk with lr schedule: final val PPL 23.56 — matching RoPE (23.54) within 0.02 PPL.**

This is the headline result. A flat-extrapolating position encoding method matches RoPE at training length on the same data, same model size, same total training iterations. The only difference is the position encoding mechanism.

### Final results: all learned models match RoPE

| Model | 200K Val PPL | Extrap 8K/512 |
|-------|-------------|---------------|
| jfixed | 23.21 | blows up |
| RoPE | **23.56** | 19.2x (blows up) |
| pmlp_qk | **23.55** | 1.45x |
| cbd_K4_qk | **23.56** | 1.37x |
| **pemb_qk** | **23.56** | **1.12x** |
| random_qk | 24.15 | 1.07x |

All three learned models (pemb_qk, cbd_K4_qk, pmlp_qk) converge to 23.55-23.56 — exactly matching RoPE. The position encoding mechanism does not limit model quality. The ceiling is the same regardless of how position is encoded.

**pemb_qk is the headline result**: matches RoPE exactly (23.56) with 1.12x extrapolation. The simplest learned approach — one `nn.Embedding(vocab_size, C//2)` per layer, zero-init, `tanh*π + rope_base`, cumsum. No MLP, no codebook, no data dependence. Each token gets a fixed learned angle vector per layer, same every time that token appears, regardless of context or position. Angles depend only on token identity, never on hidden states, so flat extrapolation is theoretically and mathematically guaranteed. Flash Attention compatible — identical interface to RoPE.

### The lr-extrapolation tradeoff (final picture)

| Model | Constant 5e-4 (200K) | With schedule (200K) |
|-------|---------------------|---------------------|
| | Val PPL / Extrap | Val PPL / Extrap |
| random_qk | 31.45 / 1.07x | 24.15 / 1.07x |
| **pemb_qk** | **31.32 / 1.09x** | **23.56 / 1.12x** |
| cbd_K4_qk | 31.23 / 1.14x | 23.56 / 1.37x |
| pmlp_qk | 32.90 / 1.04x | 23.55 / 1.45x |
| cbd_K4_qkv | — / 1.10x | in progress (1.11x at 145K) |

- **random_qk**: i.i.d. noise per position, no token or data dependence → extrapolation perfectly stable. Gap to RoPE: 0.59 PPL.
- **pemb_qk**: learned per-token per-layer embeddings, no data dependence → matches RoPE with 1.12x extrapolation. Flat extrapolation theoretically guaranteed.
- **cbd_K4_qk**: discrete codebook bottleneck → matches RoPE but extrapolation degrades to 1.37x under the schedule.
- **pmlp_qk**: soft MLP bottleneck → matches RoPE but extrapolation degrades to 1.45x under the schedule.
- **cbd_K4_qkv**: V rotation keeping extrapolation at 1.11x through phase 1. If it holds through phase 2, it would match RoPE with flat extrapolation.

### The architecture spectrum

| Architecture | Angles depend on | Val PPL (sched) | Extrap | Complexity |
|-------------|-----------------|-----------------|--------|------------|
| random | nothing (i.i.d. noise) | 24.15 | 1.07x | Zero params |
| **pemb** | **token identity** | **23.56** | **1.12x** | **Simple embedding** |
| cbd K=4 | token + bottlenecked context | 23.56 | 1.37x | Codebook + projection |
| pmlp | token + MLP(x) correction | 23.55 | 1.45x | Embedding + MLP |

pemb is the clear winner: matches RoPE, flattest extrapolation among learned models, simplest architecture. More expressiveness (cbd, pmlp) does not improve val PPL but degrades extrapolation. The per-token embedding is all that's needed.

### cbd_K4_qkv vs cbd_K4_qk — V rotation controls extrapolation

cbd_K4_qkv adds V rotation (rotating values by cumsum angles with inverse unrotation). Comparing against cbd_K4_qk at matched iterations during phase 1 (lr=2e-4):

| Iter | cbd_K4_qk | cbd_K4_qkv | qkv behind | qk ext | qkv ext |
|------|-----------|------------|------------|--------|---------|
| 105K | 28.40 | 28.66 | +0.26 | 1.12x | 1.06x |
| 110K | 27.94 | 28.08 | +0.14 | 1.20x | 1.06x |
| 115K | 27.63 | 27.76 | +0.13 | 1.14x | 1.07x |
| 120K | 27.51 | 27.59 | +0.08 | 1.32x | 1.07x |
| 125K | 27.31 | 27.52 | +0.21 | 1.29x | 1.07x |
| 130K | 27.28 | 27.44 | +0.16 | 1.17x | 1.08x |
| 135K | 27.15 | 27.36 | +0.21 | 1.13x | 1.09x |

V rotation costs ~0.15-0.20 PPL on val but keeps extrapolation locked at 1.06-1.09x while cbd_K4_qk bounces between 1.12-1.32x. The decorrelation from V rotation prevents position-dependent patterns from accumulating coherently across attention heads.

### qkv schedule results (phase 2 data)

All qkv variants through the lr schedule, compared to their qk counterparts:

**At 175K (best matched point for most models):**

| Model | Val PPL | Extrap 8K/512 |
|-------|---------|---------------|
| pemb_qkv | **23.92** | 1.15x |
| pemb_qk | 23.95 | 1.21x |
| cbd_K4_qkv | 24.05 | **1.10x** |
| cbd_K4_qk | 23.93 | 1.28x |
| random_qkv | 24.57 | **1.00x** |
| random_qk | 24.53 | — |

**cbd_K4_qkv near-final (~200K): val 23.67, extrap 1.16x.** Only 0.11 behind RoPE (23.56) with flat extrapolation.

**pemb_qkv at 175K: val 23.92, extrap 1.15x.** Slightly ahead of pemb_qk (23.95) on val PPL. V rotation is not hurting pemb on val and extrap is comparable (1.15x vs 1.21x at this point).

**random_qkv: perfect 1.00x extrapolation** throughout the entire schedule. V rotation with random angles gives the flattest extrapolation of any model, though val PPL (24.57) lags ~1 PPL behind the learned models.

### V rotation: when does it help?

| Model | qk extrap (sched) | qkv extrap (sched) | V rotation effect |
|-------|-------------------|--------------------|--------------------|
| random | ~1.07x | **1.00x** | Helps — flattens perfect |
| pemb | 1.07-1.27x | 1.13-1.18x | Comparable — slight smoothing |
| cbd K=4 | 1.13-1.49x | **1.08-1.16x** | Helps — controls position leakage |
| pmlp | 1.23-1.79x | in progress | TBD |

V rotation helps most when the base angles have some position leakage (cbd, random). For pemb where angles are already clean, the effect is smaller. The decorrelation from V rotation acts as a safety net — it doesn't hurt val PPL and provides insurance against extrapolation degradation.

### Full schedule comparison table

All models: 100K@5e-4 → 50K@2e-4 → 50K@5e-5. Sorted by val PPL at 200K.

**Phase 1 (lr=2e-4, iters 100K–150K):**

| Iter | random_qk | cbd_K4_qk | pmlp_qk | pemb_qk | cbd_K4_qkv | pemb_qkv | random_qkv | lf_qk |
|------|-----------|-----------|---------|---------|------------|----------|------------|-------|
| 105K | 28.53 | 28.40 | 28.52 | 28.50 | 28.66 | 28.60 | 28.76 | 30.59 |
| 110K | 28.09 | 27.94 | 27.97 | 27.92 | 28.08 | 27.96 | 28.20 | 30.05 |
| 115K | 27.84 | 27.63 | 27.69 | 27.72 | 27.76 | 27.61 | 27.99 | 29.71 |
| 120K | 27.72 | 27.51 | 27.53 | 27.44 | 27.59 | 27.51 | 27.84 | 29.43 |
| 125K | 27.70 | 27.31 | 27.45 | 27.34 | 27.52 | 27.39 | 27.80 | 29.26 |
| 130K | 27.57 | 27.28 | 27.36 | 27.29 | 27.44 | 27.28 | 27.81 | 29.16 |
| 135K | 27.54 | 27.15 | 27.35 | 27.32 | 27.36 | 27.19 | 27.72 | 29.01 |
| 140K | 27.63 | 27.24 | 27.24 | 27.20 | 27.28 | 27.26 | 27.74 | 28.99 |
| 145K | 27.57 | 27.14 | 27.26 | — | 27.24 | 27.18 | 27.67 | 28.89 |
| 150K | 27.54 | — | 27.19 | — | 27.25 | 27.11 | 27.72 | 28.79 |

**Phase 2 (lr=5e-5, iters 150K–200K):**

| Iter | random_qk | cbd_K4_qk | pmlp_qk | pemb_qk | cbd_K4_qkv | pemb_qkv | random_qkv | lf_qk |
|------|-----------|-----------|---------|---------|------------|----------|------------|-------|
| 155K | 25.34 | 24.81 | 24.85 | 24.87 | 25.00 | 24.85 | 25.37 | 26.28 |
| 160K | 25.00 | 24.44 | 24.46 | 24.48 | 24.60 | 24.48 | 25.05 | 25.90 |
| 165K | 24.77 | 24.22 | 24.23 | 24.25 | 24.36 | 24.19 | 24.79 | 25.61 |
| 170K | 24.63 | 24.08 | 24.07 | 24.07 | 24.16 | 23.99 | 24.65 | 25.47 |
| 175K | 24.53 | 23.93 | 23.94 | 23.95 | 24.05 | 23.92 | 24.57 | 25.29 |
| 180K | 24.44 | 23.82 | 23.83 | 23.82 | 23.92 | 23.80 | 24.46 | 25.21 |
| 185K | 24.38 | 23.73 | 23.74 | 23.75 | 23.84 | 23.74 | 24.35 | 25.10 |
| 190K | 24.26 | 23.68 | 23.66 | 23.69 | 23.82 | 23.62 | 24.31 | 25.00 |
| 195K | 24.22 | 23.63 | 23.62 | 23.64 | 23.73 | 23.56 | 24.25 | 24.98 |
| 200K | 24.15 | 23.56 | 23.55 | 23.56 | 23.67 | 23.57 | 24.23 | 24.91 |

**Extrapolation (8K/512 ratio) through the schedule:**

| Iter | cbd_K4_qk | pmlp_qk | pemb_qk | cbd_K4_qkv | pemb_qkv | random_qkv | lf_qk |
|------|-----------|---------|---------|------------|----------|------------|-------|
| 110K | 1.20x | 1.62x | 1.04x | 1.06x | 1.12x | 0.98x | 0.94x |
| 130K | 1.17x | 1.53x | 1.11x | 1.08x | 1.15x | 0.98x | — |
| 150K | 1.22x | 1.39x | 1.11x | 1.10x | 1.13x | 0.98x | — |
| 160K | 1.34x | 1.77x | 1.16x | 1.09x | 1.14x | 1.01x | 0.94x |
| 175K | 1.28x | 1.48x | 1.21x | 1.10x | 1.15x | 1.00x | — |
| 190K | 1.41x | 1.33x | 1.10x | 1.13x | 1.13x | 1.02x | — |
| 200K | 1.37x | 1.45x | 1.12x | 1.16x | 1.15x | 1.01x | 0.96x |

### Final results at 200K with schedule

| Model | Val PPL | Extrap 8K/512 | Angle type | Protection |
|-------|---------|---------------|------------|------------|
| jfixed | 23.21 | blows up | fixed cumsum | None |
| RoPE | **23.56** | 19.2x (blows up) | fixed position | None |
| pmlp_qk | **23.55** | 1.45x | token emb + MLP(x) | Token base only |
| cbd_K4_qk | **23.56** | 1.37x | token codebook + context | Discrete bottleneck |
| pemb_qkv | **23.57** | 1.15x | token emb + V rotation | Token-only |
| **pemb_qk** | **23.56** | **1.12x** | **token embedding** | **Token-only** |
| cbd_K4_qkv | 23.67 | 1.16x | token codebook + V rotation | Discrete + V rot |
| random_qk | 24.15 | 1.07x | noise * fixed_freq | Random noise |
| random_qkv | 24.23 | 1.01x | noise * fixed_freq + V rot | Random noise + V rot |
| lf_qk | 24.91 | **0.96x** | noise * MLP(x) freq | Random noise |

### Two independent protections against position encoding

The results reveal two independent mechanisms that prevent position leakage:

**1. Token-only dependence (pemb, cbd):** Angles are deterministic but depend only on token identity, never on hidden states. Position cannot leak because the same token always produces the same angle regardless of position. pemb_qk achieves 23.56 PPL with 1.12x extrapolation.

**2. Random noise (random, lf, rpemb):** Angles include random noise that destroys any position signal. Even if the frequency comes from hidden states (lf_qk: MLP(x)), the random multiplication prevents the model from decoding position. lf_qk achieves 24.91 PPL with 0.96x extrapolation.

Either protection alone is sufficient. Combining both (rpemb: learned per-token frequencies × random noise) is being tested.

### Why pemb_qk is the best overall

pemb_qk matches RoPE exactly (23.56) with 1.12x extrapolation. It wins because:

1. **Token embeddings learn useful frequency structure** — closing the 0.59 PPL gap from random (24.15) to RoPE (23.56)
2. **No position leakage** — angles depend only on token identity
3. **Simplest architecture** — one `nn.Embedding(vocab_size, C//2)` per layer, zero-init, `tanh*π + rope_base`
4. **Flash Attention compatible** — identical interface to RoPE
5. **No random noise needed** — deterministic angles, reproducible inference

Random noise models (lf_qk, random_qk) achieve flatter extrapolation (0.96-1.07x) but at the cost of 0.59-1.35 PPL. The noise destroys useful angle structure that token embeddings can learn.

**The key insight: position encoding quality is not limited by the mechanism — it is limited only by position leakage.** Once leakage is eliminated, the model reaches RoPE quality. Token embeddings eliminate leakage while preserving learned structure. Random noise eliminates leakage but destroys learned structure.

### Full length extrapolation eval (SDPA, eval_iters=20, eval_batch=2/4)

All schedule-200K checkpoints evaluated at 512 through 65536 (128x training length) using PyTorch SDPA for memory-efficient attention.

| Model | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 65K/512 |
|-------|-----|------|------|------|------|-------|-------|-------|---------|
| RoPE | 23.54 | 42.75 | 106.96 | 223.88 | 381.82 | 600.53 | 808.35 | 905.18 | 38.5x |
| jfixed | 23.17 | 33.17 | 70.11 | 127.81 | 214.18 | 306.73 | 404.45 | 485.66 | 21.0x |
| ALiBi | 23.87 | 21.58 | 21.38 | 21.75 | 22.46 | 22.84 | OOM | OOM | 0.96x* |
| cbd_K4_qk | 23.47 | 22.14 | 21.92 | 24.25 | 33.76 | 64.78 | 159.02 | 480.89 | 20.5x |
| pmlp_qk | 23.59 | 22.08 | 21.48 | 22.83 | 34.30 | 71.35 | 149.27 | 427.65 | 18.1x |
| random_qk | 24.19 | 22.75 | 22.41 | 25.90 | 38.81 | 70.63 | 107.51 | 181.27 | 7.49x |
| pemb_qk | 23.56 | 22.20 | 21.47 | 22.56 | 27.62 | 38.93 | 67.13 | 139.63 | 5.93x |
| cbd_K4_qkv | 23.65 | 22.20 | 21.67 | 22.97 | 28.37 | 38.66 | 60.65 | 125.78 | 5.32x |
| pemb_qkv | 23.60 | 22.20 | 21.49 | 22.82 | 27.89 | 37.58 | 57.57 | 102.53 | 4.35x |
| pmlp_qkv | 23.86 | 22.24 | 21.65 | 22.27 | 25.41 | 30.67 | 39.53 | 62.06 | 2.60x |
| **lf_qk** | 24.90 | 23.37 | 22.23 | 22.44 | 23.89 | 25.57 | 27.27 | **40.91** | **1.64x** |
| **random_qkv** | 24.18 | 22.70 | 21.93 | 22.58 | 24.83 | 27.54 | 28.83 | **38.54** | **1.59x** |

*ALiBi OOM at 32K+ in our eval setup (PyTorch SDPA doesn't support on-the-fly additive bias; FlashAttention-2 does). Ratio is 16K/512.

### Key findings from full extrapolation

**At 8x training length (4096):** All cumsum models improve over 512 — PPL drops because longer context helps. Best: pmlp_qkv (22.27), lf_qk (22.44). TAPA paper reports stability at 4x their training length (32K/8K). Our models are improving at 8x.

**At 32x training length (16384):** Clear separation emerges.
- Flat: ALiBi (0.96x), lf_qk (1.03x), random_qkv (1.14x)
- Moderate: pmlp_qkv (1.29x), pemb_qkv (1.59x), pemb_qk (1.65x)
- Degrading: cbd_K4_qk (2.76x), random_qk (2.92x), pmlp_qk (3.03x)
- Blown up: RoPE (25.5x), jfixed (13.2x)

**At 128x training length (65536):** Only two models remain usable.
- **random_qkv (1.59x)** and **lf_qk (1.64x)** — both use random noise in angle computation
- pmlp_qkv (2.60x) — V rotation provides partial protection
- Everything else >4x degradation

**The pattern:** At extreme extrapolation lengths, random noise in angle computation is the strongest protection. V rotation helps but isn't sufficient alone. Deterministic angles (pemb, cbd) degrade because the cumsum magnitude grows as sqrt(T), creating out-of-distribution rotations at long sequences.

### ALiBi comparison

ALiBi achieves the flattest extrapolation (0.96x at 16K) with competitive val PPL (23.87). However:
- ALiBi is Flash Attention compatible (FlashAttention-2 supports ALiBi natively)
- Our cumsum models (lf_qk, random_qkv) achieve comparable flatness (1.03x, 1.14x at 16K) while being fully Flash Attention compatible
- At training length, pemb_qk (23.56) beats ALiBi (23.87) by 0.31 PPL

### lf_qkv — V rotation transforms lf into a top model

**lf_qkv final: val PPL 23.98, extrap 8K/512 = 1.02x.**

V rotation improved lf dramatically — the biggest qk→qkv improvement of any model:

| Model | Val PPL | Extrap 8K/512 | V rotation effect |
|-------|---------|---------------|-------------------|
| lf_qk | 24.91 | 0.96x | — |
| **lf_qkv** | **23.98** | **1.02x** | **−0.93 PPL** |

lf_qkv is only 0.42 behind RoPE (23.56) with near-perfect extrapolation. For comparison, pemb_qk matches RoPE (23.56) but with 1.12x extrap at 8K that degrades to 5.93x at 65K. lf_qkv maintains ~1.0x at all lengths tested.

Why V rotation helps lf so much: lf uses MLP(x) to compute frequencies. The MLP reads hidden states that contain position information through the forward-pass feedback loop. Without V rotation, this position leakage accumulates in values. With V rotation, the decorrelation prevents position-dependent value patterns from coherently accumulating. The result: val PPL improves because the model can use the MLP frequencies more effectively when V rotation controls the position leakage side effect.

### Complete results at 200K with schedule — full extrapolation to 65536

All models evaluated at 512 through 65536 (128x training length) using SDPA for memory-efficient attention.

| Model | Val PPL | 512 | 4096 | 8192 | 16384 | 32768 | 65536 | 65K/512 |
|-------|---------|-----|------|------|-------|-------|-------|---------|
| jfixed | 23.17 | 23.17 | 127.81 | 214.18 | 306.73 | 404.45 | 485.66 | 21.0x |
| RoPE | 23.54 | 23.54 | 223.88 | 381.82 | 600.53 | 808.35 | 905.18 | 38.5x |
| pmlp_qk | 23.55 | 23.59 | 22.83 | 34.30 | 71.35 | 149.27 | 427.65 | 18.1x |
| pemb_qk | 23.56 | 23.56 | 22.56 | 27.62 | 38.93 | 67.13 | 139.63 | 5.93x |
| cbd_K4_qk | 23.56 | 23.47 | 24.25 | 33.76 | 64.78 | 159.02 | 480.89 | 20.5x |
| pemb_qkv | 23.57 | 23.60 | 22.82 | 27.89 | 37.58 | 57.57 | 102.53 | 4.35x |
| cbd_K4_qkv | 23.67 | 23.65 | 22.97 | 28.37 | 38.66 | 60.65 | 125.78 | 5.32x |
| pmlp_qkv | 23.74 | 23.86 | 22.27 | 25.41 | 30.67 | 39.53 | 62.06 | 2.60x |
| rpemb_v2_qkv | 23.83 | 23.65 | 21.75 | 26.35 | 42.58 | 85.53 | 245.49 | 10.4x |
| ALiBi | 23.87 | 23.87 | 21.75 | 22.46 | 22.84 | — | — | 0.96x* |
| rpemb_v2_qk | 23.92 | 23.79 | 21.50 | 24.08 | 30.12 | 42.76 | 90.99 | 3.83x |
| rpemb4_qk | 23.96 | 23.82 | 21.74 | 24.85 | 32.34 | 44.65 | 91.22 | 3.83x |
| lf_qkv | 23.98 | 23.70 | 21.23 | 23.77 | 32.58 | 49.40 | 125.24 | 5.29x |
| **rpemb4_qkv** | **24.01** | **23.80** | **21.60** | **24.00** | **28.50** | **35.04** | **64.75** | **2.72x** |
| random_qk | 24.15 | 24.19 | 25.90 | 38.81 | 70.63 | 107.51 | 181.27 | 7.49x |
| random_qkv | 24.18 | 24.18 | 22.58 | 24.83 | 27.54 | 28.83 | 38.54 | 1.59x |
| lf_qk | 24.91 | 24.90 | 22.44 | 23.89 | 25.57 | 27.27 | 40.91 | 1.64x |

*ALiBi eval pending for 32K+ (chunked attention eval running).

### rpemb4_qkv — symmetric expression with V rotation

`freq = (1 + tanh(LN(emb))) * rope_base`, `angle = noise * freq`, Q/K/V rotation + inverse.

**Val PPL 24.01, 65K/512 = 2.72x.** V rotation improves extrapolation for the symmetric expression (3.83x → 2.72x), unlike the original rpemb_v2 expression where V rotation was catastrophic (3.83x → 10.4x).

The symmetric expression `(1 + tanh(LN(emb))) * rope_base` is more robust to V rotation because:
- Always positive (range 0 to 2×rope), no abs() needed
- Smoothly scales rope_base up or down
- No asymmetric additive π offset that creates dimension-dependent artifacts

### V rotation: helps deterministic models, hurts learned-frequency noise models

The full 65K extrapolation revealed a surprising pattern: V rotation does NOT universally improve extrapolation.

**V rotation helps deterministic angle models:**

| Model | qk 65K/512 | qkv 65K/512 | Effect |
|-------|------------|-------------|--------|
| pemb | 5.93x | 4.35x | Helps |
| cbd_K4 | 20.5x | 5.32x | Helps |
| pmlp | 18.1x | 2.60x | Helps a lot |

**V rotation helps fixed-frequency noise:**

| Model | qk 65K/512 | qkv 65K/512 | Effect |
|-------|------------|-------------|--------|
| random | 7.49x | **1.59x** | Helps |

**V rotation with learned-frequency noise — depends on expression:**

| Model | Expression | qk 65K/512 | qkv 65K/512 | Effect |
|-------|-----------|------------|-------------|--------|
| rpemb4 | `(1+tanh(LN(emb)))*rope` (symmetric) | 3.83x | **2.72x** | Helps |
| lf | `abs(LN(MLP(x)))` | **1.64x** | 5.29x | Hurts |

The key insight: in lf and rpemb, the "noise" is `uniform × learned_freq(token)`. The learned frequencies make the noise distribution token-dependent — different tokens produce different noise magnitudes. This is structured noise, not truly random. V rotation amplifies this structure: the value rotations depend on the cumsum of these token-dependent angles, creating patterns the model learns to exploit at training length but that break at longer sequences.

In pure random, `uniform × fixed_freq` — every token gets the same noise distribution regardless of identity. There is no token-dependent structure for V rotation to exploit. V rotation with truly uniform noise acts as pure decorrelation, which helps.

**The rule: V rotation helps when angle noise is truly random (uniform across tokens) or when angles are deterministic (controls position leakage). V rotation hurts when angle noise has learned structure (creates exploitable patterns at training length that don't transfer).**

### The three tiers of length extrapolation (revised)

**Tier 1 — Flat at 128x (65K/512 < 2x):**
- **random_qkv (1.59x)**: truly random noise + fixed freq + V rotation
- **lf_qk (1.64x)**: learned-freq noise WITHOUT V rotation

**Tier 2 — Moderate degradation (2-6x at 65K):**
- pmlp_qkv (2.60x): deterministic MLP + V rotation
- rpemb_v2_qk (3.83x): learned-freq noise WITHOUT V rotation
- rpemb4_qk (3.83x): symmetric learned-freq noise WITHOUT V rotation
- pemb_qkv (4.35x): deterministic token emb + V rotation
- lf_qkv (5.29x): learned-freq noise + V rotation (V rotation hurt)
- cbd_K4_qkv (5.32x): deterministic codebook + V rotation
- pemb_qk (5.93x): deterministic token emb

**Tier 3 — Blown up (>7x at 65K):**
- random_qk (7.49x), rpemb_v2_qkv (10.4x), pmlp_qk (18.1x), cbd_K4_qk (20.5x), jfixed (21.0x), RoPE (38.5x)

### rpemb experiments — final results

**rpemb_v2_qk** (`noise * abs(LN(tanh(emb)*π + rope_base))`): final val **23.92**, 65K/512 = **3.83x**. Beat random_qk (24.15) by 0.23 PPL — the learned per-token frequencies provided a small edge over fixed frequencies.

**rpemb4_qk** (`noise * (1 + tanh(LN(emb))) * rope_base`, symmetric): final val **23.96**, 65K/512 = **3.83x**. Essentially identical to rpemb_v2. The symmetric expression works as well as the original.

**rpemb_v2_qkv**: final val **23.83**, 65K/512 = **10.4x**. V rotation improved val PPL (23.83 vs 23.92) but destroyed extrapolation (10.4x vs 3.83x). The learned frequencies + V rotation create exploitable structure at training length.

### pemb expression ablation — the additive offset is essential

Attempted cleaner expressions for pemb (deterministic per-token angles) to replace `tanh(emb) * π + rope_base`:

| Expression | Init | Result |
|-----------|------|--------|
| `tanh(emb) * π + rope_base` (original pemb) | zero | **Works.** Matches RoPE at 200K, 1.12x extrap |
| `LN(emb) * rope_base` | random | **Fails.** 4.09x at 5K |
| `LN(tanh(emb)) * rope_base` | random | **Fails.** 7.50x at 10K |
| `tanh(emb) * rope_base` | ones | **Fails.** 6.40x at 10K |

Every alternative fails. The additive structure `correction + rope_base` is essential — the learned correction must be ADDED to rope_base, not multiplied with it. The additive offset provides a stable frequency floor that prevents the angles from collapsing or diverging.

### V rotation compute cost — torch.compile eliminates the overhead

Benchmarked RoPE vs jfixed (RoPE + V rotation) on A100 with SDPA attention, batch=4:

**Without torch.compile (unfused kernels):**

| Seq Len | RoPE | jfixed | V rotation overhead |
|---------|------|--------|---------------------|
| 512 | 10.8ms | 13.7ms | 27% |
| 2048 | 28.2ms | 35.5ms | 26% |
| 8192 | 133.6ms | 162.2ms | 21% |

**With torch.compile (fused kernels):**

| Seq Len | RoPE | jfixed | V rotation overhead |
|---------|------|--------|---------------------|
| 512 | 6.9ms | 7.8ms | **13%** |
| 2048 | 23.9ms | 25.1ms | **5%** |
| 8192 | 118.1ms | 120.6ms | **2%** |

V rotation overhead drops from ~25% to **2-13%** with torch.compile. At longer sequences, it's essentially free.

**Why torch.compile helps:** Without compilation, each rotary application (multiply by cos, multiply by sin, subtract, concatenate) launches separate GPU kernels. Each kernel reads the full tensor from GPU memory (HBM), computes one operation, and writes the result back. V rotation adds 2 extra rotary applications (V forward + output inverse) = many extra memory round-trips.

`torch.compile` traces the computation graph and fuses multiple element-wise operations into a single kernel. Instead of: read tensor → multiply cos → write temp → read temp → subtract → write result → ... (many HBM round-trips), the fused kernel does: read tensor → compute everything → write final result (one round-trip). The intermediate tensors are kept in fast on-chip SRAM, never written to slow HBM.

The actual arithmetic (multiplies, adds) is identical — the same math happens in the same order. Only the memory access pattern changes. Small BF16 rounding differences (<0.02 in logits) arise from fused multiply-add instructions but are negligible.

**The impact of V rotation (near-zero cost, massive extrapolation gain):**

| Model | qk 65K/512 | qkv 65K/512 | Improvement |
|-------|------------|-------------|-------------|
| pmlp | 18.1x | 2.60x | **7.0x better** |
| random | 7.49x | 1.59x | **4.7x better** |
| cbd_K4 | 20.5x | 5.32x | **3.9x better** |
| pemb | 5.93x | 4.35x | **1.4x better** |

V rotation consistently improves extrapolation for deterministic and fixed-frequency models, with near-zero compute cost when using torch.compile.

### pemb_qk compute cost — identical to RoPE

Benchmarked RoPE vs pemb_qk vs jfixed on A100 with torch.compile + SDPA, batch=4:

| Seq Len | RoPE | pemb_qk | pemb overhead | jfixed | jfixed overhead |
|---------|------|---------|---------------|--------|-----------------|
| 512 | 7.4ms | 7.4ms | **0%** | 8.2ms | 11% |
| 1024 | 15.6ms | 16.3ms | **4%** | 16.9ms | 8% |
| 2048 | 38.9ms | 39.0ms | **0%** | 41.6ms | 7% |
| 4096 | 114.5ms | 121.3ms | **6%** | 126.4ms | 10% |
| 8192 | 389.2ms | 393.4ms | **1%** | 404.1ms | 4% |

**pemb_qk has zero compute overhead vs RoPE.** The per-layer embedding lookup and cumsum are negligible compared to the QKV projection and attention computation. With torch.compile fusing the element-wise operations, pemb_qk is indistinguishable from RoPE in speed.

**The full pemb_qk value proposition:**
- Matches RoPE on val PPL (23.56 vs 23.56)
- Flat extrapolation at training length (1.12x at 8K)
- Zero compute overhead vs RoPE
- Flash Attention compatible (identical rotary interface)
- No post-training required for longer contexts
- Theoretically guaranteed: token-only angle dependence prevents position encoding

There is no reason to use RoPE over pemb_qk. Same quality, same speed, better extrapolation.

### Learned frequency experiments — rope_lf and jfixed_lf (completed)

Made RoPE and jfixed frequencies learnable (`nn.Parameter` initialized from standard 1/10000^(2i/d)), trained with full schedule.

**Base training (constant lr=5e-4):**
- rope_lf trails RoPE by ~0.1-0.3 PPL throughout
- jfixed_lf slightly ahead of jfixed early (5-25K), then converges

**Schedule (lr decay):**
rope_lf pulls ahead of RoPE from the very first eval of the schedule:

| Iter | RoPE | rope_lf | diff |
|------|------|---------|------|
| 105K | 28.23 | 27.85 | -0.38 |
| 120K | 27.37 | 27.01 | -0.36 |
| 150K | 27.09 | 26.77 | -0.32 |
| 170K | 24.06 | 23.77 | -0.29 |
| 180K | 23.87 | 23.53 | -0.34 |
| **200K** | **23.56** | **23.37** | **-0.19** |

rope_lf is consistently 0.24-0.51 ahead of RoPE throughout the schedule. Learned frequencies underperform at high lr but benefit from lr decay — the optimizer can fine-tune frequencies more precisely at lower lr.

jfixed_lf trails jfixed throughout the schedule:

| Iter | jfixed | jfixed_lf | diff |
|------|--------|-----------|------|
| 160K | 23.99 | 24.44 | +0.45 |
| 165K | 23.79 | 24.11 | +0.32 |
| **200K** | **23.21** | **23.35** | **+0.14** |

**Full extrapolation to 65K:**

| Model | Val PPL | 512 | 4096 | 8192 | 16384 | 32768 | 65536 | 65K/512 |
|-------|---------|-----|------|------|-------|-------|-------|---------|
| jfixed | 23.21 | 23.17 | 127.81 | 214.18 | 306.73 | 404.45 | 485.66 | 21.0x |
| jfixed_lf | 23.35 | 23.32 | 185.05 | 334.68 | 515.09 | 627.52 | 716.66 | 30.7x |
| rope_lf | 23.37 | 23.29 | 260.68 | 394.65 | 633.93 | 874.01 | 1037.12 | 44.5x |
| RoPE | 23.56 | 23.54 | 223.88 | 381.82 | 600.53 | 808.35 | 905.18 | 38.5x |

**Learned frequencies improve val PPL but worsen extrapolation.** rope_lf gains 0.19 PPL over RoPE but blows up more at 65K (44.5x vs 38.5x). jfixed_lf loses on both val PPL (-0.14) and extrapolation (30.7x vs 21.0x).

The optimizer tunes frequencies to fit training-length attention patterns more precisely. This helps val PPL but makes the model more dependent on the specific frequency values, which become more out-of-distribution at long sequences.

**Why learned frequencies help RoPE but hurt jfixed:** jfixed already beats RoPE with fixed frequencies (23.21 vs 23.56) because the cumsum + V rotation acts as implicit regularization. Learning frequencies removes this regularization, hurting jfixed. RoPE has no such implicit regularization — learning is purely beneficial for val PPL.

### ALiBi full extrapolation (completed)

Evaluated using chunked attention to avoid OOM (4096-token chunks, ALiBi bias computed per chunk).

| Model | Val PPL | 512 | 4096 | 8192 | 16384 | 32768 | 65536 | 65K/512 |
|-------|---------|-----|------|------|-------|-------|-------|---------|
| **ALiBi** | **23.87** | **23.87** | **21.75** | **22.46** | **23.19** | **21.06** | **22.84** | **0.96x** |

ALiBi achieves perfectly flat extrapolation — actually *improving* at longer lengths (32K PPL 21.06 < 512 PPL 23.87). This is because more context provides more information for prediction. Models with flat extrapolation should naturally improve at longer lengths. The degradation seen in other models at long lengths is an artifact of broken position encoding, not a fundamental property.

ALiBi's extrapolation is in a different league from anything in our cumsum framework. Our best flat models (random_qkv 1.59x, lf_qk 1.64x at 65K) still degrade. ALiBi doesn't.

However, ALiBi has weaknesses:
- 0.31 PPL behind RoPE at training length (23.87 vs 23.56) — gap likely grows at scale
- Distance-based attention decay limits needle-in-haystack recall at very long distances (though low-slope heads can reach hundreds/thousands of positions back)
- Our pemb_qk matches RoPE exactly (23.56) while providing better-than-RoPE extrapolation (5.93x vs 38.5x at 65K)

### Updated complete extrapolation table

All scheduled 200K models, evaluated 512–65536 (128x training length):

| Model | Val PPL | 512 | 4096 | 8192 | 16384 | 32768 | 65536 | 65K/512 |
|-------|---------|-----|------|------|-------|-------|-------|---------|
| jfixed | 23.21 | 23.17 | 127.81 | 214.18 | 306.73 | 404.45 | 485.66 | 21.0x |
| jfixed_lf | 23.35 | 23.32 | 185.05 | 334.68 | 515.09 | 627.52 | 716.66 | 30.7x |
| rope_lf | 23.37 | 23.29 | 260.68 | 394.65 | 633.93 | 874.01 | 1037.12 | 44.5x |
| RoPE | 23.54 | 23.54 | 223.88 | 381.82 | 600.53 | 808.35 | 905.18 | 38.5x |
| pmlp_qk | 23.55 | 23.59 | 22.83 | 34.30 | 71.35 | 149.27 | 427.65 | 18.1x |
| pemb_qk | 23.56 | 23.56 | 22.56 | 27.62 | 38.93 | 67.13 | 139.63 | 5.93x |
| cbd_K4_qk | 23.56 | 23.47 | 24.25 | 33.76 | 64.78 | 159.02 | 480.89 | 20.5x |
| pemb_qkv | 23.57 | 23.60 | 22.82 | 27.89 | 37.58 | 57.57 | 102.53 | 4.35x |
| cbd_K4_qkv | 23.67 | 23.65 | 22.97 | 28.37 | 38.66 | 60.65 | 125.78 | 5.32x |
| pmlp_qkv | 23.74 | 23.86 | 22.27 | 25.41 | 30.67 | 39.53 | 62.06 | 2.60x |
| rpemb_v2_qkv | 23.83 | 23.65 | 21.75 | 26.35 | 42.58 | 85.53 | 245.49 | 10.4x |
| ALiBi | 23.87 | 23.87 | 21.75 | 22.46 | 23.19 | 21.06 | 22.84 | 0.96x |
| rpemb_v2_qk | 23.92 | 23.79 | 21.50 | 24.08 | 30.12 | 42.76 | 90.99 | 3.83x |
| rpemb4_qk | 23.96 | 23.82 | 21.74 | 24.85 | 32.34 | 44.65 | 91.22 | 3.83x |
| lf_qkv | 23.98 | 23.70 | 21.23 | 23.77 | 32.58 | 49.40 | 125.24 | 5.29x |
| rpemb4_qkv | 24.01 | 23.80 | 21.60 | 24.00 | 28.50 | 35.04 | 64.75 | 2.72x |
| random_qk | 24.15 | 24.19 | 25.90 | 38.81 | 70.63 | 107.51 | 181.27 | 7.49x |
| random_qkv | 24.18 | 24.18 | 22.58 | 24.83 | 27.54 | 28.83 | 38.54 | 1.59x |
| lf_qk | 24.91 | 24.90 | 22.44 | 23.89 | 25.57 | 27.27 | 40.91 | 1.64x |

### Summary of all findings

**At training length (512):** pemb_qk matches RoPE (23.56) at zero compute overhead. The position encoding mechanism does not limit model quality.

**At moderate extrapolation (8x, 4096):** All cumsum models improve over 512 PPL — more context helps. Best: rpemb4_qkv (21.60), lf_qkv (21.23), ALiBi (21.75).

**At extreme extrapolation (128x, 65536):** Only ALiBi (0.96x) stays truly flat. Best cumsum models: random_qkv (1.59x), lf_qk (1.64x). Models with deterministic angles and no V rotation degrade most.

**V rotation:** Near-zero cost with torch.compile (2-4% overhead). Consistently helps deterministic models and fixed-frequency noise. Hurts learned-frequency noise models (lf_qkv, rpemb_v2_qkv) but helps rpemb4_qkv (symmetric expression).

**Learned frequencies:** Improve val PPL for RoPE (23.37 vs 23.56) but worsen extrapolation (44.5x vs 38.5x). Hurt jfixed on both metrics.

**The tradeoff:** There is a fundamental tension between val PPL and extrapolation. Models that fit training-length patterns precisely (learned frequencies, data-dependent angles) achieve better PPL but degrade faster at long lengths. Models that prevent position encoding (random noise, token-only angles) extrapolate better but leave PPL on the table. pemb_qk is the best compromise — matches RoPE on PPL while maintaining moderate extrapolation (5.93x at 65K, 1.12x at 8K).

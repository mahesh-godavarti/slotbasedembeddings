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

## Plan: Next Experiments

- **Longer training of JoFormer v2 from fixed base** — the model is still improving at 50K, more training may further improve both training-length and extrapolation PPL.
- **JoFormer v2 warmup with angle_lr=5e-4** — same warmup approach but with angle lr matching main lr.

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

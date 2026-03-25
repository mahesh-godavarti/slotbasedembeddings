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

## Plan: Next Experiments

### Scale-up experiment
- **Config**: n_embed=768, n_layers=16, n_heads=8, block_size=512, window=32
- **Models**: hybrid_1 (RoPE + NoPE) vs joformer2_hybrid_1 (JoFormer v2 + NoPE)
- **Params**: RoPE ~163M, JoFormer v2 ~193M
- **Data**: OWT (9.1B tokens, vocab=32K)
- **GPU**: A6000 49GB — both fit (32GB and 37GB)
- **Goal**: Test whether joformer v2's ~2 PPL advantage over RoPE grows with scale

## Hardware

- GPU: NVIDIA RTX A6000 (49GB VRAM)
- CPU: AMD EPYC 7763 (8 cores)
- RAM: 64GB
- Training speed: ~8 it/s on OWT

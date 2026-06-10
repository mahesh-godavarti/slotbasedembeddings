# Learned Frequency Experiments: RoPE vs rope_lf, jfixed vs jfixed_lf

Comparing standard fixed frequencies (1/10000^(2i/d)) with learned frequencies initialized from the same values.

## Prior work

### Trainable RoPE frequencies
- **ComRoPE**: trainable commuting angle matrices (vision, ImageNet)
- **CARoPE**: context-dependent frequencies (similar to our lf/rpemb models)
- **Mixed-frequency RoPE**: learnable spectral variants with flexible frequency/amplitude/phase

Our experiment is a simple ablation: just make the scalar frequencies an `nn.Parameter`. Not novel as a method — we're using it to understand whether the standard RoPE frequency spacing (1/10000^(2i/d)) is optimal or if learning can improve it.

### Randomized Positional Encodings (Ruoss et al., ACL 2023)
Google DeepMind (ACL 2023). Instead of positions [0, 1, 2, ..., N-1], sample N random increasing indices from a larger range [1, L] during training. Training length N=40, test up to M=500 (12.5x), sampling range L=2048. Tiny model (249K params, 5 blocks, d=64).

Evaluated only on **algorithmic reasoning tasks** (15 synthetic tasks) — NOT language modeling, no perplexity comparison with RoPE. The authors note: "if L is much larger than N or M, performance degrades since the model is unlikely to encounter enough unique indices during training."

Fundamental limitation: extrapolation is bounded by L (the maximum position seen during training). The authors tested L up to 8192 with N=40. For language modeling at scale with training length 512 and extrapolation to 65536, L would need to be ≥65536. Sampling 512 indices from [1, 65536] gives average gaps of ~128 between consecutive positions — a very different distribution from dense sequential text.

Our approach is fundamentally different — we eliminate position encoding entirely rather than randomizing it. Position emerges from token-level composition, so there is no maximum range.

### "Round and Round We Go! What makes Rotary Positional Encodings useful?" (ICLR 2025)
Analysis of why RoPE works. Key findings relevant to our work:
- RoPE doesn't actually decay attention with distance (contrary to popular belief)
- Models mostly use **low RoPE frequencies** — high frequencies are reserved for specialized positional attention heads (diagonal, previous-token patterns)
- Truncating lowest frequencies (p-RoPE) actually improves performance on Gemma 2B
- Randomized positional encodings work because they force the model to rely on low frequencies, achieving a soft form of distance invariance

The finding that models predominantly use low frequencies is consistent with our learned frequency experiments — if the model only needs low frequencies, the standard log-spaced RoPE schedule (which includes many unused high frequencies) is already near-optimal, explaining why rope_lf doesn't improve over RoPE.

### TAPA — Token-Aware Phase Attention (ICLR 2026, rejected)
Learnable token-dependent phase function for attention. LLaMA3-7B scale. Trained at 8K, extrapolates to 32K (4x). Rejected due to efficiency concerns (1.1-1.3x slowdown), limited evaluation (single architecture), and weak ablations. Our pemb_qk achieves zero overhead with torch.compile and extrapolates to 128x.

**TAPA needle-in-haystack results** (after fine-tuning ALL models at 32K with 5B extra tokens):

| Method | 1K | 2K | 4K | 8K | 16K | 32K | 64K | Avg |
|--------|----|----|----|----|-----|-----|-----|-----|
| RoPE (b=5e5) | 99 | 100 | 100 | 99 | 96 | 95 | 0 | 84.1 |
| RoPE (b=2e8) | 98 | 97 | 99 | 99 | 96 | 93 | 0 | 83.1 |
| TAPA | 99 | 98 | 98 | 98 | 99 | 100 | **96** | **98.3** |

TAPA maintains 96% recall at 64K where RoPE collapses to 0%. However, the 64K result is only 2x beyond the 32K fine-tuning length — not a test of extrapolation from the original 8K training. All models were fine-tuned at 32K before this evaluation.

**Recall mechanism comparison:**
- **RoPE**: relative rotation = `(i-j) * freq`, deterministic function of distance. Enables precise distance-based retrieval at training-length distances. Fails at distances beyond training because rotation magnitudes go out of distribution.
- **ALiBi**: explicit distance-based attention decay. Suppresses ALL distant tokens, including needles. Flat PPL extrapolation but recall fails by design.
- **TAPA**: phase depends on Q·K content interaction. Content-based retrieval independent of distance. Explains the strong needle-in-haystack performance.
- **Our cumsum (pemb/rpemb)**: relative rotation = sum of intervening token angles. Depends on the CONTENT of tokens between query and key, not on distance. For within-training-distance retrieval (e.g., needle 100 tokens back at position 50,000), the cumsum of ~100 token angles is within training distribution regardless of absolute position. For beyond-training-distance retrieval (needle 50,000 tokens back), the cumsum is a huge random walk and recall likely fails.

**Key insight**: in our cumsum models, recall at distance D depends on whether D is within training length, not on absolute position. A needle 100 tokens back should be retrievable at position 50,000 just as well as at position 200. This is fundamentally different from RoPE (where absolute position matters through the position * freq computation) and ALiBi (where distance always hurts).

### MLA — Multi-Head Latent Attention (DeepSeek-V2/V3)

MLA compresses the KV cache by projecting keys and values into a low-rank latent space (e.g., 512 dimensions instead of 8192), reducing KV cache memory by 93.3%. Only a subset of Q/K dimensions use RoPE — the rest are position-free.

**MLA inherits RoPE's limitations.** The RoPE dimensions will produce out-of-distribution rotations beyond training length. Position information from those dimensions propagates to the non-RoPE dimensions through the forward-pass feedback loop (attention → hidden states → subsequent layers). This means:
- MLA will have the same extrapolation failures as RoPE beyond training length
- Needle-in-haystack beyond training-length distances will fail for the same reasons as standard RoPE
- DeepSeek-V3 trains at 128K, so the issue is hidden in practice but exists in principle

**Application of our work to MLA:** Replacing the RoPE dimensions in MLA with pemb (per-token cumsum angles) would fix the extrapolation limitation while preserving the latent compression benefits. pemb changes how position is encoded (cumsum of token angles vs position × freq), MLA changes how KV is stored (compressed latent vs full per-head). These are orthogonal — they can be combined directly. The pemb dimensions would provide position information that extrapolates flat, while the latent compression would maintain MLA's memory efficiency.

## Learned Frequency Experiments

- **rope_lf**: RoPE with `nn.Parameter(freqs)` — Q/K rotation only
- **jfixed_lf**: jfixed with `nn.Parameter(freqs)` — Q/K/V rotation + inverse

Both use the same schedule: 100K@5e-4 → 50K@2e-4 → 50K@5e-5.

## Constant lr=5e-4 base training comparison

| Iter | RoPE | rope_lf | jfixed | jfixed_lf | RoPE ext | rope_lf ext | jfixed ext | jfixed_lf ext |
|------|------|---------|--------|-----------|----------|-------------|------------|---------------|
| 5K | 59.14 | 59.33 | 59.16 | **58.51** | 3.79x | 3.48x | 2.43x | 2.32x |
| 10K | 46.92 | 47.12 | 46.87 | **46.43** | 6.40x | 5.40x | 3.19x | 3.18x |
| 15K | 42.53 | 42.50 | 42.50 | **42.02** | 8.16x | 6.23x | 3.77x | 3.94x |
| 20K | 39.75 | 40.06 | 39.77 | **39.69** | 8.51x | 8.19x | 4.21x | 5.02x |
| 25K | 38.04 | 38.12 | 37.93 | **37.81** | 9.78x | 8.50x | 4.45x | 5.50x |
| 30K | 36.83 | 36.91 | 36.61 | — | 13.13x | 11.45x | 4.60x | — |

*Extrap = 8K/512 PPL ratio. Lower is better.*

## Observations (base training, constant lr=5e-4)

- **jfixed_lf** is slightly ahead early (5-25K), converges with jfixed (30-100K), then trails in the schedule
- **rope_lf** is slightly behind RoPE (~0.1-0.3 worse) during base training
- **jfixed_lf extrap** is slightly worse than jfixed (5.50x vs 4.45x at 25K)
- **rope_lf extrap** is slightly better than RoPE (8.50x vs 9.78x at 25K)

## Schedule results — rope_lf pulls ahead of RoPE

Learned frequencies underperform at high lr (5e-4) but benefit from lr decay. rope_lf pulls ahead of RoPE from the **very first eval of the schedule** and maintains the lead throughout.

### RoPE vs rope_lf (full schedule)

| Iter | RoPE | rope_lf | diff |
|------|------|---------|------|
| 105K | 28.23 | 27.85 | **-0.38** |
| 110K | 27.73 | 27.33 | **-0.40** |
| 115K | 27.54 | 27.22 | **-0.32** |
| 120K | 27.37 | 27.01 | **-0.36** |
| 125K | 27.28 | 27.03 | **-0.25** |
| 130K | 27.22 | 26.91 | **-0.31** |
| 135K | 27.34 | 26.83 | **-0.51** |
| 150K | 27.09 | 26.77 | **-0.32** |
| 155K | 24.76 | 24.48 | **-0.28** |
| 160K | 24.44 | 24.15 | **-0.29** |
| 165K | 24.20 | 23.91 | **-0.29** |
| 170K | 24.06 | 23.77 | **-0.29** |
| 175K | 23.92 | 23.68 | **-0.24** |
| 180K | 23.87 | 23.53 | **-0.34** |
| 185K | 23.78 | 23.52 | **-0.26** |

rope_lf is consistently **0.24-0.51 PPL ahead** of RoPE throughout the schedule. At 185K: rope_lf = 23.52 vs RoPE = 23.78. The standard RoPE frequency spacing (1/10000^(2i/d)) is not optimal — learning frequencies improves val PPL by ~0.3 when combined with lr decay.

### jfixed vs jfixed_lf (schedule, in progress)

| Iter | jfixed | jfixed_lf | diff |
|------|--------|-----------|------|
| 155K | 24.36 | 26.59 | +2.23 |
| 160K | 23.99 | 24.44 | +0.45 |
| 165K | 23.79 | 24.11 | +0.32 |

jfixed_lf is **behind** jfixed by 0.3-0.5 PPL in the schedule. Opposite of rope_lf. Learned frequencies help RoPE but hurt jfixed under the schedule. Still running — may close the gap.

### Why learned frequencies help RoPE but not jfixed

Hypothesis: jfixed already beats RoPE at fixed frequencies (23.21 vs 23.56 at 200K). The fixed cumsum acts as implicit regularization. Learning frequencies removes this regularization benefit, hurting jfixed. RoPE has no such regularization — it uses position * freq directly — so learning frequencies is purely beneficial.

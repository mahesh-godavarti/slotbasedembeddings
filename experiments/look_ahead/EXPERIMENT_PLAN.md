# Experiment Plan

## Priority 1: Width scaling — does the D-N gap grow with C?

**Question**: Does the look-ahead architecture benefit more from increased width (C) than roformer?

**Setup**: For each N=D pair, run at multiple C values. Same block_size=64, same total tokens (Chinchilla-optimal or fixed budget). Compare D-N gap at each C.

### Experiment matrix

| N=D | C values | n_head | FLOPs per token |
|-----|----------|--------|----------------|
| 1 | 256, 512, 1024, 2048 | 1 | 20C² (D), 12C² (N) |
| 2 | 256, 512, 1024, 2048 | 1 or 4/8/16 | 32C² (D), 24C² (N) |
| 3 | 256, 512, 1024 | 8/16 | 44C² (D), 36C² (N) |

For each (N, C):
1. Train roformer N for X tokens
2. Convert to D, fine-tune for X/4 tokens
3. Record final PPL for both

**Expected result**: If the D-N gap grows with C, it means wider layers amplify the correction mechanism — the corr_ffn has richer representations to work with.

### Token budgets (Chinchilla 20:1)

| C | Params (N=2) | Optimal tokens |
|---|-------------|---------------|
| 256 | ~25M | 500M |
| 512 | ~45M | 900M |
| 1024 | ~91M | 1,800M |
| 2048 | ~200M | 4,000M |

C=2048 may be too expensive. C=256/512/1024 is feasible.

---

## Priority 2: Depth scaling at fixed C — where does D stop helping?

**Question**: At fixed C=1024, how does the D-N gap scale with N?

We have N=1,2,3,6,12,13,24 and D=1,2,3,6,12,23,24 at C=1024 (various block_sizes). But the scaling experiment (block_size=64) only covers N=1,2,3,6.

### Missing data points
- N=12 vs D=12 at block_size=64 (we have at block_size=256)
- N=4, N=5 at C=1024

### Analysis
Plot the token-matched crossover point vs N. Does crossover take linearly more tokens at larger N, or exponential?

---

## Priority 3: Optimal fine-tune budget

**Question**: What's the optimal ratio of fine-tune tokens to pretraining tokens?

**Setup**: For a fixed N=2 C=1024 roformer trained for 1,227M tokens:
- Fine-tune at 5%, 10%, 20%, 33%, 50%, 100% of pretraining tokens
- Compare final D PPL

**Expected result**: Diminishing returns after some threshold. Find the knee.

---

## Priority 4: D=x C_wide vs N=x C_narrow FLOP-matched

**Question**: At FLOP parity, is it better to have fewer wider look-ahead layers or more narrower roformer layers?

We have one data point: D=8 C=768 beat N=6 C=1024 by 1.38 PPL (different C, FLOP-matched).

### Proposed experiments

| Look-ahead | Roformer (FLOP-matched) |
|-----------|------------------------|
| D=2 C=1024 (32C²=33.6M) | N=3 C=838 (36C²=33.6M) |
| D=3 C=1024 (44C²=46.1M) | N=4 C=982 (48C²=46.1M) |
| D=6 C=1024 (80C²=83.9M) | N=7 C=1024 (84C²=88.1M) |
| D=12 C=1408 (152C²=301M) | N=24 C=1024 (288C²=302M) |

The last one is already running on the other machine (D=12 C=1408).

---

## Priority 5: MTP (Multi-Token Prediction)

**Question**: Can the correction vectors predict multiple future tokens for speculative decoding?

### Steps
1. Add MTP heads to a trained D model (predict tokens t+2, t+3 and corrections c+2, c+3)
2. Fine-tune briefly to train the MTP heads
3. Measure acceptance rate in speculative decoding
4. Compare throughput vs standard sequential K=1

### Prerequisites
- KV cache implementation for sequential inference
- Speculative decoding verification loop

---

## Priority 6: Real model conversion

**Question**: Can we convert a pretrained open-source transformer (e.g., GPT-2, Llama) to look-ahead and improve it?

### Steps
1. Load GPT-2 small (124M params, 12 layers)
2. Map weights to block_head_corr_ffn_add D=12
3. Zero-init corr_ffn
4. Fine-tune at K=2-5 on the same training data
5. Compare PPL before and after

This would be the ultimate demonstration of the practical recipe.

# Look-Ahead Architecture: Complete Findings

## The Architecture

A D-block transformer unit (D layers with separate weights) is iterated K times during training. At inference, sequential autoregressive generation means K=1 is sufficient -- each position naturally sees corrected representations from all previous positions.

The correction mechanism (`corr_ffn_add`): after each iteration, `correction = corr_ffn(ln(shift(z) + tok_emb))`, then `processed_x = tok_emb + correction`. The corr_ffn adds 8C^2 FLOPs per token at inference.

Total inference FLOPs: (12D + 8)C^2 per token.

## Finding 1: D=x beats N=x at matched token counts

At every scale tested (D=1,2,3,6), the look-ahead architecture beats the standard roformer with the same number of layers, given enough training tokens. The gap widens over training.

All experiments: C=1024, block_size=64, OWT data.

| Scale | Crossover (M tokens) | Final gap (PPL) | Still growing? |
|-------|---------------------|-----------------|----------------|
| D=1 vs N=1 | ~424M | -34.4 | Yes |
| D=2 vs N=2 | ~565M | -7.6 | Yes |
| D=3 vs N=3 | ~835M | -3.0 | Yes |
| D=6 vs N=6 | ~1,032M | -0.7 | Yes |

The correction mechanism starts with a disruption (PPL spike) then recovers and surpasses the roformer. Crossover takes longer at larger N. After crossover, the gap widens consistently.

### Token-matched tables

**D=1 vs N=1** (n_head=1):

| Total tokens (M) | N=1 PPL | D=1 PPL | Gap |
|---|---|---|---|
| 414 | 144.8 | 152.4 | +7.6 |
| 435 | 143.0 | 138.6 | -4.3 |
| 556 | 134.3 | 109.7 | -24.5 |
| 835 | 122.0 | 90.4 | -31.5 |
| 1,126 | 114.8 | 80.4 | -34.4 |

**D=2 vs N=2** (n_head=16):

| Total tokens (M) | N=2 PPL | D=2 PPL | Gap |
|---|---|---|---|
| 414 | 90.1 | 100.2 | +10.1 |
| 581 | 83.5 | 83.1 | -0.3 |
| 835 | 77.7 | 73.1 | -4.6 |
| 1,126 | 73.7 | 66.1 | -7.6 |

**D=3 vs N=3** (n_head=16):

| Total tokens (M) | N=3 PPL | D=3 PPL | Gap |
|---|---|---|---|
| 414 | 74.6 | 83.1 | +8.5 |
| 716 | 67.1 | 67.7 | +0.6 |
| 835 | 65.4 | 65.3 | -0.0 |
| 1,126 | 62.4 | 59.4 | -3.0 |

**D=6 vs N=6** (n_head=16):

| Total tokens (M) | N=6 PPL | D=6 PPL | Gap |
|---|---|---|---|
| 414 | 62.2 | 65.9 | +3.6 |
| 835 | 54.9 | 56.9 | +2.0 |
| 1,032 | 53.5 | 53.3 | -0.2 |
| 1,126 | 53.0 | 52.2 | -0.7 |

## Finding 2: D=12 matches N=14 at C=1024

At C=1024, block_size=256, 200K iters, batch=32:

| Model | Inference FLOPs | Final PPL |
|-------|----------------|-----------|
| N=12 | 144C^2 | 33.41 |
| N=13 | 156C^2 | 32.82 |
| D=12 from scratch | 152C^2 | 32.28 |
| D=12 fine-tuned from N=12 | 152C^2 | 32.21 |
| N=14 | 168C^2 | 32.34 |

D=12 (152C^2) beats N=13 (156C^2) by 0.54 PPL and essentially matches N=14 (168C^2). The correction mechanism at 8C^2 provides roughly the same benefit as 2 extra transformer layers (24C^2) -- 3x more efficient per FLOP.

The D=12 vs N=14 gap over training:

| Iter | D=12 | N=14 | Gap |
|------|------|------|-----|
| 30K | 49.89 | 49.31 | +0.58 |
| 55K | 42.04 | 41.54 | +0.50 |
| 75K | 38.87 | 38.81 | +0.06 |
| 100K | 36.59 | 36.58 | +0.01 |
| 120K | 35.34 | 35.35 | -0.01 |
| 150K | 33.94 | 33.91 | +0.03 |
| 200K | 32.28 | 32.34 | -0.06 |

Essentially tied from 75K onwards.

## Finding 3: D=23 beats N=24 at near FLOP parity

At C=1024, block_size=256, 200K-equivalent iters:

| Model | Inference FLOPs | Final PPL |
|-------|----------------|-----------|
| N=24 | 288C^2 | 29.42 |
| D=23 from scratch | 284C^2 | 28.89 |
| D=24 fine-tuned from N=24 | 296C^2 | 28.99 |

D=23 beats N=24 by 0.53 PPL with slightly fewer FLOPs. The gap was consistent throughout training (~0.4-0.5 PPL from 25K-equiv onwards).

## Finding 4: The relative advantage grows with width (C)

D=2 vs N=2 from scratch at matched Chinchilla-optimal tokens, block_size=64:

| C | N=2 PPL | D=2 PPL | Gap | Relative improvement |
|---|---------|---------|-----|---------------------|
| 256 | 158.83 | 143.57 | -15.26 | 9.6% |
| 512 | 95.48 | 84.69 | -10.79 | 11.3% |
| 1024 | 72.83 | 60.84 | -11.99 | 16.5% |

The relative improvement grows with C: 9.6% -> 11.3% -> 16.5%. Wider layers amplify the correction mechanism's benefit. This suggests the look-ahead architecture becomes MORE valuable at larger scale, not less.

## Finding 5: D=1 C=1952 approaches N=6 C=1024

At FLOP parity (both ~72 x 1024^2 FLOPs), 1,227M tokens:

| Model | Layers | Width | FLOPs | PPL |
|-------|--------|-------|-------|-----|
| N=6 C=1024 | 6 | 1024 | 72 x 1024^2 | 52.47 |
| D=1 C=1952 | 1 | 1952 | 20 x 1952^2 | 55.78 |

One wide look-ahead layer gets within 3.3 PPL of six narrow layers at the same FLOPs. Remarkable given D=1 has only a single transformer layer -- the correction mechanism and wider representation compensate for 5 missing layers of depth.

## Finding 6: Fine-tuning any roformer into look-ahead works

Convert pretrained roformer to look-ahead (zero-init corr_ffn), fine-tune at K=2-5:

| Conversion | Baseline | Fine-tuned | Gain | FT iters |
|-----------|----------|-----------|------|----------|
| N=24 -> D=24 | 29.42 | 28.99 | -0.43 | 18K |
| N=12 -> D=12 | 33.41 | 32.21 | -1.20 | 50K |

The gain is larger at smaller N. Fine-tuning from a better roformer adapts faster (the "fresh" conversion path). But continued fine-tuning eventually catches up.

### Fresh vs continued fine-tune

| Model | D fresh PPL | D cont PPL | Fresh total tokens | Cont total tokens |
|-------|------------|------------|-------------------|------------------|
| D=1 | 83.87 | 80.37 | 1,636M | 1,124M |
| D=2 | 67.64 | 66.14 | 1,636M | 1,124M |
| D=3 | 59.78 | 59.41 | 1,636M | 1,124M |
| D=6 | 50.96 | 52.25 | 1,636M | 1,124M |

Continued fine-tuning (more FT tokens from weaker roformer) eventually beats fresh conversion (fewer FT tokens from stronger roformer) at D=1,2,3. At D=6, fresh wins because the better starting point matters more when recovery is slow.

## Finding 7: Initial disruption decreases with depth

When converting roformer to look-ahead and fine-tuning, the initial PPL spike:

| Scale | Disruption | Recovery iters |
|-------|-----------|---------------|
| D=1 | +7 PPL | ~400 |
| D=2 | +7 PPL | ~4,300 |
| D=3 | +6 PPL | ~40,000 |
| D=6 | +3 PPL | ~32,000 |
| D=12 | +0.11 PPL | immediate |
| D=24 | ~0 PPL | immediate |

Larger D = less disruption. The correction mechanism perturbs a smaller fraction of the total computation when there are more layers.

## Running experiments

### GPU 0: block_head ablation (no corr_ffn)
Converting N=24 to block_head D=24 (iteration only, no corr_ffn) and fine-tuning. Tests whether the corr_ffn is essential or if iteration alone helps.

### GPU 1: N=6 C=2048 -> D=6 C=2048
FLOP-matched to N=24 C=1024 (72 x 2048^2 = 288 x 1024^2). If D=6 C=2048 beats N=24 C=1024, the story is: fewer, wider layers with corrections beat many narrow layers at the same FLOPs.

## Key open questions

1. **Is corr_ffn essential?** The block_head ablation (running) will answer this. If iteration alone helps, the story changes.

2. **Can D=6 C=2048 beat N=24 C=1024?** Running on GPU 1. This would be the strongest width-vs-depth result.

3. **How does D=1 scale with C?** D=1 C=1952 got 55.78 vs N=6 C=1024 at 52.47. Wider C might close the gap further.

4. **Does the relative advantage keep growing with C?** The 9.6% -> 11.3% -> 16.5% trend at D=2 suggests yes, but needs more data points.

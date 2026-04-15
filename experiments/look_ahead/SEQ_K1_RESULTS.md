# Sequential K=1 vs Parallel K=N Results

## Key Finding

Sequential K=1 matches parallel K=10 at inference. This confirms the core property of the architecture: at sequential inference, each position sees fully converged corrections from all predecessors, equivalent to running all K iterations in parallel.

At block_size=1024, sequential K=1 (29.43) actually slightly beats parallel K=5 (29.51) and exactly matches parallel K=10 (29.43).

## What Sequential K=1 Means

During training, K parallel iterations are run over all positions simultaneously. At sequential inference (K=1), each position is processed one at a time — position t sees the final output of positions 0 through t-1. This means position t gets fully converged corrections from all predecessors, not the partial corrections from limited iterations.

**Parallel K=1 is never valid** — it feeds raw token embeddings with no corrections. Parallel K=1 PPL is 70-84 at these scales, far worse than the actual model quality.

## D=1 C=2048 Results Across Block Sizes

All models: block_head_corr_ffn_add, D=1, C=2048, lr=2e-4, OWT data.

### block_size=256, batch=64

| Iters | Val PPL (K=5) | Seq K=1 | Gap | K=1 par | K=2 | K=3 | K=5 | K=10 |
|-------|--------------|---------|-----|---------|-----|-----|-----|------|
| 100K  | 39.95        | 40.03   | +0.08 | 84.44 | 43.77 | 40.52 | 39.95 | 40.02 |
| 200K  | 35.33        | 35.42   | +0.09 | 75.51 | 38.87 | 35.85 | 35.33 | 35.41 |
| 300K  | 33.58        | 33.67   | +0.09 | 72.42 | 37.11 | 34.12 | 33.58 | 33.66 |
| 400K  | 32.70        | 32.80   | +0.10 | 70.80 | 36.21 | 33.23 | 32.70 | 32.79 |

Seq K=1 consistently ~0.09 above K=5. Seq K=1 ≈ K=10.

### block_size=512, batch=64

| Iters | Val PPL (K=5) | Seq K=1 | Gap | K=1 par | K=2 | K=3 | K=5 | K=10 |
|-------|--------------|---------|-----|---------|-----|-----|-----|------|
| 100K  | 34.35        | 34.43   | +0.08 | 81.02 | 38.25 | 35.01 | 34.35 | 34.41 |
| 200K  | 30.61        | 30.69   | +0.08 | 73.22 | 34.33 | 31.24 | 30.61 | 30.68 |

Seq K=1 consistently ~0.08 above K=5. Seq K=1 ≈ K=10.

### block_size=1024, batch=32

| Iters | Val PPL (K=5) | Seq K=1 | Gap | K=1 par | K=2 | K=3 | K=5 | K=10 |
|-------|--------------|---------|-----|---------|-----|-----|-----|------|
| 200K  | 29.51        | **29.43** | **-0.08** | 84.13 | 34.42 | 30.39 | 29.51 | 29.43 |

**Seq K=1 beats K=5** (29.43 vs 29.51) and exactly matches K=10 (29.43). At longer block_size, the sequential inference mode is even more effective — the correction chain has more positions to propagate through, exceeding what K=5 parallel iterations achieve.

## Other Configurations

### D=5 C=1024, block_size=256, batch=64

| Iters | Val PPL (K=5) | Seq K=1 | K=1 par | K=2 | K=3 | K=5 | K=10 |
|-------|--------------|---------|---------|-----|-----|-----|------|
| 100K  | 38.60        | 38.62   | 58.17   | 40.18 | 38.80 | 38.60 | 38.61 |

### D=8 C=768, block_size=256, batch=64

| Iters | Val PPL (K=5) | Seq K=1 | K=1 par | K=2 | K=3 | K=5 | K=10 |
|-------|--------------|---------|---------|-----|-----|-----|------|
| 100K  | 39.10        | 39.10   | 51.60   | 40.12 | 39.22 | 39.10 | 39.10 |

Seq K=1 = K=5 = K=10 at D=8. Higher D means faster convergence — fewer iterations needed.

### D=12 C=1024, block_size=256, batch=32

| Iters | Val PPL (K=5) | Seq K=1 | K=1 par | K=2 | K=3 | K=5 | K=10 |
|-------|--------------|---------|---------|-----|-----|-----|------|
| 200K  | 32.29        | 32.29   | 38.33   | 32.58 | 32.30 | 32.29 | 32.29 |

Seq K=1 = K=5 = K=10 exactly.

### D=23 C=1024, block_size=256, batch=32

| Iters | Val PPL (K=5) | Seq K=1 | K=1 par | K=2 | K=3 | K=5 | K=10 |
|-------|--------------|---------|---------|-----|-----|-----|------|
| 200K  | 28.88        | 28.88   | 32.80   | 29.03 | 28.89 | 28.88 | 28.88 |

Seq K=1 = K=5 = K=10 exactly.

## Pattern

| D | Seq K=1 vs K=5 gap |
|---|-------------------|
| 1 (bs256) | +0.08 to +0.10 |
| 1 (bs512) | +0.08 |
| 1 (bs1024) | **-0.08** (seq beats K=5) |
| 5 | +0.02 |
| 8 | 0.00 |
| 12 | 0.00 |
| 23 | 0.00 |

At higher D, convergence is faster so seq K=1 matches K=5 exactly. At D=1, there's a small gap at smaller block_size that disappears (and reverses) at block_size=1024.

## Parallel K=1 is Never Valid

Parallel K=1 feeds raw token embeddings — the block never sees corrections. It's consistently 2-2.5x worse than the actual model:

| Config | Val PPL | Parallel K=1 | Ratio |
|--------|---------|-------------|-------|
| D=1 bs256 100K | 39.95 | 84.44 | 2.11x |
| D=1 bs512 200K | 30.61 | 73.22 | 2.39x |
| D=1 bs1024 200K | 29.51 | 84.13 | 2.85x |
| D=8 bs256 100K | 39.10 | 51.60 | 1.32x |
| D=23 bs256 200K | 28.88 | 32.80 | 1.14x |

The ratio shrinks with higher D because there are more blocks with separate weights that can do useful work even without corrections. At D=1, the single block is completely dependent on the correction chain.

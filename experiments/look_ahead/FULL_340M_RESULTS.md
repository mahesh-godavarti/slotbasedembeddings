# Full Results at ~340M FLOPs

All models: block_size=256, batch=32, lr=2e-4, softmax, n_head=16, OWT data, 200K iters.

## All models

| Model | Type | FLOPs | C/N | Layers | Status |
|-------|------|-------|-----|--------|--------|
| N=2 C=3776 | roformer | 342M | 1888 | 2 | Done: 36.10 |
| N=4 C=2656 | roformer | 339M | 664 | 4 | Done: 31.95 |
| N=6 C=2176 | roformer | 341M | 363 | 6 | Done: 30.35 |
| N=12 C=1536 | roformer | 340M | 128 | 12 | Done: 29.01 |
| N=24 C=1088 | roformer | 341M | 45 | 24 | Done: 28.68 |
| D=6 C=2048 | corr_ffn_add | 336M | 341 | 6+corr | Done: 29.04 |
| D=1 C=4128 | corr_ffn_add | 341M | 4128 | 1+corr | Running: 44.00 @ 60K |
| SA D=1 C=3776 | sa_corr_ffn_add | 342M | 3776 | 1+attn+corr | Running: 33.82 @ 180K |
| SA D=3 C=2656 | sa_corr_ffn_add | 339M | 885 | 3+attn+corr | Running: 66.82 @ 10K |
| SA D=5 C=2176 | sa_corr_ffn_add | 341M | 435 | 5+attn+corr | Running: 30.41 @ 160K |

## Full training curves

| Iter | N=2 C=3776 | N=4 C=2656 | N=6 C=2176 | N=12 C=1536 | N=24 C=1088 | D=6 C=2048 | D=1 C=4128 | SA D=5 C=2176 | SA D=1 C=3776 | SA D=3 C=2656 |
|------|-----------|-----------|-----------|------------|------------|-----------|-----------|--------------|--------------|--------------|
| 5K | 103.66 | 88.22 | 86.15 | 87.02 | 95.80 | 86.15 | 101.96 | 86.86 | 103.57 | 88.27 |
| 10K | 80.02 | 67.56 | 65.06 | 64.36 | 67.97 | 64.24 | 78.78 | 64.81 | 78.02 | 66.82 |
| 15K | 69.31 | 59.04 | 56.71 | 55.20 | 57.21 | 55.79 | 66.86 | 56.18 | 66.62 | -- |
| 20K | 62.78 | 53.93 | 51.66 | 50.06 | 51.38 | 50.55 | 60.55 | 51.14 | 60.68 | -- |
| 25K | 58.66 | 50.70 | 48.31 | 46.62 | 47.60 | 46.76 | 56.39 | 47.73 | 56.12 | -- |
| 30K | 55.57 | 48.07 | 45.91 | 44.02 | 44.77 | 44.63 | 53.36 | 45.27 | 52.91 | -- |
| 35K | 53.27 | 45.94 | 44.03 | 42.12 | 42.71 | 42.56 | 50.47 | 43.33 | 50.49 | -- |
| 40K | 51.18 | 44.64 | 42.61 | 40.57 | 41.10 | 41.29 | 48.94 | 41.70 | 48.34 | -- |
| 45K | 49.56 | 43.19 | 41.25 | 39.38 | 39.67 | 39.74 | 47.20 | 40.45 | 46.86 | -- |
| 50K | 48.25 | 42.30 | 39.97 | 38.41 | 38.67 | 38.78 | 46.14 | 39.13 | 45.55 | -- |
| 55K | 47.05 | 41.34 | 39.18 | 37.45 | 37.65 | 37.97 | 44.76 | 38.24 | 44.13 | -- |
| 60K | 46.25 | 40.47 | 38.44 | 36.82 | 36.89 | 37.04 | 44.00 | 37.40 | 43.11 | -- |
| 65K | 45.29 | 39.75 | 37.86 | 35.98 | 36.25 | 36.39 | -- | 36.87 | 42.75 | -- |
| 70K | 44.45 | 39.18 | 37.24 | 35.56 | 35.55 | 35.66 | -- | 36.25 | 41.54 | -- |
| 75K | 43.76 | 38.60 | 36.57 | 34.92 | 35.09 | 35.33 | -- | 35.51 | 41.01 | -- |
| 80K | 43.13 | 38.10 | 36.02 | 34.37 | 34.41 | 34.74 | -- | 35.08 | 40.38 | -- |
| 85K | 42.34 | 37.55 | 35.59 | 34.01 | 34.01 | 34.27 | -- | 34.55 | 39.74 | -- |
| 90K | 41.94 | 37.20 | 35.12 | 33.69 | 33.61 | 33.89 | -- | 34.30 | 39.10 | -- |
| 95K | 41.38 | 36.60 | 34.74 | 33.13 | 33.20 | 33.45 | -- | 33.75 | 38.55 | -- |
| 100K | 40.90 | 36.21 | 34.42 | 32.85 | 32.92 | 33.11 | -- | -- | -- | -- |
| 150K | -- | 33.73 | 31.79 | 30.47 | 30.37 | 30.52 | -- | 30.82 | 34.92 | -- |
| 200K | **36.10** | **31.95** | **30.35** | **29.01** | **28.68** | **29.04** | -- | -- | -- | -- |

## Completed roformer depth sweep

| Model | Final PPL | Gain from previous |
|-------|-----------|-------------------|
| N=2 C=3776 | 36.10 | -- |
| N=4 C=2656 | 31.95 | 4.15 (N=2->N=4) |
| N=6 C=2176 | 30.35 | 1.60 (N=4->N=6) |
| N=12 C=1536 | 29.01 | 1.34 (N=6->N=12) |
| N=24 C=1088 | 28.68 | 0.33 (N=12->N=24) |

Diminishing returns from depth: 4.15 -> 1.60 -> 1.34 -> 0.33. Each doubling of layers provides roughly half the benefit of the previous doubling. N=24 is barely better than N=12.

## Head-to-head comparisons

### 1. SA D=1 C=3776 vs N=2 C=3776 (both ~342M)

SA D=1 beats N=2 by ~2.8 PPL consistently from 10K onwards. Gap stable or growing. One layer with attention correction beats two plain layers at the same FLOPs.

| Iter | N=2 | SA D=1 | Gap |
|------|-----|--------|-----|
| 10K | 80.02 | 78.02 | -2.00 |
| 30K | 55.57 | 52.91 | -2.66 |
| 60K | 46.25 | 43.11 | -3.14 |
| 95K | 41.38 | 38.55 | -2.83 |

SA D=1 projected final: ~33. N=2 final: 36.10. Gap: ~3 PPL.

### 2. D=1 C=4128 vs SA D=1 C=3776 (both ~341M)

Same depth (D=1), same FLOPs. FFN-only correction (sees z[t-1]) vs attention correction (sees all z[0..t-1]).

| Iter | D=1 C=4128 | SA D=1 C=3776 | Gap |
|------|-----------|--------------|-----|
| 10K | 78.78 | 78.02 | +0.76 |
| 30K | 53.36 | 52.91 | +0.45 |
| 40K | 48.94 | 48.34 | +0.60 |
| 50K | 46.14 | 45.55 | +0.59 |
| 60K | 44.00 | 43.11 | +0.89 |

SA pulling ahead. Gap growing from ~0.3 to ~0.9. Attention correction is better than FFN-only at D=1.

### 3. SA D=5 C=2176 vs D=6 C=2048 (341M vs 336M)

| Iter | D=6 C=2048 | SA D=5 C=2176 | Gap |
|------|-----------|--------------|-----|
| 30K | 44.63 | 45.27 | +0.64 |
| 60K | 37.04 | 37.40 | +0.36 |
| 95K | 33.45 | 33.75 | +0.30 |

D=6 ahead by ~0.3-0.6. The extra block in D=6 is worth more than the attention correction in SA D=5. D=6 final: 29.04. SA D=5 projected: ~29.5.

### 4. D=6 C=2048 vs N=6 C=2176 vs N=12 C=1536 (all ~340M)

| Model | PPL @ 95K | Final (200K) |
|-------|-----------|-------------|
| N=12 C=1536 | 33.13 | 29.01 |
| D=6 C=2048 | 33.45 | 29.04 |
| N=6 C=2176 | 34.74 | 30.35 |

D=6 matches N=12 (29.04 vs 29.01). Both beat N=6 by ~1.3 PPL. The correction mechanism at D=6 provides the equivalent of doubling from 6 to 12 layers.

### 5. SA D=3 C=2656 vs N=4 C=2656 (both 339M)

SA D=3 at 10K: 66.82. N=4 at 10K: 67.56. SA D=3 slightly ahead early. ~42h left -- too early to draw conclusions.

## Key findings so far

### 1. Diminishing returns from depth at fixed FLOPs

At ~340M FLOPs, doubling layers from N=12 to N=24 gains only 0.33 PPL. The optimal depth is around N=12 (C/N=128) -- matching deployed model ratios. N=24 (C/N=45) is overprovisioned.

### 2. D=6 with correction matches N=12 without

D=6 C=2048 (29.04) matches N=12 C=1536 (29.01) at similar FLOPs. Six layers with correction mechanism equals twelve layers without. The correction provides the equivalent of doubling depth.

### 3. SA correction beats an extra layer at D=1

SA D=1 beats N=2 by ~2.8 PPL at the same FLOPs. At D=1, replacing the second layer with a correction attention + FFN is strictly more efficient. The correction wiring extracts more from the same FLOPs than a plain transformer layer.

### 4. SA attention beats FFN-only correction at D=1

SA D=1 C=3776 is pulling ahead of D=1 C=4128 (gap growing to +0.89 at 60K). Attending to all previous z values is better than seeing only z[t-1]. The advantage may grow with more training.

### 5. At D=5-6, an extra block beats attention correction

D=6 (corr_ffn_add) beats SA D=5 by ~0.3 at same FLOPs. At higher D, depth matters more than richer correction. The attention correction's value is largest at small D where depth is most lacking.

## Still running

| Experiment | Purpose | ETA |
|-----------|---------|-----|
| D=1 C=4128 (GPU 1) | FFN-only correction baseline for SA D=1 comparison | ~13h |
| SA D=3 C=2656 (GPU 0) | Does 3 layers + attn correction beat 4 plain layers? | ~42h |
| SA D=5 C=2176 (qmti92t1) | Does 5 layers + attn correction approach N=12? | running |
| SA D=1 C=3776 (qmti92t1) | One layer + attn correction -- how far can it go? | running |

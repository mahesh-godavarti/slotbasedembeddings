# FLOP-Matched Comparison at ~340M FLOPs

All models: block_size=256, batch=32, lr=2e-4, softmax, n_head=16, OWT data, 200K iters.

## Final results (completed)

| Model | FLOPs | C/N | Final PPL (200K) |
|-------|-------|-----|-----------------|
| N=24 C=1088 | 341M | 45 | **28.68** |
| N=12 C=1536 | 340M | 128 | **29.01** |
| D=6 C=2048 | 336M | 341 | **29.04** |
| N=6 C=2176 | 341M | 363 | **30.35** |

## Running

| Model | FLOPs | C/N | Latest PPL | ETA |
|-------|-------|-----|-----------|-----|
| N=4 C=2656 | 339M | 664 | 38.10 @ 80K | ~8h |
| N=2 C=3776 | 342M | 1888 | 38.64 @ 135K | ~3.5h |
| SA D=5 C=2176 | 341M | 435 | 33.75 @ 95K | qmti92t1 |

## Full training curves

| Iter | N=24 C=1088 | N=12 C=1536 | N=6 C=2176 | D=6 C=2048 | SA D=5 C=2176 |
|------|------------|------------|-----------|-----------|--------------|
| 5K | 95.80 | 87.02 | 86.15 | 86.15 | 86.86 |
| 10K | 67.97 | 64.36 | 65.06 | 64.24 | 64.81 |
| 15K | 57.21 | 55.20 | 56.71 | 55.79 | 56.18 |
| 20K | 51.38 | 50.06 | 51.66 | 50.55 | 51.14 |
| 25K | 47.60 | 46.62 | 48.31 | 46.76 | 47.73 |
| 30K | 44.77 | 44.02 | 45.91 | 44.63 | 45.27 |
| 35K | 42.71 | 42.12 | 44.03 | 42.56 | 43.33 |
| 40K | 41.10 | 40.57 | 42.61 | 41.29 | 41.70 |
| 45K | 39.67 | 39.38 | 41.25 | 39.74 | 40.45 |
| 50K | 38.67 | 38.41 | 39.97 | 38.78 | 39.13 |
| 55K | 37.65 | 37.45 | 39.18 | 37.97 | 38.24 |
| 60K | 36.89 | 36.82 | 38.44 | 37.04 | 37.40 |
| 65K | 36.25 | 35.98 | 37.86 | 36.39 | 36.87 |
| 70K | 35.55 | 35.56 | 37.24 | 35.66 | 36.25 |
| 75K | 35.09 | 34.92 | 36.57 | 35.33 | 35.51 |
| 80K | 34.41 | 34.37 | 36.02 | 34.74 | 35.08 |
| 85K | 34.01 | 34.01 | 35.59 | 34.27 | 34.55 |
| 90K | 33.61 | 33.69 | 35.12 | 33.89 | 34.30 |
| 95K | 33.20 | 33.13 | 34.74 | 33.45 | 33.75 |
| 100K | 32.92 | 32.85 | 34.42 | 33.11 | -- |
| 105K | 32.57 | 32.62 | 34.11 | 32.70 | -- |
| 110K | 32.34 | 32.22 | 33.83 | 32.40 | -- |
| 115K | 32.05 | 32.07 | 33.60 | 32.20 | -- |
| 120K | 31.69 | 31.80 | 33.28 | 31.97 | -- |
| 125K | 31.55 | 31.55 | 33.08 | 31.57 | -- |
| 130K | 31.22 | 31.28 | 32.77 | 31.28 | -- |
| 135K | 31.01 | 31.09 | 32.48 | 31.11 | -- |
| 140K | 30.73 | 30.80 | 32.30 | 31.07 | -- |
| 145K | 30.64 | 30.74 | 32.05 | 30.76 | -- |
| 150K | 30.37 | 30.47 | 31.79 | 30.52 | -- |
| 155K | 30.19 | 30.24 | 31.50 | 30.40 | -- |
| 160K | 29.98 | 30.08 | 31.35 | 30.21 | -- |
| 165K | 29.77 | 29.85 | 31.11 | 30.13 | -- |
| 170K | 29.57 | 29.73 | 30.90 | 29.98 | -- |
| 175K | 29.52 | 29.58 | 30.72 | 29.74 | -- |
| 180K | 29.33 | 29.46 | 30.62 | 29.58 | -- |
| 185K | 29.17 | 29.28 | 30.51 | 29.48 | -- |
| 190K | 29.01 | 29.19 | 30.51 | 29.34 | -- |
| 195K | 28.95 | 28.99 | 30.45 | 29.15 | -- |
| 200K | **28.68** | **29.01** | **30.35** | **29.04** | -- |

## Observations

### 1. Depth ordering at 200K

N=24 (28.68) > N=12 (29.01) > D=6 (29.04) > N=6 (30.35)

N=24 wins at 200K by 0.33 over N=12. But N=12 was ahead through 95K -- depth advantage only shows in the final 100K iters.

### 2. N=12 vs N=24: diminishing returns from depth

N=6 to N=12: 30.35 -> 29.01 = 1.34 PPL gain (doubling layers)
N=12 to N=24: 29.01 -> 28.68 = 0.33 PPL gain (doubling again)

The second doubling provides 4x less benefit. 24 layers is overprovisioned at this FLOP budget.

### 3. D=6 C=2048 matches N=12 C=1536

D=6 (29.04) and N=12 (29.01) are within 0.03 PPL. Six wide layers with correction match twelve medium layers. D=6 has 1% fewer FLOPs (336M vs 340M).

### 4. Correction mechanism value

D=6 C=2048 (29.04) vs N=6 C=2176 (30.35) = 1.31 PPL improvement from correction at similar depth and FLOPs. This is the isolated value of the correction mechanism.

### 5. C/N ratio and performance

| C/N | Model | PPL |
|-----|-------|-----|
| 45 | N=24 C=1088 | 28.68 |
| 128 | N=12 C=1536 | 29.01 |
| 341 | D=6 C=2048 | 29.04 |
| 363 | N=6 C=2176 | 30.35 |

The optimal C/N is around 45-128 for standard roformers. The correction mechanism allows C/N=341 to match C/N=128 performance.

### 6. SA D=5 C=2176 tracking

At 95K: SA D=5 (33.75) is between N=6 (34.74) and N=12 (33.13). The attention-based correction helps over plain N=6 but not enough to match N=12 at this point. Still running.

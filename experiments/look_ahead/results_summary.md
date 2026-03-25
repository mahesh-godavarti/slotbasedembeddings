# Split-Block Look-Ahead: Experiment Results

## C=768 Experiments (100K iters, block_size=256, batch=64, vocab=16000, L40S)

### Roformer N=6 C=768 baseline (72C², 67.1M params)

| Iter | roformer N=6 C=768 |
|------|--------------------|
| 5K   | 30.50              |
| 10K  | 25.89              |
| 15K  | 23.69              |
| 20K  | 22.46              |
| 25K  | 21.63              |
| 30K  | 20.98              |
| 35K  | 20.49              |
| 40K  | 20.09              |
| 45K  | 19.80              |
| 50K  | 19.51              |
| 55K  | 19.26              |
| 60K  | 19.09              |
| 65K  | 18.88              |
| 70K  | 18.70              |
| 75K  | 18.55              |
| 80K  | 18.37              |
| 85K  | 18.32              |
| 90K  | 18.17              |
| 95K  | 18.05              |
| 100K | **17.95**          |

**Roformer N=6 C=768 final: 17.95 PPL.** This is the baseline to beat at C=768.
For reference: C=446 N=6 was 22.06 — C=768 is 4.11 PPL better (larger C helps significantly).

### corr_ffn_concat D=5 C=768 vs roformer N=6 C=768 (both 72C²)

| Iter | concat D=5 C=768 | roformer N=6 C=768 | Gap |
|------|------------------|--------------------|----|
| 5K   | 28.88            | 30.50              | -1.62 |
| 10K  | 24.38            | 25.89              | -1.51 |
| 15K  | 22.26            | 23.69              | -1.43 |
| 20K  | 21.05            | 22.46              | -1.41 |
| 25K  | 20.21            | 21.63              | -1.42 |
| 30K  | 19.65            | 20.98              | -1.33 |
| 35K  | 19.21            | 20.49              | -1.28 |
| 40K  | 18.76            | 20.09              | -1.33 |
| 45K  | 18.50            | 19.80              | -1.30 |
| 50K  | 18.18            | 19.51              | -1.33 |
| 55K  | 17.92            | 19.26              | -1.34 |
| 60K  | 17.76            | 19.09              | -1.33 |
| 65K  | 17.56            | 18.88              | -1.32 |
| 70K  | 17.38            | 18.70              | -1.32 |
| 80K  | 17.12            | 18.37              | -1.25 |
| 85K  | 17.02            | 18.32              | -1.30 |
| 90K  | 16.92            | 18.17              | -1.25 |
| 95K  | 16.75            | 18.05              | -1.30 |
| 100K | 16.71            | 17.95              | -1.24 |

**Final: D=5 concat 16.69 vs roformer N=6 17.95 — gap of 1.26 PPL (7.0%) at matched FLOPs (72C²).** Already surpassed roformer N=6's final 100K result (17.95) at just 55K iters.
Diagnostics: Seq K=1 = 16.87. Depth: K=1→23.35, K=2→17.32, K=3→16.77, K=5→16.69, K=10→16.70. L ≈ 0.71.

At C=446, the gap at 40K was 1.12. At C=768, **1.33** — the advantage grows with C.

### corr_ffn_add D=3 C=768 (44C², 39% fewer FLOPs than roformer N=6) — in progress

| Iter | add D=3 (44C²) | roformer N=6 (72C²) | Gap |
|------|----------------|---------------------|-----|
| 5K   | 31.64          | 30.50               | +1.14 |
| 10K  | 27.03          | 25.89               | +1.14 |
| 15K  | 24.93          | 23.69               | +1.24 |
| 20K  | 23.54          | 22.46               | +1.08 |
| 25K  | 22.54          | 21.63               | +0.91 |
| 30K  | 21.97          | 20.98               | +0.99 |
| 35K  | 21.38          | 20.49               | +0.89 |
| 40K  | 20.91          | 20.09               | +0.82 |
| 45K  | 20.58          | 19.80               | +0.78 |
| 50K  | 20.25          | 19.51               | +0.74 |
| 55K  | 20.06          | 19.26               | +0.80 |
| 60K  | 19.84          | 19.09               | +0.75 |
| 65K  | 19.56          | 18.88               | +0.68 |
| 70K  | 19.43          | 18.70               | +0.73 |
| 75K  | 19.22          | 18.55               | +0.67 |
| 80K  | 19.12          | 18.37               | +0.75 |
| 85K  | 18.95          | 18.32               | +0.63 |
| 90K  | 18.84          | 18.17               | +0.67 |
| 95K  | 18.77          | 18.05               | +0.72 |
| 100K | **18.67**      | **17.95**           | **+0.72** |

**Final: D=3 add 18.66 vs roformer N=6 17.95 — gap of 0.71 PPL with 39% fewer FLOPs (44C² vs 72C²).** At C=446 the gap was 1.73 (23.79 vs 22.06); at C=768 it's 0.71 — scaling cuts the gap by more than half.
Diagnostics: Seq K=1 = 18.96. Depth: K=1→29.72, K=2→19.80, K=3→18.82, K=5→18.66, K=10→18.67. L ≈ 0.71.

#### FLOP-matched comparison: D=3 add (44C²) vs roformer N=4 (48C²) at C=768

| Iter | add D=3 (44C²) | roformer N=4 (48C²) | Gap |
|------|----------------|---------------------|-----|
| 5K   | 31.64          | 33.12               | -1.48 |
| 10K  | 27.03          | 28.51               | -1.48 |
| 100K | **18.66**      | **20.05**           | **-1.39** |

**D=3 add beats roformer N=4 by 1.39 PPL with 8% fewer FLOPs (44C² vs 48C²).** At C=446 the gap was 1.06 — at C=768 it's 1.39, growing with C.

### C=768 FLOP-matched summary

| Model | FLOPs | Params | Final PPL | vs FLOP-matched roformer |
|-------|-------|--------|-----------|--------------------------|
| corr_ffn_concat D=5 | 72C² | 67.1M | **16.69** | -1.26 vs N=6 (17.95) |
| corr_ffn_add D=3 | 44C² | 50.6M | **18.66** | -1.39 vs N=4 (20.05), 8% fewer FLOPs |
| corr_ffn_add D=5 | 68C² | — | (running, 25% done) | |
| roformer N=6 | 72C² | 67.1M | 17.95 | baseline |
| roformer N=4 | 48C² | 52.9M | 20.05 | baseline |

---

## Key Finding: corr_ffn_add is the best D>1 variant

### Evidence across all FLOP budgets (C=446, K=5, 100K iters)

| Model | FLOPs | Final PPL | Seq K=1 | L | vs roformer |
|-------|-------|-----------|---------|---|-------------|
| D=2 add | 32C² | 26.09 | 26.48 | 0.54 | beats N=3 (27.19, 36C²) by **1.10 PPL**, 11% fewer FLOPs |
| D=3 add | 44C² | 23.79 | 24.12 | ~0.55 | beats N=4 (24.85, 48C²) by **1.06 PPL**, 8% fewer FLOPs |
| D=4 add | 56C² | 20.82 (200K) | 21.14 | ~0.67 | (no 56C² roformer baseline; see note) |
| D=5 add | 68C² | **21.17** | 21.50 | ~0.70 | beats N=6 (22.06, 72C²) by **0.89 PPL**, 6% fewer FLOPs |
| D=6 add | 80C² | **20.40** | 20.68 | ~0.59 | beats rhf N=6 (21.44, 80C²) by **1.04 PPL** at FLOP parity |

### Why corr_ffn_add wins

1. **Best convergence (L ≈ 0.5).** K=5 and K=10 give identical PPL. K=3 already nearly converged (23.94 vs 23.79 at D=3). Sequential K=1 inference reliable.

2. **Zero overhead.** Same 20C² FLOPs and same params as token-blind corr_ffn. The addition `ln(shift(z) + tok_emb)` costs nothing.

3. **Catches deeper roformers over training.** D=3 add (44C²) vs roformer N=5 (60C²): gap 0.56 at 100K → **crossover at 155K** (22.71 vs 22.75). D=3 add surpassed roformer N=5's 100K result using 36% fewer FLOPs. Shared weights keep improving while roformer's separate layers plateau.

4. **Consistent across FLOP budgets.** Beats the FLOP-matched or stronger roformer at every budget tested.

---

## C=446 Master Comparison (all models, 100K iters, block_size=256, vocab=16000)

### 36C² budget (21.47M params, 36C² FLOPs/token)

Param and FLOP matched to roformer N=3.

| Model | Final PPL | Seq K=1 | Notes |
|-------|-----------|---------|-------|
| roformer N=3 | 27.19 | — | baseline |
| block_head D=3 K=5 k_min=2 | 27.32 | 28.46 | weight sharing alone ≈ roformer |
| stacked_block_head N=3 K=5 | **28.95** | 31.39 | L=1.095, K=10 worse (31.83) |

**Conclusion:** Weight sharing alone (block_head) matches roformer but doesn't beat it.

### 36C² budget — D=2 corr_ffn_concat (21.46M params, 36C² FLOPs/token)

Param AND FLOP matched to roformer N=3. The concat FFN (12C²) replaces the 3rd roformer layer.

| Model | Final PPL | Seq K=1 | Params | FLOPs |
|-------|-----------|---------|--------|-------|
| D=2 corr_ffn_concat C=446 K=5 | **25.48** | 25.82 | 21.46M | 36C² |
| roformer N=3 | 27.19 | — | 21.47M | 36C² |

### 32C² budget (20.67M params, 32C² FLOPs/token)

| Model | Final PPL | Seq K=1 | Notes |
|-------|-----------|---------|-------|
| D=2 corr_ffn C=446 K=5 | 26.68 | 26.72 | beats roformer N=3 (36C²) with 11% fewer FLOPs |
| D=2 corr_ffn_add C=446 K=5 | **26.09** | 26.48 | beats roformer N=3 by 1.10 PPL, L=0.54 |

#### D=2 variants vs roformer N=3: training curves

| Iter | D=2 corr_ffn (32C²) | D=2 corr_ffn_add (32C²) | D=2 corr_ffn_concat (36C²) | roformer N=3 (36C²) |
|------|---------------------|-------------------------|----------------------------|---------------------|
| 5K   | 43.03               | 42.50                   | 41.05                      | 43.00               |
| 10K  | 36.88               | 36.35                   | 35.44                      | 37.00               |
| 15K  | 34.19               | 33.86                   | 33.04                      | 34.41               |
| 20K  | 32.63               | 32.19                   | 31.46                      | 32.89               |
| 25K  | 31.53               | 31.11                   | 30.43                      | 31.88               |
| 30K  | 30.85               | 30.43                   | 29.62                      | 31.08               |
| 35K  | 30.16               | 29.75                   | 29.01                      | 30.42               |
| 40K  | 29.61               | 29.18                   | 28.51                      | 29.96               |
| 45K  | 29.27               | 28.68                   | 28.04                      | 29.50               |
| 50K  | 28.88               | 28.42                   | 27.63                      | 29.19               |
| 55K  | 28.58               | 28.08                   | 27.27                      | 28.83               |
| 60K  | 28.22               | 27.74                   | 27.15                      | 28.55               |
| 65K  | 27.87               | 27.41                   | 26.82                      | 28.37               |
| 70K  | 27.59               | 27.12                   | 26.60                      | 28.20               |
| 75K  | 27.52               | 26.94                   | 26.35                      | 27.89               |
| 80K  | 27.34               | 26.72                   | 26.14                      | 27.75               |
| 85K  | 27.17               | 26.53                   | 26.06                      | 27.56               |
| 90K  | 26.91               | 26.34                   | 25.78                      | 27.42               |
| 95K  | 26.89               | 26.25                   | 25.76                      | 27.28               |
| 100K | **26.68**           | **26.09**               | **25.48**                  | **27.19**           |

D=2 corr_ffn (32C²) beats roformer N=3 (36C²) by 0.51 PPL with 11% fewer FLOPs.
D=2 corr_ffn_concat (36C²) beats roformer N=3 (36C²) by **1.71 PPL** at same FLOPs. Best 36C² model.

D=2 corr_ffn diagnostics: seq K=1 = 26.72, L = 0.74. Depth: K=1→41.47, K=3→26.94, K=5→26.68, K=10→26.71.
D=2 corr_ffn_add diagnostics: seq K=1 = 26.48, L = 0.54. Better convergence than corr_ffn.
D=2 corr_ffn_concat diagnostics: seq K=1 = 25.82, L = 0.54. Depth: K=1→43.36, K=3→25.73, K=5→25.49, K=10→25.49.

### 44C² budget (23.06M params, 44C² FLOPs/token)

Param and FLOP matched to roformer_head_ffn N=3 (roformer + 8C² head FFN).

| Model | Final PPL | Seq K=1 | Notes |
|-------|-----------|---------|-------|
| D=3 corr_ffn K=10 (la5) | **23.18** | 23.23 | best overall, no k_min |
| D=3 concat v2 K=5 | 23.39 | 23.73 | |
| D=3 corr_ffn_add K=5 | **23.79** | 24.12 | beats N=4 by 1.06, 0.56 behind N=5 |
| D=3 corr_ffn K=5 k_min=2 (la6) | 23.98 | 23.96 | |
| roformer_head_ffn N=3 | 25.78 | — | baseline |

**Conclusion:** All look-ahead corr_ffn variants beat roformer_head_ffn by 1.8–2.6 PPL.
The 8C² corr_ffn is what makes look-ahead win — it's paid once (shared across iterations).

#### 44C² training curve: corr_ffn_add D=3 vs roformer baselines

| Iter | corr_ffn_add D=3 (44C²) | roformer N=4 (48C²) | roformer N=5 (60C²) | rhf N=3 (44C²) |
|------|-------------------------|--------------------|--------------------|----------------|
| 5K   | 39.49                   | 40.04              | 37.96              | 40.97          |
| 10K  | 33.92                   | 34.45              | 32.65              | 35.38          |
| 15K  | 31.36                   | 31.99              | 30.27              | 32.88          |
| 20K  | 29.81                   | 30.49              | 28.78              | 31.46          |
| 25K  | 28.77                   | 29.40              | 27.79              | 30.45          |
| 30K  | 27.87                   | 28.68              | 26.99              | 29.62          |
| 35K  | 27.34                   | 28.06              | 26.41              | 29.00          |
| 40K  | 26.78                   | 27.52              | 25.83              | 28.52          |
| 45K  | 26.37                   | 27.19              | 25.44              | 28.14          |
| 50K  | 26.02                   | 26.74              | 25.08              | 27.79          |
| 55K  | 25.67                   | 26.46              | 24.78              | 27.45          |
| 60K  | 25.35                   | 26.21              | 24.54              | 27.19          |
| 65K  | 25.11                   | 25.97              | 24.32              | 26.92          |
| 70K  | 24.82                   | 25.75              | 24.09              | 26.72          |
| 75K  | 24.68                   | 25.60              | 23.93              | 26.55          |
| 80K  | 24.51                   | 25.35              | 23.79              | 26.29          |
| 85K  | 24.29                   | 25.21              | 23.59              | 26.20          |
| 90K  | 24.15                   | 25.07              | 23.50              | 25.99          |
| 95K  | 23.95                   | 24.91              | 23.32              | 25.77          |
| 100K | **23.79**               | **24.85**          | **23.23**          | **25.78**      |

D=3 add (44C²) final at 100K: **23.79 PPL**. Seq K=1 = 24.12. Depth: K=1→36.51, K=2→25.08, K=3→23.94, K=5→23.79, K=10→23.79.
Beats roformer N=4 (48C²) by **1.06 PPL** with 8% fewer FLOPs. Beats rhf N=3 (44C²) by **1.99 PPL** at FLOP parity.

#### 200K extension: D=3 add catches roformer N=5 (C=446)

| Iter | corr_ffn_add D=3 (44C²) | roformer N=5 (60C²) ref | Gap |
|------|-------------------------|------------------------|-----|
| 100K | 23.79                   | 23.23                  | +0.56 |
| 105K | 23.65                   |                        |       |
| 110K | 23.54                   |                        |       |
| 115K | 23.49                   |                        |       |
| 120K | 23.39                   |                        |       |
| 125K | 23.26                   |                        |       |
| 130K | 23.10                   |                        |       |
| 135K | 23.06                   |                        |       |
| 140K | 22.96                   |                        |       |
| 145K | 22.98                   |                        |       |
| 150K | 22.83                   |                        |       |
| 155K | 22.71                   |                        |       |
| 160K | 22.68                   |                        |       |
| 165K | 22.63                   |                        |       |
| 170K | 22.58                   |                        |       |
| 175K | 22.45                   |                        |       |
| 180K | 22.40                   |                        |       |
| 185K | 22.43                   |                        |       |
| 190K | 22.42                   |                        |       |
| 195K | 22.22                   | (running)              |       |
| 200K | **22.27**               | (running)              |       |

D=3 add 200K final: **22.27 PPL**. Roformer N=5 200K in progress (130K, 22.60). Head-to-head below.

#### 200K head-to-head: D=3 add (44C²) vs roformer N=5 (60C²)

| Iter | corr_ffn_add D=3 (44C²) | roformer N=5 (60C²) | Gap |
|------|-------------------------|---------------------|-----|
| 5K   | 39.43                   | 37.96               | +1.47 |
| 10K  | 33.91                   | 32.65               | +1.26 |
| 15K  | 31.38                   | 30.27               | +1.11 |
| 20K  | 29.76                   | 28.78               | +0.98 |
| 25K  | 28.77                   | 27.79               | +0.98 |
| 30K  | 27.87                   | 26.99               | +0.88 |
| 35K  | 27.32                   | 26.41               | +0.91 |
| 40K  | 26.70                   | 25.83               | +0.87 |
| 45K  | 26.34                   | 25.44               | +0.90 |
| 50K  | 25.94                   | 25.08               | +0.86 |
| 55K  | 25.59                   | 24.78               | +0.81 |
| 60K  | 25.26                   | 24.54               | +0.72 |
| 65K  | 25.06                   | 24.32               | +0.74 |
| 70K  | 24.76                   | 24.09               | +0.67 |
| 75K  | 24.61                   | 23.93               | +0.68 |
| 80K  | 24.46                   | 23.79               | +0.67 |
| 85K  | 24.23                   | 23.59               | +0.64 |
| 90K  | 24.19                   | 23.50               | +0.69 |
| 95K  | 23.93                   | 23.32               | +0.61 |
| 100K | 23.79                   | 23.24               | +0.55 |
| 105K | 23.65                   | 23.04               | +0.61 |
| 110K | 23.54                   | 22.99               | +0.55 |
| 115K | 23.49                   | 22.89               | +0.60 |
| 120K | 23.39                   | 22.80               | +0.59 |
| 125K | 23.26                   | 22.76               | +0.50 |
| 130K | 23.10                   | 22.60               | +0.50 |
| 135K | 23.06                   | 22.57               | +0.49 |
| 140K | 22.96                   | 22.54               | +0.42 |
| 145K | 22.98                   | 22.42               | +0.56 |
| 150K | 22.83                   | 22.34               | +0.49 |
| 155K | 22.71                   | 22.30               | +0.41 |
| 160K | 22.68                   | 22.22               | +0.46 |
| 165K | 22.63                   | 22.15               | +0.48 |
| 170K | 22.58                   | 22.11               | +0.47 |
| 175K | 22.45                   | 22.05               | +0.40 |
| 180K | 22.40                   | 22.00               | +0.40 |
| 185K | 22.43                   |                     |       |
| 190K | 22.42                   | 21.92               | +0.50 |
| 195K | 22.22                   | 21.88               | +0.34 |
| 200K | **22.27**               | **21.83**           | **+0.44** |

**Final at 200K: D=3 add 22.27 vs roformer N=5 21.83 — gap of 0.44 PPL.** D=3 add uses 36% fewer FLOPs (44C² vs 60C²). Gap narrowed from 1.47 at 5K to 0.44 at 200K but did not cross over.

### 48C² budget (roformer N=4 matched)

| Model | Final PPL | Seq K=1 | Params | Notes |
|-------|-----------|---------|--------|-------|
| D=3 concat v2 K=5 | 23.39 | 23.73 | 23.1M | also listed under 44C² (48C² with concat FFN) |
| corr_ffn_concat D=3 K=5 | **23.41** | 23.73 | — | 48C² FLOP-matched, beats N=4 by 1.44 PPL |
| corr_ffn_concat D=3 K=5 jointln | **23.44** | 23.81 | — | joint LN over 2C; L≈0.57 (better than old L≈0.74, same PPL) |
| roformer N=4 C=446 | **24.85** | — | 23.9M | |

#### corr_ffn_concat D=3 training curve (48C²)

| Iter | corr_ffn_concat D=3 (48C²) | roformer N=4 (48C²) | Gap  |
|------|----------------------------|---------------------|------|
| 5K   | 38.94                      | 40.04               | -1.10 |
| 10K  | 33.24                      | 34.45               | -1.21 |
| 15K  | 30.88                      | 31.99               | -1.11 |
| 20K  | 29.30                      | 30.49               | -1.19 |
| 25K  | 28.30                      | 29.40               | -1.10 |
| 30K  | 27.49                      | 28.68               | -1.19 |
| 35K  | 26.91                      | 28.06               | -1.15 |
| 40K  | 26.38                      | 27.52               | -1.14 |
| 45K  | 25.91                      | 27.19               | -1.28 |
| 50K  | 25.52                      | 26.74               | -1.22 |
| 55K  | 25.08                      | 26.46               | -1.38 |
| 60K  | 24.85                      | 26.21               | -1.36 |
| 65K  | 24.65                      | 25.97               | -1.32 |
| 70K  | 24.37                      | 25.75               | -1.38 |
| 75K  | 24.12                      | 25.60               | -1.48 |
| 80K  | 23.92                      | 25.35               | -1.43 |
| 85K  | 23.78                      | 25.21               | -1.43 |
| 90K  | 23.62                      | 25.07               | -1.45 |
| 95K  | 23.50                      | 24.91               | -1.41 |
| 100K | **23.41**                  | **24.85**           | **-1.44** |

D=3 concat (48C²) final: **23.41 PPL**. Seq K=1 = 23.73. L = 0.74.
Beats roformer N=4 (48C²) by **1.44 PPL** at FLOP parity. Gap widened from 1.1 early to 1.44 at 100K.
Only 0.18 PPL behind roformer N=5 (23.23, 60C²) which uses **25% more FLOPs**.

#### roformer N=4 C=446 training curve

| Iter | roformer N=4 C=446 | roformer N=5 C=446 | block_head_corr_ffn C=508 K=5 | roformer_head_ffn N=3 C=446 |
|------|--------------------|--------------------|-------------------------------|----------------------------|
| 5K   | 40.04              | 37.96              | 46.32                         | 40.97                      |
| 10K  | 34.45              | 32.65              | 40.29                         | 35.38                      |
| 15K  | 31.99              | 30.27              | 37.90                         | 32.88                      |
| 20K  | 30.49              | 28.78              | 36.23                         | 31.46                      |
| 25K  | 29.40              | 27.79              | 34.99                         | 30.45                      |
| 30K  | 28.68              | 26.99              | 34.13                         | 29.62                      |
| 35K  | 28.06              | 26.41              | 33.47                         | 29.00                      |
| 40K  | 27.52              | 25.83              | 33.15                         | 28.52                      |
| 45K  | 27.19              | 25.44              | 32.72                         | 28.14                      |
| 50K  | 26.74              | 25.08              | 32.26                         | 27.79                      |
| 55K  | 26.46              | 24.78              | 32.15                         | 27.45                      |
| 60K  | 26.21              | 24.54              | 31.66                         | 27.19                      |
| 65K  | 25.97              | 24.32              | 31.45                         | 26.92                      |
| 70K  | 25.75              | 24.09              | 31.09                         | 26.72                      |
| 75K  | 25.60              | 23.93              | 30.97                         | 26.55                      |
| 80K  | 25.35              | 23.79              | 30.80                         | 26.29                      |
| 85K  | 25.21              | 23.59              | 30.50                         | 26.20                      |
| 90K  | 25.07              | 23.50              | 30.48                         | 25.99                      |
| 95K  | 24.91              | 23.32              | 30.30                         | 25.77                      |
| 100K | **24.85**          | **23.23**          | **30.23**                     | **25.78**                  |

Each extra roformer layer: N=3→N=4 = 2.34 PPL (27.19→24.85), N=4→N=5 = 1.62 PPL (24.85→23.23).
roformer_head_ffn N=3 (25.78, 44C²) beats roformer N=4 (24.85, 48C²) — the head FFN is more efficient than a 4th layer.

### 60C² budget (26.25M params, 60C² FLOPs/token)

Param and FLOP matched to roformer N=5.

| Model | Final PPL | Seq K=1 | Notes |
|-------|-----------|---------|-------|
| stacked corr_ffn N=3 K=10 | 25.50 | — | no k_min, L≈0.005 |
| stacked corr_ffn N=3 K=5 k_min=2 | 26.02 | 26.06 | L=0.011 |
| stacked corr_ffn_add N=3 K=5 k_min=2 | **23.04** | 23.69 | L=0.011, beats roformer N=5 |
| roformer N=5 | 23.23 | — | |

#### 60C² training curve: roformer N=5 vs stacked corr_ffn variants

| Iter | roformer N=5 | stacked corr_ffn K=10 | stacked corr_ffn K=5 k_min=2 | stacked corr_ffn_add K=5 k_min=2 |
|------|--------------|-----------------------|-------------------------------|----------------------------------|
| 5K   | 37.96        | 42.35                 | 42.69                         | 38.41                            |
| 10K  | 32.65        | 36.64                 | 36.84                         | 32.98                            |
| 15K  | 30.27        | 34.00                 | 34.21                         | 30.53                            |
| 20K  | 28.78        | 32.22                 | 32.43                         | 28.93                            |
| 25K  | 27.79        | 31.13                 | 31.36                         | 27.89                            |
| 30K  | 26.99        | 30.23                 | 30.56                         | 27.20                            |
| 35K  | 26.41        | 29.32                 | 29.91                         | 26.47                            |
| 40K  | 25.83        | 28.77                 | 29.07                         | 25.90                            |
| 45K  | 25.44        | 28.34                 | 28.71                         | 25.47                            |
| 50K  | 25.08        | 27.99                 | 28.37                         | 25.09                            |
| 55K  | 24.78        | 27.55                 | 27.90                         | 24.74                            |
| 60K  | 24.54        | 27.20                 | 27.77                         | 24.55                            |
| 65K  | 24.32        | 26.94                 | 27.53                         | 24.34                            |
| 70K  | 24.09        | 26.63                 | 27.16                         | 23.92                            |
| 75K  | 23.93        | 26.36                 | 26.83                         | 23.74                            |
| 80K  | 23.79        | 26.14                 | 26.52                         | 23.52                            |
| 85K  | 23.59        | 25.93                 | 26.49                         | 23.46                            |
| 90K  | 23.50        | 25.71                 | 26.35                         | 23.29                            |
| 95K  | 23.32        | 25.50                 | 26.04                         | 23.09                            |
| 100K | 23.23        | **25.50**             | **26.02**                     | **23.04**                        |

stacked corr_ffn K=5 k_min=2 diagnostics: seq K=1 = 26.06, L = 0.011. Depth: K=1→31.36, K=2→26.34, K=3→26.02, K=5→26.02, K=10→26.06.
stacked corr_ffn_add K=5 k_min=2 diagnostics: seq K=1 = 23.69, L = 0.011. Depth: K=1→30.08, K=2→23.47, K=3→23.10, K=5→23.04, K=10→23.04.

### 68C² budget (corr_ffn_add D=5)

D=5 add: (12×5+8)C² = 68C². 6% fewer FLOPs than roformer N=6 (72C²).

| Model | Final PPL | Seq K=1 | FLOPs | Notes |
|-------|-----------|---------|-------|-------|
| corr_ffn_add D=5 K=5 | **21.17** | 21.50 | 68C² | beats N=6 (22.06) by 0.89 PPL with 6% fewer FLOPs |
| corr_ffn_concat D=5 K=5 | 20.89 | 21.17 | 72C² | 4C² more FLOPs, 0.28 PPL better |
| roformer N=6 | 22.06 | — | 72C² | |

D=5 add (68C²) beats roformer N=6 (72C²) by **0.89 PPL** with 6% fewer FLOPs. L ≈ 0.7.

### 56C² budget (corr_ffn_add D=4)

D=4 add: (12×4+8)C² = 56C². Between roformer N=4 (48C²) and N=5 (60C²).

| Model | Iters | Final PPL | Seq K=1 | FLOPs | Notes |
|-------|-------|-----------|---------|-------|-------|
| corr_ffn_add D=4 K=5 | 200K | **20.82** | 21.14 | 56C² | nearly matches D=6 at 100K (20.40, 80C²) |
| roformer N=5 | 100K | 23.23 | — | 60C² | 7% more FLOPs, 2.41 PPL worse |

D=4 add at 200K (20.82) nearly matches D=6 at 100K (20.40). More depth (D=6) wins over more iterations (D=4×200K) at similar compute, but the gap is small (0.42 PPL).

#### 56C² training curve: corr_ffn_add D=4 (200K iters, C=446)

| Iter | corr_ffn_add D=4 |
|------|------------------|
| 5K   | 37.59 |
| 10K  | 32.15 |
| 15K  | 29.76 |
| 20K  | 28.17 |
| 25K  | 27.09 |
| 30K  | 26.25 |
| 35K  | 25.55 |
| 40K  | 25.06 |
| 45K  | 24.75 |
| 50K  | 24.37 |
| 55K  | 23.99 |
| 60K  | 23.74 |
| 65K  | 23.51 |
| 70K  | 23.18 |
| 75K  | 23.00 |
| 80K  | 22.89 |
| 85K  | 22.67 |
| 90K  | 22.57 |
| 95K  | 22.42 |
| 100K | 22.26 |
| 105K | 22.19 |
| 110K | 22.07 |
| 115K | 21.94 |
| 120K | 21.89 |
| 125K | 21.76 |
| 130K | 21.63 |
| 135K | 21.60 |
| 140K | 21.53 |
| 145K | 21.42 |
| 150K | 21.36 |
| 155K | 21.29 |
| 160K | 21.19 |
| 165K | 21.17 |
| 170K | 21.06 |
| 175K | 21.06 |
| 180K | 21.00 |
| 185K | 20.93 |
| 190K | 20.92 |
| 195K | 20.85 |
| 200K | **20.82** |

### 72C² budget (roformer N=6 matched)

| Model | Final PPL | Seq K=1 | FLOPs | Notes |
|-------|-----------|---------|-------|-------|
| stacked concat v2 N=3 K=5 | 21.96 | 23.28 | 72C² | |
| corr_ffn_concat D=5 K=5 | **20.89** | 21.17 | 72C² | beats roformer N=6 by 1.17 PPL |
| roformer_head_ffn N=6 | 21.44 | — | 80C²* | *8C² more FLOPs from head FFN |
| roformer N=6 | **22.06** (100K) / **20.76** (200K) | — | 72C² | |

Stacked concat v2 (21.96) beats roformer N=6 (22.06) at same 72C² FLOPs.

#### 72C² training curves: roformer N=6 vs stacked concat v2 N=3 vs roformer_head_ffn N=6 (80C²)

| Iter | roformer N=6 | stacked concat v2 N=3 | corr_ffn_concat D=5 | rhf N=6 (80C²) |
|------|--------------|-----------------------|---------------------|----------------|
| 5K   | 36.68        | 36.60                 | 35.82               | 35.69          |
| 10K  | 31.52        | 31.38                 | 30.44               | 30.74          |
| 15K  | 29.17        | 28.79                 | 28.15               | 28.39          |
| 20K  | 27.72        | 27.51                 | 26.67               | 26.99          |
| 25K  | 26.70        | 26.54                 | 25.63               | 25.93          |
| 30K  | 25.91        | 25.75                 | 24.73               | 25.14          |
| 35K  | 25.23        | 25.14                 | 24.03               | 24.51          |
| 40K  | 24.79        | 24.68                 | 23.67               | 23.98          |
| 45K  | 24.35        | 24.25                 | 23.25               | 23.63          |
| 50K  | 23.99        | 23.85                 | 22.78               | 23.28          |
| 55K  | 23.72        | 23.56                 | 22.47               | 22.97          |
| 60K  | 23.41        | 23.27                 | 22.27               | 22.72          |
| 65K  | 23.22        | 23.11                 | 22.05               | 22.50          |
| 70K  | 23.01        | 22.96                 | 21.82               | 22.30          |
| 75K  | 22.86        | 22.70                 | 21.58               | 22.11          |
| 80K  | 22.66        | 22.52                 | 21.51               | 21.96          |
| 85K  | 22.52        | 22.33                 | 21.35               | 21.79          |
| 90K  | 22.34        | 22.15                 | 21.13               | 21.66          |
| 95K  | 22.25        | 22.09                 | 21.06               | 21.53          |
| 100K | **22.06**    | **21.96**             | **20.89**           | **21.44**      |

Concat D=5 (72C²) beats roformer N=6 (72C²) by **1.17 PPL** at same FLOPs. Even beats rhf N=6 (21.44, 80C²) with fewer FLOPs.
Diagnostics: seq K=1 = 21.17, L ≈ 0.5. Depth: K=1→28.68, K=2→21.61, K=3→20.97, K=5→20.89, K=10→20.89.

#### Roformer N=6 200K extension (C=446, in progress)

| Iter | roformer N=6 (72C²) | corr_ffn_add D=4 (56C²) | Gap (rfm−D4) | corr_ffn_add D=6 (80C²) |
|------|--------------------|-----------------------|-------------|------------------------|
| 5K   | 36.65              | 37.59                 | -0.94       | 34.96                  |
| 10K  | 31.51              | 32.15                 | -0.64       | 29.82                  |
| 15K  | 29.19              | 29.76                 | -0.57       | 27.30                  |
| 20K  | 27.77              | 28.17                 | -0.40       | 25.77                  |
| 25K  | 26.68              | 27.09                 | -0.41       | 24.77                  |
| 30K  | 25.94              | 26.25                 | -0.31       | 24.00                  |
| 35K  | 25.25              | 25.55                 | -0.30       | 23.38                  |
| 40K  | 24.83              | 25.06                 | -0.23       | 22.92                  |
| 45K  | 24.37              | 24.75                 | -0.38       | 22.54                  |
| 50K  | 24.04              | 24.37                 | -0.33       | 22.19                  |
| 55K  | 23.69              | 23.99                 | -0.30       | 21.97                  |
| 60K  | 23.44              | 23.74                 | -0.30       | 21.70                  |
| 65K  | 23.22              | 23.51                 | -0.29       | 21.48                  |
| 70K  | 23.08              | 23.18                 | -0.10       | 21.27                  |
| 75K  | 22.86              | 23.00                 | -0.14       | 21.11                  |
| 80K  | 22.66              | 22.89                 | -0.23       | 20.95                  |
| 85K  | 22.53              | 22.67                 | -0.14       | 20.77                  |
| 90K  | 22.36              | 22.57                 | -0.21       | 20.56                  |
| 95K  | 22.26              | 22.42                 | -0.16       | 20.51                  |
| 100K | 22.09              | 22.26                 | -0.17       | **20.40**              |
| 105K | 22.01              | 22.19                 | -0.18       |                        |
| 110K | 21.92              | 22.07                 | -0.15       |                        |
| 115K | 21.83              | 21.94                 | -0.11       |                        |
| 120K | 21.76              | 21.89                 | -0.13       |                        |
| 125K | 21.62              | 21.76                 | -0.14       |                        |
| 130K | 21.56              | 21.63                 | -0.07       |                        |
| 135K | 21.48              | 21.60                 | -0.12       |                        |
| 140K | 21.42              | 21.53                 | -0.11       |                        |
| 145K | 21.35              | 21.42                 | -0.07       |                        |
| 150K | 21.24              | 21.36                 | -0.12       |                        |
| 155K | 21.23              | 21.29                 | -0.06       |                        |
| 160K | 21.14              | 21.19                 | -0.05       |                        |
| 165K | 21.13              | 21.17                 | -0.04       |                        |
| 170K | 21.05              | 21.06                 | -0.01       |                        |
| 175K | 20.99              | 21.06                 | -0.07       |                        |
| 180K | 20.92              | 21.00                 | -0.08       |                        |
| 185K | 20.85              | 20.93                 | -0.08       |                        |
| 190K | 20.81              | 20.92                 | -0.11       |                        |
| 195K | 20.78              | 20.85                 | -0.07       |                        |
| 200K | **20.76**          | **20.82**             | **-0.06**   |                        |

**Final at 200K: roformer N=6 20.76 vs D=4 add 20.82 — gap of 0.06 PPL.** Essentially tied, with D=4 using 22% fewer FLOPs (56C² vs 72C²). Gap narrowed from 0.94 at 5K to 0.06 at 200K.

### 80C² budget (corr_ffn_add D=6, C=446)

D=6 add: (12×6+8)C² = 80C². No exact roformer match (N=7 would be 84C²). Compare vs D=5 concat (72C²) and rhf N=6 (80C²).

#### 80C² training curve: corr_ffn_add D=6 vs D=5 concat vs roformer baselines

| Iter | corr_ffn_add D=6 (80C²) | corr_ffn_concat D=5 (72C²) | roformer N=6 (72C²) | rhf N=6 (80C²) |
|------|------------------------|---------------------------|--------------------|--------------------|
| 5K   | 34.96                  | 35.82                     | 35.29              | 34.09              |
| 10K  | 29.82                  | 30.44                     | 29.96              | 29.04              |
| 15K  | 27.30                  | 28.15                     | 27.55              | 26.74              |
| 20K  | 25.77                  | 26.67                     | 26.09              | 25.35              |
| 25K  | 24.77                  | 25.63                     | 25.44              | 24.57              |
| 30K  | 24.00                  | 24.73                     | 24.83              | 24.00              |
| 35K  | 23.38                  | 24.03                     | 24.32              | 23.54              |
| 40K  | 22.92                  | 23.67                     | 23.81              | 23.14              |
| 45K  | 22.54                  | 23.25                     | 23.23              | 22.74              |
| 50K  | 22.19                  | 22.78                     | 22.82              | 22.37              |
| 55K  | 21.97                  | 22.47                     | 22.47              | 22.06              |
| 60K  | 21.70                  | 22.27                     | 23.41              | 22.72              |
| 65K  | 21.48                  | 22.05                     | 23.22              | 22.50              |
| 70K  | 21.27                  | 21.82                     | 23.01              | 22.30              |
| 75K  | 21.11                  | 21.58                     | 22.86              | 22.11              |
| 80K  | 20.95                  | 21.51                     | 22.66              | 21.96              |
| 85K  | 20.77                  | 21.35                     | 22.52              | 21.79              |
| 90K  | 20.56                  | 21.13                     | 22.34              | 21.66              |
| 95K  | 20.51                  | 21.06                     | 22.25              | 21.53              |
| 100K | **20.40**              | **20.89**                 | **22.06**          | **21.44**          |

D=6 add (80C²) final: **20.40 PPL**. Seq K=1 = 20.68. L ≈ 0.59.
Beats rhf N=6 (21.44, 80C²) by **1.04 PPL** at FLOP parity.
Beats D=5 concat (20.89, 72C²) by 0.49 PPL (with 11% more FLOPs).
Beats roformer N=6 (22.06, 72C²) by **1.66 PPL**.
Depth sweep: K=1→30.27, K=2→21.45, K=3→20.60, K=5→20.41, K=10→20.41.

### D=1 param-matched variants (all trailing roformer N=3 by 3-4+ PPL)

| Model | C | Inference FLOPs | Final PPL | vs roformer N=3 (27.19, 36C²) |
|-------|---|----------------|-----------|-------------------------------|
| block_head_recompute D=1 | 554 | 20C² (6.14M) | **30.63** | -3.4 PPL behind |
| block_head_corr_ffn D=1 | 508 | 20C² (5.16M) | **30.23** | ~-3 PPL behind |
| block_head_corr_ffn_concat D=1 | 490 | 24C² (5.76M) | **28.08** (seq 28.45) | -0.9 PPL behind |

**Conclusion:** D=1 cannot compete at param parity — D>1 is essential at large C.

### D scaling vs N scaling analysis (Wiki, C=446, 100K iters)

Comparing marginal PPL gains per additional 12C² of compute:

| Step (+12C²) | Roformer | corr_ffn_add | Ratio |
|--------------|----------|--------------|-------|
| 3→4 (N or D) | 2.34 (27.19→24.85) | 1.56 (23.82→22.26) | 1.50× |
| 4→5          | 1.62 (24.85→23.23) | 1.09 (22.26→21.17) | 1.49× |
| 5→6          | 1.17 (23.23→22.06) | 0.77 (21.17→20.40) | 1.52× |

Full FLOP-matched comparison:

| corr_ffn_add | FLOPs | PPL | roformer | FLOPs | PPL | Gap (add wins by) |
|--------------|-------|-----|----------|-------|-----|-------------------|
| D=3          | 44C²  | 23.82 | N=4    | 48C²  | 24.85 | 1.03 PPL (fewer FLOPs) |
| D=4          | 56C²  | 22.26 | N=5    | 60C²  | 23.23 | 0.97 PPL (fewer FLOPs) |
| D=5          | 68C²  | 21.17 | N=6    | 72C²  | 22.06 | 0.89 PPL (fewer FLOPs) |

**Roformer gains ~1.5× more PPL per additional layer than corr_ffn_add gains per additional D block.** This ratio is remarkably stable across all three steps.

**The FLOP-matched advantage is shrinking: 1.03 → 0.97 → 0.89.** corr_ffn_add wins at every depth tested, but the margin narrows as depth increases. This is because corr_ffn_add starts from a better base (the 8C² correction FFN gives a strong initial advantage) but roformer extracts more value from each additional layer of unique weights.

**Implication for deep scaling:** If the ~0.08 PPL/step narrowing continues, the advantage would erode to zero around D≈14 vs N≈15 (~176C² vs 180C²). At that depth, roformer would catch up. This suggests the shared-weight iterative mechanism is most advantageous at moderate depths (D=3–8) where the 8C² fixed overhead is amortized but hasn't yet been overcome by roformer's stronger per-layer gains. Very deep scaling may favor standard roformer.

### Key takeaways

1. **block_head D=3 (27.32) ≈ roformer N=3 (27.19).** Stacked N=3 still running.
2. **The corr_ffn is what wins.** It adds 8C² params/FLOPs but is paid only once (shared across K iterations). All corr_ffn variants beat the FLOP-matched roformer_head_ffn baseline.
3. **D>1 is essential at large C.** D=1 trails roformer N=3 by 3-6 PPL at param parity.
4. **Stacking pays the corr_ffn cost per unit** (N×8C²), making it expensive. D>1 with shared corr_ffn is more efficient.
5. **Roformer scales better per layer than corr_ffn_add per D block** (1.5× more PPL gain per +12C²). The FLOP-matched advantage narrows with depth (1.03→0.97→0.89). The sweet spot for corr_ffn_add is moderate depth (D=3–8).

---

## D=1 Param-Matched Experiments (C=446 roformer N=3 parity, block_size=256, vocab=16000)

### block_head_recompute D=1 C=554 K=5 k_min=2 (12C² = 21.47M params)

Settings: n_embed=554, n_layers=5, block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp

Training curve (still running):

| Iter | Val PPL |
|------|---------|
| 10K  | 46.82   |
| 15K  | 40.74   |
| 20K  | 38.33   |
| 25K  | 36.63   |
| 30K  | 35.51   |
| 35K  | 34.79   |
| 40K  | 34.06   |
| 45K  | 33.54   |
| 50K  | 33.11   |
| 55K  | 32.78   |
| 60K  | 32.49   |
| 65K  | 32.10   |
| 70K  | 31.85   |
| 75K  | 31.66   |
| 80K  | 31.46   |
| 85K  | 31.19   |
| 90K  | 31.08   |
| 95K  | 30.87   |
| 100K | 30.63   |

Roformer N=3 C=446 baseline: **27.19** at 100K. Final gap: **3.44 PPL behind**.

### block_head_corr_ffn_concat D=1 C=490 K=5 k_min=2 (24C² = 21.47M params)

Settings: n_embed=490, n_layers=5, block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp

Training curve (still running):

| Iter | Val PPL |
|------|---------|
| 5K   | 43.26   |
| 10K  | 37.62   |
| 15K  | 35.11   |
| 20K  | 33.83   |
| 25K  | 32.87   |
| 30K  | 31.95   |
| 35K  | 31.43   |
| 40K  | 31.04   |
| 45K  | 30.48   |
| 50K  | 30.23   |
| 55K  | 29.94   |
| 60K  | 29.63   |
| 65K  | 29.46   |
| 70K  | 29.08   |
| 75K  | 28.82   |
| 80K  | 28.69   |
| 85K  | 28.48   |
| 90K  | 28.33   |
| 95K  | 28.21   |
| 100K | 28.08   |

Diagnostics: seq K=1 = 28.45, L = 0.85. Depth sweep: K=1→60.52, K=2→30.80, K=3→28.49, K=5→28.08, K=10→28.13.

Roformer N=3 C=446 baseline: **27.19** at 100K. Final gap: **0.89 PPL behind**.
Best D=1 variant. 24C² inference FLOPs vs roformer's 36C².

### block_head_corr_ffn D=1 C=508 K=5 k_min=2 (20C² = 21.47M params)

Settings: n_embed=508, n_layers=5, block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp

Training curve (still running), side-by-side with roformer_head_ffn N=3 C=446 (44C²):

| Iter | corr_ffn D=1 C=508 | roformer_head_ffn N=3 | Gap |
|------|--------------------|-----------------------|-----|
| 45K  | 32.72              | 28.14                 | 4.58 |
| 50K  | 32.26              | 27.79                 | 4.47 |
| 55K  | 32.15              | 27.45                 | 4.70 |
| 60K  | 31.66              | 27.19                 | 4.47 |
| 65K  | 31.45              | 26.92                 | 4.53 |
| 70K  | 31.09              | 26.72                 | 4.37 |
| 75K  | 30.97              | 26.55                 | 4.42 |
| 80K  | 30.80              | 26.29                 | 4.51 |
| 85K  | 30.50              | 26.20                 | 4.30 |
| 90K  | 30.48              | 25.99                 | 4.49 |
| 95K  | 30.30              | 25.77                 | 4.53 |
| 100K | 30.23              | 25.78                 | 4.45 |

roformer_head_ffn N=3 C=446 final: **25.78**.

Depth sweep (k_min=2):

| K | PPL |
|---|-----|
| 1 | 53.21 |
| 2 | 33.14 |
| 3 | 30.72 |
| 5 | 30.23 |
| 10 | 30.35 |
| sequential | 30.39 |

Clean depth sweep, no divergence. K=3 already close to full K=5.

### stacked_block_head N=3 vs block_head D=3 (C=446, K=5, 36C²)

Both are 36C² FLOP budget, param-matched to roformer N=3. Stacked has separate weights per unit; block_head shares weights across D=3 iterations.

| Iter | stacked_block_head N=3 | block_head D=3 | Gap |
|------|------------------------|----------------|-----|
| 5K   | 46.97                  | 44.19          | 2.78 |
| 10K  | 41.14                  | 37.90          | 3.24 |
| 15K  | 37.51                  | 35.23          | 2.28 |
| 20K  | 35.88                  | 33.55          | 2.33 |
| 25K  | 34.54                  | 32.57          | 1.97 |
| 30K  | 33.57                  | 31.50          | 2.07 |
| 35K  | 32.74                  | 31.39          | 1.35 |
| 40K  | 32.23                  | 30.43          | 1.80 |
| 45K  | 31.91                  | 29.88          | 2.03 |
| 50K  | 31.42                  | 29.63          | 1.79 |
| 55K  | 31.11                  | 29.07          | 2.04 |
| 60K  | 30.62                  | 28.99          | 1.63 |
| 65K  | 30.31                  | 28.79          | 1.52 |
| 70K  | 30.05                  | 28.36          | 1.69 |
| 75K  | 29.91                  | 28.10          | 1.81 |
| 80K  | 29.66                  | 27.96          | 1.70 |
| 85K  | 29.44                  | 27.75          | 1.69 |
| 90K  | —                      | 27.55          | — |
| 95K  | —                      | 27.40          | — |
| 90K  | 29.30                  | 27.55          | 1.75 |
| 95K  | 29.14                  | 27.40          | 1.74 |
| 100K | 28.95                  | 27.32          | 1.63 |

Both have 36C² params and 36C² inference FLOPs.

Diagnostics:
- stacked_block_head: seq K=1 = 31.39 (gap 2.44), par K=1 = 41.80, K=10 = 31.83, L = 1.095 (not converging)
- block_head D=3: seq K=1 = 28.46 (gap 0.14)

---

## Experiment: Split-Block Variants vs Nocat (C=50, K=10, block_size=256, 100K iters)

Settings: n_embed=50, n_layers=10, block_size=256, batch_size=64, lr=0.0002, softmax, convergence_weight=0.1, vocab=16000, data=full wiki

### Models

| Model | Params | Description |
|---|---|---|
| block_head_ffn | ~1,657K | Standard block + extra FFN at head |
| attn_corr_ffn | 1,646,750 | Attention-only, FFN generates corrections, head sees y |
| attn_head_ffn | 1,646,750 | Attention-only, raw attn delta, FFN at head |
| roformer_look_ahead_nocat | 1,646,750 | Baseline: standard block, head sees processed_x |

### 100K Final Results

| Model | Params | Val PPL | K=1 PPL | Seq K=1 | L |
|---|---|---|---|---|---|
| block_head_ffn | ~1,657K | **84.53** | 128.47 | 84.57 | 1.20 |
| attn_corr_ffn | 1,646,750 | 91.58 | 134.95 | 91.58 | — |
| block_head | 1,646,750 | 90.29 | 130.63 | 90.39 | 0.88 |
| attn_head_ffn | 1,646,750 | 93.88 | 107.11 | 93.89 | — |
| roformer_look_ahead_nocat | 1,646,750 | 91.85 | 106.66 | 91.98 | 0.95 |

### Training curves (all 4 models)

| Iter | block_head_ffn | attn_corr_ffn | nocat  | attn_head_ffn |
|------|----------------|---------------|--------|---------------|
| 5K   | 152.88         | 164.50        | 165.33 | 167.80        |
| 10K  | 121.56         | 128.96        | 128.43 | 139.05        |
| 15K  | 110.34         | 116.31        | 115.69 | 126.01        |
| 20K  | 103.93         | 109.24        | 109.15 | 118.30        |
| 25K  | 99.59          | 104.64        | 104.93 | 112.78        |
| 30K  | 96.59          | 101.76        | 102.12 | 108.92        |
| 35K  | 94.32          | 99.67         | 100.16 | 105.99        |
| 40K  | 92.53          | 98.09         | 98.52  | 103.83        |
| 45K  | 91.17          | 96.87         | 97.31  | 101.83        |
| 50K  | 90.08          | 95.99         | 96.42  | 100.52        |
| 55K  | 88.96          | 95.13         | 95.46  | 99.25         |
| 60K  | 88.18          | 94.55         | 94.85  | 98.46         |
| 65K  | 87.47          | 93.95         | 94.28  | 97.51         |
| 70K  | 87.05          | 93.51         | 93.72  | 96.86         |
| 75K  | 86.45          | 93.04         | 93.35  | 96.21         |
| 80K  | 85.91          | 92.65         | 92.97  | 95.60         |
| 85K  | 85.61          | 92.34         | 92.64  | 95.14         |
| 90K  | 85.22          | 92.14         | 92.38  | 94.68         |
| 95K  | 84.88          | 91.86*        | 92.19  | 94.31         |
| 100K | 84.53          | 91.58         | 91.85  | 93.88         |

*estimated from tqdm

**attn_corr_ffn ≈ nocat** — only ~0.2-0.5 PPL difference. FFN-as-correction doesn't help vs standard block correction.
**attn_head_ffn worst** — ~3 PPL behind nocat. Raw attention delta as correction is weak.
**block_head_ffn is the clear winner** — ~6-7 PPL ahead. The extra head FFN is what matters.

### 10K eval (full diagnostics)

| Model | Params | Val PPL | K=1 PPL | Seq K=1 | L |
|---|---|---|---|---|---|
| block_head_ffn | ~1,657K | 121.56 | 162.77 | 121.57 | 0.66 |
| attn_corr_ffn | 1,646,750 | 128.96 | 166.58 | 129.14 | 0.53 |
| roformer_look_ahead_nocat | 1,646,750 | 128.43 | 140.46 | 128.48 | 0.99 |
| attn_head_ffn | 1,646,750 | 139.10 | 149.91 | 139.10 | 0.53 |

### Key observations

1. **block_head_ffn wins at 84.53** — beats projhead (87.06) by 2.5 PPL with similar extra params
2. **block_head_ffn approaches concat (82.29)** — only 2.2 PPL behind, with far fewer extra params (~10K vs 800K)
3. **Perfect sequential K=1 match** for all models (block_head_ffn: 84.57 ≈ 84.53, attn_corr_ffn: 91.58 ≈ 91.58)
4. **attn_corr_ffn ≈ nocat** — only ~0.2-0.5 PPL better throughout training. Splitting attention/FFN doesn't help.
5. **attn_head_ffn is worst** — ~3 PPL behind nocat. Raw attention delta is a poor correction signal.
6. **Only block_head_ffn meaningfully beats nocat** — the key ingredient is the extra FFN at the head, not splitting the block.
7. **L=1.20 for block_head_ffn at 100K** — iterations diverging, yet sequential K=1 still matches. L doesn't predict short-sequence performance.

### block_head_ffn vs roformer baselines

| Iter | roformer N=3 | block_head_ffn | roformer N=1 |
|------|--------------|----------------|--------------|
| 5K   | 137.04       | 152.88         | 156.45       |
| 10K  | 113.90       | 121.56         | 135.58       |
| 15K  | 104.57       | 110.34         | 126.08       |
| 20K  | 99.12        | 103.93         | 119.96       |
| 25K  | 95.52        | 99.59          | 115.27       |
| 30K  | 92.83        | 96.59          | 111.35       |
| 35K  | 90.71        | 94.32          | 108.33       |
| 40K  | 88.90        | 92.53          | 105.68       |
| 45K  | 87.24        | 91.17          | 103.49       |
| 50K  | 85.95        | 90.08          | 101.82       |
| 55K  | 84.93        | 88.96          | 100.42       |
| 60K  | 83.79        | 88.18          | 99.03        |
| 65K  | 82.86        | 87.47          | 97.92        |
| 70K  | 82.14        | 87.05          | 96.98        |
| 75K  | 81.30        | 86.45          | 96.13        |
| 80K  | 80.68        | 85.91          | 95.32        |
| 85K  | 80.00        | 85.61          | 94.61        |
| 90K  | 79.31        | 85.22          | 93.90        |
| 95K  | 78.63        | 84.88          | 93.41        |
| 100K | **78.24**    | **84.53**      | **92.88**    |

block_head_ffn (1 layer, ~1,657K params) beats roformer N=1 (1 layer, 1,647K params) by 8.35 PPL.
roformer N=3 (3 layers, ~1,708K params) beats block_head_ffn by 6.3 PPL.

### stacked_block_head_ffn N=3 vs roformer N=3 (both 3 layers at inference)

| Iter | stacked_block_head_ffn N=3 | roformer N=3 | Gap   |
|------|----------------------------|--------------|-------|
| 5K   | 141.76                     | 137.04       | +4.7  |
| 10K  | 111.85                     | 113.90       | -2.1  |
| 15K  | 101.53                     | 104.57       | -3.0  |
| 20K  | 95.69                      | 99.12        | -3.4  |
| 25K  | 91.90                      | 95.52        | -3.6  |
| 30K  | 89.16                      | 92.83        | -3.7  |
| 35K  | 87.18                      | 90.71        | -3.5  |
| 40K  | 85.45                      | 88.90        | -3.5  |
| 45K  | 84.16                      | 87.24        | -3.1  |
| 50K  | 83.12                      | 85.95        | -2.8  |
| 55K  | 82.20                      | 84.93        | -2.7  |
| 60K  | 81.45                      | 83.79        | -2.3  |
| 65K  | 80.79                      | 82.86        | -2.1  |
| 70K  | 80.09                      | 82.14        | -2.1  |
| 75K  | 79.60                      | 81.30        | -1.7  |
| 80K  | 79.26                      | 80.68        | -1.4  |
| 85K  | 78.84                      | 80.00        | -1.2  |
| 90K  | 78.48                      | 79.31        | -0.8  |
| 95K  | 78.14                      | 78.63        | -0.5  |
| 100K | **77.93**                  | **78.24**    | -0.3  |

stacked_block_head_ffn N=3 beats roformer N=3 by 0.31 PPL. Empirical L=0.0001 (near-perfect convergence).
Gap narrowed from -3.5 (early) to -0.3 (final) — roformer catches up with more training but doesn't close fully.

### stacked_block_head N=3 vs stacked_block_head_ffn N=3 vs block_head_ffn vs roformer N=3

| Iter | stacked_block_head N=3 | stacked_block_head_ffn N=3 | block_head_ffn N=1 | roformer N=3 |
|------|------------------------|----------------------------|--------------------|--------------|
| 5K   | 149.33                 | 141.76                     | 152.88             | 137.04       |
| 10K  | 116.91                 | 111.85                     | 121.56             | 113.90       |
| 15K  | 105.20                 | 101.53                     | 110.34             | 104.57       |
| 20K  | 98.94                  | 95.69                      | 103.93             | 99.12        |
| 25K  | 95.00                  | 91.90                      | 99.59              | 95.52        |
| 30K  | 92.31                  | 89.16                      | 96.59              | 92.83        |
| 35K  | 90.34                  | 87.18                      | 94.32              | 90.71        |
| 40K  | 88.88                  | 85.45                      | 92.53              | 88.90        |
| 45K  | 87.71                  | 84.16                      | 91.17              | 87.24        |
| 50K  | 86.73                  | 83.12                      | 90.08              | 85.95        |
| 55K  | 85.88                  | 82.20                      | 88.96              | 84.93        |
| 60K  | 85.32                  | 81.45                      | 88.18              | 83.79        |
| 65K  | 84.68                  | 80.79                      | 87.47              | 82.86        |
| 70K  | 84.37                  | 80.09                      | 87.05              | 82.14        |
| 75K  | 83.83                  | 79.60                      | 86.45              | 81.30        |
| 80K  | 83.52                  | 79.26                      | 85.91              | 80.68        |
| 85K  | 83.12                  | 78.84                      | 85.61              | 80.00        |
| 90K  | 82.93                  | 78.48                      | 85.22              | 79.31        |
| 95K  | 82.62                  | 78.14                      | 84.88              | 78.63        |
| 100K | **82.29**              | **77.93**                  | **84.53**          | **78.24**    |

stacked_block_head N=3 100K: Val PPL 82.29, Seq K=1 82.30, L=0.0011.
- stacked_block_head_ffn beats roformer N=3 at every single checkpoint
- stacked_block_head (no head_ffn) trails roformer N=3 in later stages but beats non-stacked block_head_ffn throughout
- head_ffn worth ~4.4 PPL (77.93 vs 82.29)
- stacking alone worth ~2.2 PPL over non-stacked (82.29 vs 84.53)

### block_head_ffn D=5 K=10 (5 layers at inference, compare to roformer N=5 at 70.89)

| Iter | block_head_ffn D=5 | roformer N=5 | Gap   |
|------|--------------------|--------------| ------|
| 5K   | 140.53             | 122.91       | +17.6 |
| 10K  | 109.10             | 102.05       | +7.1  |
| 15K  | 97.88              | 94.28        | +3.6  |
| 20K  | 91.85              | 89.51        | +2.3  |
| 25K  | 87.83              | 86.22        | +1.6  |
| 30K  | 85.03              | 83.71        | +1.3  |
| 35K  | 82.94              | 81.85        | +1.1  |
| 40K  | 81.25              | 80.23        | +1.0  |
| 50K  | 78.66              | 77.70        | +1.0  |
| 55K  | 77.73              | 76.67        | +1.1  |
| 60K  | 76.89              | 75.71        | +1.2  |
| 65K  | 76.16              | 75.02        | +1.1  |
| 70K  | 75.53              | 74.30        | +1.2  |
| 75K  | 74.95              | 73.53        | +1.4  |
| 80K  | 74.47              | 72.90        | +1.6  |
| 85K  | 74.05              | 72.41        | +1.6  |
| 90K  | 73.62              | 71.90        | +1.7  |
| 95K  | 73.36              | 71.43        | +1.9  |
| 100K | **72.96**          | **70.89**    | **+2.1** |

block_head_ffn D=5 100K: Val PPL 72.96, Seq K=1 72.97, L=0.58, params 1,789,700.
Gap widened in second half (+1.0 → +2.1). D=5 ends ~2 PPL behind roformer N=5 with similar params.

### joformer_projected_block_head_ffn vs block_head_ffn (roformer)

| Iter | block_head_ffn (roformer) | joformer_projected_block_head_ffn | Gap   |
|------|---------------------------|----------------------------------|-------|
| 5K   | 152.88                    | 169.33                           | +16.5 |
| 10K  | 121.56                    | 125.33                           | +3.8  |
| 15K  | 110.34                    | 110.82                           | +0.5  |
| 20K  | 103.93                    | 103.18                           | -0.8   |
| 25K  | 99.59                     | 98.56                            | -1.0   |
| 35K  | 94.32                     | 92.51                            | -1.8   |
| 55K  | 88.96                     | 86.83                            | -2.1   |
| 60K  | 88.18                     | 85.63                            | -2.6   |
| 65K  | 87.47                     | 85.07                            | -2.4   |
| 70K  | 87.05                     | 84.21                            | -2.8   |
| 90K  | 85.22                     | 82.37                            | -2.9   |
| 100K | **84.53**                 | **81.57**                        | **-3.0** |

JoFormer projected crossed over at 20K — final gap -3.0 at 100K.
joformer_projected_block_head_ffn 100K: Val PPL 81.57, Seq K=1 81.57, L=0.98.

### joformer_projected_block_head_corr_ffn vs joformer_projected_block_head_ffn vs roformer N=3 (C=50, K=10, block_size=256)

| Iter | joformer_proj_corr_ffn | block_head_corr_ffn | joformer_proj_block_head_ffn | roformer N=3 |
|------|------------------------|---------------------|------------------------------|--------------|
| 5K   | 143.51                 | 152.10              | 169.33                       | 137.04       |
| 10K  | 114.54                 | 120.40              | 125.33                       | 113.90       |
| 15K  | 104.18                 | 109.05              | 110.82                       | 104.57       |
| 20K  | 98.48                  | 102.60              | 103.18                       | 99.12        |
| 25K  | 94.83                  | 98.29               | 98.56                        | 95.52        |
| 30K  | 92.07                  | 95.30               | 95.08                        | 92.83        |
| 35K  | 89.93                  | 93.18               | 92.51                        | 90.71        |
| 40K  | 88.64                  | 91.47               | 90.67                        | 88.90        |
| 45K  | 87.14                  | 90.03               | 89.06                        | 87.24        |
| 50K  | 86.18                  | 89.01               | 87.76                        | 85.95        |
| 55K  | 85.25                  | 88.08               | 86.83                        | 84.93        |
| 60K  | 84.24                  | 87.41               | 85.63                        | 83.79        |
| 65K  | 83.67                  | 86.73               | 85.07                        | 82.86        |
| 70K  | 83.10                  | 86.44               | 84.21                        | 82.14        |
| 75K  | 82.46                  | 85.75               | 83.57                        | 81.30        |
| 80K  | 82.08                  | 85.33               | 83.23                        | 80.68        |
| 85K  | 81.68                  | 85.15               | 82.86                        | 80.00        |
| 90K  | 81.40                  | 84.75               | 82.37                        | 79.31        |
| 95K  | 81.00                  | 84.35               | 81.95                        | 78.63        |
| 100K | **80.77**              | **84.20**           | **81.57**                    | **78.24**    |

joformer_proj_corr_ffn 100K: Val PPL 80.77, Seq K=1 80.80, L=0.66. Beats joformer_proj_block_head_ffn (81.57) by 0.8 PPL.

### stacked_block_head_corr_ffn N=3 vs stacked_block_head_ffn N=3 vs roformer N=3 (C=50, K=10, block_size=256)

| Iter | stacked_corr_ffn N=3 | stacked_block_head_ffn N=3 | roformer_head_ffn N=3 | roformer N=3 |
|------|----------------------|---------------------------|-----------------------|--------------|
| 5K   | 141.22               | 141.76                    | 134.89                | 137.04       |
| 10K  | 112.21               | 111.85                    | 112.28                | 113.90       |
| 15K  | 101.17               | 101.53                    | 103.01                | 104.57       |
| 20K  | 95.20                | 95.69                     | 97.17                 | 99.12        |
| 25K  | 91.12                | 91.90                     | 93.19                 | 95.52        |
| 30K  | 88.44                | 89.16                     | 90.25                 | 92.83        |
| 35K  | 86.40                | 87.18                     | 87.93                 | 90.71        |
| 40K  | 84.72                | 85.45                     | 86.02                 | 88.90        |
| 45K  | 83.56                | 84.16                     | 84.58                 | 87.24        |
| 50K  | 82.39                | 83.12                     | 83.10                 | 85.95        |
| 55K  | 81.48                | 82.20                     | 81.83                 | 84.93        |
| 60K  | 80.63                | 81.45                     | 80.75                 | 83.79        |
| 65K  | 79.93                | 80.79                     | 79.73                 | 82.86        |
| 70K  | 79.47                | 80.09                     | 78.91                 | 82.14        |
| 75K  | 78.97                | 79.60                     | 78.19                 | 81.30        |
| 80K  | 78.41                | 79.26                     | 77.67                 | 80.68        |
| 85K  | 77.99                | 78.84                     | 76.85                 | 80.00        |
| 90K  | 77.67                | 78.48                     | 76.25                 | 79.31        |
| 95K  | 77.31                | 78.14                     | 75.79                 | 78.63        |
| 100K | **77.00**            | **77.93**                 | **75.32**             | **78.24**    |

stacked_block_head_corr_ffn N=3 100K: 77.00 PPL. Beats stacked_block_head_ffn (77.93) by 0.93 PPL. Loses to roformer_head_ffn N=3 (75.32) by 1.7 PPL. Beats roformer N=3 (78.24) by 1.24 PPL.

### roformer_head_ffn N=3 vs roformer N=3 vs stacked_block_head_ffn N=3

| Iter | roformer_head_ffn N=3 | roformer N=3 | stacked_block_head_ffn N=3 |
|------|----------------------|--------------|---------------------------|
| 5K   | 134.89               | 137.04       | 141.76                    |
| 10K  | 112.28               | 113.90       | 111.85                    |
| 15K  | 103.01               | 104.57       | 101.53                    |
| 20K  | 97.17                | 99.12        | 95.69                     |
| 25K  | 93.19                | 95.52        | 91.90                     |
| 30K  | 90.25                | 92.83        | 89.16                     |
| 35K  | 87.93                | 90.71        | 87.18                     |
| 40K  | 86.02                | 88.90        | 85.45                     |
| 45K  | 84.58                | 87.24        | 84.16                     |
| 50K  | 83.10                | 85.95        | 83.12                     |
| 55K  | 81.83                | 84.93        | 82.20                     |
| 60K  | 80.75                | 83.79        | 81.45                     |
| 65K  | 79.73                | 82.86        | 80.79                     |
| 70K  | 78.91                | 82.14        | 80.09                     |
| 75K  | 78.19                | 81.30        | 79.60                     |
| 80K  | 77.67                | 80.68        | 79.26                     |
| 85K  | 76.85                | 80.00        | 78.84                     |
| 90K  | 76.25                | 79.31        | 78.48                     |
| 95K  | 75.79                | 78.63        | 78.14                     |
| 100K | **75.32**            | **78.24**    | **77.93**                 |

roformer_head_ffn N=3 100K: 75.32 PPL (1,728,400 params).
- head_ffn gives roformer ~2.9 PPL boost (75.32 vs 78.24)
- roformer_head_ffn beats stacked_block_head_ffn by 2.6 PPL (75.32 vs 77.93)
- head_ffn benefits roformer more than the look-ahead architecture — reinforces that block_head (no head_ffn) is the right comparison

### FLOP-matched block_head_corr_ffn C=74 vs roformer_head_ffn N=3 C=50 (20C² = (12×3+8)C²)

| Iter | block_head_corr_ffn C=74 | roformer_head_ffn N=3 | roformer N=3 | stacked_block_head_ffn N=3 |
|------|--------------------------|----------------------|--------------|---------------------------|
| 5K   | 117.35                   | 134.89               | 137.04       | 141.76                    |
| 10K  | 96.91                    | 112.28               | 113.90       | 111.85                    |
| 15K  | 88.77                    | 103.01               | 104.57       | 101.53                    |
| 20K  | 84.24                    | 97.17                | 99.12        | 95.69                     |
| 25K  | 81.11                    | 93.19                | 95.52        | 91.90                     |
| 30K  | 78.92                    | 90.25                | 92.83        | 89.16                     |
| 35K  | 77.18                    | 87.93                | 90.71        | 87.18                     |
| 40K  | 75.79                    | 86.02                | 88.90        | 85.45                     |
| 45K  | 74.73                    | 84.58                | 87.24        | 84.16                     |
| 50K  | 73.68                    | 83.10                | 85.95        | 83.12                     |
| 55K  | 72.95                    | 81.83                | 84.93        | 82.20                     |
| 60K  | 72.37                    | 80.75                | 83.79        | 81.45                     |
| 65K  | 71.65                    | 79.73                | 82.86        | 80.79                     |
| 70K  | 71.12                    | 78.91                | 82.14        | 80.09                     |
| 75K  | 70.82                    | 78.19                | 81.30        | 79.60                     |
| 80K  | 70.34                    | 77.67                | 80.68        | 79.26                     |
| 85K  | 69.94                    | 76.85                | 80.00        | 78.84                     |
| 90K  | 69.74                    | 76.25                | 79.31        | 78.48                     |
| 95K  | 69.41                    | 75.79                | 78.63        | 78.14                     |
| 100K | **69.12**                | **75.32**            | **78.24**    | **77.93**                 |

block_head_corr_ffn C=74 100K: **69.12 PPL** (2,495,148 params). Seq K=1 = 69.18. L = 0.92 (high, one ratio >1).
- 6.2 PPL better than FLOP-matched roformer_head_ffn N=3 (75.32)
- 9.1 PPL better than roformer N=3 (78.24)
- 1-layer inference model beating all 3-layer models
- Most gains by K=3 (72.13), K=5 nearly converged (69.38 vs 69.12)
- Concerning: L trended up to 1.2+ at 85-90K before dropping back. Last ratio (K=9→K=10) often >1.0.
  Iterations not truly contracting — could be problematic for long-form generation beyond block_size=256.

Depth sweep (block_head_corr_ffn C=74, 100K):

| K          | Val PPL |
|------------|---------|
| 1          | 119.91  |
| 2          | 80.12   |
| 3          | 72.13   |
| 5          | 69.38   |
| 10         | 69.12   |
| sequential | 69.18   |

### FLOP-matched block_head_corr_ffn C=68 vs roformer N=3 C=50 (20C² = 36C²)

| Iter | block_head_corr_ffn C=68 | roformer N=3 | Gap   |
|------|--------------------------|--------------|-------|
| 5K   | 123.52                   | 137.04       | -13.5 |
| 10K  | 101.25                   | 113.90       | -12.7 |
| 15K  | 92.60                    | 104.57       | -12.0 |
| 20K  | 87.75                    | 99.12        | -11.4 |
| 25K  | 84.45                    | 95.52        | -11.1 |
| 30K  | 82.10                    | 92.83        | -10.7 |
| 35K  | 80.25                    | 90.71        | -10.5 |
| 40K  | 78.83                    | 88.90        | -10.1 |
| 45K  | 77.75                    | 87.24        | -9.5  |
| 50K  | 76.78                    | 85.95        | -9.2  |
| 55K  | 76.02                    | 84.93        | -8.9  |
| 60K  | 75.17                    | 83.79        | -8.6  |
| 65K  | 74.59                    | 82.86        | -8.3  |
| 70K  | 74.16                    | 82.14        | -8.0  |
| 75K  | 73.57                    | 81.30        | -7.7  |
| 80K  | 73.10                    | 80.68        | -7.6  |
| 85K  | 72.72                    | 80.00        | -7.3  |
| 90K  | 72.43                    | 79.31        | -6.9  |
| 95K  | 72.17                    | 78.63        | -6.5  |
| 100K | **71.95**                | **78.24**    | **-6.3** |

block_head_corr_ffn C=68 100K: **71.95 PPL** (2,285,976 params). Seq K=1 = 72.02. L = 0.89.
- 6.3 PPL better than FLOP-matched roformer N=3 (78.24)

### Deep block_head_corr_ffn D=3 vs stacked_block_head_corr_ffn N=3 (C=50, K=10, block_size=256)

| Iter | deep corr_ffn D=3 | stacked corr_ffn N=3 | roformer_head_ffn N=3 |
|------|-------------------|---------------------|----------------------|
| 5K   | 138.39            | 141.22              | 134.89               |
| 10K  | 111.22            | 112.21              | 112.28               |
| 15K  | 101.33            | 101.17              | 103.01               |
| 20K  | 95.67             | 95.20               | 97.17                |
| 25K  | 91.86             | 91.12               | 93.19                |
| 30K  | 89.05             | 88.44               | 90.25                |
| 35K  | 86.87             | 86.40               | 87.93                |
| 40K  | 85.06             | 84.72               | 86.02                |
| 45K  | 83.78             | 83.56               | 84.58                |
| 50K  | 82.50             | 82.39               | 83.10                |
| 55K  | 81.48             | 81.48               | 81.83                |
| 60K  | 80.62             | 80.63               | 80.75                |
| 65K  | 79.85             | 79.93               | 79.73                |
| 70K  | 79.13             | 79.47               | 78.91                |
| 75K  | 78.59             | 78.97               | 78.19                |
| 80K  | 78.14             | 78.41               | 77.67                |
| 85K  | 77.64             | 77.99               | 76.85                |
| 90K  | 77.17             | 77.67               | 76.25                |
| 95K  | 76.79             | 77.31               | 75.79                |
| 100K | **76.60**         | **77.00**           | **75.32**            |

Diagnostics at 100K:

| Metric         | deep corr_ffn D=3 | stacked corr_ffn N=3 |
|----------------|-------------------|---------------------|
| Parallel K=1   | 115.55            | 87.00               |
| Parallel K=2   | 82.40             | 78.36               |
| Parallel K=3   | 77.95             | 77.29               |
| Parallel K=5   | 76.68             | 77.04               |
| Parallel K=10  | 76.60             | 77.00               |
| Sequential K=1 | 76.60             | 77.01               |
| Final val      | 76.60             | 77.00               |

### block_head_corr_ffn D=1 C=50: convergence_weight ablation (K=10, block_size=256)

| Iter | cw=0    | cw=0.1  | cw=0.5  |
|------|---------|---------|---------|
| 5K   | 154.39  | 152.10  | 153.15  |
| 10K  | 121.47  | 120.40  | 121.16  |
| 15K  | 109.39  | 109.05  | 109.65  |
| 20K  | 102.83  | 102.60  | 103.18  |
| 25K  | 98.44   | 98.29   | 98.77   |
| 30K  | 95.41   | 95.30   | 95.68   |
| 35K  | 93.43   | 93.18   | 93.65   |
| 40K  | 91.77   | 91.47   | 91.86   |
| 45K  | 90.29   | 90.03   | 90.42   |
| 50K  | 89.31   | 89.01   | 89.31   |
| 55K  | 88.35   | 88.08   |         |
| 60K  | 87.63   | 87.41   |         |
| 65K  | 86.92   | 86.73   |         |
| 70K  | 86.61   | 86.44   |         |
| 75K  | 85.89   | 85.75   |         |
| 80K  | 85.47   | 85.33   |         |
| 85K  | 85.14   | 85.15   |         |
| 90K  | 84.83   | 84.75   |         |
| 95K  | 84.38   | 84.35   |         |
| 100K | **84.16** | **84.20** |       |

cw=0 diagnostics: Seq K=1 = 84.19, L = 0.94, Par K=1 = 130.95, Par K=5 = 84.50.
cw=0.1 diagnostics: Seq K=1 = 84.20, L = ~0.92.
Conclusion: convergence_weight makes negligible difference at C=50. cw=0 slightly better final PPL and much better Par K=1.

### FLOP-matched block_head_corr_ffn C=660 vs roformer_head_ffn N=3 C=446 vs roformer N=3 C=492 (~8.7M multiplies/token)

| Iter | corr_ffn C=660 | roformer_head_ffn N=3 C=446 | roformer N=3 C=492 | Gap (corr vs head_ffn) |
|------|----------------|-----------------------------|--------------------|------------------------|
| 5K   | 40.14          | 40.97                       | 41.38              | -0.8                   |
| 10K  | 34.80          | 35.38                       | 35.81              | -0.6                   |
| 15K  | 32.38          | 32.88                       | 33.33              | -0.5                   |
| 20K  | 30.94          | 31.46                       | 31.77              | -0.5                   |
| 25K  | 30.02          | 30.45                       | 30.69              | -0.4                   |
| 30K  | 29.28          | 29.62                       | 29.98              | -0.3                   |
| 35K  | 28.78          | 29.00                       | 29.32              | -0.2                   |
| 40K  | 28.11          | 28.52                       | 28.87              | -0.4                   |
| 45K  | 27.84          | 28.14                       | 28.44              | -0.3                   |
| 50K  | 27.46          | 27.79                       | 28.09              | -0.3                   |
| 55K  | 27.06          | 27.45                       | 27.82              | -0.4                   |
| 60K  | 26.75          | 27.19                       | 27.49              | -0.4                   |
| 65K  | 26.61          | 26.92                       | 27.23              | -0.3                   |
| 70K  | 26.29          | 26.72                       | 27.01              | -0.4                   |
| 75K  | 26.18          | 26.55                       | 26.82              | -0.4                   |
| 80K  | 25.97          | 26.29                       | 26.69              | -0.3                   |
| 85K  | 25.74          | 26.20                       | 26.51              | -0.5                   |
| 90K  | 25.66          | 26.09                       | 26.39              | -0.4                   |
| 95K  |                | 25.99                       | 26.26              |                        |
| 100K |                | **25.78**                   | **26.12**          |                        |

### Deep D=3 corr_ffn C=446 vs roformer_head_ffn N=3 C=446 (same FLOPs 44C², same params, big machine)

| Iter | D=3 corr_ffn C=446 | roformer_head_ffn N=3 C=446 | Gap  |
|------|--------------------|-----------------------------|------|
| 5K   | 38.95              | 40.97                       | -2.0 |
| 10K  | 33.22              | 35.38                       | -2.2 |
| 15K  | 30.62              | 32.88                       | -2.3 |
| 20K  | 29.06              | 31.46                       | -2.4 |
| 25K  | 27.97              | 30.45                       | -2.5 |
| 30K  | 27.07              | 29.62                       | -2.6 |
| 35K  | 26.47              | 29.00                       | -2.5 |
| 40K  | 25.95              | 28.52                       | -2.6 |
| 45K  | 25.54              | 28.14                       | -2.6 |
| 50K  | 25.10              | 27.79                       | -2.7 |
| 55K  | 24.83              | 27.45                       | -2.6 |
| 60K  | 24.53              | 27.19                       | -2.7 |
| 65K  | 24.33              | 26.92                       | -2.6 |
| 70K  | 24.05              | 26.72                       | -2.7 |
| 75K  | 23.95              | 26.55                       | -2.6 |
| 80K  | 23.77              | 26.29                       | -2.5 |
| 85K  | 23.59              | 26.20                       | -2.6 |
| 90K  | 23.46              | 26.09                       | -2.6 |
| 95K  | 23.28              | 25.99                       | -2.7 |
| 100K | **23.18**          | **25.78**                   | **-2.6** |
| 75K  |                    | 26.55                       |      |
| 80K  |                    | 26.29                       |      |
| 85K  |                    | 26.20                       |      |
| 90K  |                    | 26.09                       |      |
| 95K  |                    | 25.99                       |      |
| 100K |                    | **25.78**                   |      |

D=3 corr_ffn C=446 final: **23.18 PPL**. Seq K=1 = 23.23 (gap 0.05). Beats roformer_head_ffn N=3 C=446 (25.78) by **2.6 PPL**.
Gap steady at ~2.6 throughout training. At 70K already below roformer_head_ffn's 100K final.

### D=3 concat v2 C=446 (K=5) vs D=3 non-concat C=446 (K=10) (big machine)

Concat v2 has n_layers=15 (K=5) + random K training. Non-concat has n_layers=30 (K=10), no random K.
Two confounds: K=5 vs K=10, and random K training (~0.5 PPL penalty at small C).

| Iter | Concat v2 (K=5) | Non-concat (K=10) | Gap  |
|------|-----------------|-------------------|------|
| 5K   | 38.51           | 38.95             | -0.4 |
| 10K  | 33.00           | 33.22             | -0.2 |
| 15K  | 30.70           | 30.62             | +0.1 |
| 20K  | 29.21           | 29.06             | +0.2 |
| 25K  | 28.20           | 27.97             | +0.2 |
| 30K  | 27.47           | 27.07             | +0.4 |
| 35K  | 26.80           | 26.47             | +0.3 |
| 40K  | 26.28           | 25.95             | +0.3 |
| 45K  | 25.83           | 25.54             | +0.3 |
| 50K  | 25.49           | 25.23             | +0.3 |
| 55K  | 25.06           | 24.83             | +0.2 |
| 60K  | 24.83           | 24.53             | +0.3 |
| 65K  | 24.68           | 24.33             | +0.4 |
| 70K  | 24.38           | 24.05             | +0.3 |
| 75K  | 24.21           | 23.95             | +0.3 |
| 80K  | 23.91           | 23.77             | +0.1 |

Non-concat pulling ahead despite concat v2 winning at small C. Gap ~0.3 PPL, narrowing.
Possible cause: K=5 + random K hurting concat v2 training.
**Planned**: D=3 non-concat C=446 with K=5 + random K (n_layers=15) to isolate confounds.

### Stacked N=3 corr_ffn C=446 vs D=3 corr_ffn C=446 (big machine)

| Iter | Stacked N=3 C=446 | D=3 C=446 | Gap |
|------|------------------|-----------|-----|
| 5K   | 42.35 | 38.95 | +3.4 |
| 10K  | 36.64 | 33.22 | +3.4 |
| 15K  | 34.00 | 30.62 | +3.4 |
| 20K  | 32.22 | 29.06 | +3.2 |
| 25K  | 31.13 | 27.97 | +3.2 |
| 30K  | 30.23 | 27.07 | +3.2 |
| 35K  | 29.32 | 26.47 | +2.9 |
| 40K  | 28.77 | 25.95 | +2.8 |
| 45K  | 28.34 | 25.54 | +2.8 |
| 50K  | 27.99 | 25.23 | +2.8 |
| 55K  | 27.55 | 24.83 | +2.7 |
| 60K  | 27.20 | 24.53 | +2.7 |
| 65K  | 26.94 | 24.33 | +2.6 |
| 70K  | 26.63 | 24.05 | +2.6 |
| 75K  | 26.36 | 23.95 | +2.4 |
| 80K  | 26.14 | 23.77 | +2.4 |
| 85K  | 25.93 | 23.59 | +2.3 |
| 90K  | 25.71 | 23.46 | +2.3 |
| 95K  | 25.50 | 23.28 | +2.2 |
| 100K | 25.46 | 23.18 | +2.3 |

**Complete.** D=3: **23.18 PPL**, seq K=1 = 23.23. Stacked N=3: **25.46 PPL**, seq K=1 = 25.52.
D=3 wins by 2.28 PPL. Stacked has better parallel K=1 (41.42 vs 87.73) indicating faster convergence, but D=3 achieves better final quality.

### D=3 concat v2 vs D=3 add vs Stacked N=3 non-concat vs Stacked N=3 concat v2 (C=446, big machine)

| Iter | D=3 concat v2 (K=5) | D=3 add (K=5) | Stacked N=3 non-concat (K=10) | Stacked N=3 concat v2 (K=5) |
|------|---------------------|---------------|-------------------------------|----------------------------|
| 5K   | 38.51               | 39.42         | 42.35                         | 36.60                      |
| 10K  | 33.00               | 33.81         | 36.64                         | 31.38                      |
| 15K  | 30.70               | 31.43         | 34.00                         | 28.79                      |
| 20K  | 29.21               | 29.88         | 32.22                         | 27.51                      |
| 25K  | 28.20               | 28.77         | 31.13                         | 26.54                      |
| 30K  | 27.47               | 27.87         | 30.23                         | 25.75                      |
| 35K  | 26.80               | 27.23         | 29.32                         | 25.14                      |
| 40K  | 26.28               | 26.74         | 28.77                         | 24.68                      |
| 45K  | 25.83               | 26.36         | 28.34                         | 24.25                      |
| 50K  | 25.49               | 25.98         | 27.99                         | 23.85                      |
| 55K  | 25.06               | 25.60         | 27.55                         | 23.56                      |
| 60K  | 24.83               | 25.32         | 27.20                         | 23.27                      |
| 65K  | 24.68               | 25.08         | 26.94                         | 23.11                      |
| 70K  | 24.38               | 24.82         | 26.63                         | 22.96                      |
| 75K  | 24.21               | 24.67         | 26.36                         | 22.70                      |
| 80K  | 23.91               | 24.45         | 26.14                         | 22.52                      |
| 85K  | 23.78               | 24.31         | 25.93                         | 22.33                      |
| 90K  | 23.62               | 24.15         | 25.71                         | 22.15                      |
| 95K  | 23.57               | 23.97         | 25.50                         | 22.09                      |
| 100K | 23.39               | 23.82         | 25.46                         | **21.96**                  |

Final diagnostics (C=446, 100K):

| Model | Final PPL | Seq K=1 | Par K=1 | Par K=5 | L |
|-------|-----------|---------|---------|---------|---|
| Stacked N=3 concat v2 (K=5) | **21.96** | 23.28 | 34.65 | 21.96 | 0.10 |
| D=3 concat v2 (K=5) | 23.39 | 23.73 | 35.64 | 23.39 | — |
| D=3 non-concat (K=30) | 23.18 | 23.23 | 87.73 | 23.64 | — |
| D=3 add (K=5) | 23.82 | 24.14 | 36.90 | 23.80 | — |
| Stacked N=3 non-concat (K=10) | 25.46 | 25.52 | 41.42 | 25.45 | — |

Stacked concat v2 wins: 21.96 vs 23.39 (D=3 concat v2) vs 23.18 (D=3 non-concat) vs 23.82 (D=3 add) vs 25.46 (stacked non-concat).
Excellent convergence: L=0.10, K sweep monotonically decreasing, seq K=1 gap only 1.32 PPL.
BUT stacked concat v2 has ~50% more params (72C² vs 48C²) due to 3 corr_ffns vs 1.

### C=96/C=100 scale: D=1, D=3 concat v2 vs roformer_head_ffn baselines

- D=1 concat v2 C=136: 24 × 136² = 443,904 inference FLOPs (FLOP-matched to N=3 C=100)
- D=3 concat v2 C=96: 48 × 96² = 442,368 inference FLOPs (FLOP-matched to N=3 C=100)
- roformer_head_ffn N=3 C=100: 44 × 100² = 440,000 inference FLOPs
- roformer_head_ffn N=3 C=96: 44 × 96² = 405,504 inference FLOPs (same-C control)

| Iter | D=1 C=136 | D=3 C=96 | rhf N=3 C=100 | rhf N=3 C=96 |
|------|-----------|----------|---------------|--------------|
| 5K   | 78.54     | 85.52    | 84.72         | 87.22        |
| 10K  | 67.12     | 72.51    | 71.17         | 73.65        |
| 15K  | 62.65     | 67.37    | 65.83         | 68.10        |
| 20K  | 59.82     | 64.02    | 62.80         | 64.80        |
| 25K  | 57.95     | 61.80    | 60.70         | 62.63        |
| 30K  | 56.50     | 60.03    | 59.05         | 60.96        |
| 35K  | 55.47     | 58.89    | 58.05         | 59.73        |
| 40K  | 54.52     | 57.76    | 56.98         | 58.53        |
| 45K  | 53.68     | 56.90    | 56.09         | 57.70        |
| 50K  | 53.13     | 56.08    | 55.36         | 56.89        |
| 55K  | 52.54     | 55.43    | 54.75         | 56.26        |
| 60K  | 52.01     | 54.86    | 54.17         | 55.71        |
| 65K  | 51.62     | 54.46    | 53.76         | 55.10        |
| 70K  | 51.20     | 53.81    | 53.25         | 54.70        |
| 75K  | 50.98     | 53.53    | 52.93         | 54.22        |
| 80K  | 50.62     | 53.10    | 52.57         | 53.80        |
| 85K  | 50.37     | 52.81    | 52.15         | 53.61        |
| 90K  | 50.07     | 52.47    | 51.92         | 53.12        |
| 95K  | 49.82     | 52.34    | 51.54         | 52.81        |
| 100K | 49.54     | 51.99    | 51.38         | 52.51        |

D=1 C=136 complete: **49.54 PPL**. Seq K=1 = 49.80 (gap 0.26). Par K=1 = 85.82. L = 0.70. Params = 4,814,896.
D=3 C=96 complete: **51.99 PPL**. Beats rhf C=96 (52.51) by 0.52 PPL at same C.
rhf N=3 C=96 complete: **52.51 PPL**.
D=3 C=96 slightly trails roformer_head_ffn C=100 (51.38) — the C=96 vs C=100 width gap explains it.

### Param-matched: roformer_head_ffn N=3 C=100 vs D=1 concat v2 C=105
roformer_head_ffn N=3 C=100: 3,660,800 params, 440,000 inference FLOPs/token
D=1 concat v2 C=105: 3,679,996 params, 264,600 inference FLOPs/token (40% fewer)

| Iter | D=1 C=105 | roformer_head_ffn N=3 C=100 |
|------|-----------|----------------------------|
| 5K   | 91.19     | 84.72 |
| 10K  | 76.68     | 71.17 |
| 15K  | 71.06     | 65.83 |
| 20K  | 67.65     | 62.80 |
| 25K  | 65.30     | 60.70 |
| 30K  | 63.72     | 59.05 |
| 35K  | 62.36     | 58.05 |
| 40K  | 61.33     | 56.98 |
| 45K  | 60.45     | 56.09 |
| 50K  | 59.78     | 55.36 |
| 55K  | 58.95     | 54.75 |
| 60K  | 58.55     | 54.17 |
| 65K  | 57.86     | 53.76 |
| 70K  | 57.49     | 53.25 |
| 75K  | 57.18     | 52.93 |
| 80K  | 56.75     | 52.57 |
| 85K  | 56.44     | 52.15 |
| 90K  | 56.21     | 51.92 |
| 95K  | 55.95     | 51.54 |
| 100K | 55.70     | 51.38 |

D=1 C=105 final: **55.68 PPL** (K=5). Seq K=1 = 55.87. Converged (K=5 = K=10 = 55.68/55.72).
roformer_head_ffn N=3 C=100 final: **51.38 PPL**.
Same params, D=1 uses 40% fewer inference FLOPs but loses by **4.3 PPL**.

### Param-matched: roformer_head_ffn N=6 C=100 vs D=1/D=2/D=3 concat v2
roformer_head_ffn N=6 C=100: 4,024,700 params, 800,000 inference FLOPs/token
D=1 concat v2 C=115: 4,015,930 params, 317,400 inference FLOPs/token (60% fewer)
D=2 concat v2 C=111: 4,015,441 params, 443,556 inference FLOPs/token (44% fewer)
D=3 concat v2 C=108: ~4,037,056 params, 559,872 inference FLOPs/token (30% fewer)

| Iter | rhf N=6 C=100 | D=3 C=108 | D=2 C=111 | D=1 C=115 |
|------|---------------|-----------|-----------|-----------|
| 5K   | 74.83 | 79.98 | 82.02 | 86.25 |
| 10K  | 63.16 | 68.08 | 69.70 | 73.20 |
| 15K  | 58.54 | 62.92 | 64.70 | 67.72 |
| 20K  | 55.66 | 60.01 | 61.66 | 64.90 |
| 25K  | 53.65 | 58.17 | 59.67 | 62.57 |
| 30K  | 52.27 | 56.63 | 58.17 | 61.14 |
| 35K  | 51.17 | 55.40 | 56.95 | 59.79 |
| 40K  | 50.27 | 54.48 | 56.17 | 58.72 |
| 45K  | 49.43 | 53.46 | 55.08 | 57.98 |
| 50K  | 48.83 | 52.78 | 54.34 | 57.33 |
| 55K  | 48.32 | 52.21 | 53.81 | 56.64 |
| 60K  | 47.72 | 51.81 | 53.14 | 56.19 |
| 65K  | 47.29 | 51.22 | 52.80 | 55.46 |
| 70K  | 46.86 | 50.93 | 52.36 | 55.11 |
| 75K  | 46.51 | 50.48 | 51.95 | 54.76 |
| 80K  | 46.17 | 50.13 | 51.53 | 54.29 |
| 85K  | 45.89 | 49.85 | 51.30 | 54.13 |
| 90K  | 45.59 | 49.60 | 51.15 | 53.91 |
| 95K  | 45.33 | 49.20 | 50.68 | 53.57 |
| 100K | 45.05 | 48.98 | 50.47 | 53.47 |

D=3 C=108 final: **48.98 PPL**. L = 0.34, ratios [0.72, 0.61, 0.34].
Each D step narrows the gap to N=6 (45.05): D=1 trails by 8.4, D=2 by 5.4, D=3 by 3.9.

### D=4 block_head C=108 vs D=3 concat v2 C=108 (same 48C² inference FLOPs, ~4M params)

Does a 4th block beat the corr_ffn at the same FLOP budget?

| Iter | roformer N=4 C=108 | D=3 concat v2 C=108 | corr_ffn_add D=4 C=108 | sync D=4 C=108 | D=4 block_head C=108 | D=4 nosub C=108 | Stacked N=4 C=108 |
|------|--------------------|---------------------|------------------------|----------------|----------------------|-----------------|-------------------|
| 5K   | 79.01              | 79.98               | 78.47                  | 83.06          | 90.07                | 88.32           | 89.48             |
| 10K  | 67.04              | 68.08               | 66.53                  | 70.19          | 74.53                | 72.24           | 73.75             |
| 15K  | 62.07              | 62.92               | 61.80                  | 64.94          | 68.09                | 66.40           | 67.56             |
| 20K  | 59.28              | 60.01               | 58.71                  | 61.80          | 64.73                | 63.21           | 64.17             |
| 25K  | 57.30              | 58.17               | 56.71                  | 59.83          | 62.33                | 60.96           | 61.80             |
| 30K  | 55.91              | 56.63               | 55.25                  | 58.10          | 60.42                | 59.15           | 60.09             |
| 35K  | 54.70              | 55.40               | 54.13                  | 56.89          | 58.90                | 57.86           | 58.76             |
| 40K  | 53.90              | 54.48               | 53.23                  | 56.02          | 57.85                | 56.83           | 57.80             |
| 45K  | 53.01              | 53.82               | 52.36                  | 55.06          | 56.93                | 56.01           | 56.79             |
| 50K  | 52.37              | 52.78               | 51.63                  | 54.43          | 56.11                | 55.10           | 56.08             |
| 55K  | 51.81              | 52.21               | 51.09                  | 53.90          | 55.60                | 54.44           | 55.45             |
| 60K  | 51.29              | 51.81               | 50.53                  | 53.30          | 54.92                | 54.12           | 54.92             |
| 65K  | 50.83              | 51.22               | 50.19                  | 52.95          | 54.23                | 53.62           | 54.45             |
| 70K  | 50.39              | 50.93               | 49.67                  | 52.41          | 54.01                | 53.06           | 54.00             |
| 75K  | 50.01              | 50.48               | 49.28                  | 51.94          | 53.46                | 52.75           | 53.60             |
| 80K  | 49.61              | 50.13               | 48.90                  | 51.57          | 53.21                | 52.28           | 53.27             |
| 85K  | 49.35              | 49.85               | 48.22                  | 51.33          | 52.95                | 52.01           | 53.08             |
| 90K  | 49.01              | 49.60               | 47.98                  | 51.15          | 52.62                | 51.92           | 52.70             |
| 95K  | 48.77              | 49.20               | 47.98                  | 50.78          | 52.47                | 51.42           | 52.37             |
| 100K | 48.58              | 48.98               | **47.76**              | **50.67**      | 52.22                | 51.25           | 52.39             |

**Note on param/FLOP matching:** roformer N=4, D=3 concat v2, sync D=4, block_head D=4, stacked N=4 are all 48C² FLOPs, ~4.02M params.
corr_ffn_add D=4 has **56C²** FLOPs, ~4.12M params (extra corr_ffn adds 8C²). NOT param/FLOP matched — 17% more FLOPs.
D=4 nosub has 48C² FLOPs, ~4.04M params (no extra FFN).

Roformer N=4 C=108 final: **48.58 PPL** (4.04M params, 48C² FLOPs) — the param+FLOP matched baseline.
D=3 concat v2 C=108 final: **48.98 PPL**. Only 0.40 PPL behind roformer N=4.
D=4 block_head C=108 final: **52.22 PPL**. Seq K=1 = 53.69. Par K=1 = 81.63. L = 0.95.
Contraction ratios [1.017, 1.063, 0.946] — iterations overshoot before contracting. K=3 (52.05) beats K=5 (52.22) beats K=10 (52.90).
D=4 nosub C=108 final: **51.25 PPL**. Seq K=1 = 51.42. Par K=1 = 81.67. L = 0.91.
Contraction ratios [0.727, 0.585, 0.915] — no overshoot, better convergence than block_head. K=10 (51.31) ≈ K=5 (51.25).
Sync D=4 C=108 final (kmin=2): **50.67 PPL**. Seq K=1 = 50.69. Par K=1 = 63.83. Par K=2 = 51.50. L = 0.58.
Excellent convergence — seq K=1 nearly identical to K=5. 2.09 PPL behind roformer N=4. Best among 48C²-matched look-ahead variants.
corr_ffn_add D=4 C=108 final (kmin=2): **47.76 PPL**. Seq K=1 = 47.87. Par K=1 = 68.20. Par K=2 = 48.91. L = 0.50.
Best in table but NOT param/FLOP matched (56C², 17% more FLOPs). Beats roformer N=4 (48.58) by 0.82 PPL.
Stacked N=4 C=108 final: **52.39 PPL** (no kmin). Divergent at low K: K=2=151.74, K=3=251.74, K=10=63.05, seq=63.53.
All block_head variants ran WITHOUT kmin — convergence issues likely from this. Rerunning stacked with kmin=2.

D=1 C=115 final: **53.47 PPL** (K=5). Seq K=1 = 53.68. Par K=1 = 89.95. L = 0.60.
D=2 C=111 final: **50.47 PPL** (K=5). Seq K=1 = 50.58. Par K=1 = 75.14. L = 0.42.
D=3 C=108 at 90K: 49.60. Still running.
Each D step narrows the gap to N=6 (45.05): D=1 trails by 8.4, D=2 by 5.4, D=3 by ~4.

### Depth scaling: roformer_head_ffn N=6 vs N=3 at C=446

How much does doubling depth (N=3→N=6) buy at large C?
At C=100, N=6 beat N=3 by 6.3 PPL (45.05 vs 51.38). At C=446, N=6 nearly matches N=3's final 100K number at only 25K iters.

- rhf N=3 C=446: 30,240,082 params (baseline from big machine, final 25.78)
- rhf N=6 C=446: 30,240,082 params (running on A10G)

| Iter | rhf N=6 C=446 | rhf N=3 C=446 |
|------|---------------|---------------|
| 5K   | 35.69         | 40.97         |
| 10K  | 30.74         | 35.38         |
| 15K  | 28.39         | 32.88         |
| 20K  | 26.99         | 31.46         |
| 25K  | 25.93         | 30.45         |
| 30K  | 25.14         | 29.62         |
| 35K  | 24.51         | 29.00         |
| 40K  | 23.98         | 28.52         |
| 45K  | 23.63         | 28.14         |
| 50K  | 23.28         | 27.79         |
| 55K  | 22.97         | 27.45         |
| 60K  | 22.72         | 27.19         |
| 65K  | 22.50         | 26.92         |
| 70K  | 22.30         | 26.72         |
| 75K  | 22.11         | 26.55         |
| 80K  | 21.96         | 26.29         |
| 85K  | 21.79         | 26.20         |
| 90K  | 21.66         | 26.09         |
| 95K  | 21.53         | 25.99         |
| 100K | **21.44**     | **25.78**     |

**Complete.** N=6 C=446: **21.44 PPL**. Beats N=3 (25.78) by **4.34 PPL**. Doubling depth buys substantial quality at large C.

corr_ffn C=660 at 75K: 26.18. roformer_head_ffn at 75K: 26.55. roformer N=3 at 75K: 26.82.
All three models within ~0.6 PPL of each other. At small C (50-74), corr_ffn had 6+ PPL advantage over FLOP-matched baselines. At large C (660), gap shrinks to <0.5 PPL.
Still running — need remaining corr_ffn numbers.

### Random K training: block_head_corr_ffn C=50 k_min=2 vs baseline cw=0 (K=10, block_size=256)

K sampled uniformly from [2, 10] each batch during training. Eval always uses full K=10.

| Iter | random K (k_min=2, cw=0) | baseline cw=0 | Gap  |
|------|--------------------------|---------------|------|
| 5K   | 154.34                   | 154.39        | -0.1 |
| 10K  | 122.06                   | 121.47        | +0.6 |
| 15K  | 109.95                   | 109.39        | +0.6 |
| 20K  | 103.39                   | 102.83        | +0.6 |

| 40K  | 92.32                    | 91.77         | +0.6 |
| 45K  | 90.88                    | 90.29         | +0.6 |
| 50K  | 89.86                    | 89.31         | +0.6 |
| 55K  | 88.82                    | 88.35         | +0.5 |
| 60K  | 88.11                    | 87.63         | +0.5 |
| 65K  | 87.11                    | 86.92         | +0.2 |
| 70K  | 86.77                    | 86.61         | +0.2 |
| 75K  | 86.16                    | 85.89         | +0.3 |
| 80K  | 85.69                    | 85.47         | +0.2 |
| 85K  | 85.29                    | 85.14         | +0.2 |
| 90K  | 84.92                    | 84.83         | +0.1 |
| 95K  | 84.62                    | 84.38         | +0.2 |
| 100K | **84.32**                | **84.16**     | **+0.2** |

Final diagnostics at 100K:

| Metric         | random K (k_min=2) | baseline cw=0 |
|----------------|-------------------|---------------|
| Val PPL (K=10) | 84.32             | 84.16         |
| Sequential K=1 | 84.61             | 84.19         |
| Parallel K=1   | 118.35            | 130.95        |
| Parallel K=2   | 88.59             |               |
| Parallel K=3   | 85.29             |               |
| Parallel K=5   | 84.64             | 84.50         |
| Empirical L    | 0.72              | 0.94          |

Random K improves convergence (L=0.72 vs 0.94) and parallel K=1 dramatically (118 vs 131),
but does not improve sequential K=1 (84.61 vs 84.19). Costs ~0.2 PPL at full K=10.

### K=5 training vs K=10 baseline (block_head_corr_ffn C=50, block_size=256)

Training with K=5 (n_layers=5) instead of K=10. 2x faster training, same params.

| Iter | K=5 training | K=10 baseline (cw=0) | K=10 random K (k_min=2) |
|------|-------------|---------------------|------------------------|
| 5K   | 154.01      | 154.39              | 154.34                 |
| 10K  | 121.06      | 121.47              | 122.06                 |
| 15K  | 109.26      | 109.39              | 109.95                 |
| 20K  | 102.79      | 102.83              | 103.39                 |
| 25K  | 98.51       | 98.44               | 98.99                  |
| 30K  | 95.51       | 95.41               | 96.01                  |
| 35K  | 93.53       | 93.43               | 93.98                  |
| 40K  | 91.84       | 91.77               | 92.32                  |
| 45K  | 90.42       | 90.29               | 90.88                  |
| 50K  | 89.49       | 89.31               | 89.86                  |
| 55K  | 88.52       | 88.35               | 88.82                  |
| 60K  | 87.87       | 87.63               | 88.11                  |
| 65K  | 87.11       | 86.92               | 87.11                  |
| 70K  | 86.77       | 86.61               | 86.77                  |
| 75K  | 86.16       | 85.89               | 86.16                  |
| 80K  | 85.69       | 85.47               | 85.69                  |
| 85K  | 85.29       | 85.14               | 85.29                  |
| 90K  | 84.92       | 84.83               | 84.92                  |
| 95K  | 84.62       | 84.38               | 84.62                  |
| 100K | **84.32**   | **84.16**           | **84.32**              |

Final diagnostics at 100K:

| Metric         | K=5 training | K=10 baseline (cw=0) | K=10 random K |
|----------------|-------------|---------------------|---------------|
| Val PPL        | 84.32       | 84.16               | 84.32         |
| Sequential K=1 | 84.46       | 84.19               | 84.61         |
| Parallel K=1   | 128.19      | 130.95              | 118.35        |
| Parallel K=3   | 85.75       |                     | 85.29         |
| Parallel K=5   | 84.32       | 84.50               | 84.64         |
| Empirical L    |             | 0.94                | 0.72          |

### Deep D=3 corr_ffn C=50 with random K (k_min=2) vs D=3 baseline vs roformer_head_ffn N=3

| Iter | D=3 random K | D=3 baseline (cw=0) | roformer_head_ffn N=3 |
|------|-------------|--------------------|-----------------------|
| 5K   | 139.20      | 138.39             | 134.89                |
| 10K  | 111.57      | 111.22             | 112.28                |
| 15K  | 101.56      | 101.33             | 103.01                |
| 20K  | 95.98       | 95.67              | 97.17                 |
| 25K  | 92.17       | 91.86              | 93.19                 |
| 30K  | 89.31       | 89.05              | 90.25                 |
| 35K  | 87.12       | 86.87              | 87.93                 |
| 40K  | 85.40       | 85.06              | 86.02                 |
| 45K  | 84.17       | 83.78              | 84.58                 |
| 50K  | 82.88       | 82.50              | 83.10                 |
| 55K  | 81.95       | 81.48              | 81.83                 |
| 60K  | 80.99       | 80.62              | 80.75                 |
| 65K  | 80.20       | 79.85              | 79.73                 |
| 70K  | 79.48       | 79.13              | 78.91                 |
| 75K  | 78.99       | 78.59              | 78.19                 |
| 80K  | 78.51       | 78.14              | 77.67                 |
| 85K  | 78.09       | 77.64              | 76.85                 |
| 90K  | 77.53       | 77.17              | 76.25                 |
| 95K  | 77.10       | 76.79              | 75.79                 |
| 100K | **76.83**   | **76.60**          | **75.32**             |

### Deep D=3 corr_ffn C=74 vs D=1 C=74 (block_size=256)

D=3 C=74: 44C² FLOPs/token. FLOP-matched to roformer_head_ffn N=3 C=74 (no baseline yet).

| Iter | D=3 C=74 | roformer_head_ffn N=3 C=74 | D=1 C=74 | Gap (D=3 vs rhf) |
|------|----------|-----------------------------|----------|-------------------|
| 5K   | 104.88   | 103.39                      | 117.35   | +1.5              |
| 10K  | 86.66    | 85.86                       | 96.91    | +0.8              |
| 15K  | 79.66    | 79.18                       | 88.77    | +0.5              |
| 20K  | 75.74    | 75.31                       | 84.24    | +0.4              |
| 25K  | 72.79    | 72.62                       | 81.11    | +0.2              |
| 30K  | 70.91    | 70.73                       | 78.92    | +0.2              |
| 35K  | 69.31    | 69.08                       | 77.18    | +0.2              |
| 40K  | 68.24    | 67.92                       | 75.79    | +0.3              |
| 45K  | 66.97    | 66.75                       | 74.73    | +0.2              |
| 50K  | 66.09    | 65.83                       | 73.68    | +0.3              |
| 55K  | 65.21    | 65.03                       | 72.95    | +0.2              |
| 60K  | 64.52    | 64.29                       | 72.37    | +0.2              |
| 65K  | 64.06    | 63.69                       | 71.65    | +0.4              |
| 70K  | 63.51    | 63.14                       | 71.12    | +0.4              |
| 75K  | 62.98    | 62.57                       | 70.82    | +0.4              |
| 80K  | 62.55    | 62.07                       | 70.34    | +0.5              |
| 85K  | 62.23    | 61.60                       | 69.94    | +0.6              |
| 90K  | 61.95    | 61.20                       | 69.74    | +0.8              |
| 95K  | 61.65    | 60.82                       | 69.41    | +0.8              |
| 100K | **61.21**| **60.47**                   | **69.12**| **+0.7**          |

D=3 C=74 loses to roformer_head_ffn N=3 C=74 by **0.7 PPL** (61.21 vs 60.47, 2,628,496 params each).
Gap closed from +1.5 to +0.2 around 25-60K, then widened back to +0.7 at 100K.
D=3 beats D=1 by ~8 PPL (61.21 vs 69.12) — depth matters enormously within the look-ahead architecture.

### Stacked N=3 vs Deep D=3 vs D=1 at C=74 (block_head_corr_ffn, block_size=256)

Stacked N=3: 3 units × K=5 iters, each unit has 1 block + 1 corr_ffn. More params and FLOPs than D=3.

| Iter | Stacked N=3 | D=3 | D=1 | roformer_head_ffn N=3 |
|------|-------------|------|------|----------------------|
| 5K   | 108.02      | 104.88 | 117.35 | 103.39             |
| 10K  | 88.65       | 86.66  | 96.91  | 85.86              |
| 15K  | 81.15       | 79.66  | 88.77  | 79.18              |
| 20K  | 76.68       | 75.74  | 84.24  | 75.31              |
| 25K  | 73.90       | 72.79  | 81.11  | 72.62              |
| 30K  | 71.97       | 70.91  | 78.92  | 70.73              |
| 35K  | 70.25       | 69.31  | 77.18  | 69.08              |
| 40K  | 68.92       | 68.24  | 75.79  | 67.92              |
| 45K  | 68.00       | 66.97  | 74.73  | 66.75              |
| 50K  | 67.07       | 66.09  | 73.68  | 65.83              |
| 55K  | 66.14       | 65.21  | 72.95  | 65.03              |
| 60K  | 65.63       | 64.52  | 72.37  | 64.29              |
| 65K  | 65.11       | 64.06  | 71.65  | 63.69              |
| 70K  | 64.64       | 63.51  | 71.12  | 63.14              |
| 75K  | 64.10       | 62.98  | 70.82  | 62.57              |
| 80K  | 63.84       | 62.55  | 70.34  | 62.07              |
| 85K  | 63.30       | 62.23  | 69.94  | 61.60              |
| 90K  | 62.93       | 61.95  | 69.74  | 61.20              |
| 95K  | 62.65       | 61.65  | 69.41  | 60.82              |
| 100K | **62.44**   | **61.21** | **69.12** | **60.47**       |

Stacked N=3 final: **62.44 PPL**. Trails D=3 by 1.2 PPL (61.21), trails roformer_head_ffn by 2.0 PPL (60.47). D=3 clearly better architecture.

### D=3 corr_ffn_concat v2 (tok_emb) C=74 vs D=3 corr_ffn C=74 (k_min=2, block_size=256)

concat v2: 48C² FLOPs/iter vs corr_ffn 44C² (~9% more).

| Iter | D=3 concat v2 C=74 | D=3 corr_ffn C=74 | roformer_head_ffn N=3 C=74 | Gap (concat vs rhf) |
|------|-------------------|-------------------|---------------------------|-------------------|
| 5K   | 102.09            | 104.88            | 103.39                    | -1.3              |
| 10K  | 85.15             | 86.66             | 85.86                     | -0.7              |
| 15K  | 78.63             | 79.66             | 79.18                     | -0.6              |
| 20K  | 74.74             | 75.74             | 75.31                     | -0.6              |
| 25K  | 72.10             | 72.79             | 72.62                     | -0.5              |
| 30K  | 69.97             | 70.91             | 70.73                     | -0.8              |
| 35K  | 68.42             | 69.31             | 69.08                     | -0.7              |
| 40K  | 67.13             | 68.24             | 67.92                     | -0.8              |
| 45K  | 66.08             | 66.97             | 66.75                     | -0.7              |
| 50K  | 65.15             | 66.09             | 65.83                     | -0.7              |
| 55K  | 64.37             | 65.21             | 65.03                     | -0.7              |
| 60K  | 63.73             | 64.52             | 64.29                     | -0.6              |
| 65K  | 63.17             | 64.06             | 63.69                     | -0.5              |
| 70K  | 62.62             | 63.51             | 63.14                     | -0.5              |
| 75K  | 62.19             | 62.98             | 62.57                     | -0.4              |
| 80K  | 61.79             | 62.55             | 62.07                     | -0.3              |
| 85K  | 61.30             | 62.23             | 61.60                     | -0.3              |
| 90K  | 61.07             | 61.95             | 61.20                     | -0.1              |
| 95K  | 60.73             | 61.65             | 60.82                     | -0.1              |
| 100K | **60.41**         | **61.21**         | **60.47**                 | **-0.1**          |

D=3 concat v2 C=74 final: **60.41 PPL** (2,650,400 params). Seq K=1 = 60.48 (gap +0.07). Par K=1 = 83.07. L=0.51.
Beats roformer_head_ffn (60.47) by 0.06 and D=3 corr_ffn (61.21) by 0.8.
Gap vs roformer_head_ffn narrowed from -0.8 to -0.1 at end — essentially tied. Concat v2 uses 48C² FLOPs vs roformer_head_ffn's 44C².

D=3 random K 100K diagnostics: Val PPL 76.83, Seq K=1 76.83 (exact match!), Par K=1 102.67, L=0.83, 1,728,400 params.
- Random K costs ~0.2 PPL vs D=3 baseline (76.83 vs 76.60)
- Better parallel K=1 (102.67 vs 115.55 baseline)
- Sequential K=1 matches K=10 exactly (76.83), even better than baseline (76.60 vs 76.60)
- Both D=3 variants lose to roformer_head_ffn N=3 (75.32) at 100K

### block_head_corr_ffn_concat C=50 K=5 vs corr_ffn K=5 baseline

concat variant: correction = corr_ffn(concat(ln(shift(z)), processed_x)). corr_ffn input is 2C. 24C² FLOPs vs 20C².

| Iter | corr_ffn_concat K=5 | corr_ffn K=5 (baseline) | Gap  |
|------|--------------------|-----------------------|------|
| 5K   | 148.20             | 154.01                | -5.8 |
| 10K  | 115.89             | 121.06                | -5.2 |
| 15K  | 104.68             | 109.26                | -4.6 |
| 20K  | 98.34              | 102.79                | -4.5 |
| 25K  | 94.21              | 98.51                 | -4.3 |
| 30K  | 91.48              | 95.51                 | -4.0 |
| 35K  | 89.35              | 93.53                 | -4.2 |
| 40K  | 87.75              | 91.84                 | -4.1 |
| 45K  | 86.47              | 90.42                 | -4.0 |
| 50K  | 85.37              | 89.49                 | -4.1 |
| 55K  | 84.45              | 88.52                 | -4.1 |
| 60K  | 83.83              | 87.87                 | -4.0 |
| 65K  | 83.06              | 87.11                 | -4.1 |
| 70K  | 82.50              | 86.77                 | -4.3 |
| 75K  | 82.13              | 86.16                 | -4.0 |
| 80K  | 81.69              | 85.69                 | -4.0 |
| 85K  | 81.39              | 85.29                 | -3.9 |
| 90K  | 80.99              | 84.92                 | -3.9 |
| 95K  | 80.65              | 84.62                 | -4.0 |
| 100K | **80.44**          | **84.32**             | **-3.9** |

corr_ffn_concat v1 (processed_x) final: **80.44 PPL** (1,677,100 params). L=0.71. Seq K=1 = 84.80. Par K=1 = 126.69.
- 3.9 PPL better than corr_ffn K=5 baseline (84.32)
- **BROKEN**: Sequential K=1 (84.80) is 4.4 PPL worse than val PPL (80.44). Circular dependency on processed_x.

### block_head_corr_ffn_concat v2 (tok_emb fix) C=50 K=5

Fix: use tok_emb instead of processed_x in concat input. tok_emb is constant — no circular dependency.
correction = corr_ffn(concat(ln(shift(z)), tok_emb)). Same params (1,677,100).

| Iter | concat v2 (tok_emb) | concat v1 (processed_x) | corr_ffn K=5 (baseline) | v2 vs baseline |
|------|--------------------|-----------------------|-----------------------|----------------|
| 5K   | 148.49             | 148.20                | 154.01                | -5.5           |
| 10K  | 116.73             | 115.89                | 121.06                | -4.3           |
| 15K  | 105.28             | 104.68                | 109.26                | -4.0           |
| 20K  | 98.80              | 98.34                 | 102.79                | -4.0           |
| 25K  | 94.56              | 94.21                 | 98.51                 | -4.0           |
| 30K  | 91.76              | 91.48                 | 95.51                 | -3.8           |
| 35K  | 89.60              | 89.35                 | 93.53                 | -3.9           |
| 40K  | 87.90              | 87.75                 | 91.84                 | -3.9           |
| 45K  | 86.65              | 86.47                 | 90.42                 | -3.8           |
| 50K  | 85.55              | 85.37                 | 89.49                 | -3.9           |
| 55K  | 84.65              | 84.45                 | 88.52                 | -3.9           |
| 60K  | 83.89              | 83.83                 | 87.87                 | -4.0           |
| 65K  | 83.14              | 83.06                 | 87.11                 | -4.0           |
| 70K  | 82.59              | 82.50                 | 86.77                 | -4.2           |
| 75K  | 82.25              | 82.13                 | 86.16                 | -3.9           |
| 80K  | 81.67              | 81.69                 | 85.69                 | -4.0           |
| 85K  | 81.42              | 81.39                 | 85.29                 | -3.9           |
| 90K  | 80.98              | 80.99                 | 84.92                 | -3.9           |
| 95K  | 80.73              | 80.65                 | 84.62                 | -3.9           |
| 100K | **80.42**          | **80.44**             | **84.32**             | **-3.9**       |

corr_ffn_concat v2 (tok_emb) final: **80.42 PPL** (1,677,100 params). L=0.57.
- **Sequential K=1 = 80.51** — gap of only 0.09 PPL vs val PPL (80.42). FIX WORKS.
  (v1 with processed_x had 4.4 PPL gap — broken. v2 with tok_emb has 0.09 gap — fixed.)
- Parallel K=1 = 127.93, K=2 = 85.28, K=3 = 81.27, K=5 = 80.42
- 3.9 PPL better than corr_ffn K=5 baseline (84.32)
- Contraction ratios: [0.88, 0.59, 0.57] — excellent convergence

corr_ffn K=5 baseline diagnostics: Seq K=1 = 84.46, Par K=1 = 128.19.

**Key findings:**
- **K=5 matches K=10** within 0.16 PPL (84.32 vs 84.16) with 2x faster training
- **K=5 sequential K=1** is tight: 84.46 vs 84.32 (0.14 gap)
- **Random K matches K=5** exactly (84.32) — the randomization effectively trains at avg K=6
- **Recommendation: use K=5 for training.** Nearly identical results at half the compute.
  Random K (k_min=2) adds no benefit over fixed K=5. The iterations converge fast enough
  that K>5 provides diminishing returns at C=50.

### block_head_corr_ffn_add C=50 K=5 vs concat v2 vs corr_ffn (token-blind)

add variant: correction = corr_ffn(ln(shift(z) + tok_emb)). Same 20C² FLOPs as corr_ffn, but token-aware.
concat v2: correction = corr_ffn(concat(ln(shift(z)), tok_emb)). 24C² FLOPs.

| Iter | corr_ffn_add (20C²) | concat v2 (24C²) | corr_ffn (20C²) | add vs corr_ffn | add vs concat |
|------|--------------------|-----------------|-----------------|----|-----|
| 5K   | 154.14 | 148.49 | 154.01 | +0.1 | +5.7 |
| 10K  | 120.16 | 116.73 | 121.06 | -0.9 | +3.4 |
| 15K  | 108.14 | 105.28 | 109.26 | -1.1 | +2.9 |
| 20K  | 101.43 | 98.80  | 102.79 | -1.4 | +2.6 |
| 25K  | 97.15  | 94.56  | 98.51  | -1.4 | +2.6 |
| 30K  | 94.18  | 91.76  | 95.51  | -1.3 | +2.4 |
| 35K  | 92.07  | 89.60  | 93.53  | -1.5 | +2.5 |
| 40K  | 90.31  | 87.90  | 91.84  | -1.5 | +2.4 |
| 45K  | 88.76  | 86.65  | 90.42  | -1.7 | +2.1 |
| 50K  | 87.71  | 85.55  | 89.49  | -1.8 | +2.2 |
| 55K  | 86.67  | 84.65  | 88.52  | -1.9 | +2.0 |
| 60K  | 86.00  | 83.89  | 87.87  | -1.9 | +2.1 |
| 65K  | 85.26  | 83.14  | 87.11  | -1.9 | +2.1 |
| 70K  | 84.85  | 82.59  | 86.77  | -1.9 | +2.3 |
| 75K  | 84.35  | 82.25  | 86.16  | -1.8 | +2.1 |

| 80K  | 83.92 | 81.67  | 85.69  | -1.8 | +2.3 |
| 85K  | 83.47 | 81.42  | 85.29  | -1.8 | +2.1 |
| 90K  | 83.15 | 80.98  | 84.92  | -1.8 | +2.2 |
| 95K  | 82.80 | 80.73  | 84.62  | -1.8 | +2.1 |
| 100K | **82.59** | **80.42** | **84.32** | **-1.7** | **+2.2** |

corr_ffn_add: 1,667,100 params (same as corr_ffn). concat v2: 1,677,100 params.

corr_ffn_add 100K diagnostics: Seq K=1 = 82.57 (gap -0.02!), Par K=1 = 140.57, L=0.51.
- Falls between concat v2 and token-blind corr_ffn as expected
- Beats corr_ffn by 1.7 PPL, trails concat v2 by 2.2 PPL
- Addition provides token-awareness but concatenation preserves more information
- Excellent convergence (L=0.51) and near-perfect seq K=1 match (gap 0.02)

### block_head_corr_ffn_add_px D=1 C=50 K=5 — FAILED (head sees processed_x instead of z)

Same params/FLOPs as corr_ffn_add. Only difference: classifier head reads processed_x (tok_emb + shift(correction)) instead of z (block output).

| Iter | corr_ffn_add | corr_ffn_add_px | Gap  |
|------|-------------|-----------------|------|
| 5K   | 120.52      | 159.48          | +39.0 |
| 10K  | 102.93      | 126.46          | +23.5 |
| 15K  | 97.05       | 113.62          | +16.6 |
| 20K  | 93.92       | 106.65          | +12.7 |
| 25K  | 91.63       | 101.92          | +10.3 |
| 30K  | 90.10       | 98.84           | +8.7 |
| 35K  | 88.90       | 96.45           | +7.6 |
| 40K  | 87.69       | 94.71           | +7.0 |
| 45K  | 87.02       | 93.40           | +6.4 |
| 50K  | 86.67       | 92.40           | +5.7 |
| 55K  | 86.00       | 91.15           | +5.2 |
| 60K  | 85.26       | 90.36           | +5.1 |
| 65K  | 84.85       | 89.66           | +4.8 |
| 70K  | 84.35       | 89.27           | +4.9 |
| 75K  | 83.92       | 88.81           | +4.9 |
| 80K  | 83.47       | 88.23           | +4.8 |
| 85K  | 83.15       | 88.05           | +4.9 |
| 90K  | 82.80       | 87.58           | +4.8 |
| 95K  | 82.59       | 87.26           | +4.7 |
| 100K | **82.59**   | **86.82**       | **+4.2** |

px diagnostics: Seq K=1 = 87.02, Par K=1 = 93.57, L = 0.59.
Baseline diagnostics: Seq K=1 = 82.61, Par K=1 = 115.26, L = 0.51.

**Verdict: dud.** 4.2 PPL worse at 100K. The head needs the self-inclusive context from z (block output), not just the past-only processed_x. The only upside is better parallel K=1 (93.57 vs 115.26), which is irrelevant — sequential is the real metric.

### FLOP-matched D=1 concat v2 vs roformer baselines (C=50 baseline scale)

D=1 concat v2: 24C² FLOPs/token at seq K=1.
FLOP-matched C: C=62 vs roformer N=3 C=50 (36C²=90K), C=68 vs roformer_head_ffn N=3 C=50 (44C²=110K).

**D=1 concat v2 C=62 vs roformer N=3 C=50 (90K FLOPs)**

| Iter | D=1 concat v2 C=62 | roformer N=3 C=50 | Gap |
|------|-------------------|------------------|-----|
| 5K   | 127.56 | 137.04 | -9.5 |
| 10K  | 103.51 | 113.90 | -10.4 |
| 15K  | 94.42  | 104.57 | -10.2 |
| 20K  | 89.03  | 99.12  | -10.1 |
| 25K  | 85.54  | 95.52  | -10.0 |
| 30K  | 83.11  | 92.83  | -9.7 |
| 35K  | 81.36  | 90.71  | -9.4 |
| 40K  | 79.86  | 88.90  | -9.0 |
| 45K  | 78.58  | 87.24  | -8.7 |
| 50K  | 77.65  | 85.95  | -8.3 |
| 55K  | 76.81  | 84.93  | -8.1 |
| 60K  | 76.14  | 83.79  | -7.7 |
| 65K  | 75.51  | 82.86  | -7.4 |
| 70K  | 75.02  | 82.14  | -7.1 |
| 75K  | 74.58  | 81.30  | -6.7 |
| 80K  | 74.12  | 80.68  | -6.6 |
| 85K  | 73.80  | 80.00  | -6.2 |
| 90K  | 73.34  | 79.31  | -6.0 |
| 95K  | 73.05  | 78.63  | -5.6 |
| 100K | **72.71** | **78.24** | **-5.5** |

D=1 concat v2 C=62 final: **72.71 PPL** (2,093,620 params). Seq K=1 = 72.82 (gap 0.11). L=0.40.
Beats roformer N=3 C=50 by **5.5 PPL** at FLOP parity.

**D=1 concat v2 C=68 vs roformer_head_ffn N=3 C=50 (110K FLOPs)**

| Iter | D=1 concat v2 C=68 | roformer_head_ffn N=3 C=50 | Gap |
|------|-------------------|---------------------------|-----|
| 5K   | 119.77 | 134.89 | -15.1 |
| 10K  | 97.86  | 112.28 | -14.4 |
| 15K  | 89.40  | 103.01 | -13.6 |
| 20K  | 84.76  | 97.17  | -12.4 |
| 25K  | 81.74  | 93.19  | -11.5 |
| 30K  | 79.37  | 90.25  | -10.9 |
| 35K  | 77.76  | 87.93  | -10.2 |
| 40K  | 76.50  | 86.02  | -9.5 |
| 45K  | 75.14  | 84.58  | -9.4 |
| 50K  | 74.30  | 83.10  | -8.8 |
| 55K  | 73.43  | 81.83  | -8.4 |
| 60K  | 72.69  | 80.75  | -8.1 |
| 65K  | 72.14  | 79.73  | -7.6 |
| 70K  | 71.62  | 78.91  | -7.3 |
| 75K  | 71.11  | 78.19  | -7.1 |
| 80K  | 70.70  | 77.67  | -7.0 |
| 85K  | 70.52  | 76.85  | -6.3 |
| 90K  | 69.98  | 76.25  | -6.3 |
| 95K  | 69.64  | 75.79  | -6.2 |
| 100K | **69.36** | **75.32** | **-6.0** |

D=1 concat v2 C=68 final: **69.36 PPL** (2,357,540 params). Seq K=1 = 69.43 (gap 0.07). L=0.50.
Beats roformer_head_ffn N=3 C=50 by **6.0 PPL** at FLOP parity.

Convergence check (re-run with K=10 eval): K=5 PPL 69.54, K=10 PPL 69.54 — converged. Seq K=1 = 69.63.
Parallel K breakdown: K=1=105.53, K=2=71.79, K=3=69.84, K=5=69.54, K=10=69.54.

**Key finding**: A single shared-weight block with concat v2 correction crushes 3-layer roformer at FLOP parity.
Gap narrows over training (from ~10-15 early to ~5.5-6.0 at 100K) but remains very large.


### FLOP-matched D=3 concat v2 vs roformer baselines (C=50 baseline scale)

D=3 concat v2: 48C² FLOPs/token at seq K=1.
FLOP-matched C: C=44 vs roformer N=3 C=50 (36C²=90K), C=48 vs roformer_head_ffn N=3 C=50 (44C²=110K).

**D=3 concat v2 C=44 vs roformer N=3 C=50 (90K FLOPs)**

| Iter | D=3 concat v2 C=44 | roformer N=3 C=50 | Gap |
|------|-------------------|------------------|-----|
| 5K   | 148.39 | 137.04 | +11.4 |
| 10K  | 117.84 | 113.90 | +3.9 |
| 15K  | 106.80 | 104.57 | +2.2 |
| 20K  | 100.66 | 99.12  | +1.5 |
| 25K  | 96.62  | 95.52  | +1.1 |
| 30K  | 93.68  | 92.83  | +0.9 |
| 35K  | 91.36  | 90.71  | +0.7 |
| 40K  | 89.83  | 88.90  | +0.9 |
| 45K  | 88.24  | 87.24  | +1.0 |
| 50K  | 87.12  | 85.95  | +1.2 |
| 55K  | 86.09  | 84.93  | +1.2 |
| 60K  | 85.50  | 83.79  | +1.7 |

D=3 concat v2 C=44 losing to roformer N=3 C=50. C=44 too small for 3 blocks. Still running.

### block_head D=3 vs roformer N=3 (both 3 layers at inference, C=50)

| Iter | block_head D=3 | roformer N=3 | Gap   |
|------|----------------|--------------|-------|
| 5K   | 152.39         | 137.04       | +15.4 |
| 10K  | 118.06         | 113.90       | +4.2  |
| 15K  | 105.88         | 104.57       | +1.3  |
| 20K  | 99.27          | 99.12        | +0.2  |
| 25K  | 95.15          | 95.52        | -0.4  |
| 30K  | 92.22          | 92.83        | -0.6  |
| 35K  | 90.09          | 90.71        | -0.6  |
| 40K  | 88.46          | 88.90        | -0.4  |
| 45K  | 87.09          | 87.24        | -0.2  |
| 50K  | 85.99          | 85.95        | +0.0  |
| 55K  | 85.10          | 84.93        | +0.2  |
| 60K  | 84.43          | 83.79        | +0.6  |
| 65K  | 83.82          | 82.86        | +1.0  |
| 70K  | 83.30          | 82.14        | +1.2  |
| 75K  | 82.76          | 81.30        | +1.5  |
| 80K  | 82.38          | 80.68        | +1.7  |
| 85K  | 81.99          | 80.00        | +2.0  |
| 90K  | 81.65          | 79.31        | +2.3  |
| 95K  | 81.19          | 78.63        | +2.6  |
| 100K | **80.94**      | **78.24**    | **+2.7** |

### Inference FLOP matching: block_head vs roformer

At inference, block_head uses 1 block application (sequential K=1), roformer N uses N block applications.

**FLOPs per block per token** (dominant linear projection cost):
- QKV projection: 3C² multiplies
- Attention output projection: C² multiplies
- FFN (up + down): 4C² + 4C² = 8C² multiplies
- **Total: 12C² multiplies per block per token**

**FLOP matching**: to match roformer N=3 at C=50:
- roformer N=3 C=50: 3 × 12 × 50² = 90,000 multiplies/token
- block_head C=86: 1 × 12 × 86² = 88,752 multiplies/token (≈ equal)

block_head C=86 has more params (2,858K vs 1,708K) due to the larger embedding table (vocab × C),
but the same inference compute. If it matches roformer N=3's PPL (78.24), the architecture
trades params for speed — more weights stored, fewer FLOPs at inference.

### FLOP-matched: block_head C=86 vs roformer N=3 C=50

| Iter | block_head C=86 | roformer N=3 C=50 | Gap    |
|------|-----------------|-------------------|--------|
| 5K   | 119.18          | 137.04            | -17.9  |
| 10K  | 99.24           | 113.90            | -14.7  |
| 15K  | 91.20           | 104.57            | -13.4  |
| 20K  | 86.24           | 99.12             | -12.9  |
| 25K  | 83.00           | 95.52             | -12.5  |
| 30K  | 80.62           | 92.83             | -12.2  |
| 35K  | 78.90           | 90.71             | -11.8  |
| 40K  | 77.56           | 88.90             | -11.3  |
| 45K  | 76.36           | 87.24             | -10.9  |
| 50K  | 75.43           | 85.95             | -10.5  |
| 55K  | 74.59           | 84.93             | -10.3  |
| 60K  | 74.02           | 83.79             | -9.8   |
| 65K  | 73.56           | 82.86             | -9.3   |
| 70K  | 72.76           | 82.14             | -9.4   |
| 75K  | 72.50           | 81.30             | -8.8   |
| 80K  | 72.14           | 80.68             | -8.5   |
| 85K  | 71.80           | 80.00             | -8.2   |
| 90K  | 71.42           | 79.31             | -7.9   |
| 95K  | 71.18           | 78.63             | -7.5   |
| 100K | **70.97**       | **78.24**         | **-7.3** |

block_head C=86: 2,858K params, roformer N=3 C=50: 1,708K params.
Same inference FLOPs (~89-90K multiplies/token). block_head beats roformer by 7.3 PPL (70.97 vs 78.24).
block_head crossed roformer's 100K final (78.24) at ~35K iters.

### FLOP-matched: block_head C=850 vs roformer N=3 C=492

| Iter | block_head C=850 | roformer N=3 C=492 | Gap   |
|------|------------------|--------------------|-------|
| 5K   | 40.47            | 41.38              | -0.9  |
| 10K  | 35.70            | 35.81              | -0.1  |
| 15K  | 33.34            | 33.33              | +0.0  |
| 20K  | 32.15            | 31.77              | +0.4  |
| 25K  | 30.92            | 30.69              | +0.2  |
| 30K  | 30.21            | 29.98              | +0.2  |
| 35K  | 29.50            | 29.32              | +0.2  |
| 40K  | 29.05            | 28.87              | +0.2  |
| 45K  | 28.50            | 28.44              | +0.1  |
| 50K  | 28.15            | 28.09              | +0.1  |
| 55K  | 27.79            | 27.82              | -0.0  |
| 60K  | 27.44            | 27.49              | -0.1  |
| 65K  | 27.26            | 27.23              | +0.0  |
| 70K  | 27.01            | 27.01              | +0.0  |
| 75K  | 26.73            | 26.82              | -0.1  |
| 80K  | 26.50            | 26.69              | -0.2  |
| 85K  | 26.36            | 26.51              | -0.2  |
| 90K  | 26.15            | 26.39              | -0.2  |
| 95K  | 25.98            | 26.26              | -0.3  |
| 100K | **25.91**        | **26.12**          | **-0.2** |

block_head C=850 (35.9M params) vs roformer N=3 C=492 (20.2M params).
Same inference FLOPs (~8.7M multiplies/token). Nearly identical PPL through 25K.

### Param-matched: block_head C=850 vs roformer N=3 C=648 (~35.9M params each)

| Iter | block_head C=850 | roformer N=3 C=648 | Gap   |
|------|------------------|--------------------|-------|
| 5K   | 40.47            | 37.59              | +2.9  |
| 10K  | 35.70            | 32.55              | +3.2  |
| 15K  | 33.34            | 30.24              | +3.1  |
| 20K  | 32.15            | 28.83              | +3.3  |
| 25K  | 30.92            | 27.89              | +3.0  |
| 30K  | 30.21            | 27.18              | +3.0  |
| 35K  | 29.50            | 26.57              | +2.9  |
| 40K  | 29.05            | 26.01              | +3.0  |
| 45K  | 28.50            | 25.66              | +2.8  |
| 50K  | 28.15            | 25.35              | +2.8  |
| 55K  | 27.79            | 25.06              | +2.7  |
| 60K  | 27.44            | 24.76              | +2.7  |
| 65K  | 27.26            | 24.57              | +2.7  |
| 70K  | 27.01            | 24.37              | +2.6  |
| 75K  | 26.73            | 24.18              | +2.6  |
| 80K  | 26.50            | 24.06              | +2.4  |
| 85K  | 26.36            | 23.92              | +2.4  |
| 90K  | 26.15            | 23.75              | +2.4  |
| 95K  | 25.98            | 23.62              | +2.4  |
| 100K | **25.91**        | **23.49**          | **+2.4** |

Both ~35.9M params. block_head uses 1 block at inference, roformer uses 3.

### block_head C=850 final diagnostics (100K)

| Metric | PPL |
|--------|-----|
| Final val (K=10) | 25.94 |
| Sequential K=1 | 28.99 |
| Parallel K=1 | 84.07 |
| Parallel K=2 | 49.50 |
| Parallel K=3 | 41.44 |
| Parallel K=5 | 38.98 |

Sequential K=1 gap: ~3 PPL vs full K=10. Iterations don't converge smoothly — ratios at iterations 6-7 consistently expand (1.3-1.9) before contracting back. L stabilized ~0.52 after 25K. Oscillating, not converging.

### Prior baselines (C=50, block_size=256, 100K iters)

| Model | Params | 100K PPL |
|---|---|---|
| roformer N=5 | 1,769,350 | 70.89 |
| concat head (D=1 K=10) | 2,446,850 | 82.29 |
| projhead (D=1 K=10) | 1,651,800 | 87.06 |
| corrhead (D=1 K=10) | 1,646,750 | 97.38 |
| roformer N=1 | 1,646,750 | 100.24 (block_size=64) |

### block_head_corr_ffn / delta_ffn D=1 vs block_head_ffn vs roformer_head_ffn N=1 (C=50, K=10, block_size=256)

block_head_corr_ffn: correction = corr_ffn(ln(z)). block_head_delta_ffn: correction = corr_ffn(ln(z - processed_x)).

| Iter | block_head_corr_ffn | block_head_delta_ffn | block_head_ffn | roformer_head_ffn N=1 | block_head | roformer N=3 |
|------|---------------------|----------------------|----------------|----------------------|------------|--------------|
| 5K   | 152.10              | 157.61               | 152.88         | 152.52               | 165.33     | 137.04       |
| 10K  | 120.40              | 125.09               | 121.56         | 131.17               | 128.43     | 113.90       |
| 15K  | 109.05              | 112.98               | 110.34         | 120.76               | 115.69     | 104.57       |
| 20K  | 102.60              | 106.49               | 103.93         | 113.77               | 109.15     | 99.12        |
| 25K  | 98.29               | 102.04               | 99.59          | 108.78               | 104.93     | 95.52        |
| 30K  | 95.30               | 99.00                | 96.59          | 105.06               | 102.12     | 92.83        |
| 35K  | 93.18               | 96.78                | 94.32          | 101.95               | 100.16     | 90.71        |
| 40K  | 91.47               | 94.99                | 92.53          | 99.71                | 98.52      | 88.90        |
| 45K  | 90.03               | 93.56                | 91.17          | 97.57                | 97.31      | 87.24        |
| 50K  | 89.01               | 92.63                | 90.08          | 95.93                | 96.42      | 85.95        |
| 55K  | 88.08               | 91.50                | 88.96          | 94.52                | 95.46      | 84.93        |
| 60K  | 87.41               | 90.79                | 88.18          | 93.39                | 94.85      | 83.79        |
| 65K  | 86.73               | 90.18                | 87.47          | 92.27                | 94.28      | 82.86        |
| 70K  | 86.44               | 89.85                | 87.05          | 91.18                | 93.72      | 82.14        |
| 75K  | 85.75               | 89.32                | 86.45          | 90.29                | 93.35      | 81.30        |
| 80K  | 85.33               | 88.59                | 85.91          | 89.67                | 92.97      | 80.68        |
| 85K  | 85.15               | 88.39                | 85.61          | 89.03                | 92.64      | 80.00        |
| 90K  | 84.75               | 88.06                | 85.22          | 88.32                | 92.38      | 79.31        |
| 95K  | 84.35               | 87.71                | 84.88          | 87.70                | 92.19      | 78.63        |
| 100K | **84.20**           | **87.32**            | **84.53**      | **87.29**            | **91.85**  | **78.24**    |

### Final diagnostics (D=1 variants, 100K)

| Metric | block_head_corr_ffn | block_head_delta_ffn | block_head_ffn |
|--------|---------------------|----------------------|----------------|
| Val PPL (K=10) | 84.17 | 87.32 | 84.53 |
| Sequential K=1 | 84.20 | 87.38 | 84.57 |
| Parallel K=1 | 130.06 | 127.60 | 128.47 |
| Empirical L | ~0.8-1.0 | 0.98 | 1.20 |
| Params | 1,667,100 | 1,667,100 | ~1,657,000 |

corr_ffn: best PPL, excellent sequential match (84.20 vs 84.17). L oscillates 0.75-0.99.
delta_ffn: better convergence (L=0.98) but worse PPL. FFN on delta stabilizes but limits expressiveness.

### D=1 C=50 corr_ffn_add vs delta_ffn_add (K=5, k_min=2, block_size=256, 10K iters)

| Iter | corr_ffn_add | delta_ffn_add | Gap |
|------|--------------|---------------|-------|
| 1K   | 445.69       | 446.15        | +0.46 |
| 2K   | 269.89       | 267.10        | -2.79 |
| 3K   | 198.38       | 197.94        | -0.44 |
| 4K   | 168.84       | 169.93        | +1.09 |
| 5K   | 153.06       | 155.01        | +1.95 |
| 6K   | 142.55       | 145.08        | +2.53 |
| 7K   | 135.01       | 138.01        | +3.00 |
| 8K   | 129.06       | 132.03        | +2.97 |
| 9K   | 124.80       | 127.89        | +3.09 |
| 10K  | 121.14       | 124.31        | +3.17 |

corr_ffn_add wins by 3.17 PPL at 10K. Delta_ffn_add starts competitive but falls behind — shifting the delta (z - processed_x) instead of z directly loses information. The gap is widening at 10K, so it would likely be worse at 100K.

### D=1 C=50 variant comparison: corr_ffn_add, tied, pure variants (K=5, k_min=2, block_size=256, 10K iters)

Testing whether tying the correction FFN to the block's FFN (saving 8C² params) or using a "pure" residual pattern (f(tok_emb, shift(z)) instead of f(processed_x, shift(z))) helps or hurts.

**Variants tested:**
- **corr_ffn_add** (20C²): correction = corr_ffn(ln_corr(shift(z) + tok_emb)). Separate corr_ffn.
- **corr_ffn_add_tied** (12C²): Same as add but corr_ffn = block.ffn, ln_corr = block.ln2. Shares FFN weights.
- **corr_ffn_add_pure** (20C²): processed_x = tok_emb + shift(z) + corr_ffn(ln_corr(tok_emb + shift(z))). Uses f(tok_emb, shift(z)) pattern.
- **corr_ffn_add_tied_pure** (12C²): Same as add_pure but corr_ffn = block.ffn, ln_corr = block.ln2.

**Training curves:**

| Iter | corr_ffn_add | corr_ffn_add_tied | add_pure | add_tied_pure |
|------|--------------|-------------------|----------|---------------|
| 1K   | 445.07       | 426.36            | 432.55   | 437.75        |
| 2K   | 270.08       | 265.66            | 308.79   | 316.47        |
| 3K   | 198.59       | 203.35            | 233.24   | 240.06        |
| 4K   | 168.86       | 176.58            | 194.77   | 200.33        |
| 5K   | 153.04       | 161.50            | 173.69   | 178.73        |
| 6K   | 142.63       | 151.51            | 160.06   | 165.20        |
| 7K   | 135.02       | 143.95            | 150.82   | 156.24        |
| 8K   | 129.17       | 137.85            | 143.95   | 148.93        |
| 9K   | 124.71       | 133.16            | 138.75   | 143.74        |
| 10K  | 120.96       | 129.43            | 134.27   | 139.13        |

**Diagnostics at 10K:**

| Metric | corr_ffn_add | corr_ffn_add_tied | add_pure | add_tied_pure |
|--------|--------------|-------------------|----------|---------------|
| Final (K=5) | 120.96 | 129.43 | 134.27 | 139.13 |
| Seq K=1 | 120.95 | 129.44 | 135.07 | 140.31 |
| Empirical L | 0.44 | 0.42 | 0.88 | 0.91 |

**Full comparison at 10K (all D=1 C=50 variants):**

| Model | FLOPs/iter | 10K PPL | Seq K=1 | L |
|-------|-----------|---------|---------|---|
| corr_ffn_concat | 24C² | 117.25 | — | — |
| corr_ffn_add | 20C² | 120.96 | 120.95 | 0.44 |
| corr_ffn_add_tied | 12C² | 129.43 | 129.44 | 0.42 |
| block_head_recompute | 12C² (20C² FLOPs) | 129.38 | 129.42 | 0.47 |
| block_head | 12C² | 129.97 | 130.04 | 0.60 |
| block_aligned | 12C² | 133.98 | — | — |
| add_pure | 20C² | 134.27 | 135.07 | 0.88 |
| add_tied_pure | 12C² | 139.13 | 140.31 | 0.91 |

**Key findings:**
- **Pure variants are dead — direct skip connection defeats contraction.** The f(tok_emb, shift(z)) pattern has a direct path from shift(z) to processed_x. Since z ≈ processed_x + delta (block residuals), this creates a near-identity iteration map (L ≈ 0.9). The previous iteration's processed_x leaks through shift(z) without being compressed or subtracted out. Block_head avoids this via `z - processed_x` (cancels identity). corr_ffn_add avoids this by routing through the FFN bottleneck (no direct skip). Pure has neither mechanism.
- **Tied ≈ block_head at 12C².** corr_ffn_add_tied (129.43) ≈ block_head (129.97). Shared FFN can't specialize — must serve both representation enrichment (inside block) and correction generation (dual objective compromise).
- **Separate corr_ffn earns its 8C².** The 20C²→12C² gap is ~9 PPL (120.96 vs 129.43). Independent weights let the correction FFN specialize.
- **Block_head is optimal at 12C².** Delta subtraction is the cheapest contraction mechanism — zero extra params.
- **Hierarchy is clear:** concat (24C²) > add (20C²) > tied/recompute/block_head (12C²) >> pure variants.

### block_head_recompute vs block_head (C=50, K=5, k_min=2, block_size=256)

block_head_recompute: shifts delta (z-x), reapplies block.ffn at destination with tok_emb.
12C² params (same as block_head), 20C² FLOPs (FFN runs twice per iteration).

| Iter | block_head | block_head_recompute | Gap  |
|------|------------|---------------------|------|
| 5K   | 163.93     | 161.01              | -2.9 |
| 10K  | 130.01     | 129.26              | -0.8 |
| 15K  | 117.80     | 117.55              | -0.3 |
| 20K  | 110.89     | 111.15              | +0.3 |
| 25K  | 106.35     | 106.73              | +0.4 |
| 30K  | 103.48     | 103.65              | +0.2 |
| 35K  | 100.98     | 101.36              | +0.4 |
| 40K  | 99.27      | 99.47               | +0.2 |
| 45K  | 97.94      | 98.14               | +0.2 |
| 50K  | 96.99      | 97.01               | +0.0 |
| 55K  | 96.12      | 95.98               | -0.1 |
| 60K  | 95.32      | 95.21               | -0.1 |
| 65K  | 94.80      | 94.64               | -0.2 |
| 70K  | 94.15      | 94.02               | -0.1 |
| 75K  | 93.86      | 93.42               | -0.4 |
| 80K  | 93.34      | 92.97               | -0.4 |
| 85K  | 92.87      | 92.49               | -0.4 |
| 90K  | 92.63      | 92.29               | -0.3 |
| 95K  | 92.27      | 91.93               | -0.3 |
| 100K | 91.97      | 91.54               | -0.4 |

100K diagnostics: recompute Final=91.54, seq K=1=91.57, L=0.57. block_head Final=91.97, seq K=1=92.08, L=0.54.
Essentially identical — recompute is ~0.4 PPL better at 100K, well within noise.
Recompute wins early (gap narrows by 20K), then they oscillate, then recompute pulls slightly ahead late.
Token awareness via recompute provides no meaningful benefit when the FFN is tied.
The gap to corr_ffn_add (separate FFN) is the real difference — independent weights matter, not token awareness.

### k_min training: robustness across depths (stacked_block_head N=4 D=3 C=108, 100K iters)

Random K training: each batch samples K uniformly from [k_min, K_max]. Eval always at full K.
k_min=2 means the model always sees at least 2 iterations, so corrections are always meaningful.

**Depth robustness comparison (4.04M params):**

| K (eval) | la6 (k_min=2) | la5 (no k_min) |
|----------|---------------|----------------|
| 1        | 61.22         | 83.65          |
| 2        | 55.29         | 151.74         |
| 3        | 55.00         | 251.74         |
| 5 (full) | 55.18         | 52.39          |
| 10       | 55.30         | 63.05          |
| seq      | 55.30         | 63.53          |

Night and day difference. Without k_min, K=2 and K=3 completely diverge (151, 251 PPL).
With k_min=2, all depths K=2..10 are within 0.3 PPL of each other. K=1 also much better (61 vs 84).

**Training curve:**

| Iter | k_min=2 (la6) | no k_min (la5) |
|------|---------------|----------------|
| 65K  | 57.15         | 54.45          |
| 70K  | 56.75         | 54.00          |
| 75K  | 56.43         | 53.60          |
| 100K | 55.18         | 52.39          |

**Tradeoff:** k_min=2 costs ~2.8 PPL at full K=5 (55.18 vs 52.39), but the model is usable at any depth.
K=3 (55.00) is slightly better than K=5 (55.18) — the model converges faster with k_min training.

**vs concat D=3 (param-matched, 4.04M):**

| Model | Final PPL | K=2 | K=3 | Params |
|-------|-----------|------|------|--------|
| concat C=108 D=3 | 48.98 | 50.25 | 49.10 | 4.04M |
| stacked_block_head k_min=2 | 55.18 | 55.29 | 55.00 | 4.04M |

### Large-scale comparison: add vs concat vs stacked concat (D=3, C=446, K=5, 100K iters)

| Iter | Add (K=5) | Concat v2 (K=5) | Stacked concat v2 (K=5) |
|------|-----------|-----------------|-------------------------|
| 5K   | 39.42     | 38.51           | 36.60                   |
| 10K  | 33.81     | 33.00           | 31.38                   |
| 15K  | 31.43     | 30.70           | 28.79                   |
| 20K  | 29.88     | 29.21           | 27.51                   |
| 25K  | 28.77     | 28.20           | 26.54                   |
| 30K  | 27.87     | 27.47           | 25.75                   |
| 35K  | 27.23     | 26.80           | 25.14                   |
| 40K  | 26.74     | 26.28           | 24.68                   |
| 45K  | 26.36     | 25.83           | 24.25                   |
| 50K  | 25.98     | 25.49           | 23.85                   |
| 55K  | 25.60     | 25.06           | 23.56                   |
| 60K  | 25.32     | 24.83           | 23.27                   |
| 65K  | 25.08     | 24.68           | 23.11                   |
| 70K  | 24.82     | 24.38           | 22.96                   |
| 75K  | 24.67     | 24.21           | 22.70                   |
| 80K  | 24.45     | 23.91           | 22.52                   |
| 85K  | 24.31     | 23.78           | 22.33                   |
| 90K  | 24.15     | 23.62           | 22.15                   |
| 95K  | 23.97     | 23.57           | 22.09                   |
| 100K | 23.82     | 23.39           | 21.96                   |

Add = block_head_corr_ffn_add D=3 C=446. Concat v2 = D=1 concat v2 C=446. Stacked concat v2 = stacked D=3.
Add vs concat gap: ~0.4-0.5 PPL throughout. Stacked concat v2 wins by ~1.4 over concat, ~1.9 over add.

Concat D=3 still leads by ~6 PPL at param parity. The concat head's architectural advantage persists.

### roformer N=3 C=446 baseline (21,467,262 params, 36C² FLOPs/token, 100K iters)

| Iter | Val PPL |
|------|---------|
| 5K   | 43.00   |
| 10K  | 37.00   |
| 15K  | 34.41   |
| 20K  | 32.89   |
| 25K  | 31.88   |
| 30K  | 31.08   |
| 35K  | 30.42   |
| 40K  | 29.96   |
| 45K  | 29.50   |
| 50K  | 29.19   |
| 55K  | 28.83   |
| 60K  | 28.55   |
| 65K  | 28.37   |
| 70K  | 28.20   |
| 75K  | 27.89   |
| 80K  | 27.75   |
| 85K  | 27.56   |
| 90K  | 27.42   |
| 95K  | 27.28   |
| 100K | 27.19   |

Param-matched to block_head D=3 C=446 and stacked_block_head N=3 C=446 (both 36C² block params).
roformer_head_ffn N=3 C=446 (25.78) has 8C² more params from the head FFN.

### block_head D=3 C=446 K=5 k_min=2 (21,467,262 params, 100K iters)

| Iter | block_head D=3 | roformer N=3 | Gap   |
|------|----------------|--------------|-------|
| 5K   | 44.19          | 43.00        | +1.2  |
| 10K  | 37.90          | 37.00        | +0.9  |
| 15K  | 35.23          | 34.41        | +0.8  |
| 20K  | 33.55          | 32.89        | +0.7  |
| 25K  | 32.57          | 31.88        | +0.7  |
| 30K  | 31.50          | 31.08        | +0.4  |
| 35K  | 31.39          | 30.42        | +1.0  |
| 40K  | 30.43          | 29.96        | +0.5  |
| 45K  | 29.88          | 29.50        | +0.4  |
| 50K  | 29.63          | 29.19        | +0.4  |
| 55K  | 29.07          | 28.83        | +0.2  |
| 60K  | 28.99          | 28.55        | +0.4  |
| 65K  | 28.79          | 28.37        | +0.4  |
| 70K  | 28.36          | 28.20        | +0.2  |
| 75K  | 28.10          | 27.89        | +0.2  |
| 80K  | 27.96          | 27.75        | +0.2  |
| 85K  | 27.75          | 27.56        | +0.2  |
| 90K  | 27.48          | 27.42        | +0.1  |
| 95K  | 27.45          | 27.28        | +0.2  |
| 100K | 27.32          | 27.19        | +0.1  |

Seq K=1 = 28.46. block_head D=3 essentially ties roformer at param parity (27.32 vs 27.19).
Without corr_ffn, weight sharing alone barely matches separate weights.

### block_head_corr_ffn D=3 C=446 K=5 k_min=2 (23,061,712 params, 100K iters)

| Iter | corr_ffn D=3 | block_head D=3 | roformer N=3 | Gap (corr vs roformer) |
|------|-------------|----------------|--------------|------------------------|
| 5K   | 39.84       | 44.19          | 43.00        | -3.2                   |
| 10K  | 34.16       | 37.90          | 37.00        | -2.8                   |
| 15K  | 31.56       | 35.23          | 34.41        | -2.9                   |
| 20K  | 29.98       | 33.55          | 32.89        | -2.9                   |
| 25K  | 28.91       | 32.57          | 31.88        | -3.0                   |
| 30K  | 28.07       | 31.50          | 31.08        | -3.0                   |
| 35K  | 27.59       | 31.39          | 30.42        | -2.8                   |
| 40K  | 26.94       | 30.43          | 29.96        | -3.0                   |
| 45K  | 26.53       | 29.88          | 29.50        | -3.0                   |
| 50K  | 26.17       | 29.63          | 29.19        | -3.0                   |
| 55K  | 25.82       | 29.07          | 28.83        | -3.0                   |
| 60K  | 25.51       | 28.99          | 28.55        | -3.0                   |
| 65K  | 25.32       | 28.79          | 28.37        | -3.1                   |
| 70K  | 25.00       | 28.36          | 28.20        | -3.2                   |
| 75K  | 24.89       | 28.10          | 27.89        | -3.0                   |
| 80K  | 24.66       | 27.96          | 27.75        | -3.1                   |
| 85K  | 24.54       | 27.75          | 27.56        | -3.0                   |
| 90K  | 24.39       | 27.48          | 27.42        | -3.0                   |
| 95K  | 24.26       | 27.45          | 27.28        | -3.0                   |
| 100K | 23.98       | 27.32          | 27.19        | -3.2                   |

100K diagnostics:

| Model | Final PPL | Seq K=1 | Par K=1 | Params |
|-------|-----------|---------|---------|--------|
| block_head_corr_ffn D=3 | 23.98 | 23.96 | 35.60 | 23.06M (+8C²) |
| block_head D=3 | 27.36 | 28.46 | 49.91 | 21.47M |
| roformer N=3 | 27.19 | — | — | 21.47M |
| roformer_head_ffn N=3 | 25.78 | — | — | 23.06M (+8C²) |

corr_ffn D=3 beats roformer by 3.21 PPL and roformer_head_ffn by 1.80 PPL.
block_head D=3 ties roformer (27.36 vs 27.19) — weight sharing alone doesn't help.
The 3.4 PPL gap (block_head → corr_ffn) is what the extra 8C² corr_ffn buys.

### D=3 C=446 comparison (all param-matched to roformer except where noted)

| Model | Final PPL | Seq K=1 | Params | Notes |
|-------|-----------|---------|--------|-------|
| Stacked concat v2 | 21.96 | 23.28 | 23.06M* | *extra params from concat head |
| D=3 corr_ffn | 23.18 | 23.23 | 23.06M* | *K=10, +8C² from corr_ffn |
| D=3 concat v2 | 23.39 | 23.73 | 23.06M* | *+8C² from corr_ffn |
| D=3 corr_ffn_add | 23.82 | 24.14 | 23.06M* | *+8C² from corr_ffn |
| roformer_head_ffn N=3 | 25.78 | — | 23.06M* | *+8C² from head_ffn |
| roformer N=3 | 27.19 | — | 21.47M | baseline |
| block_head D=3 | 27.32 | 28.46 | 21.47M | param-matched |

The corr_ffn is what makes look-ahead beat roformer — without it, block_head just matches.

### stacked_block_head_corr_ffn N=3 C=446 K=10 (26,250,612 params = 60C², 100K iters)

No k_min, no random K. Param-matched to roformer N=5 (60C²).

| Iter | stacked corr_ffn N=3 |
|------|---------------------|
| 5K   | 42.35               |
| 10K  | 36.64               |
| 15K  | 34.00               |
| 20K  | 32.22               |
| 25K  | 31.13               |
| 30K  | 30.23               |
| 35K  | 29.49               |
| 40K  | 28.80               |
| 45K  | 28.34               |
| 50K  | 27.99               |
| 55K  | 27.49               |
| 60K  | 27.32               |
| 65K  | 26.94               |
| 70K  | 26.62               |
| 75K  | 26.37               |
| 80K  | 26.15               |
| 85K  | 26.08               |
| 90K  | 25.90               |
| 95K  | 25.60               |
| 100K | 25.50               |

Empirical L stable at ~0.005 — pristine convergence across all 3 units.
Awaiting roformer N=5 C=446 baseline for comparison.

---

## OpenWebText C=1024 Experiments (vocab=32000)

Scaling test on OpenWebText (~9.1B tokens, vocab=32000) with C=1024.
All runs: block_size=256, batch_size=64, lr=2e-4, AMP (bfloat16), 100K iters.

### Model configurations

| Model | Params | FLOP budget | Notes |
|-------|-------:|-------------|-------|
| roformer N=6 | ~113M | 72C² | Baseline, 6 separate layers |
| block_head_corr_ffn_add D=3 | ~112M | 44C² | 3 shared blocks × 5 iters, additive correction |

### Roformer N=6 C=1024 OWT (completed)

| Iter | Val PPL |
|-----:|--------:|
| 5K   | 88.14   |
| 10K  | 68.00   |
| 15K  | 60.11   |
| 20K  | 55.80   |
| 25K  | 52.80   |
| 30K  | 50.75   |
| 35K  | 48.91   |
| 40K  | 47.72   |
| 45K  | 46.39   |
| 50K  | 45.56   |
| 55K  | 44.72   |
| 60K  | 44.05   |
| 65K  | 43.39   |
| 70K  | 42.84   |
| 75K  | 42.42   |
| 80K  | 41.80   |
| 85K  | 41.42   |
| 90K  | 40.97   |
| 95K  | 40.78   |
| 100K | 40.48   |

### block_head_corr_ffn_add D=3 C=1024 OWT (completed)

| Iter | Val PPL |
|-----:|--------:|
| 5K   | 102.37  |
| 10K  | 77.40   |
| 15K  | 67.51   |
| 20K  | 61.68   |
| 25K  | 58.10   |
| 30K  | 55.39   |
| 35K  | 53.32   |
| 40K  | 51.77   |
| 45K  | 50.12   |
| 50K  | 48.96   |
| 55K  | 47.86   |
| 60K  | 46.94   |
| 65K  | 46.38   |
| 70K  | 45.60   |
| 75K  | 45.00   |
| 80K  | 44.31   |
| 85K  | 43.87   |
| 90K  | 43.47   |
| 95K  | 42.95   |
| 100K | 42.71   |

Final diagnostics: val PPL 42.67, sequential K=1: 43.69 (1.0 PPL penalty), empirical L: 0.66, contraction ratios: [0.36, 0.43, 0.66].

### block_head_corr_ffn_add D=4 C=1024 OWT (completed)

56C² FLOPs — 22% fewer than roformer N=6 (72C²). n_layers=20, d_block=4, K=5.

| Iter | Val PPL |
|-----:|--------:|
| 5K   | 95.16   |
| 10K  | 72.02   |
| 15K  | 63.09   |
| 20K  | 57.83   |
| 25K  | 54.29   |
| 30K  | 51.97   |
| 35K  | 49.96   |
| 40K  | 48.71   |
| 45K  | 46.93   |
| 50K  | 46.16   |
| 55K  | 45.25   |
| 60K  | 44.45   |
| 65K  | 43.70   |
| 70K  | 43.02   |
| 75K  | 42.31   |
| 80K  | 41.97   |
| 85K  | 41.46   |
| 90K  | 40.96   |
| 95K  | 40.60   |
| 100K | 40.29   |

### Head-to-head comparison (OWT C=1024)

| Iter | Roformer N=6 (72C²) | D=3 Add (44C²) | D=4 Add (56C²) | D=5 Add (68C²) |
|-----:|---------------------:|----------------:|----------------:|----------------:|
| 5K   | 88.14                | 102.37          | 95.16           | 91.49           |
| 10K  | 68.00                | 77.40           | 72.02           | 68.61           |
| 15K  | 60.11                | 67.51           | 63.09           | 60.04           |
| 20K  | 55.80                | 61.68           | 57.83           | 55.08           |
| 25K  | 52.80                | 58.10           | 54.29           | 51.92           |
| 30K  | 50.75                | 55.39           | 51.97           | 49.32           |
| 35K  | 48.91                | 53.32           | 49.96           | 47.70           |
| 40K  | 47.72                | 51.77           | 48.71           | 46.13           |
| 45K  | 46.39                | 50.12           | 46.93           | 45.09           |
| 50K  | 45.56                | 48.96           | 46.16           | 44.11           |
| 55K  | 44.72                | 47.86           | 45.25           | 43.12           |
| 60K  | 44.05                | 46.94           | 44.45           | 42.30           |
| 65K  | 43.39                | 46.38           | 43.70           | 41.94           |
| 70K  | 42.84                | 45.60           | 43.02           | 41.17           |
| 75K  | 42.42                | 45.00           | 42.31           | 40.39           |
| 80K  | 41.80                | 44.31           | 41.97           | 40.12           |
| 85K  | 41.42                | 43.87           | 41.46           | 39.54           |
| 90K  | 40.97                | 43.47           | 40.96           | 39.33           |
| 95K  | 40.78                | 42.95           | 40.60           | 38.83           |
| 100K | **40.48**            | 42.71           | **40.29**       | **38.61**       |

**D=5 Add beats roformer N=6 at 100K: 38.61 vs 40.48 PPL with 6% fewer FLOPs (68C² vs 72C²).** Seq K=1 = 38.62 (zero penalty).
**D=4 Add beats roformer N=6 at 100K: 40.29 vs 40.48 PPL with 22% fewer FLOPs (56C² vs 72C²).**
D=3 Add finishes at 42.71 — 2.2 PPL behind with 39% fewer FLOPs (44C² vs 72C²).
D=5 crossed roformer's final result at ~75K. D=4 crosses over at ~88K.

### block_head_delta_ffn_add D=4 C=1024 OWT (completed)

Same as corr_ffn_add but shifts delta (z - processed_x) instead of z.
56C² FLOPs, n_layers=20, d_block=4, K=5.

| Iter | Val PPL |
|-----:|--------:|
| 5K   | 95.60   |
| 10K  | 72.31   |
| 15K  | 63.56   |
| 20K  | 57.84   |
| 25K  | 54.42   |
| 30K  | 52.10   |
| 35K  | 49.93   |
| 40K  | 48.65   |
| 45K  | 46.95   |
| 50K  | 46.20   |
| 55K  | 45.41   |
| 60K  | 44.65   |
| 65K  | 43.76   |
| 70K  | 43.10   |
| 75K  | 42.42   |
| 80K  | 42.09   |
| 85K  | 41.45   |
| 90K  | 41.09   |
| 95K  | 40.75   |
| 100K | 40.35   |

Final diagnostics: val PPL 40.30, empirical L ~0.6, parallel K=5: 40.30, K=10: 40.31 (converged).
Sequential eval crashed due to missing seq_k parameter (now fixed).

**corr_ffn_add vs delta_ffn_add D=4 C=1024 OWT**: essentially identical (40.29 vs 40.35).
Shifting delta instead of z makes no meaningful difference at this scale.

### block_head_corr_ffn_add D=5 C=1024 OWT (completed)

Settings: n_embed=1024, n_layers=25, d_block=5, K=5, k_min=2, block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp
FLOPs: (12×5+8)C² = 68C². Roformer N=6 = 72C² (6% more FLOPs).

| Iter | D=5 corr_ffn_add (68C²) | Roformer N=6 (72C²) | Gap |
|------|------------------------|---------------------|------|
| 5K   | 91.49                  | 91.69               | -0.20 |
| 10K  | 68.61                  | 68.41               | +0.20 |
| 15K  | 60.04                  | 60.41               | -0.37 |
| 20K  | 55.08                  | 55.69               | -0.61 |
| 25K  | 51.92                  | 52.80               | -0.88 |
| 30K  | 49.32                  | 50.75               | -1.43 |
| 35K  | 47.70                  | 48.91               | -1.21 |
| 40K  | 46.13                  | 47.72               | -1.59 |
| 45K  | 45.09                  | 46.39               | -1.30 |
| 50K  | 44.11                  | 45.56               | -1.45 |
| 55K  | 43.12                  | 44.72               | -1.60 |
| 60K  | 42.30                  | 44.05               | -1.75 |
| 65K  | 41.94                  | 43.39               | -1.45 |
| 70K  | 41.17                  | 42.84               | -1.67 |
| 75K  | 40.39                  | 42.42               | -2.03 |
| 80K  | 40.12                  | 41.80               | -1.68 |
| 85K  | 39.54                  | 41.42               | -1.88 |
| 90K  | 39.33                  | 40.97               | -1.64 |
| 95K  | 38.83                  | 40.78               | -1.95 |
| 100K | **38.61**              | **40.48**           | **-1.87** |

D=5 corr_ffn_add (68C²) beats roformer N=6 (72C²) by **1.87 PPL** at 100K with **6% fewer FLOPs**.
D=5 crossed roformer's final 100K result (40.48) at ~75K iters.
Gap grew from near-zero at 5-10K to ~1.5-2.0 from 30K onwards.

Depth diagnostics: parallel K=1→58.17, K=2→40.18, K=3→38.80, K=5→38.60, K=10→38.61.
Sequential K=1 = 38.62, K=2 = 38.62. **Zero sequential penalty** (position 0 init fix working).
Empirical L ≈ 0.55 (contraction ratios ~0.65-0.72, 0.45-0.55, 0.48-0.66 across 3 iterations).

### block_head_corr_ffn_add D=8 C=768 OWT (completed)

Settings: n_embed=768, n_layers=40, d_block=8, K=5, k_min=2, block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp
FLOPs: (12×8+8)C² = 104C². Absolute FLOPs: 104 × 768² = 61.3M.
Compare: roformer N=6 C=1024 = 72 × 1024² = 75.5M FLOPs (19% more).

| Iter | D=8 C=768 (61.3M FLOPs) | Roformer N=6 C=1024 (75.5M FLOPs) | Gap |
|------|-------------------------|----------------------------------|------|
| 5K   | 94.80                   | 91.69                            | +3.11 |
| 10K  | 71.25                   | 68.41                            | +2.84 |
| 15K  | 61.48                   | 60.41                            | +1.07 |
| 20K  | 55.98                   | 55.69                            | +0.29 |
| 25K  | 52.69                   | 52.80                            | -0.11 |
| 30K  | 50.11                   | 50.75                            | -0.64 |
| 35K  | 48.32                   | 48.91                            | -0.59 |
| 40K  | 46.95                   | 47.72                            | -0.77 |
| 45K  | 45.66                   | 46.39                            | -0.73 |
| 50K  | 44.72                   | 45.56                            | -0.84 |
| 55K  | 43.89                   | 44.72                            | -0.83 |
| 60K  | 43.14                   | 44.05                            | -0.91 |
| 65K  | 42.45                   | 43.39                            | -0.94 |
| 70K  | 41.72                   | 42.84                            | -1.12 |
| 75K  | 41.40                   | 42.42                            | -1.02 |
| 80K  | 40.77                   | 41.80                            | -1.03 |
| 85K  | 40.30                   | 41.42                            | -1.12 |
| 90K  | 39.86                   | 40.97                            | -1.11 |
| 95K  | 39.58                   | 40.78                            | -1.20 |
| 100K | **39.10**               | **40.48**                        | **-1.38** |

D=8 C=768 (61.3M FLOPs) beats roformer N=6 C=1024 (75.5M FLOPs) by **1.38 PPL with 19% fewer FLOPs**.
Crossed over at ~25K. Seq K=1 = 39.10 (zero penalty).
Depth diagnostics: parallel K=1→51.60, K=2→40.12, K=3→39.22, K=5=K=10=39.10.

### Roformer N=12 C=768 OWT (completed)

Settings: n_embed=768, n_layers=12, block_size=256, batch_size=64, lr=2e-4, softmax, amp
FLOPs: 144C². Absolute FLOPs: 144 × 768² = 84.9M. D=8 uses 28% fewer inference FLOPs.

Final: **37.83 PPL**.

### Roformer N=11 C=768 OWT (completed)

Settings: n_embed=768, n_layers=11, block_size=256, batch_size=64, lr=2e-4, softmax, amp
FLOPs: 132C². Absolute FLOPs: 132 × 768² = 77.9M. D=8 uses 21% fewer inference FLOPs.

Final: **38.74 PPL**.

### D=8 vs N=11 vs N=12 head-to-head (all C=768 OWT)

| Iter | D=8 (104C²) | N=11 (132C²) | N=12 (144C²) | D=8 vs N=11 | N=11 vs N=12 |
|------|------------|-------------|-------------|-------------|-------------|
| 5K   | 94.80      | 87.78       | 87.07       | +7.02       | +0.71       |
| 10K  | 71.25      | 65.75       | 65.46       | +5.50       | +0.29       |
| 15K  | 61.48      | 57.85       | 57.48       | +3.63       | +0.37       |
| 20K  | 55.98      | 53.42       | 52.80       | +2.56       | +0.62       |
| 25K  | 52.69      | 50.47       | 49.86       | +2.22       | +0.61       |
| 30K  | 50.11      | 48.50       | 47.95       | +1.61       | +0.55       |
| 35K  | 48.32      | 46.76       | 46.25       | +1.56       | +0.51       |
| 40K  | 46.95      | 45.66       | 44.97       | +1.29       | +0.69       |
| 45K  | 45.66      | 44.49       | 43.94       | +1.17       | +0.55       |
| 50K  | 44.72      | 43.41       | 42.94       | +1.31       | +0.47       |
| 55K  | 43.89      | 42.75       | 42.15       | +1.14       | +0.60       |
| 60K  | 43.14      | 42.17       | 41.44       | +0.97       | +0.73       |
| 65K  | 42.45      | 41.52       | 40.96       | +0.93       | +0.56       |
| 70K  | 41.72      | 40.94       | 40.20       | +0.78       | +0.74       |
| 75K  | 41.40      | 40.51       | 39.74       | +0.89       | +0.77       |
| 80K  | 40.77      | 39.97       | 39.33       | +0.80       | +0.64       |
| 85K  | 40.30      | 39.70       | 39.02       | +0.60       | +0.68       |
| 90K  | 39.86      | 39.34       | 38.64       | +0.52       | +0.70       |
| 95K  | 39.58      | 38.93       | 38.23       | +0.65       | +0.70       |
| 100K | **39.10**  | **38.74**   | **37.83**   | **+0.36**   | **+0.91**   |

**D=8 vs N=11**: D=8 (104C²) only **0.36 PPL behind** N=11 (132C²) — 21% fewer inference FLOPs.
**N=11 vs N=12**: The 12th roformer layer (+12C², +9%) adds 0.91 PPL.
**D=8 vs N=12**: D=8 (104C²) 1.27 PPL behind N=12 (144C²) — 28% fewer inference FLOPs.

### block_head_corr_ffn_concat D=8 C=768 OWT (completed)

Settings: n_embed=768, n_layers=40, d_block=8, K=5, k_min=2, block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp
FLOPs: (12×8+16)C² = 112C². Absolute FLOPs: 112 × 768² = 66.1M.
Compare: D=8 add = 104C² = 61.3M FLOPs. Concat adds 8C² (8% more FLOPs) for the 2C→C input projection in the correction FFN.

Final: **39.22 PPL** (vs D=8 add's 39.10).

#### D=8 add vs concat head-to-head (both C=768 OWT)

| Iter | D=8 add (104C²) | D=8 concat (112C²) | Gap (add − concat) |
|------|-----------------|--------------------|--------------------|
| 5K   | 94.80           | 94.23              | +0.57              |
| 10K  | 71.25           | 71.03              | +0.22              |
| 15K  | 61.48           | 61.73              | -0.25              |
| 20K  | 55.98           | 56.10              | -0.12              |
| 25K  | 52.69           | 52.51              | +0.18              |
| 30K  | 50.11           | 49.82              | +0.29              |
| 35K  | 48.32           | 48.15              | +0.17              |
| 40K  | 46.95           | 46.67              | +0.28              |
| 45K  | 45.66           | 45.69              | -0.03              |
| 50K  | 44.72           | 44.67              | +0.05              |
| 55K  | 43.89           | 43.79              | +0.10              |
| 60K  | 43.14           | 43.00              | +0.14              |
| 65K  | 42.45           | 42.30              | +0.15              |
| 70K  | 41.72           | 41.69              | +0.03              |
| 75K  | 41.40           | 41.19              | +0.21              |
| 80K  | 40.77           | 40.70              | +0.07              |
| 85K  | 40.30           | 40.44              | -0.14              |
| 90K  | 39.86           | 40.00              | -0.14              |
| 95K  | 39.58           | 39.51              | +0.07              |
| 100K | **39.10**       | **39.22**          | **-0.12**          |

**At D=8, concat ≈ add.** The gap is noise-level (±0.3 PPL throughout, final difference 0.12). The extra 8C² from concat's 2C input to the correction FFN provides no benefit when D is already large. **Use add at high D** — fewer FLOPs, same or slightly better PPL.

This confirms the pattern: at low D, concat helps because the correction FFN's 2C input (tok_emb + shifted correction) provides richer information. At high D, the D=8 block already has enough capacity to extract information from the shifted correction alone, making the tok_emb concatenation redundant.

### D=8 C=768 K=2 compiled OWT (completed)

Settings: n_embed=768, n_layers=16 (D=8 × K=2), d_block=8, K=2, k_min=0 (fixed K, no random sampling), block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp, **torch.compile enabled**
FLOPs: 104C² (same as K=5 — inference FLOPs don't depend on K).

**Training speed**: 2.67 it/s (vs ~1.6 it/s for K=5 uncompiled) — **1.7× speedup**.
- 60% fewer block applications per iteration (16 vs 40)
- torch.compile optimizes the static computation graph
- Wall clock: 10h 27min vs ~17.4h for K=5

**Note on torch.compile**: Required `--k_min 0` to ensure fixed K (no `random.randint` call which breaks static graph). Also required `sudo apt-get install python3.12-dev` and `sudo ldconfig` to fix Triton compilation (missing `Python.h` and stale library cache).

Final: **40.17 PPL** (K=5 baseline: 39.10). **Cost: 1.07 PPL for 1.7× faster training.**

| Iter | D=8 K=2 (compiled) | D=8 K=5 (baseline) | Gap |
|------|--------------------|--------------------|-----|
| 5K   | 96.05              | 94.80              | +1.25 |
| 10K  | 72.19              | 71.25              | +0.94 |
| 15K  | 62.77              | 61.48              | +1.29 |
| 20K  | 57.18              | 55.98              | +1.20 |
| 25K  | 53.87              | 52.69              | +1.18 |
| 30K  | 51.34              | 50.11              | +1.23 |
| 35K  | 49.48              | 48.32              | +1.16 |
| 40K  | 47.96              | 46.95              | +1.01 |
| 45K  | 46.77              | 45.66              | +1.11 |
| 50K  | 45.86              | 44.72              | +1.14 |
| 55K  | 44.98              | 43.89              | +1.09 |
| 60K  | 44.07              | 43.14              | +0.93 |
| 65K  | 43.51              | 42.45              | +1.06 |
| 70K  | 42.76              | 41.72              | +1.04 |
| 75K  | 42.43              | 41.40              | +1.03 |
| 80K  | 41.77              | 40.77              | +1.00 |
| 85K  | 41.33              | 40.30              | +1.03 |
| 90K  | 40.94              | 39.86              | +1.08 |
| 95K  | 40.59              | 39.58              | +1.01 |
| 100K | **40.17**          | **39.10**          | **+1.07** |

The gap is remarkably stable at ~1.0-1.2 PPL throughout training. K=2 captures most of the model's capacity — the extra 3 iterations in K=5 contribute only 1 PPL.

Diagnostics:
- Parallel K=1: 62.58 (no iteration — bad)
- Parallel K=2: 40.16 (optimal — matches training K)
- Parallel K=3: 40.64, K=4: 40.53 (slightly worse — overtrained for K=2)
- **Sequential K=1: 40.59** (0.43 PPL penalty vs parallel K=2)

With K=5 training, sequential K=1 had zero penalty (39.10 = parallel K=5). With K=2 training, there's a 0.43 PPL gap — the model doesn't converge as deeply in 2 iterations, so sequential inference doesn't perfectly replicate the training regime.

### D=8 C=768 K-schedule OWT (in progress)

Settings: same as K=2 but with curriculum K schedule, **no torch.compile** (random K phase breaks static graph).
Schedule: K=1 (iter 0–50K) → K=2 (50K–90K) → K=random(2,5) (90K–100K).

CLI: `--k_schedule "0:1,50000:2,90000:2-5" --n_layers 40` (model created with full K=5 weights, schedule controls which K is used during training).

Final: **41.63 PPL** (9h 50min, 2.83 it/s average).

| Iter | K-schedule | K=2 (compiled) | K=5 (baseline) | Phase |
|------|-----------|----------------|----------------|-------|
| 5K   | 100.92    | 96.05          | 94.80          | K=1   |
| 10K  | 76.39     | 72.19          | 71.25          | K=1   |
| 15K  | 66.88     | 62.77          | 61.48          | K=1   |
| 20K  | 61.14     | 57.18          | 55.98          | K=1   |
| 25K  | 57.89     | 53.87          | 52.69          | K=1   |
| 30K  | 55.28     | 51.34          | 50.11          | K=1   |
| 35K  | 53.29     | 49.48          | 48.32          | K=1   |
| 40K  | 51.70     | 47.96          | 46.95          | K=1   |
| 45K  | 50.47     | 46.77          | 45.66          | K=1   |
| 50K  | 49.44     | 45.86          | 44.72          | K=1→K=2 |
| 55K  | 48.19     | 44.98          | 43.89          | K=2   |
| 60K  | 46.91     | 44.07          | 43.14          | K=2   |
| 65K  | 45.99     | 43.51          | 42.45          | K=2   |
| 70K  | 45.06     | 42.76          | 41.72          | K=2   |
| 75K  | 44.52     | 42.43          | 41.40          | K=2   |
| 80K  | 43.81     | 41.77          | 40.77          | K=2   |
| 85K  | 43.13     | 41.33          | 40.30          | K=2   |
| 90K  | 42.87     | 40.94          | 39.86          | K=2→K=2-5 |
| 95K  | 42.10     | 40.59          | 39.58          | K=2-5 |
| 100K | **41.63** | **40.17**      | **39.10**      |       |

Diagnostics:
- Parallel K=2: 41.83, K=3: 41.62, K=5: 41.62, K=10: 41.62
- **Sequential K=1: 41.62** (zero penalty — the K=2-5 random phase restored convergence)

#### K training comparison summary

| Run | Final PPL | Seq K=1 | Wall time | Speedup |
|-----|-----------|---------|-----------|---------|
| K=5 (baseline) | 39.10 | 39.10 (0.00) | ~17.4h | 1.0× |
| K=2 (compiled) | 40.17 | 40.59 (+0.43) | 10.5h | 1.7× |
| K-schedule (1→2→2-5) | 41.63 | 41.62 (0.00) | 9.8h | 1.8× |

**Findings**: K-schedule is fastest (1.8×) and restores zero seq K=1 penalty, but 2.5 PPL behind K=5. The K=1 phase (50K iters) created too large a gap to recover from in only 40K of K=2 and 10K of K=2-5. A less aggressive schedule (e.g. shorter K=1 phase, or starting at K=2) might close the gap.

---

## Training Speedup Investigation

### Bottleneck analysis

The look-ahead architecture's training bottleneck is the K-iteration loop. For D=8 K=5, this means 40 sequential block applications per training step (vs 8 layers in a roformer). Each iteration is a Python for-loop body, limiting GPU utilization.

### Approaches tested

1. **torch.compile** (`--compile` flag): Compiles the model graph for Inductor/Triton optimization. Requires static graph — incompatible with `random.randint` for k_min, so use `--k_min 0`. Setup: `sudo apt-get install python3.12-dev && sudo ldconfig`.

2. **Reduced K**: Training with K=2 instead of K=5 reduces block applications from 40 to 16 (60% reduction). Combined with torch.compile, achieves 1.7× speedup. Quality impact TBD.

### Approaches considered but not yet tested

- **Flash Attention**: Would speed up the attention computation in each block, but benefits both look-ahead and roformer equally. Not a differentiator. Current code uses manual attention (Q×K^T scaling, masking, softmax, ×V) in `joformer/train_wiki.py:185`.

- **Fixed K with random loss iteration**: Instead of varying K during training, always run K=5 iterations but randomly sample which iteration's output to compute loss on. This gives the compiler a fixed graph while still training convergence across depths. Not yet implemented.

### Key insight: training cost vs inference cost

Look-ahead is more expensive to train than roformer (40 block apps vs 8-12 layers). **The value proposition is in inference FLOPs**, not training efficiency. Training speedups help iteration speed during research but don't change the deployment advantage.

---

## Future Work

- **Mid-FFN insertion**: At D=8, roformer gains 1.5× more PPL per additional layer. Hypothesis: inserting an extra FFN between blocks 4 and 5 (mid-point of D=8) could boost quality for moderate FLOP cost. The FFN would break the monotony of the shared-weight iteration by adding a non-shared transformation mid-way. Not yet tested.

- **K=2 training quality**: Currently running D=8 K=2 with torch.compile. If quality drop from K=5 is small (< 1-2 PPL), K=2 training enables 1.7× faster experimentation. This would dramatically speed up the research cycle.

- **Thunder Compute parallel experiments**: tnr-cli installed, code and OWT data (34GB) synced to tnr machine. Ready for parallel runs. RTX A6000 on-demand at $0.27/hr or A100 at $0.78/hr available.

- **GPT-2 scale comparison**: A100 (80GB) could train GPT-2 XL scale (1.5B params). At $0.78/hr this is feasible for short runs. Would provide a more meaningful comparison point.

- **Multi-token prediction (MTP) via correction vectors**: The correction at position t already encodes contextual information from the sequence. This could be used to predict not just token t+1 but also t+2, t+3, etc., enabling speculative drafts from a single forward pass without a separate draft model. Requires retraining with multi-token prediction heads.

- **Compatibility with Look-Ahead Decoding (Fu et al.)**: The sequential K=1 inference mode is incompatible with parallel speculation/verification (position t+1 depends on the correction from position t). However, two workarounds exist:
  1. **Unrolled parallel mode**: Run the model in its training mode (K iterations, raw embeddings) which is a standard parallel forward pass, fully compatible with speculative decoding.
  2. **MTP from corrections**: If multi-token prediction works, speculative tokens come from the model itself — no external draft model or parallel verification needed.

---

## Single-Head C=1024 Scaling Experiments (killed — ThunderCompute preemption)

These experiments were running on 2× H100 PCIe but the instance was preempted, killing both. No checkpoints saved. Results below are from log files up to the point of termination.

### Roformer N=24 C=1024 single-head OWT (killed at 56%)

Settings: n_embed=1024, n_layers=24, block_size=256, batch_size=64, lr=2e-4, softmax, amp, n_head=1, cuda:0
Params: 367,879,424. FLOPs: 288C².

| Iter | Val PPL |
|-----:|--------:|
| 5K   | 72.70   |
| 10K  | 53.75   |
| 15K  | 46.64   |
| 20K  | 42.77   |
| 25K  | 40.20   |
| 30K  | 38.52   |
| 35K  | 37.07   |
| 40K  | 35.89   |
| 45K  | 35.04   |
| 50K  | 34.33   |
| 55K  | 33.63   |

Killed at iter 55,983. Last eval at 55K: **33.63 PPL**.

### block_head_corr_ffn_add D=12 C=1024 single-head OWT (killed at 33%)

Settings: n_embed=1024, n_layers=60 (D=12 × K=5), d_block=12, k_min=2, block_size=256, batch_size=64, lr=2e-4, softmax, convergence_weight=0.1, amp, n_head=1, cuda:1
Params: 225,120,512. FLOPs: (12×12+8)C² = 152C².

| Iter | Val PPL |
|-----:|--------:|
| 5K   | 78.34   |
| 10K  | 58.48   |
| 15K  | 50.80   |
| 20K  | 46.81   |
| 25K  | 43.94   |
| 30K  | 41.98   |

Killed at iter 33,465. Last eval at 30K: **41.98 PPL**.

### Single-head comparison at 30K

| Model | FLOPs | Val PPL @ 30K |
|-------|------:|----------:|
| D=12 add (152C²) | 152C² | 41.98 |
| Roformer N=24 (288C²) | 288C² | 38.52 |

D=12 trails by 3.46 PPL but uses **47% fewer FLOPs**. For reference, D=5 C=1024 (68C²) achieved 38.61 at 100K — D=12 at 30K (41.98) is on a faster trajectory and would likely beat roformer N=24 at FLOP-matched comparison by 100K.

### Multi-head equivalents (n_head=16, in progress)

Relaunched as multi-head (16 heads, head_dim=64) after preemption. batch_size=32 (D=12 OOM'd at batch=64 with 16 heads due to attention activation memory), 200K iters to match total tokens.
- GPU 0: roformer N=24 C=1024 n_head=16 → log: `logs/roformer_n24_c1024_h16_owt.log`
- GPU 1: corr_ffn_add D=12 C=1024 n_head=16 → log: `logs/corr_ffn_add_d12_c1024_h16_owt.log`

**Second preemption (2026-03-23)**: Both h16 runs killed again. Roformer reached 80K/200K, D=12 reached 50K/200K. No checkpoints were saved. Restarted with auto-resume infrastructure (see below).

#### Multi-head vs single-head comparison (token-matched: h16 batch=32 iter N = h1 batch=64 iter N/2)

| Iter (h16, b32) | Roformer N=24 h16 | Roformer N=24 h1 (token-matched) | D=12 h16 | D=12 h1 (token-matched) | D=12 vs Roformer gap (h16) |
|------|-------------------|----------------------------------|----------|-------------------------|---------|
| 5K   | 96.72 | — | 98.83 | — | +2.11 |
| 10K  | 69.30 | 72.70 (h1 @ 5K) | 73.47 | 78.34 (h1 @ 5K) | +4.17 |
| 15K  | 58.18 | — | 62.80 | — | +4.62 |
| 20K  | 52.46 | 53.75 (h1 @ 10K) | 56.53 | 58.48 (h1 @ 10K) | +4.07 |
| 25K  | 48.49 | — | 52.70 | — | +4.21 |
| 30K  | 45.54 | 46.64 (h1 @ 15K) | 49.49 | 50.80 (h1 @ 15K) | +3.95 |
| 35K  | 43.37 | — | 47.24 | — | +3.87 |
| 40K  | 41.77 | 42.77 (h1 @ 20K) | 45.52 | 46.81 (h1 @ 20K) | +3.75 |
| 45K  | 40.33 | — | 44.05 | — | +3.72 |
| 50K  | 39.27 | 40.20 (h1 @ 25K) | — | 43.94 (h1 @ 25K) | — |
| 55K  | 38.17 | — | — | — | — |
| 60K  | 37.37 | 38.52 (h1 @ 30K) | — | 41.98 (h1 @ 30K) | — |
| 65K  | 36.67 | — | — | — | — |
| 70K  | 36.10 | — | — | — | — |
| 75K  | 35.67 | — | — | — | — |
| 80K  | 34.95 | — | — | — | — |

Multi-head beats single-head at matched tokens for both models (~1 PPL for roformer, ~5 PPL for D=12).
D=12 vs roformer h16 gap: peaked at 4.62 (15K), narrowing since → 3.72 at 45K.

### Multi-head h16 restart (2026-03-24, in progress)

Third launch after two preemptions. Same config, same seed (42). Checkpointing enabled (rolling saves every 5K iters). Logs: `logs/roformer_n24_c1024_h16_owt_restart.log`, `logs/corr_ffn_add_d12_c1024_h16_owt_restart.log`.

Roformer reproduces previous run's PPL exactly (same seed, deterministic). D=12 diverges slightly (~+0.3 PPL) because `k_min=2` uses Python's `random.randint` which was not seeded — only `torch.manual_seed` was set. Fixed for future runs by adding `random.seed(args.seed)`.

| Iter | Roformer N=24 (288C²) | D=12 add (152C²) | Gap |
|------|----------------------|-------------------|-----|
| 5K   | 96.72  | 98.80  | +2.08 |
| 10K  | 69.30  | 73.62  | +4.32 |
| 15K  | 58.18  | 63.10  | +4.92 |
| 20K  | 52.46  | 56.82  | +4.36 |
| 25K  | 48.49  | 53.04  | +4.55 |
| 30K  | 45.54  | 49.89  | +4.35 |
| 35K  | 43.37  | 47.68  | +4.31 |
| 40K  | 41.77  | 45.86  | +4.09 |
| 45K  | 40.33  | 44.33  | +4.00 |
| 50K  | 39.27  | 43.10  | +3.83 |
| 55K  | 38.17  | 42.04  | +3.87 |
| 60K  | 37.37  | 41.02  | +3.65 |
| 65K  | 36.67  | 40.39  | +3.72 |
| 70K  | 36.10  | 39.66  | +3.56 |
| 75K  | 35.67  | 38.87  | +3.20 |
| 80K  | 34.95  | 38.42  | +3.47 |
| 85K  | 34.49  | 37.87  | +3.38 |
| 90K  | 34.11  | 37.39  | +3.28 |

Gap peaked at 4.92 (15K) and has narrowed to 3.28 at 90K. D=12 uses **47% fewer inference FLOPs** (152C² vs 288C²).

### D=23 C=1024 h16 FLOP-matched vs roformer N=24 (in progress)

Settings: n_embed=1024, n_layers=115 (D=23 × K=5), d_block=23, k_min=2, block_size=256, lr=2e-4, softmax, convergence_weight=0.1, amp, n_head=16.
Params: 363,678,976. FLOPs: (12×23+8)C² = 284C² — near FLOP-matched to roformer N=24 (288C²).

Phase 1 (iters 0–30K): batch=16, no flash attention, eval_interval=10K. Log: `logs/corr_ffn_add_d23_c1024_h16_owt.log`.
Phase 2 (iters 30K+): batch=32, flash attention (`blocks_flash.py`), eval_interval=10K, 200K max iters. Log: `logs/corr_ffn_add_d23_c1024_h16_flash_owt.log`.

Switched to flash attention at 30K to fit batch=32 (61GB vs 80GB available). `F.scaled_dot_product_attention` replaces manual attention — same weights, checkpoint-compatible.

**Eval batch size caveat**: Phase 1 evals used batch=16, but roformer/D=12 evals used batch=32. The eval function (`estimate_loss`) seeds with 42 and draws `eval_iters` batches of `batch_size` — different batch sizes produce different validation samples. Evaluating the 30K checkpoint at batch=16 gave 61.07; at batch=32 gave 59.51 — a **~1.5 PPL difference** from eval batch size alone. Phase 1 numbers are therefore ~1.5-2 PPL too pessimistic vs roformer/D=12. Phase 2 numbers are directly comparable (all batch=32).

| Tokens (equiv b32 iters) | Roformer N=24 (288C²) | D=23 add (284C²) | D=23 gap | D=12 gap (ref) | Note |
|---|---|---|---|---|---|
| 5K | 96.72 | 100.33 | +3.61 | +2.08 | batch=16 eval, true gap ~+2 |
| 10K | 69.30 | 71.59 | +2.29 | +4.32 | batch=16 eval, true gap ~+1 |
| 15K | 58.18 | 61.07 | +2.89 | +4.92 | batch=16 eval, true gap ~+1.3 |
| 15K | 58.18 | **59.51** | **+1.33** | +4.92 | batch=32 eval (flash), directly comparable |

**Key finding**: D=23 at 15K-equiv tokens is only +1.33 PPL behind roformer N=24, vs D=12's +4.92 at the same point. D=23 is near FLOP-matched (284C² vs 288C²), so this gap must close and cross over for the architecture to win. D=12's gap narrowed from +4.92 to ~+3.0 over training; D=23 starting from +1.33 is in a much stronger position.

## Checkpointing & Auto-Resume Infrastructure

Added 2026-03-23 after two preemptions lost all training progress.

### Problem
ThunderCompute instances are preempted/rebooted without warning. Previous runs had no checkpointing, so all progress was lost on each reboot.

### Solution

#### 1. Checkpoint saving (in `train_wiki_streaming.py`)
- Rolling checkpoint saved at every eval (every 5K iters) to `checkpoints/{model_name}_latest.pt`
- Saves: model state, optimizer state, scheduler state, GradScaler state, ppl_log, diagnostics_log, best_val_loss, current iteration
- Only keeps latest checkpoint per model (overwrites previous)
- `--checkpoint_dir` defaults to `checkpoints/` (previously empty/disabled)
- On startup, automatically loads checkpoint if it exists — resumes from `iter + 1`

#### 2. Auto-start on boot (sshd wrapper)
The container runs `tini -s -- /usr/sbin/sshd -D -e` as PID 1. No systemd, no cron.

- `/usr/sbin/sshd.real` — the original sshd binary (renamed)
- `/usr/sbin/sshd` — wrapper script that launches training in background, then exec's sshd.real
- `/home/ubuntu/look_ahead6/train_both.sh` — launcher that starts both experiments in detached `screen` sessions

**Boot sequence**: tini → sshd wrapper → `train_both.sh &` (background) → exec sshd.real
Training starts immediately on boot. No SSH login required.

#### 3. Key files

| File | Purpose |
|------|---------|
| `train_both.sh` | Launches both experiments in screen sessions |
| `checkpoints/{model}_latest.pt` | Rolling checkpoint (auto-resume) |
| `/usr/sbin/sshd` | Wrapper script (starts training + exec's real sshd) |
| `/usr/sbin/sshd.real` | Original sshd binary |

#### 4. Monitoring

```bash
# Check progress
bash check_progress.sh logs/roformer_n24_c1024_h16_owt.log
bash check_progress.sh logs/corr_ffn_add_d12_c1024_h16_owt.log

# Check checkpoints
ls -lh checkpoints/

# Attach to screen sessions
screen -r roformer
screen -r corr_ffn

# GPU status
nvidia-smi
```

#### 5. Manual restart
```bash
bash /home/ubuntu/look_ahead6/train_both.sh
```
Kills any existing training processes first, then relaunches with auto-resume from checkpoint.

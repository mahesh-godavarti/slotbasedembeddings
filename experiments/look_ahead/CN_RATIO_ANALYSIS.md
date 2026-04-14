# C/N Ratio Analysis

## Deployed model ratios

| Model | Params | N (depth) | C (width) | C/N |
|-------|--------|-----------|-----------|-----|
| GPT-2 Small | 124M | 12 | 768 | 64 |
| GPT-2 XL | 1.5B | 48 | 1600 | 33 |
| Llama 7B | 7B | 32 | 4096 | 128 |
| Llama 70B | 70B | 80 | 8192 | 102 |
| GPT-3 175B | 175B | 96 | 12288 | 128 |
| PaLM 540B | 540B | 118 | 18432 | 156 |

Typical range: C/N = 100-150 at scale. Smaller models (GPT-2) use lower ratios (33-64).

## Our experiments by C/N ratio

### At ~302M inference FLOPs (our main FLOP budget)

| Model | N | C | C/N | PPL | Realistic? |
|-------|---|---|-----|-----|------------|
| N=24 C=1024 | 24 | 1024 | 43 | 29.42 | No -- too deep |
| N=12 C=1408 | 12 | 1408 | 117 | 29.92 | Yes |
| N=6 C=2048 | 6 | 2048 | 341 | 30.86 | No -- too wide |
| D=12 C=1408 | 12 | 1408 | 117 | 29.00 | Yes |

### At ~336M inference FLOPs

| Model | N | C | C/N | PPL | Realistic? |
|-------|---|---|-----|-----|------------|
| N=24 C=1088 | 24 | 1088 | 45 | running | No -- too deep |
| N=12 C=1536 | 12 | 1536 | 128 | running | Yes |
| N=6 C=2176 | 6 | 2176 | 363 | running | No -- too wide |
| D=6 C=2048 | 6 | 2048 | 341 | 29.04 | Wide, but with correction |

## What the data confirms

### 1. Depth has diminishing returns at fixed FLOPs

At ~302M FLOPs:
- N=6 C=2048: 30.86
- N=12 C=1408: 29.92 (gain of 0.94 from 6->12 layers)
- N=24 C=1024: 29.42 (gain of only 0.50 from 12->24 layers)

Doubling from 6 to 12 layers gains 0.94 PPL. Doubling again from 12 to 24 gains only 0.50. The second doubling is half as useful. N=24 is overprovisioned.

### 2. The realistic baseline is N=12, not N=24

N=12 C=1408 (C/N=117) is within 0.50 PPL of N=24 C=1024 (C/N=43) at lower FLOPs. At matched FLOPs (N=12 C=1456-1536), the gap would be even smaller.

Any comparison against N=24 C=1024 flatters the alternative because N=24 is not the optimal depth for this FLOP budget.

### 3. D=12 C=1408 beats the realistic baseline

D=12 C=1408 (29.00) vs N=12 C=1408 (29.92) = 0.92 PPL improvement from the correction mechanism at matched depth and width (only 5% FLOP difference from the corr_ffn). This is the honest measure of look-ahead's value at a realistic architecture.

### 4. The correction mechanism's value at realistic C/N

At realistic C/N (~117):
- Correction adds 0.92 PPL (D=12 C=1408 vs N=12 C=1408)
- For 8C^2 additional inference FLOPs (5.6% overhead)

This is the real story -- not "D beats overprovisioned N=24" but "D adds ~1 PPL at realistic architectures for minimal overhead."

## Pending comparisons

| Comparison | Purpose | Status |
|-----------|---------|--------|
| D=6 C=2048 vs N=12 C=1536 | Can 6 wide layers + correction beat 12 medium layers? | N=12 running |
| D=12 C=1408 vs N=12 C=1536 | Correction value at FLOP parity, realistic C/N | N=12 running |
| N=24 C=1088 vs D=6 C=2048 | Deep narrow vs wide+correction (both ~336M) | N=24 running |

## Waiting on

- N=12 C=1536 to finish -- the realistic FLOP-matched baseline for D=6 C=2048 and D=12 C=1408
- N=24 C=1088 to finish -- confirms whether N=24 is overprovisioned at higher C too
- N=6 C=2176 to finish -- confirms width alone can't match depth

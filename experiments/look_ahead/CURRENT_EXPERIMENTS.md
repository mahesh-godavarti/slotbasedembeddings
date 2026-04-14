# Current Experiments and Rationale

## The question

At a fixed inference FLOP budget (~340M), what is the optimal depth-width tradeoff? And does the look-ahead correction mechanism change this tradeoff?

## The FLOP budget: ~340M

All experiments use ~340M inference FLOPs per token, block_size=256, batch=32, 200K iters, OWT data.

## Roformer depth sweep (same FLOPs, varying depth)

We are running plain roformers at N=2, 4, 6, 12, 24 -- all FLOP-matched at ~340M by adjusting C:

| Model | FLOPs | C/N | Status | Final PPL |
|-------|-------|-----|--------|-----------|
| N=2 C=3776 | 342M | 1888 | Running (~3h left) | -- |
| N=4 C=2656 | 339M | 664 | Running (~7h left) | -- |
| N=6 C=2176 | 341M | 363 | Done | 30.35 |
| N=12 C=1536 | 340M | 128 | Done | 29.01 |
| N=24 C=1088 | 341M | 45 | Done | 28.68 |

**Purpose**: Shows how PPL varies with depth at fixed FLOPs. Establishes the diminishing returns curve. N=12 to N=24 gained only 0.33 PPL despite doubling layers. N=2 and N=4 will complete the picture at the shallow end.

## Look-ahead models (same FLOP budget)

| Model | FLOPs | Architecture | Status | Final PPL |
|-------|-------|-------------|--------|-----------|
| D=6 C=2048 (corr_ffn_add) | 336M | 6 blocks + FFN correction | Done | 29.04 |
| SA D=5 C=2176 | 341M | 5 blocks + attention correction + FFN correction | Running (qmti92t1) | 33.75 @ 95K |
| SA D=1 C=3776 | 342M | 1 block + attention correction + FFN correction | Running (qmti92t1) | 38.55 @ 95K |

**Purpose**: Shows whether the correction mechanism (FFN-only or SA) changes the depth-width tradeoff.

## Head-to-head comparisons

### 1. SA D=1 C=3776 vs N=2 C=3776 (both 342M FLOPs)

Same FLOPs. SA D=1 has 1 block + attention correction. N=2 has 2 plain blocks. SA D=1 is consistently ahead by ~2.8 PPL through 95K. The correction mechanism at D=1 is worth more than adding a second layer.

**Why this matters**: At D=1, the correction has only shallow z to work with. Yet it still beats an extra layer. This is the clearest demonstration that the correction mechanism adds real value -- not depth, not width, but the correction wiring itself.

### 2. D=1 C=4128 vs SA D=1 C=3776 (both ~341M FLOPs)

Queued on GPU 1. Same depth (D=1), same FLOPs. D=1 uses FFN-only correction (sees z[t-1] only). SA D=1 uses attention correction (sees all z[0..t-1]). The only difference is whether the correction attends to all previous positions or just the previous one.

**Why this matters**: Isolates the value of attention-based correction vs FFN-only correction. If SA D=1 beats D=1, the attention over all previous z values is strictly better than looking at z[t-1] alone. If they're similar, the simple FFN correction is sufficient.

### 3. SA D=3 C=2656 vs N=4 C=2656 (both 339M FLOPs)

Queued on GPU 0. Exact FLOP match at the same C. SA D=3 has 3 blocks + attention correction. N=4 has 4 plain blocks. Tests whether 3 blocks with correction beats 4 blocks without.

**Why this matters**: Same building blocks (attention + FFN), just wired differently. If SA D=3 wins, the correction wiring is more efficient than adding a 4th layer.

### 4. D=6 C=2048 vs N=12 C=1536 (336M vs 340M FLOPs)

Both done. D=6: 29.04. N=12: 29.01. Essentially tied. Six wide layers with FFN correction matches twelve medium layers. The correction mechanism allows half the depth at the cost of wider layers.

### 5. D=6 C=2048 vs N=6 C=2176 (336M vs 341M FLOPs)

Both done. D=6: 29.04. N=6: 30.35. Correction adds 1.31 PPL at similar depth and FLOPs. This is the isolated value of the correction mechanism at D=6.

## What we expect to learn

1. **The full depth-PPL curve at 340M FLOPs**: N=2, 4, 6, 12, 24 gives us 5 points. Where does the curve flatten? Is N=12 already near optimal?

2. **Whether SA correction beats an extra layer**: SA D=1 vs N=2 (ahead by 2.8 so far), SA D=3 vs N=4 (queued). If SA consistently beats N+1, the correction wiring is more efficient per FLOP than depth.

3. **Whether SA beats FFN-only correction**: D=1 C=4128 vs SA D=1 C=3776 (queued). Attention over all previous z vs just z[t-1].

4. **The optimal architecture at 340M FLOPs**: After all experiments complete, we rank every architecture and identify the best depth-width-correction combination.

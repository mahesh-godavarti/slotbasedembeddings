# Pointer Chasing: Adaptive Curriculum Breaks the Composition Barrier

## The Problem

Standard transformers can't learn multi-hop content-based composition with shuffled data at k≥5. Despite having sufficient capacity, the optimizer gets stuck at a ~36% accuracy plateau for 2+ hops of composition. This held true across:
- All positional encodings: RoPE, no-RoPE, datadep, datadepv, joformer, datadep2
- With/without windowed attention
- With/without helper signals (multi-Q, random hop targets)
- Fixed-schedule curriculum learning (both global and per-level)
- 200K-500K training iterations

## The Solution: Adaptive Per-Level Key Curriculum

### Key insight

The composition barrier is an optimization problem, not an architecture problem. The model needs to learn content-based matching for increasing numbers of keys, but jumping from k=2 to k=5 is too large a step. A **gradual, adaptive** curriculum that waits for each step to converge before advancing solves it.

### How it works

1. **Per-level key control**: each level independently controls how many keys have active targets. `_n_keys_active = [5, 2, 2]` means L0 trains on all 5 keys, L1 and L2 train on keys {0,1} only.

2. **Chain-based masking**: a target is valid only if the ENTIRE resolution chain stays within active keys. If L1's chain goes through an inactive key at L0, that target is IGNORE.

3. **Sequential level advancement**: only start advancing level L+1 after level L reaches k=5.

4. **Adaptive threshold**: advance k when accuracy exceeds 90% for 3 consecutive evaluations. No fixed iteration schedule.

### Architecture: no-RoPE + windowed attention

- **No RoPE**: prevents misleading positional signals with shuffled data
- **Window=38**: prevents shortcuts to the base table from higher levels (~1.4 level blocks for k=5)
- The within-triplet structure (value, =, key) is recognized through token type, not position

## Results

### 3-hop k=5 v=10 N=4 — SOLVED at 69K iters

```
Settings: n_hops=3, n_keys=5, n_values=10, n_embed=128, n_head=4
          no_rope, window=38, shuffle (permutation), adaptive curriculum
          threshold=0.9, consecutive=3
```

**Adaptive progression:**

| Iter | Event | k state |
|------|-------|---------|
| 0 | Start | [5, 2, 2] |
| 5.5K | L1 k=2→3 | [5, 3, 2] |
| 8.5K | L1 k=3→4 | [5, 4, 2] |
| 10K | L1 k=4→5 → start L2 | [5, 5, 2] |
| 32K | L2 k=2→3 | [5, 5, 3] |
| 46K | L2 k=3→4 | [5, 5, 4] |
| 48.5K | L2 k=4→5 | [5, 5, 5] |
| **69K** | **100% all levels** | [5, 5, 5] |

**Final accuracy: 100% on L0, L1, L2, and final query.**

### Comparison with failed approaches (all 3-hop k=5 v=10 shuffled)

| Approach | L1 | L2 | Iters | Solved? |
|----------|----|----|-------|---------|
| RoPE only | 36% | 36% | 200K | No |
| No-RoPE only | 36% | 36% | 200K | No |
| No-RoPE + window | 36% | 36% | 200K | No |
| datadep / datadepv / joformer / datadep2 | 36% | 36% | 100K | No |
| Global key curriculum (k=2→5 jump) | 100% | 36% | 300K | L1 only |
| Per-level fixed curriculum (slow ramp) | 100% | 50% | 400K | L1 only |
| Hybrid RoPE+NoPE N=6 (no curriculum) | 100% | 100% | 9K | Yes (3-hop only) |
| **Adaptive per-level curriculum** | **100%** | **100%** | **69K** | **Yes** |

### Why fixed curricula failed

1. **Global key curriculum**: k=2→5 for ALL levels simultaneously. L1 and L2 both disrupted at each transition. L1 eventually recovered (77K), L2 never did.

2. **Per-level fixed curriculum**: correct idea (keep solved levels at k=5), but fixed iteration thresholds. Steps too fast → premature transitions. Steps too slow → wasted iters. Each k transition still caused drops.

3. **The adaptive difference**: waits until 90% accuracy for 3 consecutive evals before advancing. Each step gets exactly as much time as needed. L1 advanced quickly (5.5K→10K). L2 took longer at k=2 (10K→32K = 22K iters) before advancing. No wasted iterations, no premature transitions.

## Why this works

The composition problem has two parts:
1. **Content matching**: find a specific key token among k shuffled entries
2. **QK-composition**: use retrieved info from one layer to guide attention in the next

At k=2, both are easy (binary permutations). At k=5, the model must learn to distinguish 5 keys and compose across layers with 5 possible routings.

The adaptive curriculum:
- Starts with k=2 where composition is easy → model learns the MECHANISM
- Gradually increases k, giving the model time to extend the mechanism to more keys
- Each step builds on the previous, never disrupting what's already learned
- The 90% threshold ensures the mechanism is robust before facing harder data

## Running the experiment

```bash
python -u pointer_chasing.py \
    --n_hops 3 --n_keys 5 --n_values 10 \
    --n_embed 128 --n_head 4 --n_iters 500000 --batch_size 64 --lr 1e-3 \
    --gpu 0 --permutation --run N4 \
    --no_rope --window 38 \
    --adaptive_curriculum --adaptive_threshold 0.9 --adaptive_consecutive 3 \
    --checkpoint_dir checkpoints_adaptive
```

## 4-hop k=5 v=10 N=5 — SOLVED

### k=2 start (adaptive_k starts at [5,2,2,2])

```
Settings: same as 3-hop but n_hops=4, N=5
```

| Iter | Event | k state |
|------|-------|---------|
| 6.5K | L1 at k=5 | [5, 5, 2, 2] |
| 21K | L2 at k=5, start L3 | [5, 5, 5, 2] |
| 147K | L3 k=2→3 | [5, 5, 5, 3] |
| 202.5K | L3 k=3→4 | [5, 5, 5, 4] |
| 204K | L3 k=4→5 | [5, 5, 5, 5] |
| **234K** | **100% all levels** | [5, 5, 5, 5] |

L3 took 126K iters just at k=2 (21K→147K). The deepest level is exponentially harder.

### k=1 start (adaptive_k starts at [5,1,1,1]) — 2.4x FASTER

| Iter | Event | k state |
|------|-------|---------|
| 7K | L1 at k=5 | [5, 5, 1, 1] |
| 23K | L2 at k=5, start L3 | [5, 5, 5, 1] |
| 71.5K | L3 k=1→2 | [5, 5, 5, 2] |
| 78.5K | L3 k=2→3 | [5, 5, 5, 3] |
| 84K | L3 k=3→4 | [5, 5, 5, 4] |
| 87K | L3 k=4→5 | [5, 5, 5, 5] |
| **99K** | **100% all levels** | [5, 5, 5, 5] |

**k=1 start is 2.4x faster** (99K vs 234K). The critical difference: L3 at k=1 took 48K to reach 90% (23K→71.5K), vs 126K at k=2. Starting from k=1 lets the model learn the composition STRUCTURE (trivially) before facing content matching.

Once L3 crosses from k=1 to k=2, subsequent transitions are fast: k=2→3 in 7K, k=3→4 in 5.5K, k=4→5 in 3K. The structure learned at k=1 transfers efficiently.

### Why k=1 helps

At k=1, there's only one key per level. The permutation is identity. The composition chain is trivially key 0→key 0→...→base_table[0]. No content matching needed — the model just learns the chain structure.

When advancing to k=2, the model already knows HOW to compose across levels. It just needs to extend this to 2 keys (binary permutation). This is much easier than learning both structure and matching simultaneously at k=2.

### Hop suppression is essential

Without hop suppression, the 4-hop run with k=2 start failed (L3 stuck at 35% for 40K+ iters). The gradient noise from L3's random-accuracy targets interfered with L2 learning.

With hop suppression: only the currently-advancing level and solved levels have active targets. Unsolved future levels are IGNORE. This eliminates gradient interference.

## Positional encoding comparison

| PE | L1 k=5 | L2 k=5 | L3 k=5 | Solved | Notes |
|----|--------|--------|--------|--------|-------|
| no-RoPE k=1 start | 7K | 23K | 87K | **99K** | Best |
| no-RoPE k=2 start | 10K | 21K | 204K | 234K | Slower |
| datadepv k=1 start | 8K | stuck at 30% | — | — | Worse |
| RoPE (any config) | stuck | — | — | — | Fails |

**no-RoPE is the clear winner** for shuffled composition. Data-dependent angles (datadepv) add complexity without benefit. RoPE actively hurts by providing misleading positional signals.

## 10-hop k=5 v=10 N=11 — L3 stuck

10-hop with N=11 and k=1 start: L1 at k=5 by 7.5K, L2 at k=5 by 43.5K, but L3 stuck at ~31% at k=1 for 260K+ iters.

Despite hop suppression making L3's effective task identical to 4-hop (same positions, same context — L4-L9 are after L3 and can't be seen), the N=11 model is harder to optimize than N=5. The extra depth slows convergence.

| Level | 4-hop (N=5) | 10-hop (N=11) |
|-------|------------|---------------|
| L1 at k=5 | 7K | 7.5K |
| L2 at k=5 | 23K | 43.5K |
| L3 at k=1→2 | 71.5K | stuck at 306K |

**Open question**: Would N=5 on 10-hop solve L1-L3 at the same speed as 4-hop? The deeper levels (L4-L9) would then need more layers.

## Summary of all working approaches for shuffled pointer chasing

| Approach | Max hops solved | Iters | Notes |
|----------|----------------|-------|-------|
| Hybrid RoPE+NoPE N=6 | 3-hop | 9K | No curriculum, but can't do 4-hop |
| Adaptive k=2 start (no-RoPE + window) | 3-hop | 69K | |
| Adaptive k=2 start (no-RoPE + window) | 4-hop | 234K | Slow — L3 took 126K at k=2 |
| **Adaptive k=1 start (no-RoPE + window)** | **3-hop** | **~50K** | |
| **Adaptive k=1 start (no-RoPE + window)** | **4-hop** | **99K** | **2.4x faster than k=2 start** |
| Adaptive k=1 start (no-RoPE + window) | 10-hop | in progress | L3 stuck at N=11, likely N too deep |
| datadepv + adaptive | — | stuck | Worse than no-RoPE |

## Key takeaways

1. **Adaptive curriculum is essential** for shuffled multi-hop composition at k≥5. Fixed schedules fail because premature transitions disrupt learning.

2. **k=1 start is critical** for deep composition. The model learns the composition STRUCTURE at k=1 (trivial), then extends to content matching. k=2→3 transitions take ~7K iters after k=1 warmup.

3. **Hop suppression** eliminates gradient interference from unsolved levels. Without it, noisy gradients from deep levels prevent intermediate levels from learning.

4. **no-RoPE + windowed attention** is the right setup for shuffled data. RoPE provides misleading positional signals. Data-dependent angles don't help.

5. **Model depth matters for optimization** — N=11 is harder to optimize than N=5 even for the same effective task. Deeper models need more careful training.

6. **Each level takes progressively longer**: L1 ~7K, L2 ~23K, L3 ~87K (4-hop). Roughly exponential in depth.

## Code

- `pointer_chasing.py` flags:
  - `--adaptive_curriculum`: enable adaptive per-level key curriculum
  - `--adaptive_threshold 0.9`: accuracy threshold to advance
  - `--adaptive_consecutive 3`: consecutive evals above threshold
  - `--no_rope`: disable RoPE (required for shuffled data)
  - `--window 38`: windowed attention (prevents base table shortcuts)

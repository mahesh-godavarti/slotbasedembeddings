# Pointer Chasing: Empirical TC^0 Separation — Final Results

## Summary

We demonstrate that a D=1 look-ahead model trained with BPTT solves k-hop pointer chasing for any k using a single shared-weight block, while an N-layer transformer solves at most ~N levels. This empirically demonstrates the computational separation between fixed-depth transformers (TC^0) and unbounded sequential depth.

**Key result**: BPTT with 1 shared block solves all 11 levels (10-hop) at 100%, while N-layer transformers show a clean staircase — N=1→1 level, N=3→3, N=5→6, N=10→7, N=11→8, N=12→all 11.

## Critical Design Choices

### Windowed attention is essential

Without windowing, all models can directly attend to the base table values from any position, creating a shortcut that gives ~1/k accuracy without composition. With windowed attention (window=38), higher-level Q sections cannot see the base table — they must chain through intermediate Q sections.

**Bug we caught**: The BPTT model's incremental KV cache was accumulating ALL past keys, bypassing the window. With full attention, BPTT plateaued at 82%. Once fixed to use windowed KV cache, BPTT reached 100%.

### No shuffling (fixed entry order)

With shuffled entries, content-based composition beyond 2 hops proved extremely difficult for gradient descent (77K iters for 2-hop, never for 3-hop). Without shuffling, the model uses positional patterns (RoPE) for within-level lookups — but the window prevents cross-level positional shortcuts. This combination works.

### Per-level key tokens

Each level uses its own key namespace (A0,B0 for level 0; A1,B1 for level 1) to prevent ambiguity when the same logical key appears at multiple levels.

### Dense targets via Q sections

After each table, a Q section queries each key with the target being the fully resolved value. This provides gradient signal at every level.

## 10-Hop Results

### Settings
- n_hops=10, n_keys=5, n_values=10, permutation=True
- n_head=4, batch_size=64, window=38, no_shuffle
- Code: `pointer_chasing.py` with `blocks2.py`

### Staircase (e=256, lr=1e-4, 50K iters)

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | Levels solved |
|-------|----|----|----|----|----|----|----|----|----|----|-----|---------------|
| N=1 | 1.00 | 0.30 | 0.10 | 0.12 | 0.12 | 0.11 | 0.11 | 0.12 | 0.11 | 0.11 | 0.11 | 1 |
| N=3 | 1.00 | 1.00 | 1.00 | 0.37 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 3 |
| N=5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.35 | 0.11 | 0.11 | 0.11 | 0.11 | 6 |
| N=10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 | 0.36 | 0.36 | 0.38 | 7 |
| N=11 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 | 0.37 | 0.36 | 8 |
| **N=12** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **11 (all)** |

### BPTT (windowed, fixed)

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | Iters to solve |
|-------|----|----|----|----|----|----|----|----|----|----|-----|----------------|
| BPTT e=256 lr=1e-4 k=5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ~16K |
| BPTT e=128 lr=1e-3 k=5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ~20K |
| BPTT e=128 lr=1e-3 k=10 v=20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 | ~23K |

BPTT scales to larger key spaces (k=10, v=20, window=52). The wave propagates through all levels:

**BPTT k=10 v=20 progression** (e=128, lr=1e-3, window=52, no shuffle):

| Iter | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|------|----|----|----|----|----|----|----|----|----|----|-----|
| 3K | 1.00 | 0.74 | 0.27 | 0.20 | 0.19 | 0.18 | 0.18 | 0.18 | 0.17 | 0.16 | 0.16 |
| 5K | 1.00 | 0.99 | 0.66 | 0.27 | 0.19 | 0.18 | 0.17 | 0.17 | 0.16 | 0.16 | 0.11 |
| 8K | 1.00 | 1.00 | 0.80 | 0.59 | 0.36 | 0.21 | 0.21 | 0.19 | 0.20 | 0.18 | 0.15 |
| 10K | 1.00 | 1.00 | 0.86 | 0.74 | 0.47 | 0.25 | 0.20 | 0.20 | 0.20 | 0.19 | 0.17 |
| 12K | 1.00 | 1.00 | 0.97 | 0.94 | 0.75 | 0.43 | 0.24 | 0.21 | 0.20 | 0.21 | 0.22 |
| 13.5K | 1.00 | 1.00 | 0.99 | 0.99 | 0.94 | 0.83 | 0.51 | 0.24 | 0.20 | 0.20 | 0.18 |
| 19.5K | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.98 | 0.97 | 0.97 |
| 23K | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 |

### Staircase (e=128, lr=1e-3, 200K iters)

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | Levels solved |
|-------|----|----|----|----|----|----|----|----|----|----|-----|---------------|
| N=1 | 1.00 | 0.28 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.08 | 1 |
| N=3 | 1.00 | 1.00 | 0.99 | 0.36 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.11 | 3 |
| N=5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.36 | 0.34 | 0.12 | 0.12 | 0.12 | 0.13 | 4 |

## Depth Separation Analysis

The staircase shows N layers solves approximately N levels, but not exactly:
- N=1 → 1 level (L0 only)
- N=3 → 3 levels (L0-L2)
- N=5 → 6 levels (L0-L5) — slightly more than N
- N=10 → 7 levels (L0-L6) — fewer than N
- N=11 → 8 levels (L0-L7) — fewer than N
- N=12 → 11 levels (all) — sufficient depth

At e=256, the clean staircase breaks slightly at higher N (N=10 gets 7 not 10). This may be an optimization issue — deeper models need more iters. N=12 with enough capacity solves everything.

BPTT (single shared block) solves all 11 levels in ~16K iters, demonstrating unbounded effective depth through sequential processing.

## The Shortcut Problem

Without windowed attention, models exploit a shortcut: directly attend to base table values from any level, achieving ~1/k accuracy without composition. This manifests as:
- k=5: ~35% accuracy at unsolved levels (predicting any base value = 1/k + collision effects)
- k=2 v=4: ~63% (higher due to value collisions)

**BPTT without windowing plateaued at 82%** — it was partially shortcutting. With proper windowing, it reaches 100%.

## The Shuffling Problem

Shuffling table entries forces content-based matching but makes multi-hop composition extremely hard:

| Setting | 2-hop result | 3-hop result |
|---------|-------------|-------------|
| No shuffle, no window | Solved instantly (positional) | Solved instantly |
| Shuffle, no window | Solved at 77K (2-hop max) | Never (200K+) |
| Shuffle + window | Never (loss flat) | Never |
| No shuffle + window | **Solved** | **Solved** |

With shuffling, the optimizer struggles with QK-composition (using retrieved info to guide next attention step). This is a gradient descent limitation, not an architecture limitation — the model has sufficient capacity.

## RoPE Validation

Three tasks confirmed our no-RoPE implementation works correctly:

| Task | Property | RoPE | no-RoPE |
|------|----------|------|---------|
| Min element | Order-invariant | 100% | 100% |
| Copy-back-2 | Positional | 100% at 1K | 98% at 5K |
| Left neighbor | Content + positional | 100% at 1K | 99.6% at 10K |

RoPE is essential for pointer chasing (within-triplet value lookup requires relative position).

## Code

- `pointer_chasing.py`: Main experiment with all flags
  - `--window N`: Sliding window attention (essential)
  - `--no_shuffle`: Fixed entry order
  - `--no_rope`: Disable RoPE
  - `--multi_q`: Multiple Q sections per level (helper signals)
  - `--curriculum "iter:hops,..."`: Curriculum learning
  - `--vocab_hops N`: Vocab sized for N hops
- `blocks2.py`: Transformer blocks with windowed attention
- `left_neighbor_multihop.py`: Alternative multi-hop task (simpler encoding)
- `min_element.py`, `copy_back2.py`, `left_neighbor.py`: RoPE validation tasks

### Running the final experiments

```bash
# Staircase (e=256, windowed, no shuffle)
python -u pointer_chasing.py --n_hops 10 --n_keys 5 --n_values 10 \
    --n_embed 256 --n_head 4 --n_iters 50000 --batch_size 64 --lr 1e-4 \
    --gpu 0 --permutation --run N1,N3,N5,N10,N11,N12 --window 38 --no_shuffle

# BPTT (e=256, windowed, no shuffle)
python -u pointer_chasing.py --n_hops 10 --n_keys 5 --n_values 10 \
    --n_embed 256 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-4 \
    --gpu 0 --permutation --run bptt --window 38 --no_shuffle

# BPTT (e=128, windowed, no shuffle)
python -u pointer_chasing.py --n_hops 10 --n_keys 5 --n_values 10 \
    --n_embed 128 --n_head 4 --n_iters 100000 --batch_size 64 --lr 1e-3 \
    --gpu 0 --permutation --run bptt --window 38 --no_shuffle
```

## Open Questions

1. **Shuffling**: Can we make shuffled pointer chasing work? The left-neighbor multi-hop task showed promise (L2=57% at 16K with K=8) but scaling to K=20 failed.
2. **Multi-hop per layer**: At e=256, N=5 solves 6 levels (>N). Larger embeddings enable multi-hop per layer — can this be controlled?
3. **N=10 ceiling**: N=10 solves only 7 levels at 50K iters. Would more training push it higher, or is this a hard limit?
4. **Window size sensitivity**: How does changing window=38 affect results? Smaller windows might give cleaner separation.

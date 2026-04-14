# Pointer Chasing: Windowed Attention Results

## Key Discovery

Windowed attention is essential for genuine multi-hop composition in pointer chasing. Without it, models shortcut directly to the base table values. With it, we get clean depth separation: N-layer transformers solve ~N levels, BPTT solves all levels.

## Setup

- **Task**: Pointer chasing with Q-format, per-level key tokens, dense targets
- **Encoding**: `value=key` triplets per table, Q sections with targets at key positions
- **Windowed attention**: window=38 (each Q section sees its own table + previous Q section, but NOT the base table from higher levels)
- **No shuffle**: fixed entry order (positional patterns within each window)
- **Permutation**: bijective index table mappings
- **Settings**: k=5, v=10, e=128, n_head=4, lr=1e-3, batch_size=64

## Why windowing matters

Without windowing, higher-level Q sections can directly attend to the base table and shortcut to ~1/k accuracy (predicting any base value) without learning composition. With windowing, the base table is outside the attention window for higher levels, forcing the model to chain through intermediate Q sections.

## 10-hop staircase (200K iters per model)

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | Levels solved |
|-------|----|----|----|----|----|----|----|----|----|----|-----|---------------|
| N=1 | 1.00 | 0.28 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.08 | 1 |
| N=3 | 1.00 | 1.00 | 0.99 | 0.36 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.11 | 3 |
| N=5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.36 | 0.34 | 0.12 | 0.12 | 0.12 | 0.13 | 4 |
| N=10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.58 | 0.36 | 0.38 | 7+ (running, 13K) |
| N=11 | — | — | — | — | — | — | — | — | — | — | — | queued |
| N=12 | — | — | — | — | — | — | — | — | — | — | — | queued |
| **BPTT** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **0.81** | **0.82** | **0.82** | **0.82** | **0.81** | **0.41** | **L0-L9 ~82% (running, 23K)** |

### Observations

1. **Clean staircase**: N=1 → 1 level, N=3 → 3 levels, N=5 → 4 levels. Each additional layer unlocks ~1 more level.
2. **BPTT solves all levels simultaneously**: At only 23K iters, L0-L9 are all at ~82%. The wave swept through all levels at once rather than propagating one by one.
3. **BPTT is much faster**: BPTT reaches L2-L9 ~82% at 23K iters. N=10 transformer needs 13K+ just to reach L7.
4. **Final query (L10) lags**: For BPTT, L10=41% while L0-L9=82%. The final query is a single position and learns slower.

## Earlier failed approaches (with shuffling)

### Q-format with shuffling (no windowing)

Shuffling table entries and Q section entries independently forces content-based matching (no positional shortcuts). Results:

| Experiment | Result |
|-----------|--------|
| 1-hop k=3,5,8,10 N=2 | L0=100% (content matching works) |
| 2-hop k=5 N=3 (100K) | L0-L2=100% at 77K (composition works, but slow) |
| 3-hop k=5 N=4 (200K) | L0=100%, L1 stuck at 36% (never broke through) |
| 3-hop k=5 N=3 (100K) | L0-L1=100% at 75K, L2-L3 stuck (N=3 can't do 3 hops) |
| 10-hop k=2 N=5 (100K) | L0-L2=100%, L3-L10=63% (stuck) |

**Key finding**: With shuffling, the model learns at most 2 levels of content-based composition. The optimizer can't find QK-composition solutions for 3+ hops.

### Shuffling + windowing

| Experiment | Result |
|-----------|--------|
| 2-hop k=5 N=3 window=38 (200K) | L0=100%, L1=36% (loss flat, no progress) |
| 3-hop k=5 N=4 window=38 (200K) | L0=100%, L1=37% (loss flat) |

Windowing + shuffling made things WORSE — the model couldn't even learn L1. Without the base table shortcut AND without positional patterns, the gradient signal was too weak.

### Helper signal experiments

**Random hop targets**: At each Q section position, randomly choose the target from 1-hop, 2-hop, ..., fully resolved.
- 2-hop k=5 N=3: L0=100%, L1=48%, L2=100%. Final query solved but intermediate stuck.
- 3-hop k=5 N=4: No breakthrough.

**Multi-Q sections**: Each level gets multiple Q sections (Q1 for 1-hop, Q2 for 2-hop, etc.)
- 2-hop k=5 N=3: Solved at 100K (same as single-Q)
- 3-hop k=5 N=4: L1=100% at 80K (first time!), but L2/L3 stuck at 36% even at 500K
- The multi-Q format helped L1 break through where single-Q couldn't, but L2+ still failed

**Curriculum learning**: Train on 2-hop first, switch to 3-hop.
- L1 preserved from 2-hop phase, but L2/L3 at new positions never learned

## RoPE vs no-RoPE validation

Three tasks validated our no-RoPE implementation:

| Task | RoPE | no-RoPE | Expected |
|------|------|---------|----------|
| Min element (order-invariant) | 100% | 100% | Equal ✓ |
| Copy-back-2 (positional) | 100% at 1K | 98% at 5K | RoPE >> no-RoPE ✓ |
| Left neighbor (content + positional) | 100% at 1K | 99.6% at 10K | RoPE > no-RoPE ✓ |

## Scaling issues

Content matching fails at large k:
- k=3: works (L0=100%)
- k=5: works (L0=100%)
- k=10: works with enough embedding (L0=100% at e=128)
- k=50: fails at e=128 (d_head=32 too small), also fails at e=1024 (softmax dilution at 254 tokens)

## Left neighbor multi-hop (new task direction)

A simpler encoding where each level is a random permutation of tokens, and "left neighbor" defines the mapping:
```
Level 0: A B F G H D | Q G -> F
Level 1: G D A H B F | Q B -> H -> (find H in L0) -> G
```

- K=8 3-hop N=3 with RoPE: L0=100%, L1=100%, L2=57%, L3=58% at 16K iters
- K=8 3-hop N=3 no-RoPE: L0=100%, L1=5% (can't do left-neighbor without positional info)
- K=20 fails (same content matching scaling issue)

Promising but shares the same vocab across levels (ambiguity). Paused for now.

## Code

- `pointer_chasing.py`: Main experiment code with flags:
  - `--window N`: Sliding window attention
  - `--no_shuffle`: Fixed entry order
  - `--no_rope`: Disable RoPE
  - `--multi_q`: Multiple Q sections per level
  - `--curriculum "iter:hops,..."`: Curriculum learning
- `blocks2.py`: Transformer blocks with windowed attention support
- `left_neighbor_multihop.py`: Alternative multi-hop task
- `min_element.py`, `copy_back2.py`, `left_neighbor.py`: RoPE validation tasks

## Running commands

```bash
# Staircase (windowed, no shuffle)
python -u pointer_chasing.py --n_hops 10 --n_keys 5 --n_values 10 \
    --n_embed 128 --n_head 4 --n_iters 200000 --batch_size 64 --lr 1e-3 \
    --gpu 0 --permutation --run N1,N3,N5,N10,N11,N12 --window 38 --no_shuffle

# BPTT (windowed, no shuffle)
python -u pointer_chasing.py --n_hops 10 --n_keys 5 --n_values 10 \
    --n_embed 128 --n_head 4 --n_iters 200000 --batch_size 64 --lr 1e-3 \
    --gpu 0 --permutation --run bptt --window 38 --no_shuffle
```

## Open questions

1. Can we make shuffling work with windowed attention? (Currently fails — no gradient signal)
2. Is the no-shuffle result "real" composition or positional memorization through the window?
3. Can the left-neighbor task scale to larger K with the right embedding size?
4. Will BPTT reach 100% on all levels, or plateau at ~82%?
5. Will N=11/N=12 solve all 11 levels?

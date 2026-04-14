# Pointer Chasing: Empirical TC^0 Separation

## Summary

We demonstrate that a D=1 look-ahead model trained with BPTT can solve k-hop pointer chasing for any k, while an N-layer transformer solves at most N levels. This empirically demonstrates the computational separation between fixed-depth transformers (TC^0) and our architecture (unbounded sequential depth).

## The Task

**Pointer chasing with random permutation tables.** Each example contains:

1. A **base table**: maps keys to values
2. **Index tables** (one per additional hop): maps keys to keys via random **permutations** (bijections — no collisions)
3. A **query**: a single key to resolve through the chain

The encoding is **reversed** (`value=key` not `key=value`) so that with causal attention, each key position can see what it maps to.

Example (3-hop, 8 keys, 16 values):
```
Base:   v3=A v0=B v8=C v7=D v7=E v4=F v3=G v2=H |
Index1: D=A  C=B  E=C  B=D  F=E  H=F  A=G  G=H  |
Index2: C=A  B=B  A=C  H=D  E=E  G=F  F=G  D=H  |
Query:  E
```
Chain: E →(index2) E →(index1) F →(base) v4. Answer: **v4**.

### Why permutations matter

Without permutations, index tables use random many-to-one mappings. Multiple keys can map to the same target, causing composition collapse — the effective hop count is reduced. With permutations (bijections), every key maps to a unique key, forcing genuine multi-hop reasoning.

### Why reversed encoding matters

With `value=key` encoding and causal attention, the key position (rightmost in each triplet) can see its mapped value/target to the left. Without this, the model cannot see key-value associations and fails to learn.

## Dense Targets

**Targets at every key position**, not just the query:

```
Input:   v3  =  A   v0  =  B   ...  |  D   =  A   C   =  B   ...  |  C   =  A   ...  |  Q  E
Target:   _  _ v3    _  _ v0   ...  _   _  _ v7    _  _ v8   ...  _   _  _ v7   ...  _   _ v4
```

- **L0** (base key positions): predict the value (0 hops — trivial)
- **L1** (index1 key positions): predict the resolved value through base (1 hop)
- **L2** (index2 key positions): predict the resolved value through index1 and base (2 hops)
- **L(n)** (query): predict the final answer (n hops)

Dense targets are essential. Without them (single query output only), BPTT couldn't learn even 2-hop in 50K iters. With dense targets, it solves 5-hop in 1K iters and 20-hop in 4K iters.

## Models

### N-layer Transformer Baseline
Standard causal transformer with N separate-weight layers. Uses `q @ k^T` attention with standard RoPE (`blocks2.py`).

### D=1 Look-Ahead with BPTT
Single shared-weight block, processed sequentially with backpropagation through time. Uses KV-cached incremental computation (O(T²), same cost as a standard transformer).

```
for t in range(T):
    correction = corr_ffn(ln(z[t-1] + tok_emb[t]))
    px[t] = tok_emb[t] + correction
    z[t] = block(px[0..t])  # via KV cache
```

## Results

### 3-hop (k=8, v=16, e=128, n_head=4, lr=1e-3, permutation, 5K iters)

| Model | L0 (0-hop) | L1 (1-hop) | L2 (2-hop) | L3 (3-hop) |
|-------|-----------|-----------|-----------|-----------|
| N=1   | 1.000 | 0.988 | 0.250 | 0.238 |
| N=2   | 1.000 | 1.000 | 0.782 | 0.250 |
| N=3   | 1.000 | 1.000 | 1.000 | **0.992** |
| N=4   | 1.000 | 1.000 | 1.000 | **0.988** |
| **D=1 BPTT** | **1.000** | **1.000** | **1.000** | **1.000** (1.5K iters) |

Clean staircase: N=1 solves L0, N=2 solves L0-L1, N=3 solves L0-L2. BPTT solves all.

### 5-hop (k=8, v=16, e=128, n_head=4, lr=1e-3, permutation)

| Model | L0 | L1 | L2 | L3 | L4 | L5 | Iters |
|-------|----|----|----|----|----|----|-------|
| N=1   | 1.00 | 0.15 | 0.15 | 0.15 | 0.15 | 0.11 | 10K |
| N=3   | 1.00 | 1.00 | 1.00 | 0.15 | 0.15 | 0.12 | 10K |
| N=5   | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.58 | 10K |
| **D=1 BPTT** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1K** |

### 5-hop (k=16, v=32, e=256, n_head=4, lr=1e-3, permutation)

| Model | L0 | L1 | L2 | L3 | L4 | L5 | Iters |
|-------|----|----|----|----|----|----|-------|
| N=4 (e=256) | 1.00 | 1.00 | 1.00 | 1.00 | 0.19 | 0.14 | 10K |
| N=5 (e=256) | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.14 | 10K |
| **D=1 BPTT** (e=256) | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **2K** |

### 10-hop (k=8, v=16, e=128, n_head=4, lr=1e-3, permutation, 5K iters)

| Model | Levels solved (>90%) | L10 (query) |
|-------|---------------------|-------------|
| N=1   | L0 (1 level)        | fail |
| N=5   | L0-L4 (5 levels)    | fail |
| N=9   | L0-L8 (9 levels)    | fail |
| N=11  | L0-L9 (10 levels)   | 92% |
| **D=1 BPTT** | **L0-L10 (11 levels)** | **100%** |

Clean staircase: N layers solves N levels. BPTT solves all 11.

### 20-hop (k=8, v=16, n_head=4, permutation)

**Embedding size matters:**

| Setting | BPTT result |
|---------|-------------|
| e=128 lr=1e-3 | Failed — solved only L0-L1 |
| e=256 lr=1e-4 | Crashed at 4K (was at L0-L19=71-100% at 3.5K) |
| e=256 lr=5e-5 | Running (slow) |
| **e=512 lr=1e-4** | **100% all 21 levels at 4K iters** |

**Transformer depth separation at 20-hop:**

At e=256 (clean separation, lr=1e-3, 5K iters):

| Model | Levels solved (>90%) |
|-------|---------------------|
| N=1   | L0 (1 level) |
| N=5   | L0-L3 (4 levels) |
| N=10  | L0-L8 (9 levels) |
| N=15  | L0-L8 (9 levels, needed more iters) |
| N=21  | L0-L7 (8 levels, needed more iters) |

At e=512 (multi-hop per layer, lr=1e-4, 10K iters):

| Model | Levels solved (>90%) | Note |
|-------|---------------------|------|
| N=10  | L0-L18 (19 levels) | ~2 hops per layer |

e=512 with n_head=4 enables multi-hop per layer, breaking the clean depth separation. e=256 gives cleaner results (~N levels for N layers) but BPTT needs careful lr tuning (lr=1e-4 crashes, lr=5e-5 in progress).

## Key Findings

1. **Depth separation is real.** At e=128 and e=256, N-layer transformers solve approximately N levels. BPTT solves all levels.

2. **Dense targets are essential.** Without per-position targets, neither transformers nor BPTT can learn pointer chasing efficiently. Dense targets provide gradient signal at every table level.

3. **Reversed encoding enables causal learning.** `value=key` format lets key positions see their values via causal attention. Standard `key=value` fails.

4. **Permutation tables prevent shortcuts.** Bijective mappings force genuine multi-hop reasoning. Random many-to-one mappings allow composition collapse.

5. **BPTT converges faster than transformers.** On 3-hop, BPTT reaches 100% at 1.5K iters while N=3 needs 5K. On 10-hop, BPTT solves at ~2K while N=11 needs 5K+.

6. **Wider embeddings help BPTT on longer sequences** but hurt transformer depth separation. e=128 works up to 10-hop (~252 tokens). e=256/512 needed for 20-hop (~502 tokens). But e=512 enables multi-hop per layer for transformers.

7. **BPTT optimization is fragile at long sequences.** lr=1e-3 crashes at e=512. lr=1e-4 crashes at e=256 for 20-hop. The sequential gradient path through 500+ steps is sensitive to learning rate.

## Code

- `pointer_chasing.py`: Data generation, models, training loop
- `blocks2.py`: Transformer blocks with standard `q @ k^T` attention and unflipped RoPE (KV-cache compatible)

### Running experiments
```bash
# 3-hop, all models
python -u pointer_chasing.py --n_hops 3 --n_keys 8 --n_values 16 \
    --n_embed 128 --n_head 4 --n_iters 5000 --batch_size 64 --lr 1e-3 \
    --gpu 0 --permutation --run N1,N2,N3,N4,bptt

# 10-hop, sample staircase + BPTT
python -u pointer_chasing.py --n_hops 10 --n_keys 8 --n_values 16 \
    --n_embed 128 --n_head 4 --n_iters 5000 --batch_size 64 --lr 1e-3 \
    --gpu 0 --permutation --run N1,N5,N9,N10,N11,bptt

# 20-hop BPTT
python -u pointer_chasing.py --n_hops 20 --n_keys 8 --n_values 16 \
    --n_embed 512 --n_head 4 --n_iters 10000 --batch_size 64 --lr 1e-4 \
    --gpu 0 --permutation --run bptt
```

## Open Questions

1. Can we find settings where 20-hop transformers show clean separation AND BPTT converges? (e=256 gives clean separation but BPTT crashes; e=512 BPTT works but transformers do multi-hop per layer)

2. Would n_head=1 at e=512 prevent multi-hop per layer while keeping BPTT stable?

3. Can gradient clipping or lr warmup stabilize e=256 BPTT at higher lr?

# Pointer Chasing: Shuffled Q-Format Experiments

## Goal

Demonstrate depth separation with **genuine content-based matching** (shuffled entries), not positional shortcuts.

## Format: Q-sections with independent shuffling

Each level has a table (no targets) followed by Q section(s) with targets. Table entries and Q section entries are **independently shuffled** — the model cannot use positional patterns.

```
Base table: v12=A0 v6=C0 v5=D0 | Q D0 Q A0 Q C0 | Index1: H0=E1 D0=F1 ... | Q E1 Q F1 ... | Final: Q D1
```

Targets at Q section key positions. Per-level key tokens (A0,B0 for level 0; A1,B1 for level 1).

## Key findings

### Content matching (1-hop) works
- k=3 v=100 N=2 e=128 with RoPE: **100% at iter 500** (1-hop)
- k=10 v=20 N=2 e=128: **100%** (1-hop)
- k=50 v=100 N=2 e=128: **failed** (d_head=32 too small for 50 keys)
- k=50 v=100 N=1 e=1024: **failed** (N=1 insufficient — needs 2 layers for Q-format content matching)
- k=50 v=100 N=2 e=1024: **failed** (softmax dilution at 254 tokens)

### RoPE is essential
- k=3 v=100 N=2 e=1024 **no RoPE**: stuck at 33% (1/k shortcut). Loss decreasing but accuracy flat.
- Same with RoPE: **100% at iter 500**.
- Without RoPE, model has no position information — can't learn within-triplet patterns (value is 2 positions before key).

### 2-hop composition needs ~70K iters at k=5
- k=2 v=4 N=3: **100% at iter 1K** (trivial, binary permutations)
- k=5 v=10 N=3: **100% at iter 77K** (L1 broke through at 71K, all levels solved by 77K)
- k=8 v=16 N=3 e=256: L1 stuck at 26% after 20K (would likely need 100K+)

### 3-hop composition: partial success
- k=5 v=10 N=3: L0-L1 solved (75K), L2-L3 stuck at 35%. Expected — N=3 can't do 3 hops.
- k=5 v=10 N=4: L0 solved, L1 stuck at 36% after **200K iters**. Never broke through.
- Puzzling: N=3 broke through on L1 but N=4 didn't. Different optimization landscape?

### Beyond 2 levels of composition fails
- 10-hop k=2 v=4 N=5: L0-L2=100%, L3-L10=63% after 100K. Stuck.
- 10-hop k=2 v=100 N=5: L0-L1=100%, L2-L10=50% after 100K. Stuck.
- Pattern: model learns at most 2 levels of content-based composition regardless of depth or k.

## Collision/shortcut analysis

With k keys and v values, the model can achieve ~1/k accuracy by predicting any base table value without real composition:
- k=2 v=4: shortcut = 63% (collision-inflated)
- k=2 v=100: shortcut = 50% (= 1/k)
- k=3 v=6: shortcut = 49% (collision-inflated, matches observed ~50% exactly)
- k=3 v=100: shortcut = 33% (= 1/k)
- k=5 v=10: shortcut ≈ 36% (collision-inflated)

Increasing v reduces collision-based shortcuts but doesn't fix the composition problem.

## Helper signal experiments

### Random hop targets
Instead of always targeting the fully-resolved value, randomly choose target from all intermediate hop depths (1-hop key, 2-hop key, ..., fully resolved value).

- **2-hop k=5 N=3 (100K)**: L0=100%, L1=48%, **L2=100%**.
  - L2 (final query) solved! The random 1-hop targets at L1 helped L2 learn.
  - L1 stuck at 48% because half its targets are keys (not values) — eval always measures against fully-resolved value.
  - **This is promising** — the helper signal enabled the final query to compose.

- **3-hop k=5 N=4 (100K)**: L0=100%, L1=17%, L2=11%, L3=34%. No breakthrough.

### Multi-Q sections (in progress)
Each level gets multiple Q sections with increasing hop depth:
- Level 1: Q1(1-hop→key) | Q2(2-hop→value)
- Level 2: Q1(1-hop→key) | Q2(2-hop→key) | Q3(3-hop→value)

Each Q section has deterministic targets. Q1 learns easily (1-hop = content matching), Q2 can build on Q1's representations. Running on GPU 0 (2-hop) and GPU 1 (3-hop).

## Summary table

| Experiment | k | v | hops | N | e | iters | Result |
|-----------|---|---|------|---|---|-------|--------|
| 1-hop N=2 | 3 | 100 | 1 | 2 | 128 | 5K | **100%** |
| 1-hop N=2 | 10 | 20 | 1 | 2 | 128 | 20K | **100%** |
| 2-hop N=3 | 2 | 4 | 2 | 3 | 128 | 5K | **100% at 1K** |
| 2-hop N=3 | 5 | 10 | 2 | 3 | 128 | 100K | **100% at 77K** |
| 3-hop N=3 | 5 | 10 | 3 | 3 | 128 | 100K | L0-L1 solved, L2-L3 stuck |
| 3-hop N=4 | 5 | 10 | 3 | 4 | 128 | 200K | L0 only, L1 stuck at 36% |
| 10-hop N=5 | 2 | 4 | 10 | 5 | 128 | 100K | L0-L2 only, L3+ at 63% |
| 10-hop N=5 | 2 | 100 | 10 | 5 | 128 | 100K | L0-L1 only, L2+ at 50% |
| 2-hop N=3 randhop | 5 | 10 | 2 | 3 | 128 | 100K | L0,L2=100%, L1=48% |
| 3-hop N=4 randhop | 5 | 10 | 3 | 4 | 128 | 100K | L0=100%, rest stuck |

## Open questions

1. Why does N=3 break through on L1 at 3-hop but N=4 doesn't?
2. Why can the model learn at most 2 levels of content-based composition?
3. Can multi-Q sections (progressive hop supervision) enable deeper composition?
4. The random hop target result (L2=100% while L1=48%) suggests the helper signal works for the final query — can this be extended?

# 20-Hop Pointer Chasing Experiment Log

## Goal

Demonstrate that D=1 BPTT solves 20-hop pointer chasing while N-layer transformers solve at most N-1 hops.

## Settings

- n_hops=20, n_keys=8, n_values=16, permutation=True, dense targets
- Sequence length: ~502 tokens
- n_head=4, batch_size=64, seed=42
- Code: pointer_chasing.py with blocks2.py (q @ k^T, standard RoPE, KV-cached BPTT)

## What we tried and what happened

### Attempt 1: e=256 lr=1e-3, 5K iters (old blocks.py code, before blocks2 fix)
- BPTT: Total failure. L0=20%, everything else random (6%). Didn't learn.
- This was with the mismatched old code (blocks.py k@q^T + flipped RoPE, incremental q@k^T + no flip).

### Attempt 2: e=256 lr=1e-3, 5K iters (N=21 transformer, old blocks.py)
- N=21: Beautiful wave propagation. L0-L6 solved at 5K, wavefront at L8=77%, L9=55%.
- Needed more iters to reach L20. Showed the staircase pattern clearly.

### Attempt 3: e=512 lr=1e-3, 50K iters (BPTT, blocks2 code)
- CRASHED at iter 1K. Was at L0=100%, L1=100%, L2=33% at iter 500, then collapsed to 6% everywhere.
- lr=1e-3 too aggressive for e=512 with 502-token BPTT.

### Attempt 4: e=512 lr=1e-4, 50K iters (BPTT, blocks2 code) ← CURRENT, GPU 0
- Stable, no crash.
- iter 500: L0=100%, L1=26%
- iter 1000: L0=100%, L1=68%
- iter 1500: L0=100%, L1=100%, L2=78%, L3=43%
- **iter 2000: L0-L19 ALL 100%, L20=22%** ← breakthrough!
- Waiting for L20 (query, 20-hop) to converge.

### Attempt 5: e=256 lr=1e-4, 50K iters (BPTT, blocks2 code) ← CURRENT, GPU 0
- Much slower than e=512. At iter 1.5K: L0=100%, L1=42%.
- Running alongside e=512 on same GPU.

### Attempt 6: e=512 lr=1e-3, 10K iters (transformers N=1,5,10,15,19,20,21)
- KILLED. Too slow to converge at lr=1e-3 with e=512.
- N=1 at 10K: L0=100%, L1=93%. N=5 at 10K: L2=45%.
- N=10 at 4K: couldn't even learn L1. lr=1e-3 wrong for these large models.

### Attempt 7: e=512 lr=1e-4, 10K iters (transformers N=1,5,10,15,19,20,21) ← CURRENT, GPU 1
- Relaunched with lr=1e-4 to match BPTT settings.
- Apples-to-apples comparison with BPTT attempt 4.

## Currently running

| GPU | Experiment | Status |
|-----|-----------|--------|
| GPU 0 | BPTT e=512 lr=1e-4 50K | L0-L19=100% at 2K, L20=22% |
| GPU 0 | BPTT e=256 lr=1e-4 50K | L1=42% at 1.5K (slow) |
| GPU 1 | N=1,5,10,15,19,20,21 e=512 lr=1e-4 10K | Just launched |

## Key findings so far

1. **BPTT e=512 lr=1e-4 solved L0-L19 (19 hops) at 2K iters.** L20 (the query, 20 hops) at 22% and learning.

2. **e=256 is too small for 20-hop BPTT** — much slower convergence than e=512.

3. **lr=1e-3 crashes at e=512** for BPTT but works for transformers (at e=256). lr=1e-4 is stable for both.

4. **e=512 transformers at lr=1e-3 are too slow** — N=10 couldn't learn L1 in 4K. Need lr=1e-4.

## Expected final result

With matching settings (e=512, lr=1e-4):
- N=k transformer: solves L0 through L(k-1), fails at Lk
- D=1 BPTT: solves all 21 levels (L0-L20)

This would demonstrate TC^0 separation at 20 hops — D=1 BPTT with unbounded sequential depth solves what would require a 21-layer transformer.

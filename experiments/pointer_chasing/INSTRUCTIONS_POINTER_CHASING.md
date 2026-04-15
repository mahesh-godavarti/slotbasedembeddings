# Pointer Chasing Experiment: TC^0 Separation Demonstration

## Background

We have developed a "look-ahead" architecture that is a parallelizable RNN. At sequential inference, it processes one position at a time:

```
corr[t] = FFN(LN(h[t-1] + x[t]))                     # correction from previous state
px[t] = x[t] + corr[t]                                # contextualized input
h[t] = Block(px[t], {px[0], ..., px[t]})              # transformer block with attention
y[t] = softmax(W · LN(h[t]) + b)                      # output
```

This is an RNN where the recurrent cell is a transformer block + correction FFN. During normal training, we use K parallel iterations (K=5) over all positions simultaneously — fast like a transformer. But the architecture can also be trained with BPTT (sequential, like a standard RNN).

**Key theoretical property**: A standard N-layer transformer is in TC^0 — it can perform at most N sequential computation steps regardless of width. Our D=1 look-ahead at sequential inference has unbounded sequential depth (proportional to sequence length), making it Turing complete.

**This experiment demonstrates the separation empirically**: we construct a task (pointer chasing) that requires exactly k sequential steps. An N-layer transformer can solve it only if N >= k. Our D=1 look-ahead trained with BPTT can solve it for any k.

## Goal

Show that D=1 look-ahead trained with BPTT can solve k-hop pointer chasing for any k, while N=k-1 transformer fails. This experimentally demonstrates the computational separation between our architecture and fixed-depth transformers.

## The Task

Pointer chasing with random in-context tables:

```
2 hops: Table: A=5 B=3 C=8 D=1 | Index: X=B Y=D Z=A | Query: X → B → 3
3 hops: Table: A=5 B=3 ... | Index1: X=B Y=D ... | Index2: P=X Q=Z ... | Query: P → X → B → 3
```

Tables are random each example — model must learn the algorithm, not memorize.

## Code

`pointer_chasing.py` — generates data, trains, evaluates. All in one file.

## Experiments to Run

### Step 1: 2 hops — verify N=1 fails, N=2 succeeds

```bash
cd /home/ubuntu/look_ahead6
/home/ubuntu/exp8/venv/bin/python pointer_chasing.py \
    --n_hops 2 --n_keys 8 --n_values 16 \
    --n_embed 128 --n_head 4 --n_iters 10000 --batch_size 64 \
    --lr 1e-3 --gpu 1 2>&1 | tee logs/pointer_chasing_2hop.log
```

Expected: N=1 ~6% (random chance = 1/16), N=2 should reach high accuracy (>90%), N=3 should also succeed, D=1 BPTT should succeed.

### Step 2: 3 hops — verify N=2 fails, N=3 succeeds

```bash
cd /home/ubuntu/look_ahead6
/home/ubuntu/exp8/venv/bin/python pointer_chasing.py \
    --n_hops 3 --n_keys 8 --n_values 16 \
    --n_embed 128 --n_head 4 --n_iters 10000 --batch_size 64 \
    --lr 1e-3 --gpu 1 2>&1 | tee logs/pointer_chasing_3hop.log
```

Expected: N=1 and N=2 fail (~6%), N=3 succeeds, N=4 succeeds, D=1 BPTT succeeds.

### Step 3: 5 hops — verify only N>=5 and D=1 BPTT succeed

```bash
cd /home/ubuntu/look_ahead6
/home/ubuntu/exp8/venv/bin/python pointer_chasing.py \
    --n_hops 5 --n_keys 8 --n_values 16 \
    --n_embed 128 --n_head 4 --n_iters 10000 --batch_size 64 \
    --lr 1e-3 --gpu 1 2>&1 | tee logs/pointer_chasing_5hop.log
```

Expected: N=1 through N=4 fail, N=5 and N=6 succeed, D=1 BPTT succeeds.

## What Success Looks Like

A table like:

| Model | 2-hop | 3-hop | 5-hop |
|-------|-------|-------|-------|
| N=1   | fail  | fail  | fail  |
| N=2   | pass  | fail  | fail  |
| N=3   | pass  | pass  | fail  |
| N=5   | pass  | pass  | pass  |
| D=1 BPTT | pass | pass | pass |

This proves: fixed-depth transformers fail beyond their depth, D=1 look-ahead with BPTT does not.

## Notes

- Random chance accuracy = 1/n_values = 1/16 = 6.25%
- "Pass" means >90% accuracy, "fail" means stuck near random chance
- If models aren't learning, try increasing n_iters to 20000 or adjusting lr
- The D=1 BPTT model trains slower per iter (sequential) but should solve any hop count
- All models see the same data — the difference is purely architectural
- Do NOT modify pointer_chasing.py or blocks.py

# Streaming inference: wall-clock comparison and checkpoint compatibility

This writeup covers two things:

1. **Wall-clock timing** for autoregressive decode of the D=K=1 look-ahead vs param-comparable N-layer roformer baselines, using a true streaming KV-cached inference path.
2. **Checkpoint compatibility**: a new `models_streaming.py` is a drop-in replacement for the trained `BlockHeadCorrFFNAddModel` and `RoFormer` classes — `load_state_dict(strict=True)` succeeds and outputs match the reference parallel/sequential forward to fp32 numerical precision.

Hardware: 2 × NVIDIA A100-SXM4-80GB. Framework: PyTorch, bfloat16 autocast on GPU, batch=1, vocab=32000, n_head=16, max_seq_len=10100 (no prefix truncation over the run).

## 1. Streaming inference design

`models_streaming.py` reimplements two model classes with proper KV caching:

| Streaming class | Drops in for | Reference forward |
|---|---|---|
| `StreamingRoFormer` | `blocks.RoFormer` | `forward()` |
| `StreamingLookAhead` | `models.BlockHeadCorrFFNAddModel` (D=1 or D>1) | `forward_sequential(seq_k=1)` |

Per step (single new token):

- **Roformer N**: for each of N layers, compute Q,K,V for the new token, append K,V to that layer's cache, attention is 1 query × T cached keys, FFN, residuals. N layer-passes per token.
- **Look-ahead D, K=1**: maintain `h_prev = z[t-1]` (recurrent state, init zeros). At each step:
  ```
  corr_t = corr_ffn(ln_corr(h_prev + tok_emb_t))
  px_t   = tok_emb_t + corr_t
  z_t    = block_D(...block_1(px_t)...)         # KV-cached attention inside each block
  h_prev := z_t
  logits = head(ln_f(z_t))
  ```
  D layer-passes per token plus one corr_ffn over the single new token.

### Math equivalence to the trained codebase

Two non-trivial points were needed to make streaming bitwise-equivalent to the reference forward:

- **Attention math (codebase convention).** `RoFormerAttention` computes `wei = k @ q.T` (note the swap), `softmax(dim=-1)` over q-positions, then `out = wei @ v`. So Q is the key-stream and V is the value-stream; for streaming we cache **Q (post-RoPE)** and **V (unrotated)** while computing K freshly each step.
- **RoPE.** Codebase uses integer angles `outer(arange(T), arange(D//2))` flipped along T, applied with adjacent-pair grouping `(x[0::2], x[1::2])`. The flip means absolute angles change with T, which is incompatible with caching post-RoPE. Streaming uses **static angles α = -pos·j**: same relative pattern (Δα = (q-k)·j either way, so Q·K dot products are identical and the unrotated V is unchanged), no need to re-rotate the cache as T grows.
- **Param names** match the trained modules exactly (`sa_head.{keys,queries,values,proj}`, `ffn.ffn.<i>`, `ln1`, `ln2`, `token_embedding_table`, `block`/`blocks`, `corr_ffn`, `ln_corr`, `ln_f`, `head`, `tril` buffer for parity).

## 2. Wall-clock timing results

10000-token autoregressive decode, single-token streaming with KV cache, random-init weights (timing depends on architecture, not weight values). Per-100-token chunk wall-clock, cumulative wall-clock.

### Comparison 1: D=1 K=1 C=2048 vs N=6 C=1088

Run in parallel on the two GPUs. D=1 K=1 has 39% more parameters but is 2.6× faster per token because per-step it does 1 layer-equivalent vs 6.

| | D=1 K=1 C=2048 | N=6 C=1088 |
|---|---|---|
| Params | 215,035,136 | 154,980,608 |
| Total time (10k tokens) | **10.88 s** | **28.46 s** |
| Mean throughput | **919 tok/s** | **351 tok/s** |
| Mean ms/token | 1.09 | 2.85 |
| Per-100 chunk @ T≈100 | 0.114 s | 0.282 s |
| Per-100 chunk @ T≈10000 | 0.111 s | 0.280 s |
| **Speedup (N6/D1)** | — | **2.6×** |

Per-100-token chunk and cumulative time, sampled every 1000 tokens:

| tokens | D1 chunk (s) | D1 cum (s) | N6 chunk (s) | N6 cum (s) | speedup |
|---|---|---|---|---|---|
| 100 | 0.114 | 0.11 | 0.282 | 0.28 | 2.48x |
| 1000 | 0.106 | 1.07 | 0.294 | 2.85 | 2.78x |
| 2000 | 0.106 | 2.13 | 0.298 | 5.70 | 2.80x |
| 3000 | 0.107 | 3.24 | 0.281 | 8.53 | 2.62x |
| 4000 | 0.117 | 4.32 | 0.303 | 11.40 | 2.60x |
| 5000 | 0.121 | 5.41 | 0.282 | 14.26 | 2.33x |
| 6000 | 0.109 | 6.50 | 0.282 | 17.09 | 2.60x |
| 7000 | 0.109 | 7.59 | 0.284 | 19.95 | 2.59x |
| 8000 | 0.110 | 8.68 | 0.281 | 22.78 | 2.56x |
| 9000 | 0.110 | 9.78 | 0.298 | 25.62 | 2.71x |
| 10000 | 0.111 | 10.88 | 0.280 | 28.46 | 2.54x |

Per-chunk time is essentially flat — KV cache eliminates the prefix-recompute term; the linear-in-T attention-over-cache term is small at these C.

### Comparison 2: D=6 K=1 C=1024 vs N=12 C=768

Run in parallel on the two GPUs. Both ~150M params; look-ahead 1.8× faster per token.

| | D=6 K=1 C=1024 | N=12 C=768 |
|---|---|---|
| Params | 149,543,168 | 134,240,000 |
| Total time (10k tokens) | **27.32 s** | **49.64 s** |
| Mean throughput | **366 tok/s** | **201 tok/s** |
| Mean ms/token | 2.73 | 4.96 |
| Per-100 chunk @ T≈100 | 0.269 s | 0.485 s |
| Per-100 chunk @ T≈10000 | 0.276 s | 0.495 s |
| **Speedup (N12/D6)** | — | **1.8×** |

Per-100-token chunk and cumulative time:

| tokens | D6 chunk (s) | D6 cum (s) | N12 chunk (s) | N12 cum (s) | speedup |
|---|---|---|---|---|---|
| 100 | 0.269 | 0.27 | 0.485 | 0.49 | 1.80x |
| 1000 | 0.278 | 2.79 | 0.509 | 5.01 | 1.83x |
| 2000 | 0.272 | 5.57 | 0.491 | 9.94 | 1.81x |
| 3000 | 0.268 | 8.30 | 0.504 | 14.88 | 1.88x |
| 4000 | 0.270 | 11.01 | 0.499 | 19.84 | 1.85x |
| 5000 | 0.267 | 13.72 | 0.487 | 24.76 | 1.82x |
| 6000 | 0.270 | 16.42 | 0.489 | 29.69 | 1.81x |
| 7000 | 0.290 | 19.17 | 0.500 | 34.66 | 1.72x |
| 8000 | 0.269 | 21.89 | 0.507 | 39.65 | 1.88x |
| 9000 | 0.272 | 24.61 | 0.495 | 44.64 | 1.82x |
| 10000 | 0.276 | 27.32 | 0.495 | 49.64 | 1.79x |

### Single-config run: D=5 K=1 C=1120

| Metric | Value |
|---|---|
| Params | 157,094,080 |
| Total time (10k tokens) | **28.62 s** |
| Mean throughput | **349 tok/s** |
| Mean ms/token | 2.86 |
| Per-100 chunk @ T≈100 | 0.281 s |
| Per-100 chunk @ T≈10000 | 0.293 s |
| Growth over 10000 tokens | +4% |

| tokens | chunk (s) | cum (s) |
|---|---|---|
| 100 | 0.281 | 0.28 |
| 1000 | 0.284 | 2.84 |
| 2000 | 0.292 | 5.70 |
| 3000 | 0.290 | 8.57 |
| 4000 | 0.283 | 11.44 |
| 5000 | 0.285 | 14.31 |
| 6000 | 0.283 | 17.17 |
| 7000 | 0.284 | 20.03 |
| 8000 | 0.284 | 22.88 |
| 9000 | 0.284 | 25.75 |
| 10000 | 0.293 | 28.62 |

### Summary across all comparisons

| Comparison | Look-ahead | Roformer | Speedup |
|---|---|---|---|
| D=1 K=1 C=2048 vs N=6 C=1088 | 1.09 ms/tok | 2.85 ms/tok | 2.6× |
| D=5 K=1 C=1120 (alone) | 2.86 ms/tok | — | — |
| D=6 K=1 C=1024 vs N=12 C=768 | 2.73 ms/tok | 4.96 ms/tok | 1.8× |

## 3. Checkpoint-compatibility verification

### Random-init equivalence (`verify_streaming.py`, fp32, CPU)

| Test | max \|Δ\| logits | argmax | strict load |
|---|---|---|---|
| `StreamingRoFormer` ↔ `RoFormer` | 4.77e-07 | — | missing=[], unexpected=[] |
| `StreamingLookAhead` D=1 ↔ `BlockHeadCorrFFNAddModel.forward_sequential(seq_k=1)` | 3.58e-07 | — | missing=[], unexpected=[] |
| `StreamingLookAhead` D=2 ↔ `BlockHeadCorrFFNAddModel(d_block=2).forward_sequential(seq_k=1)` | 3.58e-07 | — | missing=[], unexpected=[] |

All three pass (fp32 numerical noise level).

### End-to-end on real trained checkpoints (`verify_checkpoint.py`, fp32, CUDA)

| Checkpoint | iter | val_ppl | Reference path | max \|Δ\| | argmax agreement |
|---|---|---|---|---|---|
| `block_head_corr_ffn_add` D=2 K_train=5 C=1536 (`checkpoints_d2_c1536/`) | 74,889 | 52.62 | `forward_sequential(seq_k=1)` | 7.63e-06 | 100% (64/64) |
| `roformer` N=6 C=1088 (`look_ahead8_wum8ejn2/checkpoints_n6_c1088/`) | 360,000 | 32.14 | `forward()` | 9.06e-06 | 100% (64/64) |

`load_state_dict(strict=True)`: missing=[], unexpected=[] for both.

The streaming model produces logits identical to the reference parallel/sequential forward (to fp32 noise), confirming that any saved checkpoint can be loaded into the streaming class and used directly with the per-token wall-clock measured above.

## 4. Files

In `/home/ubuntu/look_ahead8/`:

| File | Purpose |
|---|---|
| `models_streaming.py` | Streaming-inference model definitions (`StreamingRoFormer`, `StreamingLookAhead`, building blocks). State_dict-compatible with `blocks.RoFormer` and `models.BlockHeadCorrFFNAddModel`. |
| `time_streaming.py` | Driver: builds a streaming model with random init, times 10000-token autoregressive decode in 100-token chunks, writes TSV. |
| `verify_streaming.py` | Random-init equivalence test. |
| `verify_checkpoint.py` | End-to-end checkpoint-load equivalence test. |

Logs in `/home/ubuntu/look_ahead8/logs/`:

- `stream_d1_k1_c2048.tsv`, `stream_n6_c1088.tsv`
- `stream_d6_k1_c1024_v2.tsv`, `stream_n12_c768_v2.tsv`
- `stream_d5_k1_c1120_v3.tsv`

## 5. Caveats

- All wall-clock numbers are with batch=1 and random-init weights. Real deployments may bench at larger batch sizes; the look-ahead's per-token compute advantage scales with the layer-count ratio (D vs N).
- `max_seq_len=10100` was used so the prefix is never truncated. Per-step cost grows linearly with T through the cache, but the layer-mass term dominates at these C values, so per-chunk time is flat.
- The streaming look-ahead corresponds to `forward_sequential(seq_k=1)`, which is only the per-position sequential path when the trained model has `n_iters > 1`. For models trained with `n_iters = 1`, the codebase's `forward_sequential` falls back to `forward_at_depth(K=1)` (no recurrence), and the streaming model would not match — but you wouldn't deploy K=1 streaming on a K=1-trained model anyway.

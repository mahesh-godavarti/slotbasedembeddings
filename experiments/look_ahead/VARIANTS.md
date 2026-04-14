# Correction Variants — look_ahead7

All variants share the same outer loop:
```
for k in range(K):
    z = blocks(processed_x)
    shifted_z[t] = z[t-1]              # position 0 gets zeros
    correction = <variant>
    processed_x = tok_emb + correction  # non-cumulative reset
head sees z
```

---

## 1. base (`block_head_corr_ffn_add`)

```
z = blocks(processed_x)
shifted_z[t] = z[t-1]
correction = corr_ffn(ln(shifted_z + tok_emb))
processed_x = tok_emb + correction
Position t sees only z[t-1]. FLOPs: (12D + 8)C²
```

---

## 2. incorrect (`block_head_attn_corr_ffn_add`)

```
z = blocks(processed_x)
shifted_z[t] = z[t-1]
attn_out = cross_attn(Q=ln(tok_emb), KV=ln(shifted_z))
correction = corr_ffn(ln(attn_out))
processed_x = tok_emb + correction
Position t attends to z[0..t-1]. FLOPs: (12D + 20)C²
```

Problem: tok_emb only enters through Q. attn_out = weighted sum of V(shifted_z) — no tok_emb in FFN input.

---

## 3. xattn (`block_head_xattn_corr_ffn_add`)

```
z = blocks(processed_x)
shifted_z[t] = z[t-1]
attn_out = cross_attn(Q=ln(tok_emb), KV=ln(shifted_z))
correction = corr_ffn(ln(attn_out + tok_emb))
processed_x = tok_emb + correction
Position t attends to z[0..t-1], tok_emb added back before FFN. FLOPs: (12D + 20)C²
```

Strict superset of base in theory. Fails at n_head=16 with small C (head_dim=4 too small for cross-representation matching). Works at n_head=1.

---

## 4. xattn_self (`block_head_xattn_self_corr_ffn_add`)

```
z = blocks(processed_x)
shifted_z[t] = z[t-1]
attn_out = cross_attn(Q=ln(tok_emb[t]), KV=ln([shifted_z[0..t-1], tok_emb[t]]))
correction = corr_ffn(ln(attn_out))
processed_x = tok_emb + correction
Position t attends to z[0..t-1] AND tok_emb[t] itself. FLOPs: (12D + 20)C²
```

tok_emb[t] is in both Q and KV. No need to add tok_emb after attention — it's already in the attention output via V(tok_emb[t]).

---

## 5. xattn2 (`block_head_xattn2_corr_ffn_add`)

```
z = blocks(processed_x)
shifted_z[t] = z[t-1]
attn_out = cross_attn(Q=ln(tok_emb), KV=ln(shifted_z + tok_emb))
correction = corr_ffn(ln(attn_out + tok_emb))
processed_x = tok_emb + correction
Position t attends to (z[j-1] + tok_emb[j]) for j=0..t. FLOPs: (12D + 20)C²
```

Token identity baked into every KV entry.

---

## 6. SA (`block_head_sa_corr_ffn_add`)

```
z = blocks(processed_x)
shifted_z[t] = z[t-1]
corr_input = shifted_z + tok_emb
h = corr_input + self_attn(ln1(corr_input))
correction = corr_ffn(ln2(h))
processed_x = tok_emb + correction
Position t self-attends over (z[j-1] + tok_emb[j]) for j=0..t. FLOPs: (12D + 20)C²
```

Strict superset of base (attention → 0 recovers base). tok_emb in Q, K, and V. Residual connection around attention.

---

## Results (C=64, D=1, K=5, block_size=64, batch=256, OWT, 74.9K iters)

### n_head=16

| Iter | base | incorrect | xattn | SA |
|------|------|-----------|-------|----|
| 1K | 1212 | 1301 | 1132 | 1220 |
| 2K | 797 | 842 | 787 | 794 |
| 3K | 661 | 684 | 664 | 660 |
| 4K | 582 | 601 | 591 | 580 |
| 5K | 527 | 543 | 538 | 525 |
| 6K | 488 | 503 | 498 | 485 |
| 7K | 456 | 470 | 466 | 454 |
| 8K | 431 | 443 | 439 | 428 |
| 9K | 409 | 421 | 419 | 406 |
| 10K | 391 | 403 | 400 | 388 |
| 11K | 374 | 388 | 385 | 373 |
| 12K | 361 | 374 | 371 | 359 |
| 13K | 351 | 362 | 360 | 348 |
| 14K | 340 | 352 | 349 | 338 |
| 15K | 332 | 342 | 341 | 328 |
| 16K | 324 | 335 | 333 | 321 |
| 17K | 316 | 326 | 325 | 313 |
| 18K | 309 | 320 | 319 | 307 |
| 19K | 303 | 314 | 313 | 301 |
| 20K | 298 | 309 | 308 | 295 |
| 21K | 293 | 303 | 302 | 290 |
| 22K | 288 | 298 | 298 | 285 |
| 23K | 284 | 294 | 293 | 281 |
| 24K | 280 | 290 | 289 | 276 |
| 25K | 276 | 285 | 286 | 272 |
| 26K | 272 | 282 | 283 | 269 |
| 27K | 269 | 279 | — | 265 |
| 28K | 266 | 275 | — | 262 |
| 29K | 263 | 272 | — | 259 |
| 30K | 260 | 269 | — | 256 |
| 35K | 248 | 257 | — | 244 |
| 40K | 238 | 246 | — | 234 |
| 45K | 229 | 238 | — | 226 |
| 50K | 223 | 231 | — | 219 |
| 55K | 217 | 225 | — | 213 |
| 27K | 269 | 279 | 280 | 265 |
| 28K | 266 | 275 | 276 | 262 |
| 29K | 263 | 272 | 273 | 259 |
| 30K | 260 | 269 | 270 | 256 |
| 31K | 257 | 266 | 268 | 253 |
| 32K | 255 | 264 | 265 | 251 |
| 33K | 252 | 261 | 262 | 248 |
| 34K | 250 | 259 | 260 | 246 |
| 35K | 248 | 257 | 258 | 244 |
| 36K | 245 | 255 | 256 | 242 |
| 40K | 238 | 246 | — | 234 |
| 45K | 229 | 238 | — | 226 |
| 50K | 223 | 231 | — | 219 |
| 55K | 217 | 225 | — | 213 |
| 60K | 213 | 220 | — | 208 |
| 59K | 213 | 221 | 224 | 209 |
| 65K | 208 | 216 | — | 204 |
| 70K | 205 | 212 | — | 200 |
| 74.9K | **201** | **208** | killed 59K | **197** |

- xattn tracks incorrect at n_head=16 (killed at 59K, PPL 224 vs incorrect 221 at same iter)
- SA beats base by ~4 PPL consistently. Final: SA 197.20 vs base 201.42 = **-4.2 PPL**
- Cross-attention fails at head_dim=4 (C=64, n_head=16). SA robust.

### n_head=1 (partial, killed early)

| Iter | xattn (n_h=1) | SA (n_h=1) |
|------|---------------|------------|
| 1K | 1121 | 1221 |
| 5K | 513 | 522 |
| 7K | 448 | 449 |
| 9K | 407 | 402 |
| 11K | 377 | 370 |

xattn works at n_head=1 — tracks SA closely. Cross-attention needs head_dim >> 4.

### Key findings

1. **incorrect always worst** — missing tok_emb from FFN input
2. **xattn fails at n_head=16 (head_dim=4)** — cross-attn between different representations needs larger head_dim
3. **xattn works at n_head=1 (head_dim=64)** — comparable to SA
4. **SA robust across n_head** — Q/K/V from same representation
5. **SA consistently beats base by ~3-4 PPL** at n_head=16

### TODO

- [ ] Run base, xattn, SA, xattn_self, xattn2 all at n_head=1
- [ ] Scale up winning variant to C=1024 or C=1952

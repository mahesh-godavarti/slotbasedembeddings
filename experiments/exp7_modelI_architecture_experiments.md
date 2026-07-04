# Model I Architecture Experiments: Fixing Divergence at Scale

## Background

Model I uses per-layer MLP projections to compute angles from the residual stream,
then applies cumsum to get positional angles (like Model E's learned cumsum, but
content-dependent). Model I' adds V rotation.

**Problem**: I/I' diverge at n250_l8 and above. The old per-layer MLP architecture
(one MLP per transformer layer) produces unbounded, sign-varying outputs that
cumsum amplifies into chaotic angles at scale.

**Key observation**: Model K uses the identical MLP architecture but never diverges —
because K uses `projected + RoPE` (additive, no cumsum). The instability is in
cumsum on unbounded MLP outputs, not the MLP itself.

## Old Architecture (diverges)

```python
# Per-layer MLPs (one per transformer layer)
self.angle_projectors = nn.ModuleList([
    nn.Sequential(
        nn.Linear(n_embed, n_embed),
        nn.GELU(),
        nn.Linear(n_embed, n_embed // 2),
    )
    for _ in range(n_layers)
])

# In forward:
for l, block in enumerate(self.blocks):
    raw_angles = self.angle_projectors[l](x)
    angles = cumsum(raw_angles)
    x = block(x, angles, ...)
```

### Old Divergence Points (n250_l8, 100K iters)

| Model | Diverged at | Peak loss |
|-------|------------|-----------|
| I     | iter 9000  | text=2.99, kg=3.39 (partially recovered, re-diverged at 14K) |
| I'    | iter 13000 | text=17.5, kg=15.3 (catastrophic, never recovered) |

## Experiment 1: Shared MLP + Positive Angles (abs)

Changes:
1. **Shared MLP**: Single MLP used across all layers (instead of per-layer)
2. **Positive angles**: `.abs()` on MLP output before cumsum

```python
self.angle_projector = nn.Sequential(...)  # single shared MLP

# In forward:
raw_angles = self.angle_projector(x).abs()
angles = cumsum(raw_angles)
```

### Results (n250_l8, 100K iters)

**I (unprimed)**: Fully stable, completed 100K iters.

| Tier | Text h@5 | Text PPL | KG h@5 | KG PPL |
|------|----------|----------|--------|--------|
| memorization | .145 | 5.11 | .248 | 4.01 |
| transfer | .100 | 5.36 | .233 | 3.95 |
| generalization | .133 | 5.26 | .178 | 4.38 |
| kg_excl_mem | .083 | 7.09 | .200 | 4.03 |
| kg_excl_gen | .083 | 7.33 | .267 | 4.02 |
| text_excl_mem | .067 | 5.19 | .000 | 6.67 |
| text_excl_gen | .117 | 5.33 | .117 | 6.88 |

Matches E (unprimed) almost exactly.

**I' (with V rotation)**: Diverged at iter 45000 (later than old iter 13000, but still diverged).

Conclusion: `.abs()` alone stabilizes the unprimed model but is insufficient for V rotation.

## Experiment 2: Shared MLP + LayerNorm + Positive Angles (abs)

Added LayerNorm on the projected angles before `.abs()`:

```python
self.angle_projector = nn.Sequential(...)  # single shared MLP
self.angle_ln = nn.LayerNorm(n_embed // 2)

# In forward:
raw_angles = self.angle_ln(self.angle_projector(x)).abs()
angles = cumsum(raw_angles)
```

### Results (n250_l8, 100K iters)

**I' (with V rotation)**: Fully stable through 100K iters!

| Tier | Text h@5 | Text PPL | KG h@5 | KG PPL |
|------|----------|----------|--------|--------|
| memorization | .285 | 4.57 | .767 | 2.32 |
| transfer | .233 | 4.79 | .767 | 2.31 |
| generalization | .211 | 4.84 | .589 | 3.42 |
| kg_excl_mem | .117 | 8.11 | .767 | 2.42 |
| kg_excl_gen | .217 | 8.32 | .517 | 3.33 |
| text_excl_mem | .050 | 5.09 | .033 | 10.81 |
| text_excl_gen | .150 | 5.39 | .100 | 13.14 |

### Comparison with E/E' at n250_l8

| Model | KG mem h@5 | KG mem PPL | Text mem h@5 | Text mem PPL |
|-------|-----------|------------|-------------|-------------|
| E     | .217      | 4.05       | .130        | 5.08        |
| New I | .248      | 4.01       | .145        | 5.11        |
| D'    | .593      | 2.95       | .229        | 4.74        |
| **New I'** | **.767** | **2.32** | **.285** | **4.57** |
| E'    | .777      | 2.34       | .215        | 4.84        |

**I' nearly matches E'** on KG (.767 vs .777 h@5, 2.32 vs 2.34 PPL) and
**beats E' on text** (.285 vs .215 h@5, 4.57 vs 4.84 PPL).

## Experiment 3: Shared MLP + LayerNorm, NO abs (best result)

Removed `.abs()`, keeping only LayerNorm:

```python
self.angle_projector = nn.Sequential(...)  # single shared MLP
self.angle_ln = nn.LayerNorm(n_embed // 2)

# In forward:
raw_angles = self.angle_ln(self.angle_projector(x))
angles = cumsum(raw_angles)
```

### Results (n250_l8, 100K iters)

**Both I and I' fully stable through 100K iters.**

**I (unprimed):**

| Tier | Text h@5 | Text PPL | KG h@5 | KG PPL |
|------|----------|----------|--------|--------|
| memorization | .139 | 5.12 | .236 | 4.02 |
| transfer | .067 | 5.23 | .244 | 3.87 |
| generalization | .100 | 5.24 | .178 | 4.30 |
| kg_excl_mem | .050 | 7.20 | .167 | 4.10 |
| kg_excl_gen | .133 | 6.70 | .267 | 4.01 |
| text_excl_mem | .100 | 5.16 | .067 | 6.76 |
| text_excl_gen | .033 | 5.39 | .050 | 8.13 |

**I' (with V rotation):**

| Tier | Text h@5 | Text PPL | KG h@5 | KG PPL |
|------|----------|----------|--------|--------|
| memorization | .313 | 4.46 | .819 | 2.12 |
| transfer | .244 | 4.72 | .767 | 2.09 |
| generalization | .233 | 4.83 | .556 | 3.31 |
| kg_excl_mem | .133 | 7.78 | .817 | 2.12 |
| kg_excl_gen | .100 | 7.73 | .567 | 3.18 |
| text_excl_mem | .100 | 5.03 | .033 | 13.42 |
| text_excl_gen | .250 | 5.00 | .100 | 14.72 |

### Full Comparison at n250_l8

| Model | KG mem h@5 | KG mem PPL | Text mem h@5 | Text mem PPL |
|-------|-----------|------------|-------------|-------------|
| E     | .217      | 4.05       | .130        | 5.08        |
| I (abs)    | .248 | 4.01  | .145        | 5.11        |
| I (no abs) | .236 | 4.02  | .139        | 5.12        |
| D'    | .593      | 2.95       | .229        | 4.74        |
| I' (LN+abs) | .767 | 2.32  | .285        | 4.57        |
| E'    | .777      | 2.34       | .215        | 4.84        |
| **I' (LN, no abs)** | **.819** | **2.12** | **.313** | **4.46** |

## Experiment 4: Per-layer MLP + LayerNorm, NO abs (original architecture + LN)

Reverted to the original per-layer MLP architecture, just adding LayerNorm:

```python
self.angle_projectors = nn.ModuleList([
    nn.Sequential(
        nn.Linear(n_embed, n_embed),
        nn.GELU(),
        nn.Linear(n_embed, n_embed // 2),
    )
    for _ in range(n_layers)
])
self.angle_ln = nn.LayerNorm(n_embed // 2)

# In forward:
raw_angles = self.angle_ln(self.angle_projectors[l](x))
angles = cumsum(raw_angles)
```

### Results (n250_l8, 100K iters)

**Both I and I' fully stable through 100K iters.** LayerNorm alone fixes the divergence.

**I (unprimed):**

| Tier | Text h@5 | Text PPL | KG h@5 | KG PPL |
|------|----------|----------|--------|--------|
| memorization | .153 | 5.04 | .249 | 3.99 |
| transfer | .111 | 5.09 | .256 | 3.73 |
| generalization | .089 | 5.03 | .222 | 4.17 |
| kg_excl_mem | .200 | 6.70 | .300 | 3.88 |
| kg_excl_gen | .050 | 6.97 | .200 | 4.07 |
| text_excl_mem | .133 | 5.01 | .067 | 7.38 |
| text_excl_gen | .150 | 5.36 | .133 | 7.04 |

**I' (with V rotation):**

| Tier | Text h@5 | Text PPL | KG h@5 | KG PPL |
|------|----------|----------|--------|--------|
| memorization | .139 | 5.07 | .681 | 2.66 |
| transfer | .133 | 5.20 | .622 | 2.59 |
| generalization | .111 | 5.07 | .567 | 3.51 |
| kg_excl_mem | .117 | 8.30 | .733 | 2.63 |
| kg_excl_gen | .050 | 10.24 | .467 | 3.49 |
| text_excl_mem | .050 | 5.08 | .033 | 11.41 |
| text_excl_gen | .017 | 5.31 | .117 | 14.23 |

### Comparison: Per-layer vs Shared MLP (both with LN, no abs)

| Model | Per-layer (Exp 4) | Shared (Exp 3) |
|-------|------------------|----------------|
| I KG mem h@5 / PPL | .249 / 3.99 | .236 / 4.02 |
| I Text mem h@5 / PPL | .153 / 5.04 | .139 / 5.12 |
| I' KG mem h@5 / PPL | .681 / 2.66 | **.819 / 2.12** |
| I' Text mem h@5 / PPL | .139 / 5.07 | **.313 / 4.46** |

I (unprimed): Per-layer is slightly better.
I' (V rotation): Shared is dramatically better — .819 vs .681 KG, .313 vs .139 text.

## Key Findings

1. **LayerNorm alone fixes divergence** in both per-layer and shared MLP variants.
2. **abs() is unnecessary** and slightly harmful — removing it improves results.
3. **Shared MLP is critical for V rotation (I')**: Per-layer I' gets .681 KG h@5 vs shared I' .819.
   V rotation needs consistent angles across layers to build coherent geometric structure.
   Per-layer MLPs give each layer different angles, which V rotation can't leverage.
4. **Per-layer is slightly better for unprimed (I)**: .249 vs .236 KG, .153 vs .139 text.
   Without V rotation, depth-dependent angle strategies help marginally.
5. **I' (shared + LN) is the new best model** at n250_l8:
   - KG: .819 h@5 / 2.12 PPL (beats E' .777 / 2.34)
   - Text: .313 h@5 / 4.46 PPL (beats E' .215 / 4.84)

## Architecture Summary

| Variant | Shared MLP | LayerNorm | abs() | I stable | I' stable | I' KG h@5 | I' Text h@5 |
|---------|-----------|-----------|-------|----------|-----------|-----------|-------------|
| Old (per-layer, no LN) | No | No | No | No (9K) | No (13K) | diverged | diverged |
| Exp 1: shared+abs | Yes | No | Yes | Yes | No (45K) | diverged | diverged |
| Exp 2: shared+LN+abs | Yes | Yes | Yes | Yes* | Yes | .767 | .285 |
| **Exp 3: shared+LN** | **Yes** | **Yes** | **No** | **Yes** | **Yes** | **.819** | **.313** |
| Exp 4: per-layer+LN | No | Yes | No | Yes | Yes | .681 | .139 |

*I was not re-run in Exp 2; only I' was tested.

## Recommended Architecture (Exp 3: shared MLP + LN)

```python
# In __init__:
self.angle_projector = nn.Sequential(
    nn.Linear(n_embed, n_embed),
    nn.GELU(),
    nn.Linear(n_embed, n_embed // 2),
)
self.angle_ln = nn.LayerNorm(n_embed // 2)

# In forward:
for l, block in enumerate(self.blocks):
    raw_angles = self.angle_ln(self.angle_projector(x))
    angles = cumsum(raw_angles)
    x = block(x, angles, ...)
```

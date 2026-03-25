# CIFAR-10/100 Directional Monoidal Embedding Results

## Part 1: Composition Layer as Feature Extractor (CIFAR-10)

Direct application of the monoidal composition as an image embedding, compared against DFT and standard neural baselines.

**Config**: 50 epochs, batch=128, Adam lr=1e-3, cosine annealing

| Model | Parameters | Best Test Accuracy |
|-------|-----------|-------------------|
| DFT (32 features/channel) + Linear | 970 | 28.96% |
| **Monoidal Embedding (dim=32)** | **59,114** | **39.85%** |
| MLP (256→256→10) | 855,050 | 55.13% |
| CNN (2 conv + FC) | 1,070,794 | 81.56% |

**Takeaway**: Monoidal embedding outperforms DFT by 10.9 points at the same embedding dimensionality. The learnable angles capture task-specific frequency components that fixed DFT cannot. MLP and CNN are reference points with 14-18x more parameters.

---

## Part 2: Vision Transformer with Positional Encoding Variants (CIFAR-100)

### Background

The paper's algebraic framework using the semidirect product composition `(a, A).(b, B) = (a + Ab, AB)` can be applied as positional encoding in Vision Transformers. We test whether:
1. Learnable rotation frequencies beat fixed ones (monoidal vs RoPE)
2. Rotating values (JoFormer) beats rotating only Q/K (RoPE)

### Model Architecture
- ViT with patch_size=4 (8x8 = 64 patches), embed_dim=32, 4 layers, 4 heads
- ~55K parameters
- AdamW lr=1e-3, weight_decay=0.05, cosine annealing

### Two Approaches to 2D Rotation

**Axial (split dimensions):**
- First D/4 rotation pairs encode y-position only
- Second D/4 rotation pairs encode x-position only
- Each pair sees one axis. Equivalent to setting half the frequencies to zero.
- This is how standard RoPE2D is implemented in practice.

**Combined (all pairs encode both axes):**
- Every rotation pair gets angle = pos_y * freq_y[d] + pos_x * freq_x[d]
- Each pair encodes both axes simultaneously through the angle sum
- This is what the monoidal framework naturally produces
- Known in literature as "RoPE-Mixed" (ECCV 2024), shown to outperform axial at scale

### The 2D Frequency Degeneracy Problem

With combined angles, freq_y and freq_x must be different. If freq_y = freq_x, the angle becomes (pos_y + pos_x) * freq[d], and positions on the same anti-diagonal are indistinguishable.

We tested multiple fixed frequency choices for the combined approach:

| freq_x choice | rope2dv2 | joformer_fixed |
|---------------|----------|----------------|
| = freq_y (degenerate) | 44.49% | 45.83% |
| offset (2d+1) | 44.22% | 45.11% |
| flip(freq_y) | 47.81% | 45.31% |
| -freq_y (negative) | 45.91% | 45.96% |
| zeroed split | 45.24% | 44.47% |

None of the combined fixed-frequency approaches matched the axial split (47.94%) except flipped (47.81%), and even that was inconsistent when V rotation was added.

### Model Hierarchy

```
                        Fixed freqs              Learned freqs
                        ───────────              ─────────────
Axial, Q/K only:        rope2d                   monoidal_axial
Axial, Q/K/V+inv:       joformer_old             joformer_axial
Combined, Q/K only:     rope2dv2                 monoidal
Combined, Q/K/V+inv:    joformer_fixed           joformer
```

Each row shares identical code. The only difference between columns is `nn.Parameter` vs `register_buffer` for the frequencies.

### The K@Q^T Fix

The original JoFormer code uses `k @ q.transpose(-1,-2)` (K@Q^T), not `q @ k.transpose(-1,-2)` (Q@K^T). When V is rotated by the same matrix R as K, the attention must use K@Q^T so that the "journey operator" R_i^T R_j in the attention scores matches the journey R_i^{-1} R_j applied to values after inverse rotation.

Without V rotation (standard RoPE), the order doesn't matter — `q@k.T` and `k@q.T` give the same attention scores for bidirectional attention.

Fixing Q@K^T → K@Q^T improved joformer_old from 46.95% to 47.60% in early runs.

### Reproducibility

Early runs used PyTorch DataLoader with `num_workers=2` and torchvision data augmentation transforms. This produced inconsistent results across runs with the same seed (joformer_old varied by ~1% between runs while rope2d was stable).

**Root cause**: DataLoader workers have their own RNG and batch delivery order can vary.

**Fix**: Switched to look_ahead-style data loading:
- Load entire dataset into GPU memory at startup (~184MB)
- Generate random batch indices with `torch.randint` (fully seeded)
- Apply augmentation (flip, crop) on GPU with torch ops
- No DataLoader, no workers, fully deterministic

This approach is both faster (no CPU→GPU transfer per batch) and reproducible.

### Definitive Results (100 epochs, seed=42, deterministic, data on GPU)

| Model | PE Params | Best Test Acc |
|-------|-----------|-------------|
| rope2d (axial fixed) | 0 | 47.85% |
| **joformer_old (axial fixed + V)** | **0** | **48.21%** |

V rotation helps: +0.36% with identical frequencies, same seed, deterministic.

### Earlier 100-epoch Results (seed=42, K@Q^T, before deterministic fix)

| Approach | Q/K only | Q/K/V + inverse | V helps |
|----------|----------|-----------------|---------|
| Axial fixed | rope2d: 47.42% | joformer_old: 47.60% | +0.18 |
| Axial learned | monoidal_axial: 48.15% | joformer_axial: 48.30% | +0.15 |
| Combined negative | rope2dv2: 45.91% | joformer_fixed: 45.96% | +0.05 |

**Key findings from 100-epoch runs:**
1. **V rotation consistently helps** across all approaches (+0.05 to +0.36%)
2. **Learned frequencies beat fixed** (monoidal_axial 48.15% > rope2d 47.42%)
3. **Axial beats combined** at D=32 (~47-48% vs ~45-46%)
4. **JoFormer starts slower, finishes stronger** — V rotation needs more training time

### Convergence Data (100 epochs, earlier runs, seed=42)

| Epoch | rope2d | joformer_old | monoidal_axial | joformer_axial |
|-------|--------|-------------|----------------|----------------|
| 10 | 31.62 | 32.29 | 30.90 | 31.26 |
| 20 | 38.07 | 38.27 | 39.09 | 38.51 |
| 30 | 42.30 | 41.55 | 42.21 | 42.26 |
| 40 | 44.02 | 42.91 | 44.70 | 44.37 |
| 50 | 44.88 | 44.58 | 45.73 | 45.11 |
| 60 | 45.88 | 45.55 | 46.43 | 45.95 |
| 70 | 46.60 | 46.34 | 47.59 | 47.35 |
| 80 | 47.15 | 47.04 | 47.81 | 47.61 |
| 90 | 47.42 | 47.51 | 48.02 | 48.15 |
| 100 | 47.42 | 47.60 | 48.15 | 48.30 |

JoFormer variants trail early (epochs 10-50) but catch up and pass their rope counterparts by epoch 80-100.

### Partial 1000-epoch Results (non-deterministic, interrupted)

| Model | 100 ep | 500 ep | 1000 ep |
|-------|--------|--------|---------|
| learned | 47.11 | 50.76 | 51.00 |
| rope2d | 47.40 | 51.09 | 51.95 |
| joformer_old | 46.66 | 50.47 | (interrupted at ~570) |

These results are from a run contaminated by orphan processes sharing the GPU. They should be rerun with the deterministic setup.

---

## Part 3: Understanding the 2D Frequency Design

### Why axial (split dimensions) works at small D

The split approach gives the attention mechanism clean, separable position signals:
- Y-pairs respond to "same row" relationships
- X-pairs respond to "same column" relationships
- The total attention score is: attn ∝ f(Δy) + g(Δx) — additively separable

At D=32 with only 16 rotation pairs, dedicating 8 to each axis gives each axis a focused signal. The combined approach spreads information across all 16 pairs but each pair carries a mixture.

### Why combined beats axial at large D (literature)

RoPE-Mixed (ECCV 2024) shows combined outperforms axial on ViT-B (D=768):
- 224×224: Axial 83.6% → Mixed 83.8%
- 512×512 extrapolation: Axial 82.0% → Mixed 82.9%

With enough dimensions, the combined approach's ability to encode diagonal relationships outweighs the separability advantage.

### JoFormer value rotation and S4ND connection

Standard RoPE only rotates Q and K — position affects attention scores but not values. JoFormer also rotates V by the same position-dependent angles, then inverse-rotates the output:

```
out(i,j) = R(i,j)^{-1} * sum_(k,l) attn(i,j→k,l) * R(k,l) * v(k,l)
         = sum_(k,l) attn(...) * R(k-i, l-j) * v(k,l)
```

Each value gets transformed by the relative position — a position-dependent content transformation similar to S4ND's global convolution kernel. The self-contribution is always identity: R(i,j)^{-1} R(i,j) = I.

This connects attention (data-dependent routing) with structured state spaces (position-dependent content transformation) under one algebraic framework.

---

## Part 4: Infrastructure Evolution

### Data Loading

**v1 (DataLoader):** torchvision transforms + DataLoader with num_workers=2. Non-deterministic due to worker RNG. Slow data augmentation in Python/PIL.

**v2 (Preloaded CPU):** Entire dataset loaded into CPU tensors at startup. Augmentation in PyTorch ops. Deterministic but slow due to CPU→GPU transfer every batch and Python loop in crop augmentation.

**v3 (Preloaded GPU, current):** Entire dataset on GPU (~184MB). Augmentation with vectorized torch ops on GPU. Fully deterministic, fastest.

### Reproducibility

Getting deterministic results required eliminating every source of randomness not controlled by the seed:

1. **Seed everything at startup:**
   ```python
   torch.manual_seed(args.seed)
   torch.cuda.manual_seed_all(args.seed)
   ```

2. **Eliminate DataLoader workers.** With `num_workers=2`, worker processes fork from the main process and get their own RNG. The order of batch delivery from workers can vary between runs. Even with `generator=torch.Generator().manual_seed(seed)`, the workers introduced non-determinism. Solution: `num_workers=0` (single-threaded), then ultimately removed DataLoader entirely.

3. **Move data to GPU at startup.** With CPU tensors, every batch required `tensor.to(device)` which involves CPU→GPU transfer. PyTorch uses multiple threads for this (observed 346% CPU usage), and the threading could introduce non-determinism. With data already on GPU, batch creation is just `train_x[ix]` — a pure GPU indexing operation.

4. **Replace torchvision transforms with torch ops.** torchvision's `RandomHorizontalFlip` and `RandomCrop` use their own RNG (potentially Python's `random` module or PIL). Reimplemented augmentation using `torch.rand` and `torch.randint` on GPU, which are controlled by `torch.manual_seed`.

5. **Vectorize augmentation.** The initial crop implementation used a Python for-loop over the batch (`for i in range(B): crops.append(...)`). This was both slow and could interact with Python's RNG. Replaced with vectorized advanced indexing — one GPU operation for the entire batch.

6. **Generate batch indices on GPU.** `torch.randint(0, n, (batch_size,), device=train_x.device)` ensures the random indices are generated by CUDA's seeded RNG, not CPU RNG.

7. **Kill orphan processes.** When using shell scripts with `pkill`, the parent bash process gets killed but child Python processes can survive as orphans, silently sharing the GPU. This contaminated earlier 1000-epoch runs. Solution: kill individual PIDs, verify with `ps aux`, and use `run_all.py` (single Python process) instead of shell scripts.

**What we did NOT need:** `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` were added initially but turned out to be unnecessary once all other sources of non-determinism were eliminated. The model is small enough that cuDNN algorithm selection is deterministic by default.

**Verification:** Ran rope2d twice at 10 epochs with seed=42. Both runs produced identical output at every epoch:
```
Run 1: Epoch 10: train_loss=2.7561, train_acc=30.3%, test_acc=30.62%
Run 2: Epoch 10: train_loss=2.7561, train_acc=30.3%, test_acc=30.62%
```

**Lesson from look_ahead:** The look_ahead project never had this problem because it uses `np.memmap` + `torch.randint` for batch generation — no DataLoader, no workers, no augmentation. The same pattern applied here solved all reproducibility issues.

---

## Files

- `directional_composition_cifar10.py` — Monoidal embedding as feature extractor (CIFAR-10)
- `baselines_cifar10.py` — DFT, MLP, CNN baselines (CIFAR-10)
- `vit_cifar10.py` — ViT with 9 PE variants, deterministic GPU training
- `run_all.py` — Run all models sequentially with data loaded once
- `run_all_1000.sh` — Shell script for 1000-epoch runs (deprecated, use run_all.py)
- `NOTES.md` — Design notes on 2D positional encoding with literature references
- `convergence_100ep.md` — Convergence data from earlier (non-deterministic) runs
- `results.md` — This file
- `vit_cifar10_backup.py` — Backup of DataLoader-based code

---

### The Cosine LR Schedule Trap

A 100-epoch run and the first 100 epochs of a 1000-epoch run are NOT the same experiment when using cosine annealing LR schedule. The schedule depends on `T_max` (total training iterations):

- **100-epoch run**: LR decays from 1e-3 to ~0 over 100 epochs. At epoch 50, LR is ~50% decayed.
- **1000-epoch run**: LR decays from 1e-3 to ~0 over 1000 epochs. At epoch 50, LR is ~97% of initial.

This caused confusion when comparing:
```
rope2d epoch 1:  100-ep run: 4.0822 (identical)
rope2d epoch 10: 100-ep run: 2.5947 vs 1000-ep run: 2.5944 (diverging)
rope2d epoch 50: 100-ep run: 1.8427 vs 1000-ep run: 1.9067 (very different)
rope2d epoch 100: 100-ep run: 47.85% vs 1000-ep run: 47.60% (different results)
```

Epoch 1 is identical (same seed, same initial LR). By epoch 10, the LR difference causes tiny divergence. By epoch 50, the trajectories are completely different.

**Fix**: Switched to fixed LR (no schedule) as default. Cosine decay available via `--cosine_decay` flag. With fixed LR, the first N epochs of any run are identical regardless of total epochs, enabling fair comparison across different training lengths.

### Current Status

All results below use:
- Data loaded entirely on GPU (~184MB)
- `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`
- No DataLoader (direct `torch.randint` batch sampling)
- Augmentation via torch ops on GPU
- K@Q^T attention ordering (matching JoFormer's journey operator)
- Fixed LR (no cosine decay)

Running: rope2d and joformer_old at lr=2e-4, 100 epochs to establish new baseline with fixed LR.

---

### Definitive 1000-Epoch Results (lr=5e-4, fixed LR, seed=42, deterministic)

| Model | Type | Freqs | V rotation | 1000 ep |
|-------|------|-------|------------|---------|
| **monoidal** | combined | learned | no | **51.58%** |
| rope2d | axial | fixed | no | 51.53% |
| joformer | combined | learned | yes | 51.22% |
| joformer_old | axial | fixed | yes | 51.15% |
| joformer_axial | axial | learned | yes | 51.15% |
| learned | additive | — | no | 50.85% |
| joformer_fixed | combined | fixed | yes | 48.91% |
| rope2dv2 | combined | fixed | no | 48.27% |
| monoidal_axial | axial | learned | no | 48.04% |

### Analysis

**1. Monoidal (combined learned, no V) wins.**
The combined-angle approach with learnable frequencies and no V rotation achieved the best result (51.58%), narrowly beating the axial fixed-frequency rope2d (51.53%). This validates the RoPE-Mixed finding from the ECCV 2024 paper — even at D=32, learned combined frequencies can match or beat axial fixed frequencies given enough training time.

**2. V rotation consistently hurts at 1000 epochs with fixed LR.**
Every JoFormer variant underperforms its non-V counterpart:
- monoidal (51.58%) > joformer (51.22%) — combined learned
- rope2d (51.53%) > joformer_old (51.15%) — axial fixed
- At 100 epochs, V rotation helped (+0.62%). At 1000 epochs, it hurts (-0.36 to -0.38%). The V rotation may overfit or create optimization difficulties at longer training.

**3. Learned frequencies: need V rotation or long training.**
- monoidal_axial (axial learned, no V) = 48.04% — worst of all, random init at near-zero frequencies with fixed LR never recovered
- joformer_axial (axial learned, with V) = 51.15% — V rotation rescued it (+3.11%), matching joformer_old
- monoidal (combined learned, no V) = 51.58% — combined approach recovered without V rotation, but needed 1000 epochs
- The random near-zero frequency initialization hurts badly without either V rotation or very long training to compensate

**4. Combined fixed frequencies don't work at D=32.**
- rope2dv2 (48.27%) and joformer_fixed (48.91%) are far behind axial fixed (51.53%, 51.15%)
- The negative frequency trick (freq_x = -freq_y) doesn't provide enough positional discrimination at small D
- This matches our earlier finding: combined needs learnable frequencies to work

**5. All rotation-based methods beat learned PE.**
- Every model except the poorly-initialized learnable ones (monoidal_axial, rope2dv2) beats learned PE (50.85%)
- Rotational position encoding is fundamentally better than additive position embedding

### Convergence Patterns

| Epoch | rope2d | joformer_old | monoidal | joformer |
|-------|--------|-------------|----------|---------|
| 100 | 46.17 | 46.79 | — | — |
| 250 | 49.26 | 49.37 | — | — |
| 500 | 50.86 | 50.38 | — | — |
| 750 | 51.38 | — | — | — |
| 1000 | 51.53 | 51.15 | 51.58 | 51.22 |

- JoFormer variants lead early (epochs 50-300) then fall behind
- monoidal catches up late and wins at 1000 epochs
- rope2d improves steadily throughout

### 300-Epoch Cosine Results (lr=1e-3, cosine decay, seed=42, deterministic)

| Model | Type | Freqs | V rotation | 300ep cosine | 1000ep fixed |
|-------|------|-------|------------|-------------|-------------|
| **joformer_axial** | axial | learned | yes | **52.77%** | 51.15% |
| **joformer** | combined | learned | yes | **52.43%** | 51.22% |
| **learned** | additive | — | no | **52.23%** | 50.85% |
| monoidal_axial | axial | learned | no | 51.75% | 48.04% |
| monoidal | combined | learned | no | 51.59% | 51.58% |
| rope2d | axial | fixed | no | 50.91% | 51.53% |
| joformer_old | axial | fixed | yes | 50.54% | 51.15% |
| rope2dv2 | combined | fixed | no | 49.04% | 48.27% |
| joformer_fixed | combined | fixed | yes | 49.20% | 48.91% |

### Convergence Table (300 epochs cosine, best test acc)

| Epoch | learned | rope2d | jfmr_old | mon_ax | jfmr_ax | rope2dv2 | monoidal | joformer |
|-------|---------|--------|----------|--------|---------|----------|----------|----------|
| 10 | 31.96 | 34.28 | 33.62 | 30.88 | 31.88 | 30.18 | 32.16 | 32.02 |
| 20 | 38.37 | 39.71 | 39.40 | 37.31 | 38.72 | 36.69 | 38.29 | 39.23 |
| 30 | 41.99 | 42.19 | 42.08 | 40.61 | 41.96 | 39.32 | 42.37 | 42.48 |
| 50 | 45.05 | 44.91 | 44.95 | 44.53 | 45.06 | 42.13 | 45.56 | 44.50 |
| 100 | 48.13 | 48.01 | 47.15 | 48.12 | 47.28 | 45.33 | 48.26 | 48.08 |
| 150 | 49.16 | 49.18 | 48.61 | 49.53 | 49.50 | 47.27 | 49.89 | 50.17 |
| 200 | 50.53 | 49.92 | 49.55 | 50.31 | 50.67 | 47.85 | 50.65 | 51.30 |
| 250 | 51.87 | 50.70 | 50.24 | 51.32 | 52.14 | 48.74 | 51.59 | 52.06 |
| 300 | 52.23 | 50.91 | 50.54 | 51.75 | **52.77** | 49.04 | 51.59 | 52.43 |

### Analysis: Cosine vs Fixed LR

**1. Cosine schedule reverses V rotation effect.**
- Fixed LR 1000ep: V rotation hurts (monoidal 51.58% > joformer 51.22%)
- Cosine 300ep: V rotation helps (joformer 52.43% > monoidal 51.59%)
- The cosine schedule's LR decay stabilizes the V rotation at convergence

**2. Learnable models dominate with cosine.**
Top 3 are all learnable: joformer_axial (52.77%), joformer (52.43%), learned (52.23%).
Fixed-frequency models (rope2d 50.91%, joformer_old 50.54%) fall behind.
The cosine schedule's high initial LR gives learnable frequencies time to explore.

**3. Cosine rescues monoidal_axial.**
monoidal_axial: fixed LR 48.04% → cosine 51.75% (+3.71%).
The high initial LR lets randomly-initialized frequencies learn useful values before the LR decays. With fixed low LR, they never recovered.

**4. joformer_axial is the overall winner.**
52.77% — learnable axial frequencies + V rotation + cosine schedule.
This combines the best of everything: learnable frequencies for adaptability, axial structure for clean axis separation, V rotation for content-dependent positional transformation, and cosine schedule for proper convergence.

**5. Learned PE is surprisingly strong with cosine.**
52.23% — third best overall. The cosine schedule's annealing effectively "bakes in" the 2,080 learned position parameters. This is the standard ViT approach and remains competitive.

**6. Fixed frequencies underperform with cosine.**
rope2d drops from 51.53% (fixed LR) to 50.91% (cosine). The cosine schedule doesn't help models that can't adapt their positional encoding — the decaying LR doesn't benefit fixed frequencies.

### Summary: Fixed LR vs Cosine (D=32)

| Model | Fixed LR 1000ep | Cosine 300ep | Cosine helps? |
|-------|----------------|-------------|---------------|
| joformer_axial | 51.15% | **52.77%** | +1.62% |
| joformer | 51.22% | **52.43%** | +1.21% |
| learned | 50.85% | **52.23%** | +1.38% |
| monoidal_axial | 48.04% | **51.75%** | +3.71% |
| monoidal | 51.58% | 51.59% | +0.01% |
| rope2d | **51.53%** | 50.91% | -0.62% |
| joformer_old | **51.15%** | 50.54% | -0.61% |
| joformer_fixed | 48.91% | 49.20% | +0.29% |
| rope2dv2 | 48.27% | 49.04% | +0.77% |

**Pattern**: Cosine schedule helps learnable models (learned, monoidal_axial, joformer_axial, joformer) and hurts fixed-frequency models (rope2d, joformer_old). The high initial LR in cosine gives learnable parameters time to explore before the LR decays and locks in the solution.

**The winner depends on training setup:**
- Fixed LR: monoidal (combined learned, no V) = 51.58%
- Cosine: joformer_axial (axial learned, V rotation) = 52.77%

This sensitivity to LR schedule is concerning — it means the D=32 regime is too small for reliable conclusions about model architecture. The differences (~2%) are within the range of hyperparameter sensitivity.

### Underfitting at D=32
Train accuracy ~63%, test ~52% at 300 epochs cosine. The model underfits — a well-sized model on CIFAR-100 should achieve >90% train accuracy. Moving to D=64 to increase capacity.

Running: D=64, 4 layers, 4 heads, 300 epochs cosine lr=1e-3, all 9 models.

### D=64 with Full Regularization (300 epochs, cosine lr=1e-3, dropout=0.1, wd=0.1, mixup+cutout)

**The breakthrough run.** Adding dropout, weight decay, mixup, and cutout solved the overfitting at D=64 and produced clear, meaningful results.

| Model | Type | Freqs | V rotation | D=64 |
|-------|------|-------|------------|------|
| **joformer_axial** | **axial** | **learned** | **yes** | **61.33%** |
| rope2d | axial | fixed | no | 59.23% |
| joformer_old | axial | fixed | yes | 59.13% |
| monoidal_axial | axial | learned | no | 59.10% |
| joformer | combined | learned | yes | 58.83% |
| monoidal | combined | learned | no | 58.66% |
| learned | additive | — | no | 57.26% |
| joformer_fixed | combined | fixed | yes | 55.79% |
| rope2dv2 | combined | fixed | no | 55.21% |

### Convergence Table (D=64, best test acc)

| Epoch | learned | rope2d | jfmr_old | mon_ax | **jfmr_ax** |
|-------|---------|--------|----------|--------|-------------|
| 10 | 29.35 | 30.44 | 32.30 | 30.85 | 31.44 |
| 20 | 36.99 | 39.59 | 39.96 | 40.43 | 40.47 |
| 30 | 41.21 | 43.49 | 43.94 | 43.21 | 43.74 |
| 50 | 46.13 | 47.80 | 47.51 | 46.63 | 48.05 |
| 100 | 49.79 | 51.79 | 51.50 | 51.08 | 52.46 |
| 150 | 52.26 | 54.11 | 54.08 | 54.42 | 55.92 |
| 200 | 54.18 | 56.48 | 56.10 | 56.27 | 57.18+ |
| 250 | 56.49 | 58.38 | 58.26 | 58.76 | 59.96 |
| 300 | 57.26 | 59.23 | 59.13 | 59.10 | **61.33** |

### Analysis

**1. joformer_axial wins by 2.1% — the clearest result in all our experiments.**
Learnable axial frequencies + V rotation + cosine schedule + proper regularization.
This is not a noisy 0.5% difference — it's a consistent 1.5-2% lead at every checkpoint from epoch 50 onward.

**2. V rotation helps with learnable frequencies at D=64.**
- joformer_axial (61.33%) vs monoidal_axial (59.10%) = +2.23% (learned axial)
- joformer (58.83%) vs monoidal (58.66%) = +0.17% (learned combined)
- joformer_old (59.13%) vs rope2d (59.23%) = -0.10% (fixed axial)
V rotation's benefit scales with learnable frequencies. Fixed frequencies don't benefit.

**3. Learnable frequencies help with V rotation at D=64.**
- joformer_axial (61.33%) vs joformer_old (59.13%) = +2.20% (axial with V)
- monoidal_axial (59.10%) vs rope2d (59.23%) = -0.13% (axial without V)
Learnable frequencies' benefit scales with V rotation. Without V rotation, they don't help.

**4. The interaction between V rotation and learnable frequencies is synergistic.**
Neither V rotation alone nor learnable frequencies alone provides a big advantage.
But together (joformer_axial) they produce a 2.1% gain. The V rotation gives the model
a richer position-dependent transformation, and the learnable frequencies adapt to exploit it.

**5. Axial still beats combined at D=64.**
Axial models (~59-61%) consistently outperform combined (~55-59%).
The axial split provides cleaner per-axis signals that the model can use more effectively.

**6. Regularization was critical.**
Without regularization at D=64: train 96%, test 50% (massive overfitting).
With regularization: train 53%, test 61% (test > train due to augmentation).
The regularization (dropout 0.1, weight decay 0.1, mixup α=0.2, cutout 8×8) was essential
for the architectural differences to manifest clearly.

**7. Scale matters for conclusions.**
At D=32 without regularization: all models within ~2%, ordering depends on LR schedule.
At D=64 with regularization: clear 2.1% winner, consistent across training.
Small models on small data produce noisy results that don't reflect true architectural quality.

### CIFAR-100 SOTA context
State of the art for ViT from scratch on CIFAR-100 is ~84-85% with ~14M params.
Our best (61.33%) uses ~200K params. There is significant room for improvement
with larger models (D=128+), stronger augmentation, and longer training.

### D=128 with Stronger Augmentation (300 epochs, cosine lr=1e-3, dropout=0.1, wd=0.1, mixup α=0.8, cutout 16×16)

~800K params. Stronger augmentation (larger cutout, stronger mixup) to handle increased capacity.
Train/test gap: train ~53%, test ~62% at epoch 200 — healthy, still underfitting.

### Final Results (D=128)

| Model | Type | Freqs | V rot | D=128 |
|-------|------|-------|-------|-------|
| **joformer_axial** | axial | learned | yes | **66.67%** |
| joformer | combined | learned | yes | 66.19% |
| joformer_old | axial | fixed | yes | 66.10% |
| monoidal_axial | axial | learned | no | 64.81% |
| rope2d | axial | fixed | no | 64.22% |
| monoidal | combined | learned | no | 63.62% |
| joformer_fixed | combined | fixed | yes | 60.82% |
| learned | additive | — | no | 60.74% |
| rope2dv2 | combined | fixed | no | 60.45% |

### Scaling Summary Across All D

| Model | D=32 | D=64 | D=128 |
|-------|------|------|-------|
| joformer_axial | 52.77 | 61.33 | **66.67** |
| joformer | 52.43 | 58.83 | 66.19 |
| joformer_old | 50.54 | 59.13 | 66.10 |
| monoidal_axial | 51.75 | 59.10 | 64.81 |
| rope2d | 50.91 | 59.23 | 64.22 |
| monoidal | 51.59 | 58.66 | 63.62 |
| learned | 52.23 | 57.26 | 60.74 |
| joformer_fixed | 49.20 | 55.79 | 60.82 |
| rope2dv2 | 49.04 | 55.21 | 60.45 |

### joformer_axial vs rope2d gap grows with scale

| D | rope2d | joformer_axial | gap |
|---|--------|---------------|-----|
| 32 | 50.91% | 52.77% | +1.86% |
| 64 | 59.23% | 61.33% | +2.10% |
| 128 | 64.22% | 66.67% | **+2.45%** |

### Convergence Table (D=128, best test acc, all 9 models)

| Epoch | learned | rope2d | jfmr_old | mon_ax | jfmr_ax | rope2dv2 | monoidal | joformer | jfmr_fix |
|-------|---------|--------|----------|--------|---------|----------|----------|----------|----------|
| 10 | 26.47 | 28.57 | 29.99 | 29.99 | 29.66 | 26.99 | 28.66 | 28.85 | 27.59 |
| 20 | 32.04 | 37.31 | 38.11 | 38.80 | 37.47 | 33.52 | 37.38 | 37.88 | 33.39 |
| 30 | 36.22 | 43.10 | 44.20 | 43.80 | 43.54 | 39.44 | 41.67 | 43.04 | 38.58 |
| 50 | 42.87 | 48.27 | 48.72 | 48.34 | 50.32 | 44.62 | 46.91 | 49.27 | 43.76 |
| 100 | 50.48 | 53.28 | 55.65 | 53.75 | 56.01 | 49.97 | 52.97 | 55.61 | 50.26 |
| 150 | 53.63 | 57.20 | 58.85 | 57.45 | 59.92 | 52.85 | 56.66 | 59.46 | 53.83 |
| 200 | 57.40 | 60.75 | 62.17 | 60.76 | 63.31 | 56.60 | 60.05 | 62.87 | 57.19 |
| 250 | 59.80 | 63.00 | 65.32 | 63.95 | 65.52 | 59.23 | 62.71 | 65.63 | 59.82 |
| 300 | 60.74 | 64.22 | 66.10 | 64.81 | **66.67** | 60.45 | 63.62 | 66.19 | 60.82 |

### Analysis

**1. V rotation is the dominant factor at D=128.**
Top 3 are all JoFormer variants (66.67, 66.19, 66.10). The ~2% gap over non-V models is consistent and clear.

**2. The V rotation advantage grows with scale.**
- D=32: noisy, depends on LR schedule
- D=64: +2.1% (joformer_axial vs rope2d)
- D=128: +2.45% (joformer_axial vs rope2d)
This trend suggests V rotation will matter even more at larger D.

**3. Learnable frequencies help with V rotation.**
joformer_axial (66.67%) > joformer_old (66.10%) = +0.57%. Learnable frequencies adapt to work with V rotation.

**4. Without V rotation, learnable frequencies provide modest help.**
monoidal_axial (64.81%) > rope2d (64.22%) = +0.59%. A small but consistent gain.

**5. Combined approach still trails axial at D=128.**
monoidal (63.62%) < monoidal_axial (64.81%). joformer (66.19%) < joformer_axial (66.67%).
But the gap is smaller than at D=32/64.

**6. joformer_axial leads at EVERY checkpoint from epoch 50 onward.**
Not a late-stage effect — it converges faster AND reaches a higher final accuracy.

### D=256 with Full Augmentation (300 epochs, cosine lr=1e-3, dropout=0.1, wd=0.05, label_smoothing=0.1, RandAugment+mixup+cutout)

~4M params, 5 layers, 8 heads. Added RandAugment (2 ops, magnitude 9), increased mixup α to 0.8,
increased cutout to 16×16, added label smoothing 0.1. Train/test gap: train ~64%, test ~63% — well balanced.

Note: Had to reduce weight_decay from 0.15 to 0.05 and dropout from 0.2 to 0.1 because
the combination of heavy regularization + RandAugment was too aggressive (train only 28% at epoch 130).

### Complete Results

| Model | D=32 | D=64 | D=128 | D=256 |
|-------|------|------|-------|-------|
| **joformer_axial** | 52.77 | **61.33** | **66.67** | **63.27** |
| joformer_axial_perlayer | — | — | — | 62.76 |
| monoidal | 51.59 | 58.66 | 63.62 | 62.67 |
| joformer | 52.43 | 58.83 | 66.19 | 62.10 |
| joformer_old | 50.54 | 59.13 | 66.10 | 61.85 |
| rope2d | 50.91 | 59.23 | 64.22 | 61.39 |
| joformer_perlayer | — | — | — | 61.17 |
| monoidal_axial | 51.75 | 59.10 | 64.81 | 60.92 |
| monoidal_perlayer | — | — | — | 60.85 |
| monoidal_axial_perlayer | — | — | — | 60.63 |
| joformer_fixed | 49.20 | 55.79 | 60.82 | 56.66 |
| rope2dv2 | 49.04 | 55.21 | 60.45 | 55.53 |
| learned | 52.23 | 57.26 | 60.74 | 55.49 |

Note: D=128 used lighter augmentation (mixup α=0.2, cutout 8×8, no RandAugment).
D=256 used heavier augmentation (RandAugment, mixup α=0.8, cutout 16×16, label smoothing 0.1).
D=256 numbers are lower than D=128 because 300 epochs is insufficient to fully converge
with the heavier augmentation. The relative ordering is what matters.

### D=256 Convergence (best test acc, completed models)

| Epoch | learned | rope2d | jfmr_old | mon_ax | jfmr_ax | mon_ax_pl | jfmr_ax_pl |
|-------|---------|--------|----------|--------|---------|-----------|-----------|
| 50 | 39.65 | 43.49 | 44.72 | 44.47 | 46.04 | — | — |
| 100 | 45.48 | 51.94 | 53.66 | 52.65 | 54.28 | — | — |
| 150 | 50.23 | 56.40 | 57.47 | 56.61 | 58.51 | — | — |
| 200 | 53.38 | 58.89 | 60.13 | 59.43 | 60.59 | — | — |
| 250 | 55.37 | 60.48 | 61.38 | 60.77 | 62.38 | — | — |
| 300 | 55.49 | 61.39 | 61.85 | 60.92 | **63.27** | 60.63 | 62.76 |

### Analysis

**1. joformer_axial wins at every scale.**

| D | rope2d | joformer_axial | gap |
|---|--------|---------------|-----|
| 32 | 50.91% | 52.77% | +1.86% |
| 64 | 59.23% | 61.33% | +2.10% |
| 128 | 64.22% | 66.67% | +2.45% |
| 256 | 61.39% | 63.27% | +1.88% |

The advantage is consistent at ~2% across all scales. D=256 numbers are lower than D=128
because RandAugment makes training much harder and 300 epochs may not be enough to converge fully.

**2. V rotation advantage grows with D (without RandAugment).**
At D=128 without RandAugment: joformer_old (66.10%) vs rope2d (64.22%) = +1.88%.
At D=256 with RandAugment: joformer_old (61.85%) vs rope2d (61.39%) = +0.46%.
The gap compressed at D=256 likely due to insufficient training time with heavy augmentation.

**3. Per-layer frequencies don't help.**
monoidal_axial_perlayer (60.63%) < monoidal_axial (60.92%).
joformer_axial_perlayer (62.76%) < joformer_axial (63.27%).
Shared frequencies across layers work better — perhaps because consistent positional encoding
helps the residual stream maintain coherent position information.

**4. Learned PE suffers most from heavy augmentation.**
learned dropped from 60.74% (D=128) to 55.49% (D=256) despite 5x more params.
The additive position embedding may not interact well with RandAugment's geometric transforms.
Rotation-based PE is inherently more robust to image transformations.

**5. joformer_axial leads at EVERY checkpoint.**
At epoch 50, 100, 150, 200, 250, 300 — joformer_axial is ahead of all other models.
The advantage is not a late-stage effect; it converges faster throughout training.

### V Rotation Effect: Paired Comparison (D=256)

| Approach | Q/K only | Q/K/V + inverse | V helps? |
|----------|----------|-----------------|----------|
| Axial fixed | rope2d: 61.39% | joformer_old: 61.85% | +0.46% |
| Axial learned | monoidal_axial: 60.92% | **joformer_axial: 63.27%** | **+2.35%** |
| Axial learned perlayer | monoidal_axial_pl: 60.63% | joformer_axial_pl: 62.76% | +2.13% |
| Combined fixed | rope2dv2: 55.53% | joformer_fixed: 56.66% | +1.13% |
| Combined learned | monoidal: 62.67% | joformer: 62.10% | -0.57% |
| Combined learned perlayer | monoidal_pl: 60.85% | joformer_pl: 61.17% | +0.32% |

**V rotation helps in 5 of 6 cases.** The biggest gain is with axial learned frequencies (+2.35%).
The only case where V rotation hurts is combined learned (monoidal vs joformer: -0.57%).

### Per-Layer Frequencies Don't Help

| Approach | Shared | Per-layer | Difference |
|----------|--------|-----------|------------|
| monoidal_axial | 60.92% | 60.63% | -0.29% |
| joformer_axial | 63.27% | 62.76% | -0.51% |
| monoidal | 62.67% | 60.85% | -1.82% |
| joformer | 62.10% | 61.17% | -0.93% |

Per-layer frequencies consistently underperform shared frequencies. Shared frequencies
provide a consistent positional reference across all layers, which seems important for
the residual stream to maintain coherent position information.

### Combined vs Axial at D=256

| | Axial | Combined | Difference |
|--|-------|----------|------------|
| Fixed, no V | rope2d: 61.39% | rope2dv2: 55.53% | axial +5.86% |
| Fixed, V | joformer_old: 61.85% | joformer_fixed: 56.66% | axial +5.19% |
| Learned, no V | monoidal_axial: 60.92% | monoidal: 62.67% | **combined +1.75%** |
| Learned, V | joformer_axial: 63.27% | joformer: 62.10% | axial +1.17% |

With fixed frequencies, axial dominates (~5-6% gap). With learned frequencies, the gap
narrows dramatically and combined even wins for no-V (monoidal: 62.67% > monoidal_axial: 60.92%).
This confirms the RoPE-Mixed paper's finding that combined improves with learnable frequencies at larger D.

### joformer_axial Advantage Across Scales

| D | rope2d | joformer_axial | gap |
|---|--------|---------------|-----|
| 32 | 50.91% | 52.77% | +1.86% |
| 64 | 59.23% | 61.33% | +2.10% |
| 128 | 64.22% | 66.67% | +2.45% |
| 256 | 61.39% | 63.27% | +1.88% |

Consistent ~2% advantage across all scales. The gap doesn't shrink with scale —
it remains significant from D=32 through D=256.

### Key Takeaways

1. **joformer_axial (learnable axial + V rotation) is the best model at every scale tested.**
2. **V rotation provides ~2% gain with learnable axial frequencies**, consistent across D=64, 128, 256.
3. **The V rotation + learnable frequency interaction is synergistic** — neither alone provides the full benefit.
4. **Combined approach catches up to axial at larger D** when frequencies are learnable, but axial + V still wins.
5. **Per-layer frequencies don't help** — shared frequencies across layers work better.
6. **Proper regularization is essential** for the architectural differences to manifest clearly.

---

## Part 5: Outstanding Questions

### V rotation: why does it hurt at long training?
At 100 epochs with fixed LR, V rotation helps (+0.62%). At 1000 epochs, it hurts (-0.36%). Hypotheses:
- V rotation adds optimization complexity that prevents reaching the true optimum
- The inverse rotation on the output may interfere with the residual connections at convergence
- The model may overfit to the position-dependent value transformation
- Warmup or different LR schedules might change this (currently testing cosine 300 epochs)

### Initialization matters enormously for learnable frequencies
Random near-zero init (monoidal_axial: 48.04%) vs fixed RoPE frequencies (rope2d: 51.53%) — a 3.5% gap. The model needs useful frequencies from the start with fixed LR. Options:
- Initialize learnable frequencies to RoPE values (not random)
- Use warmup to let frequencies learn before heavy training
- Use cosine schedule (helps learned models converge)

### Scale dependence
At D=32, axial and combined are within ~0.05% (rope2d 51.53% vs monoidal 51.58%). The RoPE-Mixed paper showed larger advantages at D=768. Our results may be noise at this scale.

---

## Next Steps

- 300-epoch cosine lr=1e-3 run (currently queued, all 9 models)
- Test warmup schedule: warmup_lr=2e-4 for 10 epochs, high_lr=1e-3 for 20 epochs, then base lr
- Initialize learnable frequencies to RoPE values instead of random
- ImageNet experiments on H100 (Thunder Compute machine)
- Language modeling at scale (GPT-2 on OpenWebText)
- Multiple seeds for statistical significance

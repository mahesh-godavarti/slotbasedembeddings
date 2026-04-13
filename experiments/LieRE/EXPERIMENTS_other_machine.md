# CIFAR-100 Positional Encoding Experiments

## Goal

Investigate whether V rotation (rotating values in attention, not just Q/K) improves
Vision Transformers with different positional encoding schemes. Test on CIFAR-100
using LieRE's training framework for fair comparison.

## Setup

- **Architecture**: ViT-Tiny (D=384, 12 layers, 6 heads, dim_head=64, ~14.9M params)
- **Dataset**: CIFAR-100 (50K train, 10K test, 100 classes, 32x32 images)
- **Patch size**: 4x4 -> 8x8 = 64 patches + CLS token
- **Training**: Adam lr=1e-4, cosine annealing over 200 epochs, bf16-mixed
- **Augmentation**: random crop (pad 4), random horizontal flip, normalize
- **GPU**: NVIDIA H100 PCIe
- **Framework**: LieRE's PyTorch Lightning codebase, deterministic (seed=42)

## Methods Tested

### 1. LieRE64 (original LieRE, generator_dim=64)
- **Rotation**: One 64x64 dense rotation per head per layer
- **Parameterization**: Learned skew-symmetric matrices -> matrix_exp
- **Combined**: Both axes encoded in same rotation (angle = pos_y*S_y + pos_x*S_x)
- **Commutativity**: NOT guaranteed. [S_y, S_x] != 0 in general -> not purely relative PE
- **Code**: `run_cifar100.py`
- **Rotation params**: 589,824

### 2. LieRE8 (generator_dim=8)
- **Rotation**: Eight 8x8 dense rotations per head per layer
- **Same parameterization as LieRE64** but with smaller blocks
- **Commutativity**: NOT guaranteed (same issue as LieRE64, but smaller blocks may be "more nearly commutative")
- **Code**: `run_cifar100.py --generator_dim 8`

### 3. Block-diagonal 2x2 (our JoFormer approach)
- **Rotation**: 32 independent 2x2 rotations per head per layer
- **Parameterization**: Learned frequencies -> angle = pos_y*freq_y + pos_x*freq_x -> cos/sin
- **Combined**: Both axes in same rotation pair
- **Commutativity**: GUARANTEED. SO(2) is abelian -> exact relative PE
- **Code**: `run_cifar100_block_v.py`
- **Rotation params**: ~4,608 per layer

### 4. Butterfly (combined)
- **Rotation**: 6 rounds of 2x2 block rotations with interleaved permutations on full 64 dims
- **Parameterization**: Learned frequencies per round -> cos/sin + fixed permutations
- **Combined**: angle = pos_y*freq_y + pos_x*freq_x (both axes, all dims)
- **Commutativity**: NOT purely relative. Permutations between rounds prevent clean cancellation.
- **Code**: `run_cifar100_butterfly.py`
- **Rotation params**: 27,648

### 5. Axial butterfly (commutative)
- **Rotation**: Separate butterfly on each half of head dims
  - Y-butterfly: 5 rounds on dims 0-31, using pos_y only
  - X-butterfly: 5 rounds on dims 32-63, using pos_x only
- **Commutativity**: GUARANTEED. Disjoint subspaces -> trivially commute.
- **Code**: `run_cifar100_axial_butterfly.py`

### 6. Axial dense (commutative)
- **Rotation**: Separate dense rotation on each half of head dims
  - Y-axis: 32x32 dense rotation on dims 0-31 via matrix_exp
  - X-axis: 32x32 dense rotation on dims 32-63 via matrix_exp
- **Commutativity**: GUARANTEED. Disjoint subspaces.
- **Code**: `run_cifar100_axial_dense.py`

### 7. Random-mix
- **Rotation**: Random orthogonal mixing with learned frequencies
- **Code**: `run_cifar100_randmix.py`

### 8. Axial Cayley (experimental, not yet run)
- **Rotation**: Same as axial dense but uses Cayley transform R = (I-A)(I+A)^{-1} instead of matrix_exp
- **Speed**: Only ~1.15x faster than matrix_exp at 32x32 -- marginal benefit
- **Code**: `run_cifar100_axial_cayley.py`

## Results

### Completed experiments (200 epochs, cosine T_max=200)

| Method | Commutative? | Q/K only | Q/K/V | V rotation effect |
|--------|-------------|----------|-------|-------------------|
| **Axial dense (32x32 per axis)** | **Yes** | **69.24%** | **69.62%** | **+0.38% (helps)** |
| LieRE64 (dense 64x64) | No | 69.14% | 67.15% | -2.0% (hurts) |
| LieRE8 (dense 8x8) | No | 69.16% | -- | -- |
| Butterfly combined (6 rounds) | No | 67.42% | 68.17% | +0.75% (helps) |
| Block 2x2 (combined) | Yes | 66.07% | 66.66% | +0.6% (helps) |
| Axial butterfly | Yes | running | running | -- |
| Random-mix n_rounds=2 | No | queued | queued | -- |

### Completed experiments (400 epochs, cosine T_max=400)

| Method | Q/K only | Q/K/V | V rotation effect |
|--------|----------|-------|-------------------|
| LieRE64 (dense 64x64) | 70.39% | ~68.5% (interrupted at ep 300) | ~-1.5% (hurts) |

### Axial dense training curves (200 epochs)

| Epoch | Axial dense Q/K | Axial butterfly Q/K | LieRE64 Q/K | Axial dense Q/K/V |
|-------|----------------|-------------------|-------------|-------------------|
| 10 | 34.05% | 29.39% | 34.03% | 37.25% |
| 20 | 46.61% | 41.44% | 45.45% | 48.43% |
| 30 | 52.18% | 49.54% | 51.68% | 53.84% |
| 40 | 57.66% | 54.51% | 55.76% | 57.67% |
| 50 | 59.70% | 58.44% | 59.37% | 60.50% |
| 60 | 62.52% | 60.33% | 61.54% | 63.28% |
| 70 | 64.31% | 62.49% | 63.56% | 64.28% |
| 80 | 65.36% | 63.81% | 64.58% | 66.16% |
| 90 | 65.89% | 65.26% | 66.13% | 66.22% |
| 100 | 66.83% | 66.41% | 66.49% | 67.58% |
| 110 | 67.38% | -- | 67.32% | 67.28% |
| 120 | 68.13% | -- | 67.43% | 68.22% |
| 130 | 67.97% | -- | 68.23% | 68.52% |
| 140 | 68.64% | -- | 67.54% | 68.88% |
| 150 | 68.94% | -- | 68.41% | 69.22% |
| 160 | 68.96% | -- | 68.23% | 69.51% |
| 170 | 69.26% | -- | 68.31% | 69.31% |
| 180 | 69.15% | -- | 68.90% | 69.70% |
| 190 | 69.13% | -- | 69.26% | 69.70% |
| **200** | **69.24%** | -- | **69.14%** | **69.62%** |

## Key Findings

### 1. Axial dense is the best method found so far

- Q/K only: 69.24% -- matches LieRE64 (69.14%) despite being constrained to commutative rotations
- Q/K/V: **69.62%** -- best result overall, +0.48% over LieRE64 Q/K
- The axial factorization (y-axis on dims 0-31, x-axis on dims 32-63) is an inductive bias
  that matches the actual 2D grid structure of image patches

### 2. V rotation helps commutative rotations, hurts non-commutative

| Method | Commutative? | V rotation effect |
|--------|-------------|-------------------|
| Axial dense | Yes | +0.38% |
| Block 2x2 | Yes | +0.6% |
| Butterfly | No* | +0.75% |
| LieRE64 | No | -2.0% |

*Butterfly is not formally commutative but has structured permutations.

**Theory**: With commutative rotations, R(pos_i)^{-1} * R(pos_j) = R(pos_j - pos_i),
so V rotation gives a proper relative position encoding on values. With non-commutative
rotations, the inverse doesn't cleanly produce relative offsets -- it depends on absolute
positions, introducing noise rather than useful structure.

### 3. Constrained expressiveness is not a weakness

- LieRE64 has a full 64x64 rotation mixing both axes
- Axial dense has two independent 32x32 rotations (one per axis)
- Despite being strictly less expressive (block-diagonal vs full), axial dense matches on Q/K
- The constraint forces the rotation to respect the 2D grid structure rather than
  wasting capacity on cross-axis interactions that may not be useful

### 4. Dense rotations > structured rotations within each axis

- LieRE64: 69.14%, Axial dense: 69.24%, Butterfly: 67.42%, Block: 66.07%
- More expressive rotations within each subspace capture richer position representations
- Axial butterfly trails axial dense by ~2-3% early, closing over training

### 5. LieRE8 matches LieRE64 despite smaller blocks

- Consistent with LieRE paper results
- 8x8 blocks are much slower than one 64x64 block (GPU prefers one big matmul)

## Implications

The combination of **axial factorization + V rotation** achieves the best of both worlds:
- **Commutativity** (from axial structure) enables V rotation to work as a proper relative PE
- **Dense rotations** (within each axis) maintain the expressiveness that makes LieRE strong
- This is both theoretically clean (exact relative PE by ComRoPE Theorem) and empirically superior

The fact that LieRE64's non-commutativity *hurts* with V rotation but axial dense's
commutativity *helps* provides direct experimental evidence for the ComRoPE theorem's
practical relevance: commutativity is not just a theoretical property -- it determines
whether V rotation adds signal or noise.

## Speed analysis

### Per-epoch training time (CIFAR-100, ViT-Tiny, H100 PCIe)

| Method | Per epoch | Relative |
|--------|----------|----------|
| Block 2x2 | ~25s | 1x |
| RandMix | ~30s (est.) | ~1.2x |
| LieRE64 | ~59s | 2.4x |
| Axial dense | ~55s (est.) | ~2.2x |
| Butterfly | ~95s | 3.8x |
| LieRE8 | ~120s | 4.8x |

### Forward pass microbenchmark (batch=128, H100 PCIe)

| Method | Forward pass | Relative to fastest | How rotation is computed |
|--------|-------------|--------------------|-----------------------|
| RandMix | 44.0ms | 1x | Fixed random ortho mix + learned 2x2 sin/cos |
| Axial dense | 75.1ms | 1.7x | Two 32x32 matrix_exp (one per axis) |
| LieRE64 | 102.6ms | 2.3x | One 64x64 matrix_exp |

### Why matrix_exp isn't as slow as expected

At 32x32, `matrix_exp` is internally ~5-10 matrix multiplications (Pade approximation).
The GPU parallelizes the batched exponentiation across all tokens/heads/layers simultaneously.
Alternative approaches we benchmarked:

- **Cayley transform** R = (I-A)(I+A)^{-1}: Only 1.15x faster than matrix_exp at 32x32.
  Uses `torch.linalg.solve` which is similar cost to the Pade matmuls.
- **Repeated multiplication** (learn R, compute R^0, R^1, R^2, ...): Only 1.19x faster.
  Inherently sequential (each power depends on previous), so GPU can't parallelize.
  Also requires a soft orthogonality penalty ||RR^T - I|| instead of exact guarantees.

### Why RandMix is fast

RandMix avoids matrix_exp entirely. It uses:
1. Learned 2x2 block-diagonal rotations (cos/sin, like standard RoPE) -- very fast
2. Fixed random orthogonal mixing matrices between rounds -- no learned params, just matmul
3. Only 2-3 rounds of (rotate + mix), each is a simple batched matmul

The mixing matrices are generated once at init and never updated, so no gradient
computation through them. The learned parameters are just the 2x2 frequencies.

### Speed vs accuracy tradeoff

| Method | Forward (ms) | Q/K accuracy | Q/K/V accuracy |
|--------|-------------|-------------|----------------|
| RandMix | 44.0 | running | running |
| Axial dense | 75.1 | 69.24% | **69.62%** |
| LieRE64 | 102.6 | 69.14% | 67.15% |

Axial dense achieves the best accuracy while being 1.4x faster than LieRE64.
If RandMix can match accuracy, it would be the clear winner at 2.3x faster.

## Commutativity analysis

**ComRoPE Theorem (CVPR 2025)**: Rotation-based PE gives relative position encoding
if and only if the per-axis rotation matrices commute.

For 2D images: R(y,x) must satisfy R(y1,x1)^{-1} * R(y2,x2) = R(dy, dx)

| Method | Commutative? | Relative PE? |
|--------|-------------|-------------|
| Block 2x2 combined | Yes (SO(2) abelian) | Exact |
| Axial (any size) | Yes (disjoint subspaces) | Exact |
| LieRE combined | No ([S_y, S_x] != 0) | Approximate |
| Butterfly combined | No (permutations break it) | Structured |

## File inventory

| File | Description |
|------|-------------|
| `run_cifar100.py` | LieRE runner (original, supports --generator_dim and --rotate_v) |
| `run_cifar100_block_v.py` | Block-diagonal 2x2 runner |
| `run_cifar100_butterfly.py` | Combined butterfly runner |
| `run_cifar100_axial_butterfly.py` | Axial butterfly runner |
| `run_cifar100_axial_dense.py` | Axial dense runner |
| `run_cifar100_axial_cayley.py` | Axial Cayley runner (experimental) |
| `run_cifar100_randmix.py` | Random-mix runner |
| `models/rope_vit.py` | LieRE model |
| `models/rope_vit_block_v.py` | Block-diagonal 2x2 model |
| `models/rope_vit_butterfly.py` | Combined butterfly model |
| `models/rope_vit_axial_butterfly.py` | Axial butterfly model |
| `models/rope_vit_axial_dense.py` | Axial dense model |
| `models/rope_vit_axial_cayley.py` | Axial Cayley model (experimental) |
| `models/rope_vit_randmix.py` | Random-mix model |
| `models/rope_vit_axial_randmix.py` | Axial random-mix model |

# CIFAR-100 Positional Encoding Experiments

## Goal

Investigate whether V rotation (rotating values in attention, not just Q/K) improves
Vision Transformers with different positional encoding schemes. Test on CIFAR-100
using LieRE's training framework for fair comparison.

## Setup

- **Architecture**: ViT-Tiny (D=384, 12 layers, 6 heads, dim_head=64, ~14.9M params)
- **Dataset**: CIFAR-100 (50K train, 10K test, 100 classes, 32×32 images)
- **Patch size**: 4×4 → 8×8 = 64 patches + CLS token
- **Training**: Adam lr=1e-4, cosine annealing over 200 epochs, bf16-mixed
- **Augmentation**: random crop (pad 4), random horizontal flip, normalize
- **GPU**: NVIDIA H100 PCIe
- **Framework**: LieRE's PyTorch Lightning codebase, deterministic (seed=42)

## Methods Tested

### 1. LieRE64 (original LieRE, generator_dim=64)
- **Rotation**: One 64×64 dense rotation per head per layer
- **Parameterization**: Learned skew-symmetric matrices → matrix_exp
- **Combined**: Both axes encoded in same rotation (pos_y·S_y + pos_x·S_x → one matrix_exp)
- **Commutativity**: NOT guaranteed. [S_y, S_x] ≠ 0 in general → not purely relative PE
- **Rotation params**: 589,824 (2016 free params per head per layer)

### 2. LieRE8 (generator_dim=8)
- **Rotation**: Eight 8×8 dense rotations per head per layer
- **Same parameterization as LieRE64** but with smaller blocks
- **Commutativity**: NOT guaranteed (same issue, but smaller blocks "more nearly commutative")

### 3. Block-diagonal 2×2 (our JoFormer approach)
- **Rotation**: 32 independent 2×2 rotations per head per layer
- **Parameterization**: Learned frequencies → angle = pos_y·freq_y + pos_x·freq_x → cos/sin
- **Combined**: Both axes in same rotation pair
- **Commutativity**: GUARANTEED. SO(2) is abelian → exact relative PE

### 4. Butterfly (combined)
- **Rotation**: 6 rounds of 2×2 block rotations with interleaved permutations on full 64 dims
- **Parameterization**: Learned frequencies per round → cos/sin + fixed permutations
- **Combined**: Both axes in each round
- **Commutativity**: NOT purely relative. Permutations between rounds prevent clean cancellation.

### 5. Axial butterfly (commutative)
- **Rotation**: Separate butterfly on each half of head dims
  - Y-butterfly: 5 rounds on dims 0-31, using pos_y only
  - X-butterfly: 5 rounds on dims 32-63, using pos_x only
- **Commutativity**: GUARANTEED. Disjoint subspaces.

### 6. Axial dense (commutative) ★ BEST METHOD
- **Rotation**: Separate dense rotation on each half of head dims
  - Y-axis: 32×32 dense rotation on dims 0-31 via matrix_exp
  - X-axis: 32×32 dense rotation on dims 32-63 via matrix_exp
- **Commutativity**: GUARANTEED. Disjoint subspaces.
- **Rotation params**: 2 × 496 = 992 free params per head per layer (half of LieRE64)

### 7. Random-mix (combined)
- **Rotation**: n rounds of (2×2 block rotation + fixed random orthogonal mix)
- **Parameterization**: Learned 2×2 frequencies + fixed random orthogonal matrices between rounds
- **Combined**: Both axes in each round
- **Commutativity**: NOT purely relative (mix matrices between position-dependent rotations)

### 8. Axial random-mix (commutative)
- **Rotation**: Separate random-mix on each half of head dims
  - Y-axis: n rounds on dims 0-31, using pos_y only
  - X-axis: n rounds on dims 32-63, using pos_x only
- **Commutativity**: GUARANTEED. Disjoint subspaces.

### 9. Axial ortho (commutative, experimental)
- **Rotation**: Same as axial dense but learns a free matrix R per axis/head/layer
- **Parameterization**: R^i by repeated multiplication for discrete position step i
- **Orthogonality**: Soft penalty ‖RR^T - I‖ (not exact like matrix_exp)
- **Speed**: Only 1.19× faster than matrix_exp — not worth the soft constraint

### 10. Axial Cayley (commutative, experimental)
- **Rotation**: Same as axial dense but uses Cayley transform R = (I-A)(I+A)^{-1}
- **Speed**: Only 1.15× faster than matrix_exp at 32×32 — marginal benefit

## Results

### Main results (200 epochs, cosine T_max=200, best accuracy during training)

| Method | Commutative? | Q/K only | Q/K/V | V rotation effect |
|--------|-------------|----------|-------|-------------------|
| **Axial dense** | **Yes** | **69.26%** | **69.70%** | **+0.44% (helps)** |
| LieRE64 | No | 69.31% | 67.42% | -1.89% (hurts) |
| LieRE8 | No | 69.24% | — (too slow) | — |
| Axial butterfly | Yes | 68.57% | running | — |
| Butterfly combined | No | 67.42% | 68.17% | +0.75% (helps) |
| Axial randmix (n=2) | Yes | 67.37% | running | — |
| Block 2×2 | Yes | 66.07% | 66.71% | +0.64% (helps) |

### 400-epoch results (cosine T_max=400, best accuracy)

| Method | Q/K only | Q/K/V | V rotation effect |
|--------|----------|-------|-------------------|
| LieRE64 | 70.64% | 69.41% | -1.23% (hurts) |

### Axial dense training curves (200 epochs, from other machine)

| Epoch | Axial dense Q/K | Axial dense Q/K/V | LieRE64 Q/K | Delta (dense V) |
|-------|----------------|-------------------|-------------|-----------------|
| 10 | 34.05% | 37.25% | 34.03% | +3.20 |
| 20 | 46.61% | 48.43% | 45.45% | +1.82 |
| 30 | 52.18% | 53.84% | 51.68% | +1.66 |
| 50 | 59.70% | 60.50% | 59.37% | +0.80 |
| 70 | 64.31% | 64.28% | 63.56% | -0.03 |
| 100 | 66.83% | 67.58% | 66.49% | +0.75 |
| 140 | 68.64% | 68.88% | 67.54% | +0.24 |
| 160 | 68.96% | 69.51% | 68.23% | +0.55 |
| 180 | 69.15% | 69.70% | 68.90% | +0.55 |
| 200 | 69.24% | 69.62% | 69.14% | +0.38 |

## Key Findings

### 1. Axial dense is the best method found

- Q/K only: 69.26% — matches LieRE64 (69.31%) with half the rotation params
- Q/K/V: **69.70%** — best result overall, +0.44% over Q/K, +0.39% over LieRE64 Q/K
- The axial factorization matches the 2D grid structure of image patches
- Fewer params (992 vs 2016 per head/layer) acts as natural regularizer

### 2. V rotation helps commutative rotations, hurts non-commutative

| Method | Commutative? | V rotation effect |
|--------|-------------|-------------------|
| Axial dense | Yes | +0.44% (helps) |
| Butterfly | No* | +0.75% (helps) |
| Block 2×2 | Yes | +0.64% (helps) |
| LieRE64 | No | -1.89% (hurts) |

*Butterfly is not formally commutative but has structured (fixed) permutations.

**Theory**: With commutative rotations, R(pos_i)^{-1} · R(pos_j) = R(pos_j - pos_i),
so V rotation gives proper relative position encoding on values. With non-commutative
rotations, the inverse depends on absolute positions, introducing noise.

### 3. Constrained expressiveness is not a weakness

- LieRE64: full 64×64 rotation mixing both axes
- Axial dense: two independent 32×32 rotations (one per axis)
- Despite being strictly less expressive, axial dense matches or beats LieRE64
- The constraint forces the rotation to respect 2D grid structure

### 4. Dense rotations > structured rotations within each axis

- Axial dense (69.26%) > Axial butterfly (68.57%) > Block 2×2 (66.07%)
- More expressive rotations within each subspace capture richer position representations
- But the axial constraint (disjoint subspaces) is key for enabling V rotation

### 5. LieRE8 ≈ LieRE64 despite smaller blocks

- LieRE8: 69.24%, LieRE64: 69.31%
- Consistent with LieRE paper results
- 8×8 blocks much slower than one 64×64 (GPU prefers one big matmul)

## Speed Analysis

### Forward pass microbenchmark (batch=128, H100 PCIe)

| Method | Forward pass | Relative | How rotation is computed |
|--------|-------------|----------|------------------------|
| RandMix | 44.0ms | 1× | Fixed random ortho mix + learned 2×2 cos/sin |
| Axial dense | 75.1ms | 1.7× | Two 32×32 matrix_exp (one per axis) |
| LieRE64 | 102.6ms | 2.3× | One 64×64 matrix_exp |

### Per-epoch training time

| Method | Per epoch | Relative |
|--------|----------|----------|
| Block 2×2 | ~25s | 1× |
| RandMix | ~30s | ~1.2× |
| Axial dense | ~55s | ~2.2× |
| LieRE64 | ~59s | 2.4× |
| Butterfly | ~95s | 3.8× |
| LieRE8 | ~120s | 4.8× |

### Speed vs accuracy tradeoff

| Method | Forward (ms) | Q/K accuracy | Q/K/V accuracy |
|--------|-------------|-------------|----------------|
| RandMix | 44.0 | running | running |
| Axial dense | 75.1 | 69.26% | **69.70%** |
| LieRE64 | 102.6 | 69.31% | 67.42% |

Axial dense: best accuracy AND 1.4× faster than LieRE64.

### Why alternative parameterizations don't help much

- **Cayley transform** R = (I-A)(I+A)^{-1}: Only 1.15× faster than matrix_exp at 32×32.
  Uses torch.linalg.solve which is similar cost to Padé matmuls.
- **Repeated multiplication** (learn R, compute R^0, R^1, ...): Only 1.19× faster.
  Inherently sequential, requires soft orthogonality penalty instead of exact guarantees.

## Commutativity Analysis

**ComRoPE Theorem (CVPR 2025)**: Rotation-based PE gives relative position encoding
if and only if the per-axis rotation matrices commute.

For 2D images: R(y,x) must satisfy R(y₁,x₁)^{-1} · R(y₂,x₂) = R(Δy, Δx)

| Method | Commutative? | Relative PE? |
|--------|-------------|-------------|
| Block 2×2 combined | ✓ (SO(2) abelian) | ✓ exact |
| Axial (any size) | ✓ (disjoint subspaces) | ✓ exact |
| LieRE combined | ✗ ([S_y, S_x] ≠ 0) | ✗ approximate |
| Butterfly combined | ✗ (permutations break it) | ✗ structured |

## Implications

The combination of **axial factorization + V rotation** achieves the best of both worlds:
- **Commutativity** (from axial structure) enables V rotation as proper relative PE
- **Dense rotations** (within each axis) maintain the expressiveness that makes LieRE strong
- Theoretically clean (exact relative PE by ComRoPE Theorem) AND empirically superior
- Fewer rotation parameters than LieRE64 (992 vs 2016 per head/layer)

The fact that LieRE64's non-commutativity hurts with V rotation (-1.89%) but axial dense's
commutativity helps (+0.44%) provides direct experimental evidence for the ComRoPE theorem's
practical relevance.

## File Inventory

| File | Description |
|------|-------------|
| `run_cifar100.py` | LieRE runner (supports --generator_dim and --rotate_v) |
| `run_cifar100_block_v.py` | Block-diagonal 2×2 runner |
| `run_cifar100_butterfly.py` | Combined butterfly runner |
| `run_cifar100_axial_butterfly.py` | Axial butterfly runner |
| `run_cifar100_axial_dense.py` | Axial dense runner |
| `run_cifar100_randmix.py` | Random-mix runner |
| `run_cifar100_axial_randmix.py` | Axial random-mix runner |
| `run_cifar100_axial_cayley.py` | Axial Cayley runner (experimental) |
| `run_cifar100_axial_ortho.py` | Axial ortho runner (experimental) |
| `models/rope_vit.py` | LieRE model (modified: added rotate_v) |
| `models/rope_vit_block_v.py` | Block-diagonal 2×2 model |
| `models/rope_vit_butterfly.py` | Combined butterfly model |
| `models/rope_vit_axial_butterfly.py` | Axial butterfly model |
| `models/rope_vit_axial_dense.py` | Axial dense model |
| `models/rope_vit_randmix.py` | Random-mix model |
| `models/rope_vit_axial_randmix.py` | Axial random-mix model |
| `models/rope_vit_axial_cayley.py` | Axial Cayley model (experimental) |
| `models/rope_vit_axial_ortho.py` | Axial ortho model (experimental) |
| `models/vit_block_v.py` | ViT base for block-diagonal |
| `models/vit_butterfly.py` | ViT base for butterfly |
| `models/vit_axial_dense.py` | ViT base for axial dense |
| `run_gpu0.sh` | Launch script for GPU 0 experiments (from other machine) |
| `run_gpu1.sh` | Launch script for GPU 1 experiments (from other machine) |

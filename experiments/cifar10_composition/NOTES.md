# 2D Positional Encoding: Design Notes

## How RoPE works in 1D

Each D-dimensional embedding has D/2 rotation pairs: (dim0, dim1), (dim2, dim3), etc.
Each pair d gets a fixed frequency θ_d = 1/10000^(2d/D).
At position t, pair d is rotated by angle = t * θ_d.

Q and K are rotated. The dot product Q^T K encodes relative position through
R(t)^T R(s) = R(s-t) — the rotation cancels to a relative rotation.

D/2 frequencies, D/2 angles per position. V is not rotated.

## Extending to 2D: two approaches

### Approach 1: Split dimensions (standard RoPE2D)

Split the D/2 pairs into two groups:
- First D/4 pairs: rotated by pos_y * θ_d (y-axis only)
- Second D/4 pairs: rotated by pos_x * θ_d (x-axis only)

Each pair encodes exactly ONE axis. Position (2,3) and (3,2) are distinguishable
because the y-pairs see different y-positions and the x-pairs see different x-positions.

Limitation: each axis only gets D/4 frequency components instead of D/2.

### Approach 2: Combined angles (monoidal framework)

ALL D/2 pairs encode BOTH axes:
- Pair d gets angle = pos_y * freq_y[d] + pos_x * freq_x[d]

Each pair simultaneously encodes both spatial dimensions through the angle sum.
Every pair has D/2 frequency components per axis (the full set).

Critical requirement: freq_y and freq_x must be DIFFERENT. If freq_y = freq_x,
the angle becomes (pos_y + pos_x) * freq[d], and positions on the same
anti-diagonal (e.g., (0,3) and (3,0)) get identical angles everywhere.
The model literally cannot distinguish them.

For learned frequencies (monoidal/joformer): they start at the same initialization
but diverge during training, breaking the degeneracy.

For fixed frequencies (rope2dv2/joformer_fixed): they must be initialized differently.
We use different bases: freq_y = 1/10000^(2d/D), freq_x = 1/8000^(2d/D).

## Model hierarchy

```
                    Fixed freqs              Learned freqs
                    ───────────              ─────────────
Q/K only:           rope2dv2                 monoidal
Q/K/V + inverse:    joformer_fixed           joformer
```

Each row shares identical code. The only difference between columns is
nn.Parameter vs register_buffer for the frequencies.

Each column shares identical code. The only difference between rows is
whether V is rotated and output is inverse-rotated.

## JoFormer value rotation

Standard RoPE/monoidal only rotates Q and K — position affects attention SCORES
but not the VALUES being aggregated.

JoFormer also rotates V by the same position-dependent angles, then inverse-rotates
the attention output:

```
out(i,j) = R(i,j)^{-1} * sum_(k,l) attn(i,j→k,l) * R(k,l) * v(k,l)
         = sum_(k,l) attn(...) * R(k-i, l-j) * v(k,l)
```

This transforms each value by the RELATIVE position before summing — a position-dependent
content transformation, similar to S4ND's global convolution kernel. The self-contribution
is always identity: R(i,j)^{-1} R(i,j) = I.

## Anti-diagonal degeneracy bug

When freq_y = freq_x, the combined angle is (pos_y + pos_x) * freq[d].
All positions on the same anti-diagonal (pos_y + pos_x = constant) get
identical rotation angles at every dimension pair. The model cannot
distinguish (0,5) from (5,0) from (2,3) from (3,2).

This bug affected:
- rope2dv2 (first version): 44.49% accuracy (vs 47.94% for split-dimension rope2d)
- joformer_fixed (first version): used outer(pos, dim_idx) with same dim_idx for both axes

Fix: use different frequency vectors for y and x axes.

## Literature: Axial RoPE vs RoPE-Mixed

### Axial RoPE (standard)

The standard 2D extension of RoPE, used in most Vision Transformer implementations.
Splits the embedding dimensions between axes: first half encodes one axis, second
half encodes the other. No cross-axis interaction within any rotation pair.

This is equivalent to our "split dimensions" approach (Approach 1 above).

References:
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
  Original 1D RoPE paper. https://arxiv.org/abs/2104.09864
- Heo et al., "Rotary Position Embedding for Vision Transformer" (ECCV 2024)
  Systematic study of RoPE in ViTs. https://arxiv.org/abs/2403.13298

### RoPE-Mixed (combined/learnable)

Proposed in the ECCV 2024 paper above. Instead of dedicating each dimension pair
to a single axis, RoPE-Mixed uses learnable frequencies that combine both axes
in every rotation pair. This is equivalent to our "combined angles" approach
(Approach 2 / monoidal framework).

Key result from the paper:
- RoPE-Mixed consistently outperforms Axial RoPE
- ViT-B at 224×224: Axial 83.6% → Mixed 83.8%
- ViT-B at 512×512 (extrapolation): Axial 82.0% → Mixed 82.9%
- The advantage grows at extrapolated resolutions

The paper explains: "axial frequencies cannot handle diagonal directions" while
mixed frequencies enable "diagonal direction handling." The mixed approach is
described as "a generalized version of axial frequency RoPE."

Reference:
- Heo et al., "Rotary Position Embedding for Vision Transformer" (ECCV 2024)
  https://arxiv.org/abs/2403.13298

### Spiral RoPE

An alternative 2D extension that rotates position embeddings in the 2D plane
using spiral patterns, designed to capture both radial and angular spatial
relationships.

Reference:
- "Spiral RoPE: Rotate Your Rotary Positional Embeddings in the 2D Plane" (2025)
  https://arxiv.org/abs/2602.03227

### N-dimensional RoPE

Generalization of RoPE to arbitrary dimensions, relevant to our framework's
claim of generalizing to N-dimensional compositional embeddings.

Reference:
- "On N-dimensional Rotary Positional Embeddings"
  https://jerryxio.ng/posts/nd-rope/

### Connection to our framework

Our monoidal framework naturally produces the RoPE-Mixed structure: the
composition along each spatial axis adds angles, giving
angle_d = pos_y * freq_y[d] + pos_x * freq_x[d] for all D/2 pairs.

Key differences from RoPE-Mixed:
1. Our framework derives this from algebraic first principles (semidirect product),
   not as an ad-hoc modification of axial RoPE.
2. Our framework extends beyond positional encoding to also transform values
   (JoFormer), connecting to S4ND-style global convolution.
3. The algebraic structure guarantees associativity and the interchange law
   across dimensions, enabling principled N-dimensional extension.

### Connection to S4ND

S4ND (Nguyen et al., NeurIPS 2022) extends structured state space models to
multiple dimensions by factoring the ND convolution kernel into separable
1D convolutions — one per axis. Our JoFormer's value rotation achieves a
similar effect: the relative rotation R(k-i, l-j) applied to values creates
a position-dependent content transformation that functions like a structured
global convolution kernel.

Key difference: S4ND replaces attention entirely with convolution. Our framework
keeps attention (for data-dependent routing) while adding structured content
transformation through the algebraic composition.

Reference:
- Nguyen et al., "S4ND: Modeling Images and Videos as Multidimensional Signals
  Using State Spaces" (NeurIPS 2022) https://arxiv.org/abs/2210.06583

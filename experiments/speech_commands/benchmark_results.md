# Cumsum vs Scan vs FFT: Benchmark Results

## Environment

```
GPU:       Tesla T4 (15 GB)
Driver:    580.159.03
CUDA:      12.4
PyTorch:   2.6.0+cu124
Torchaudio:2.6.0+cu124
Triton:    3.2.0
NumPy:     2.3.5
Python:    3.12
```

<details>
<summary>Full pip freeze</summary>

```
cffi==2.1.0
filelock==3.20.0
fsspec==2025.12.0
Jinja2==3.1.6
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.6.1
numpy==2.3.5
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
pycparser==3.0
setuptools==83.0.0
soundfile==0.14.0
sympy==1.13.1
torch==2.6.0+cu124
torchaudio==2.6.0+cu124
tqdm==4.66.5
triton==3.2.0
typing_extensions==4.15.0
```

</details>

## Overview

We benchmark three approaches to sequence processing in a speech-command classification model (12 classes, 16 kHz, 1-second clips):

1. **Cumsum + window** (FIR): `d[t] = Σ x[t-W..t]` — implemented via `torch.cumsum` + shifted subtraction
2. **Scan + decay** (IIR): `d[t] = λ·d[t-1] + x[t]` — implemented via custom Triton sequential scan
3. **FFT/mel spectrogram**: standard STFT front-end via `torchaudio.transforms.MelSpectrogram` (cuFFT)

Two questions:
- **Cumsum vs scan**: does learned exponential decay beat a fixed hard window?
- **Cumsum vs FFT**: can a cumsum-based front-end replace the mel spectrogram entirely?

---

## Part 1: Cumsum vs Scan (Apples-to-Apples)

Both models share an identical mel-spectrogram front-end and architecture — the *only* difference is the sequence operation in the processing layers:

```
Input: 16 kHz waveform (16000 samples)
  → Mel spectrogram (n_mels=40, hop=80) → (B, 200, 40)
  → TransposedBN + GLU
  → 8 tied cumsum/scan layers:
      Linear(80→80) → complex split → rotate → SEQUENCE_OP → unrotate → concat re/im
      → TransposedBN + GLU (residual)
  → MLP readout → maxpool → Linear → 12 classes
```

| | MelCumsumFixed | MelScanFixed |
|---|---|---|
| Sequence op | `torch.cumsum` + window subtract | Triton sequential scan |
| Window/decay | Fixed W=10 | Learned λ per frequency per layer |
| Parameters | 24,812 | 25,132 (+320 decay scalars) |

The +320 parameter difference comes from 8 layers × 40 learned decay scalars in the scan model.

### Training (40 epochs, batch=128, lr=1e-3, cosine schedule)

| Metric | MelCumsumFixed | MelScanFixed |
|--------|---------------|-------------|
| **Test accuracy** | **94.82%** | 94.33% |
| Best val accuracy | 94.8% | 94.2% |
| Wall time | 648s | 694s |
| Per epoch | 16.2s | 17.4s |
| Training speedup | **1.07x** | 1.00x |

### Convergence Curves

```
Epoch   Cumsum Val    Scan Val
  1      42.5%        48.6%      scan starts slightly ahead
  5      84.8%        82.9%      cumsum pulls ahead
 10      90.3%        88.4%
 15      91.9%        90.9%
 20      92.2%        91.7%
 25      94.0%        93.2%
 30      94.4%        93.8%
 35      94.5%        94.2%
 40      94.4%        94.2%      cumsum best: 94.8% (ep32), scan best: 94.2% (ep25)
```

### Inference Latency (single sample, median of 200 runs)

| | MelCumsumFixed | MelScanFixed |
|---|---|---|
| Latency | **5.01 ms** | 7.09 ms |
| Throughput | **200 seq/s** | 141 seq/s |
| **Speedup** | **1.42x** | 1.00x |

### Primitive Operation Comparison

Raw operation timing at the mel-layer size (T=200 frames, N=40 frequencies):

| Batch | T | N | `torch.cumsum` | Triton scan | Ratio |
|------:|----:|----:|---------------:|------------:|------:|
| 1 | 200 | 40 | 84.9 μs | 270.6 μs | 3.2x |
| 8 | 200 | 40 | 84.9 μs | 273.6 μs | 3.2x |
| 32 | 200 | 40 | 73.4 μs | 331.8 μs | 4.5x |
| 128 | 200 | 40 | 140.5 μs | 514.5 μs | 3.7x |

At T=200 (post-mel frame count), `torch.cumsum` is 3-5x faster than the Triton scan.

### Learned Decay Analysis

After training, the scan model's learned decay values (λ = sigmoid(param)):

| Layer | Mean λ | Range | Effective Window (1/(1-λ)) |
|-------|--------|-------|-----------------|
| 0 | 0.789 | [0.709, 0.947] | [3, 19] |
| 1 | 0.768 | [0.683, 0.946] | [3, 18] |
| 2 | 0.755 | [0.645, 0.950] | [3, 20] |
| 3 | 0.749 | [0.629, 0.953] | [3, 21] |
| 4 | 0.745 | [0.622, 0.951] | [3, 20] |
| 5 | 0.744 | [0.624, 0.946] | [3, 18] |
| 6 | 0.746 | [0.641, 0.941] | [3, 17] |
| 7 | 0.751 | [0.669, 0.936] | [3, 16] |

The scan converges to short effective windows (3-21 frames), comparable to cumsum's fixed W=10. The model does not learn to exploit long-range dependencies — it approximates a short FIR filter, which is exactly what cumsum provides directly.

### Part 1 Summary

| | Cumsum | Scan |
|---|---|---|
| Accuracy | **94.82%** | 94.33% |
| Training speed | **1.07x faster** | baseline |
| Inference speed | **1.42x faster** | baseline |
| Primitive speed | **3-5x faster** | baseline |
| Implementation | `torch.cumsum` (built-in) | Custom Triton kernel |
| Learned window | Fixed W=10 | Converges to W≈3-21 |

The scan's theoretical advantage — learning per-frequency decay rates — doesn't translate to accuracy gains. Cumsum is faster at every level and requires no custom kernels.

---

## Part 2: CumsumE2E — Replacing the Mel Spectrogram

CumsumEndToEnd replaces the mel spectrogram entirely. Layer 1 applies learned rotational frequencies directly to the raw 16 kHz waveform, computing windowed cumsum at full sample resolution before striding down to 100 frames.

### Architecture

```
Input: 16 kHz waveform (16000 samples)
  → Layer 1: rotate by learned freqs → cumsum(16000) → window subtract → unrotate → stride → (B, 100, 80)
  → Layers 2+: Linear → complex split → rotate → cumsum(100) → window → unrotate → BN + GLU (residual)
  → MLP readout → maxpool → Linear → 12 classes
```

Parameters: 82,292 (4 untied layers, n_freqs=40, window_l1=400, stride=160)

### The `torch.cumsum` Performance Trap

Initial profiling of CumsumE2E layer 1 showed it was **19x slower** than the mel front-end (5,634 μs vs 298 μs). Step-by-step breakdown:

| Step | Time |
|------|-----:|
| `phases = exp(1j * t * f)` | 119 μs |
| `rotated = x * phases.conj()` | 127 μs |
| **`cs = rotated.cumsum(dim=1)`** | **5,384 μs** |
| window subtract + unrotate (100 positions) | 131 μs |
| **Total** | **5,634 μs** |
| Mel spectrogram (for comparison) | 298 μs |

The cumsum was 95% of the total time. This made no sense — cumsum is O(T·N) = O(16000·40) = 640K additions. The mel FFT does comparable work. What was going on?

**The problem: `torch.cumsum` on a non-contiguous dimension is pathologically slow.**

The tensor layout was (B, T, N) = (1, 16000, 40) with cumsum along dim=1 (T). In row-major memory, elements along dim=1 are strided (stride=40), not contiguous. This apparently prevents PyTorch from using its parallel prefix-sum kernel, falling back to a sequential scan:

| Operation | Time |
|-----------|-----:|
| `cumsum` on (1, 16000, 40) along **dim=1** (strided) | 5,388 μs |
| `cumsum` on (1, 40, 16000) along **dim=2** (contiguous) | 93 μs |
| `cumsum` on (16000,) 1D | 25 μs |

**58x slowdown** just from memory layout. The width (N) barely matters — it's the strided access pattern that kills performance:

| N (width) | dim=1 time (strided) |
|----------:|---------------------:|
| 1 | 25 μs |
| 10 | 2,519 μs |
| 40 | 5,387 μs |
| 80 | 5,398 μs |
| 640 | 5,989 μs |

Going from N=1 to N=10 causes a 100x slowdown even though the work only increases 10x. The kernel switches from a fast contiguous path to a slow strided path.

### Optimization 1: Stride-first Indexing

The original code computed window subtraction, unrotation, and real/imag concatenation at all 16,000 positions, then strided to 100. This materialized three (B, 16000, 40) complex intermediates unnecessarily.

The fix: index into the cumsum output at the 100 stride positions *before* the downstream ops:

```python
# Before: 3 tensors at (B, 16000, 40) then stride
cs_shifted = F.pad(cs[:, :-window], (0, 0, window, 0))  # (B, 16000, 40)
d = (cs - cs_shifted) * phases                            # (B, 16000, 40)
h = torch.cat([d.real, d.imag], dim=-1)                   # (B, 16000, 80)
h = h[:, ::stride]                                        # (B, 100, 80)

# After: index first, compute at (B, 100, 40)
out_idx = torch.arange(0, T, stride)
cs_out = cs[:, out_idx]                                    # (B, 100, 40)
cs_delayed = cs[:, (out_idx - window).clamp(min=0)] * mask # (B, 100, 40)
d = (cs_out - cs_delayed) * phases[out_idx]                # (B, 100, 40)
h = torch.cat([d.real, d.imag], dim=-1)                    # (B, 100, 80)
```

Memory savings (layer 1):

| Batch | Original | Optimized | Saved |
|------:|---------:|----------:|------:|
| 16 | 410 MB | 166 MB | 244 MB (59%) |
| 64 | 1,638 MB | 664 MB | 974 MB (59%) |
| 128 | 3,282 MB | 1,330 MB | 1,952 MB (59%) |

### Optimization 2: Contiguous Layout

The fix: compute in (B, N, T) layout from the start so cumsum runs along the contiguous last dimension. No transpose or copy needed — just build the tensors in the right shape:

```python
# (B, n_freqs, T) layout — cumsum along dim=2 is contiguous
phases_t = exp(1j * t * f)                               # (n_freqs, T)
x_complex = x.unsqueeze(1)                               # (B, 1, T)
rotated = x_complex * phases_t.conj().unsqueeze(0)       # (B, n_freqs, T) — contiguous
cs = rotated.cumsum(dim=2)                               # fast parallel prefix sum
# ... index, window subtract, transpose only the 100 output positions
```

### Combined Speedup (layer 1 only)

| Batch | Before both | After both | Speedup |
|------:|------------:|-----------:|--------:|
| 1 | 5,815 μs | 284 μs | **20x** |
| 128 | 56,023 μs | 12,000 μs | **4.7x** |

At B=1 the contiguous layout is the main win (5.4ms → ~300μs). At B=128 stride-first also matters since it avoids materializing large intermediates.

### Result: Front-end Parity with Mel

After both optimizations, the cumsum front-end matches the mel spectrogram at B=1 inference:

| | Cumsum front-end | Mel front-end (FFT+log+embed) |
|---|---|---|
| B=1 latency | **337 μs** | 359 μs |

Both produce ~100 frames of dim-80 features from a 16K-sample input. The remaining gap in the full model (3.30 ms vs 2.90 ms) comes from CumsumE2E's MLP readout vs MelCumsumFixed's simpler magnitude→fc — not the front-end.

Full model comparison (matched: 3 untied processing layers, dim=80, window=10):

| | CumsumE2E (1+3 layers) | MelCumsumFixed (mel+3 layers) |
|---|---|---|
| Parameters | 82,292 | 62,692 |
| Inference B=1 | 3.30 ms | 2.90 ms |
| Inference B=128 | 15.75 ms | 4.16 ms |
| Peak memory B=128 | 1,403 MB | 71 MB |

At B=128 the gap widens because the (128, 40, 16000) complex rotation tensor (655 MB) must be materialized for the cumsum. A fused Triton kernel that combines rotate + cumsum + window + stride in one pass could eliminate this.

---

## Theoretical Complexity: Cumsum vs FFT

### Per-hop cost at inference

At streaming inference, both front-ends process one hop (160 samples) at a time:

| | Cumsum front-end | Mel/FFT front-end |
|---|---|---|
| Operation per hop | Update 40 running complex accumulators with 160 new samples | One 400-point FFT + mel filterbank |
| FLOPs per hop | 40 × 160 = **6,400** multiply-adds | 400 × log₂(400) ≈ **3,400** butterfly ops |
| Op type | Complex multiply-add | FFT butterfly (twiddle factors) |
| Sequential depth | O(1) per sample | O(log N) = O(9) per FFT |

Both are tiny amounts of work per hop. The cumsum approach does ~2x more FLOPs but they are simpler operations (multiply-add vs butterfly). In practice both complete in microseconds.

### Total cost for one 1-second clip

| | Cumsum | Mel/FFT |
|---|---|---|
| Formula | T × N = 16,000 × 40 | (T/hop) × N_fft × log₂(N_fft) = 100 × 400 × 9 |
| Total FLOPs | 640,000 | 360,000 |

Comparable work — cumsum is ~1.8x more total FLOPs, but the actual wall-clock times are equal (~340 μs each) because `torch.cumsum` on a contiguous dimension is a single well-optimized CUDA kernel, while mel requires FFT + power + matrix multiply (mel filterbank) + log.

---

## Streaming Inference

For real-time / streaming deployment, the two front-ends have different state requirements:

| | Cumsum front-end | Mel/FFT front-end |
|---|---|---|
| State | 40 complex running sums + ring buffer (W=400 samples) | 400-sample window buffer |
| Per-sample update | O(40): add new rotated sample, subtract expired one | Buffer sample; every 160 samples: FFT |
| Latency | 1 sample (no buffering needed) | 160 samples (must fill hop) |
| Implementation | Simple multiply-add loop | Requires FFT library |

The cumsum front-end has a fundamental streaming advantage: it can produce output after every single sample with O(40) work, while the FFT must buffer a full hop of 160 samples before producing output. For the processing layers (post-front-end), both cumsum and scan have O(1) streaming updates:

| | Cumsum + Window | Scan + Decay |
|---|---|---|
| State per layer | Ring buffer of W values | 1 hidden state vector |
| State size | W × N complex | N complex |
| Update | O(1) amortized | O(1) |

---

## Key Takeaways

1. **Cumsum matches scan accuracy** (94.8% vs 94.3%) while being faster at every level — no benefit to learned exponential decay on this task.

2. **`torch.cumsum` on non-contiguous dimensions is a performance trap.** Strided cumsum on (1, 16000, 40) along dim=1 is **58x slower** than along the contiguous dimension. Always ensure cumsum runs along the innermost (contiguous) dimension.

3. **After optimization, the cumsum front-end matches mel speed at inference** (337 μs vs 359 μs at B=1). The two optimizations — stride-first indexing and contiguous memory layout — together give a **20x speedup** at B=1.

4. **At batch=128 training, memory dominates.** The (128, 40, 16000) complex rotation tensor is 655 MB. A fused Triton kernel could eliminate this by computing rotate + cumsum + window + stride without materializing the full-resolution intermediate.

5. **Cumsum has a streaming advantage**: O(40) work per sample with 1-sample latency, vs FFT's O(400 log 400) work per hop with 160-sample latency. No FFT library required.

6. **`torch.cumsum` is a universal primitive** available on every backend (CUDA, CPU, MPS, XLA) with no custom kernels needed. The scan requires a custom Triton kernel that only works on CUDA.

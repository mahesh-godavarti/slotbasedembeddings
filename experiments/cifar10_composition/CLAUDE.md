# CLAUDE.md — Instructions for Claude on Thunder Compute

**Read this entire file before doing anything.**

## Working Style

- **Ask before acting.** If something is unclear, ask.
- **Stick with what works.** Do not refactor working code.
- **No unsolicited changes.** Do not add improvements beyond what is asked.
- **Report problems immediately.** Do not silently retry with a different approach.
- **Run long-running tasks in background** using `run_in_background`.
- **Do NOT run multiple GPU jobs in parallel** — one GPU at a time.
- **Do NOT use cosine LR schedule** unless explicitly asked. Use fixed LR (default).
- **Always use --seed 42** for reproducibility.

## What This Project Is

Empirical validation of the directional non-commutative monoidal framework for the paper "Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning" (submitted to NeurIPS, rejected for lack of experiments).

The framework uses the semidirect product composition `(a, A).(b, B) = (a + Ab, AB)` where A, B are block-diagonal 2x2 rotation matrices. This subsumes RoPE, self-attention, and SSM recurrence.

## This Machine

- **2× H100 PCIe (80GB each)** — use `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` to select GPU
- Use existing venv at `/home/ubuntu/exp8/venv/` if present, otherwise create new one with `torch`, `torchvision`, `numpy`, `tqdm`, `tokenizers`

## Key Files

- `vit_cifar10.py` — ViT with 9 positional encoding variants for CIFAR-10/100
- `run_all.py` — Run all models sequentially with data loaded once
- `results.md` — Full experiment results and analysis
- `NOTES.md` — Design notes on 2D PE with literature references
- `convergence_100ep.md` — Convergence data from earlier runs

## Model Hierarchy

```
                        Fixed freqs              Learned freqs
                        ───────────              ─────────────
Axial, Q/K only:        rope2d                   monoidal_axial
Axial, Q/K/V+inv:       joformer_old             joformer_axial
Combined, Q/K only:     rope2dv2                 monoidal
Combined, Q/K/V+inv:    joformer_fixed           joformer
Baseline:               learned (additive PE)
```

**Axial** = split dimensions (first D/4 pairs = y, second D/4 = x), per-head frequencies
**Combined** = all D/2 pairs encode both axes (angle = pos_y * freq_y + pos_x * freq_x)
**JoFormer** = rotates V in addition to Q/K, inverse-rotates the output

## Critical Implementation Details

### K@Q^T attention ordering
JoFormer uses `k @ q.transpose(-1,-2)` (NOT `q @ k.transpose(-1,-2)`). This ensures the journey operator in the attention scores matches the V rotation/inversion. Without V rotation it doesn't matter, but we use K@Q^T everywhere for consistency.

### Deterministic training
- `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` at startup
- Entire dataset loaded to GPU at startup (~184MB for CIFAR-100)
- Batches via `torch.randint` — no DataLoader, no workers
- Augmentation via torch ops on GPU (random flip + random crop)
- Fixed LR (no cosine schedule) so trajectories match regardless of total epochs
- Verified: running 50 epochs produces identical numbers as first 50 of 100-epoch run
- `builtins.print` overridden with `flush=True` for real-time output

### Frequency degeneracy in combined approach
For combined angles `pos_y * freq_y + pos_x * freq_x`, freq_y and freq_x MUST be different. If equal, positions on the same anti-diagonal are indistinguishable. Current code uses `freq_x = -freq_y`.

## Latest Results (CIFAR-100, D=32, 4 layers, 4 heads, lr=5e-4, fixed LR, seed=42)

### 100-epoch results
| Model | Best Test Acc |
|-------|-------------|
| rope2d | 46.17% |
| joformer_old | 46.79% (+0.62%) |

### 1000-epoch results (from AWS A10G, still running)
| Model | 1000 ep |
|-------|---------|
| learned | 50.85% |
| rope2d | 51.53% |
| joformer_old | 51.15% |
| monoidal_axial | running |

### Key findings so far
1. V rotation (JoFormer) helps at 100 epochs (+0.62%) but advantage shrinks at 1000 epochs
2. Learned frequencies beat fixed (monoidal_axial > rope2d at 100 ep)
3. Axial beats combined at D=32 (~47% vs ~45%)
4. Fixed LR gives reproducible, epoch-independent trajectories
5. Cosine LR schedule confounds comparisons (different total epochs = different LR curves)

## What To Do Next

### Priority 1: Reproduce RoPE2D paper results on ImageNet (why we're on this machine)
- The RoPE2D paper (ECCV 2024, https://arxiv.org/abs/2403.13298) showed:
  - RoPE-Axial beats learned PE on ViT-B at ImageNet scale
  - RoPE-Mixed (= our combined/monoidal approach) beats RoPE-Axial
  - Gap widens at resolution extrapolation (512×512)
- We need to REPRODUCE these results first, then ADD our JoFormer (V rotation) on top
- Steps:
  1. Download ImageNet-1K (use torchvision or kaggle)
  2. Adapt `vit_cifar10.py` for ImageNet (224×224, 1000 classes, patch_size=16)
  3. Train ViT-S (22M params) with: learned, rope2d (axial), rope2dv2 (mixed/combined)
  4. Verify rope2d > learned and rope2dv2 > rope2d (reproducing the paper)
  5. Then add joformer variants and show V rotation helps further
- Use DeiT-III training recipe if possible (400 epochs) or simplified version
- This is for the NeurIPS resubmission — reviewers want ImageNet-scale results

### Priority 2: Language modeling at scale
- Code in `/home/ubuntu/look_ahead6/` — `train_wiki_streaming.py`, `blocks.py`, `models.py`
- Download OpenWebText and preprocess
- Train GPT-2 Small/Medium scale models comparing RoFormer vs JoFormer
- The framework imports blocks from `/home/ubuntu/joformer/`

### Priority 3: Complete CIFAR-100 1000-epoch runs
- If not already done on AWS, run all 9 models at 1000 epochs
- Use `python vit_cifar10.py --dataset cifar100 --pe_type <type> --embed_dim 32 --n_layers 4 --n_heads 4 --epochs 1000 --seed 42 --lr 5e-4`

## Running Experiments

```bash
# Single model
python vit_cifar10.py --dataset cifar100 --pe_type rope2d --embed_dim 32 --n_layers 4 --n_heads 4 --epochs 100 --seed 42 --lr 5e-4

# All models (loads data once)
python run_all.py --dataset cifar100 --models rope2d joformer_old monoidal_axial joformer_axial --epochs 1000 --seed 42 --lr 5e-4

# Available pe_types:
# learned, rope2d, joformer_old, monoidal_axial, joformer_axial,
# rope2dv2, monoidal, joformer, joformer_fixed
```

## References

- Paper: https://arxiv.org/abs/2506.03472 (companion math paper)
- NeurIPS review: ~/nips_review.txt
- MNIST code: https://github.com/mahesh-godavarti/directional_composition_mnist
- RoPE-Mixed (ECCV 2024): https://arxiv.org/abs/2403.13298
- S4ND (NeurIPS 2022): https://arxiv.org/abs/2210.06583

# Session Log: ImageNet Reproduction + Look-Ahead D=8 Concat

## Goal
1. Reproduce RoPE2D paper (ECCV 2024) results on ImageNet-1K with ViT-B
2. Run look-ahead D=8 concat C=768 experiment on OWT

## Machine
2× NVIDIA H100 PCIe (80GB each), 295GB disk, PyTorch 2.10, torchvision 0.25

---

## Issue 1: ImageNet Download — Hugging Face Authentication

**Problem**: `load_dataset("ILSVRC/imagenet-1k")` returned 403 Forbidden.

**Cause**: The HF token was a fine-grained token without "access public gated repos" permission. ImageNet-1K is a gated dataset requiring explicit permission on the token.

**Fix**: User edited the token at huggingface.co/settings/tokens to enable "Access public gated repos".

## Issue 2: ImageNet Download — Disk Space

**Problem**: Download failed with `OSError: [Errno 28] No space left on device`.

**Cause**: `load_dataset()` (non-streaming) downloads all 294 parquet files to `~/.cache/huggingface/hub/` (~156GB), then processes them into Arrow format, needing ~2× the raw data size. 295GB disk couldn't fit both the cache and the OWT data (36GB).

**Fix**:
1. Deleted the HF cache (`rm -rf ~/.cache/huggingface/hub/datasets--ILSVRC--imagenet-1k/`)
2. Rewrote `download_imagenet.py` to use `streaming=True` — images are fetched one at a time and saved directly to disk as JPEG files in ImageFolder format. No local cache needed.

## Issue 3: Look-Ahead D=8 — H100 Not Faster Than L40S

**Problem**: H100 achieved ~1.09s/it at batch=64, nearly identical to L40S at ~0.83s/it (1.2 it/s). Expected 2-3× speedup.

**Cause**: The look-ahead model runs 40 sequential layer applications (8 blocks × 5 iterations) in a Python for-loop. Each layer is a small operation (batch=64, seq=256, dim=768) that can't saturate the H100's massive compute. The bottleneck is kernel launch overhead and memory bandwidth, not raw FLOPS. Both GPUs finish each small kernel equally fast.

**Attempted fixes**:
- `torch.compile` (default mode): Reduced to ~0.72s/it (~15% faster). Helped but didn't close the gap.
- `torch.compile(mode="reduce-overhead")`: Uses CUDA graphs to eliminate kernel launch latency. Crashed with `RuntimeError` in cudagraph_trees — the model's dynamic control flow (k_min, random K selection) is incompatible with CUDA graphs.
- Batch=128 with compile: OOMed due to torch.compile's extra memory overhead + leaked GPU memory from prior crashes.

**Final state**: Running with batch=64, default `torch.compile`, ~0.72s/it. ~15% faster than L40S per iteration. The iterative architecture fundamentally limits GPU utilization on H100.

## Issue 4: torch.compile — libcuda.so Not Found

**Problem**: `torch.compile` crashed with `AssertionError: libcuda.so cannot found!`

**Cause**: Triton (used by torch.compile's inductor backend) couldn't find `libcuda.so.1` because the linker cache was stale.

**Fix**: `sudo ldconfig` refreshed the linker cache. `libcuda.so.1` was at `/usr/lib/x86_64-linux-gnu/`.

## Issue 5: ViT ImageNet — OOM at batch=256

**Problem**: Both ViT-B runs OOMed at batch=256 on H100 (80GB).

**Cause**: Two processes ended up on the same GPU. `CUDA_VISIBLE_DEVICES` environment variable was not being inherited properly through the shell execution pipeline. Both processes defaulted to GPU 0.

**Fix**: Used `--gpu 0` and `--gpu 1` flags (which use `torch.device(f"cuda:{args.gpu}")`) instead of relying on `CUDA_VISIBLE_DEVICES`. Reduced batch to 128, then increased to 512 once confirmed working.

## Issue 6: Leaked GPU Memory After Process Kill

**Problem**: After killing training processes, GPU memory remained allocated (16-55GB) even though no Python processes were running. New runs would OOM.

**Cause**: DataLoader worker processes (num_workers=8, persistent_workers=True) survived the parent kill and held CUDA contexts. `nvidia-smi` showed memory in use but "No running processes found" because the workers had already exited but their CUDA allocations weren't freed.

**Fix**: Multiple approaches needed depending on severity:
1. `sudo nvidia-smi -pm 0` (toggle persistence mode) — worked sometimes
2. `python -c "import torch; torch.cuda.empty_cache()"` — worked when the leaked memory was from the same driver context
3. `sudo nvidia-smi --gpu-reset` — failed in this container ("could not determine primary GPU")
4. `modprobe -r nvidia_uvm` — unavailable in this container

The most reliable fix was launching a quick Python process that imports torch and calls `empty_cache()`.

## Issue 7: Accidental Kill of ImageNet Download

**Problem**: ImageNet download process was killed along with training processes.

**Cause**: Used blanket `ps aux | grep python | xargs kill` which killed ALL Python processes, including the unrelated download script.

**Fix**: Target specific process names (e.g., `grep train_wiki` or `grep vit_imagenet`) instead of all Python processes.

---

## Code Changes

### vit_cifar10.py — RoPE Frequency Base
Changed fixed frequency base from 10000 to 100 to match the RoPE2D paper (ECCV 2024):
- `RoPE2D`: `100.0 ** (-t / (half // 2))` instead of `1/10000^(2t/half)`
- `RoPE2Dv2`: same change
- JoFormer variants inherit from these, get the fix automatically

**Why**: Base 10000 was designed for 1D sequences of length ~2048. For a 14×14 spatial grid, the lowest frequency (0.0001) gives negligible rotation. Base 100 gives lowest frequency ~0.013, which still produces meaningful rotations at grid position 14.

### vit_cifar10.py — Per-Head Per-Layer Learnable Frequencies
`MonoidalAxialPE` and `MonoidalPE` now have per-head frequencies matching the paper:
- Frequencies shape: `(n_heads, n_freq)` instead of `(n_freq,)`
- For ViT-B: 9,984 PE params (0.012% of total) — matches paper's "~0.01%"
- Each transformer layer gets its own PE instance (per-layer)
- `JoFormerAxialPE` and `JoFormerPE` inherit these changes

### train_wiki_streaming.py — torch.compile Support
Added `--compile` flag for `torch.compile` acceleration. Default mode only — `reduce-overhead` mode crashes with the iterative model's dynamic control flow.

### vit_imagenet.py — New File
ImageNet training script adapted from `vit_cifar10.py`:
- Imports all PE modules from vit_cifar10.py (no duplication)
- Standard ImageNet augmentation (RandomResizedCrop, RandomHorizontalFlip)
- DataLoader with num_workers=8, pin_memory, persistent_workers
- Cosine LR with linear warmup (5 epochs)
- AMP (mixed precision) for H100
- Top-1 and Top-5 accuracy tracking
- Checkpointing best model per PE type

### download_imagenet.py — Streaming Download
Downloads ImageNet-1K via HF streaming API, saves as ImageFolder format. Supports resuming (skips existing files).

---

## Issue 8: ViT-B Too Slow on Single GPU

**Problem**: ViT-B (86M params) at batch=512 took ~60 min/epoch × 300 epochs = ~12.5 days per model on one H100.

**Cause**: ViT-B is a large model. The RoPE2D paper likely used 8× A100s with DDP and batch=1024. Single-GPU training at ImageNet scale is fundamentally slow for ViT-B.

**Fix**: Switched to ViT-S (D=384, 6 layers, 6 heads, ~22M params) — ~4× less compute per step. At batch=1024, ~24 min/epoch, 300 epochs ≈ 5 days. The paper tested ViT-S as well, so results are still comparable.

---

## Current State (2026-03-21)

### ImageNet ViT-S Training — Simplified Recipe (both interrupted at ~epoch 73-79)
- **GPU 0**: `learned` PE, ViT-S (D=384, 6 layers, 6 heads, ~22M params)
- **GPU 1**: `rope2d` PE, ViT-S (same architecture)
- 300 epochs, 1251 steps/epoch, eval every 10 epochs, ~24 min/epoch
- Simplified DeiT recipe: AdamW lr=1e-3, cosine decay, 5-epoch warmup, no mixup/cutmix

### Results — Simplified Recipe (through epoch 70 of 300)

| Epoch | Learned Top-1 | Rope2d Top-1 | Gap |
|-------|--------------|-------------|-----|
| 1     | 9.06%        | 9.56%       | +0.50% |
| 10    | 48.42%       | 52.48%      | +4.06% |
| 20    | 55.35%       | 58.88%      | +3.53% |
| 30    | 57.36%       | 60.74%      | +3.38% |
| 40    | 58.48%       | 61.58%      | +3.10% |
| 50    | 59.29%       | 61.59%      | +2.30% |
| 60    | 59.89%       | 61.96%      | +2.07% |
| 70    | 60.37%       | 62.78%      | +2.41% |

**Top-5 accuracy at epoch 70**: Learned 82.58%, Rope2d 84.11% (+1.53%).

**Key observations**:
1. **Rope2d consistently ahead** — direction matches the paper (rope2d > learned PE).
2. **Gap narrowing over training**: +4.06% at epoch 10 → +2.07% at epoch 60. Learned PE catches up as it learns positional relationships that RoPE provides structurally from the start.
3. **Rope2d reached learned's epoch-40 accuracy (58.48%) at epoch 20** — 2× faster to reach the same performance.
4. **Rope2d val accuracy flattening**: only +0.38% from epoch 40→60, while learned gained +1.41%. Convergence curves suggest final gap will be ~1-1.5% at epoch 300.
5. **Absolute accuracy lower than paper** (~62% vs ~75% for ViT-S): expected since we use simplified augmentation (no Mixup, CutMix, RandAugment, stochastic depth, label smoothing, EMA). The relative comparison is what matters.
6. **Paper's ViT-B result was +0.2%** (83.4% → 83.6%). Our ViT-S gap is much larger, likely because (a) smaller models benefit more from RoPE's inductive bias, (b) we're still early in training, (c) no strong augmentation means less regularization to compensate for learned PE's lack of structure.

---

## Issue 9: ThunderCompute Preemption — Lost Training Progress

**Problem**: Machine was rebooted by ThunderCompute (preemption or maintenance) on 2026-03-23 at ~11:30 UTC. Two DeiT-III training runs were killed mid-epoch with no way to resume:
- `rope2d` on GPU 0: killed mid-epoch 43 of 300
- `joformer_old` on GPU 1: killed mid-epoch 38 of 300
- Earlier simplified-recipe runs (`learned` at epoch 79, `rope2d` at epoch 73) were also lost

**Cause**: `vit_imagenet.py` only saved "best" checkpoints (on accuracy improvement). It did not save periodic resumable checkpoints with optimizer/scheduler/scaler state. There was no auto-restart mechanism on reboot.

**Fix**: Three-part solution:

### 1. Resumable checkpoints in `vit_imagenet.py`
- New `--resume` flag
- Saves `latest_<pe_type>.pt` every epoch with full state: model, optimizer, scheduler, scaler, best metrics, and all CLI args
- On `--resume`, loads the latest checkpoint and continues from the next epoch
- `best_<pe_type>.pt` (accuracy-only checkpoint) is still saved separately, unchanged
- Disk cost: one checkpoint file per model (~264MB for ViT-S 22M params), overwritten each epoch

### 2. `auto_resume.sh` script
- Scans `checkpoints/latest_*.pt` for incomplete runs (epoch < total epochs)
- Reads the saved `args` dict from each checkpoint to reconstruct the exact launch command
- Launches each run on its original GPU with `--resume`, logging to `logs/imagenet_<pe_type>_resumed.log`

### 3. Auto-start on reboot (two mechanisms)
- **Cron `@reboot`**: installed via `crontab`, runs `auto_resume.sh` on boot
- **`.profile` fallback**: runs `auto_resume.sh` on first SSH login after reboot (uses `/proc/sys/kernel/random/boot_id` marker in `/tmp` to run only once per boot)

---

## Current State (2026-03-23)

### DeiT-III Recipe Runs (interrupted by reboot, must restart from scratch)
- Config: ViT-S (D=384, **12 layers**, 6 heads, ~22M params), batch=1024, 300 epochs
- Full DeiT-III: RandAugment, Mixup (0.8), CutMix (1.0), Random Erasing (0.25), label smoothing (0.1), stochastic depth (0.1)
- **GPU 0 — rope2d**: reached epoch 43, best val top-1 = 68.11%, top-5 = 88.36%
- **GPU 1 — joformer_old**: reached epoch 38, best val top-1 = 66.00%, top-5 = 87.28%

### Results — DeiT-III Recipe (before interruption)

| Epoch | Rope2d Top-1 | Rope2d Top-5 | JoFormer_old Top-1 | JoFormer_old Top-5 |
|-------|-------------|-------------|--------------------|--------------------|
| 1     | 4.25%       | 12.79%      | 5.26%              | 14.69%             |
| 10    | 50.67%      | 75.57%      | 50.78%             | 75.56%             |
| 20    | 61.72%      | 84.13%      | 61.54%             | 84.05%             |
| 30    | 65.91%      | 86.89%      | 66.00%             | 87.28%             |
| 40    | 68.11%      | 88.36%      | —                  | —                  |

**Key observations (DeiT-III vs simplified recipe)**:
1. DeiT-III with 12 layers reaches ~68% at epoch 40 vs simplified 6-layer reaching ~62% at epoch 70 — much stronger.
2. Rope2d and joformer_old nearly identical through epoch 30 (66.00% vs 65.91%). V rotation doesn't help at this scale/stage.
3. No resume checkpoints exist — must restart from epoch 1, but now with auto-resume enabled.

### Look-Ahead D=8 Concat (stopped)
- Killed to free GPUs for ViT runs
- Was running at ~0.72s/it with torch.compile on H100
- Need to resume on another machine or after ViT runs complete

### ImageNet Data
- 1,281,167 train images, 50,000 val images, 1000 classes
- Saved as ImageFolder at `/home/ubuntu/cifar10_composition/data/imagenet/`

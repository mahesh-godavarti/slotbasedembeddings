# Pending Experiments — Handoff Notes (2026-03-27)

## What's running on this machine (instance-o36jty3g)

### GPU 0: D=23 K=5 (in progress)
- Log: `logs/corr_ffn_add_d23_c1024_h16_flash_owt.log`
- Checkpoint: `checkpoints_d23/`
- Settings: D=23, C=1024, h16, batch=32, K=5 (k_min=2), flash attention via PYTHONPATH override
- Uses `PYTHONPATH=/home/ubuntu/look_ahead6/flash_override:$PYTHONPATH`
- Iter counter inflated by 15K (first 30K iters were batch=16, then switched to batch=32 at iter 30K). Real token-equivalent = iter - 15K.
- ~7h left. Latest: PPL 29.84 @ 180K iter (165K-equiv).
- Near FLOP-matched to roformer N=24 (284C² vs 288C²). Consistently ~0.4 PPL ahead of roformer.

### GPU 1: Roformer N=12 (in progress)
- Log: `logs/roformer_n12_c1024_h16_owt.log`
- Checkpoint: `checkpoints_n12/`
- Settings: N=12, C=1024, h16, batch=32, 200K iters
- ~7h left. Latest: PPL 35.25 @ 145K.
- Baseline for D=12 comparison and N=12→D=12 fine-tune experiment.

## Completed experiments (checkpoints on this machine)

| Model | Checkpoint | Final PPL | Notes |
|-------|-----------|-----------|-------|
| Roformer N=24 | `checkpoints/roformer_latest.pt` | 29.42 | The main baseline |
| D=12 K=5 | `checkpoints/block_head_corr_ffn_add_latest.pt` | 32.28 | 47% fewer FLOPs than N=24 |
| D=24 fine-tuned from N=24 | `checkpoints_d24_converted/` | 28.99 | Converted N=24 + 18K iters K=2-4 fine-tune |
| D=23 K-schedule | `checkpoints_d23_ksched/` | 30.37 @ 185K | Killed at 185K (K=2-5 transition). Resume pending. |

## Pending experiments (in priority order)

### 1. N=12 → D=12 fine-tune (HIGHEST PRIORITY)
After roformer N=12 finishes (~7h):
1. Convert N=12 checkpoint to D=12 look-ahead using `convert_roformer_to_lookahead.py`:
   ```bash
   python convert_roformer_to_lookahead.py \
       --roformer_ckpt checkpoints_n12/roformer_latest.pt \
       --output_ckpt checkpoints_d12_converted/block_head_corr_ffn_add_latest.pt \
       --n_embed 1024 --n_layers 60 --d_block 12 --n_head 16 \
       --block_size 256 --vocab_size 32000
   ```
2. Fine-tune at K=2-5 random (same as D=24 fine-tune approach):
   ```bash
   PYTHONPATH=/home/ubuntu/look_ahead6/flash_override:$PYTHONPATH \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   python train_wiki_streaming.py train \
       --models block_head_corr_ffn_add --n_embed 1024 --n_layers 60 --block_size 256 --batch_size 32 \
       --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 12 --n_head 16 \
       --k_schedule "0:2-5" \
       --max_iters 50000 --eval_interval 2000 \
       --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
       --checkpoint_dir checkpoints_d12_converted \
       --gpu 1 --amp
   ```
3. **What to watch for**: N=12 should end around 34-35 PPL. D=12 trained from scratch got 32.28. The fine-tune should close that gap — any improvement over N=12 baseline validates the "convert and fine-tune" approach at this scale. The D=24 fine-tune improved by 0.43 PPL (29.42→28.99).
4. **When to stop**: Monitor eval every 2K iters. Stop when PPL plateaus (no improvement for ~6K iters).

### 2. Resume D=23 K-schedule
- Checkpoint at iter 185K in `checkpoints_d23_ksched/`
- Was entering K=2-5 phase when killed. Last PPL: 30.37.
- Resume command (use same K-schedule — it will pick up the right phase from checkpoint):
  ```bash
  PYTHONPATH=/home/ubuntu/look_ahead6/flash_override:$PYTHONPATH \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_wiki_streaming.py train \
      --models block_head_corr_ffn_add --n_embed 1024 --n_layers 115 --block_size 256 --batch_size 32 \
      --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 23 --n_head 16 \
      --k_schedule "0:1,150000:2,185000:2-5" \
      --max_iters 200000 --eval_interval 5000 \
      --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
      --checkpoint_dir checkpoints_d23_ksched \
      --gpu [0 or 1] --amp
  ```

### 3. D=12 C=1409 vs N=24 C=1024 (wider look-ahead, FLOP-matched)
- D=12 with C=1409: 152 × 1409² ≈ 288 × 1024² FLOPs — FLOP-matched to N=24 C=1024
- Tests the width vs depth tradeoff: fewer, wider layers with correction mechanism vs more, narrower layers
- At small C, wider look-ahead crushed deeper roformer (D=1 C=62 beat N=3 C=50 by 5.5 PPL)
- Not yet attempted at large scale

## Key files

| File | Purpose |
|------|---------|
| `train_wiki_streaming.py` | Training script |
| `models.py` | Model definitions |
| `blocks.py` | Base building blocks (DO NOT rename/move/swap this file) |
| `flash_override/blocks.py` | Flash attention override (use via PYTHONPATH, never swap files) |
| `convert_roformer_to_lookahead.py` | Convert roformer checkpoint to look-ahead |
| `check_progress.sh` | Check training progress: `bash check_progress.sh logfile.log` |
| `results_summary.md` | Master results document |
| `CLAUDE.md` | Architecture overview and CLI reference |

## Critical rules

1. **NEVER rename/move/symlink blocks.py** — use PYTHONPATH override for flash attention. See `flash_override/blocks.py`.
2. **Flash attention**: `PYTHONPATH=/home/ubuntu/look_ahead6/flash_override:$PYTHONPATH` before launching.
3. **All C=1024 runs use**: batch=32, lr=2e-4, softmax, amp, n_head=16.
4. **Data**: `/home/ubuntu/look_ahead/look_ahead/data_owt/` (must exist on the machine, ~34GB).
5. **Venv**: `/home/ubuntu/exp8/venv/` (needs torch, numpy, tqdm, tokenizers).
6. **Sequential K=1 is the only valid inference eval.** Never report parallel K=1.

## Data dependency
The training data is at `/home/ubuntu/look_ahead/look_ahead/data_owt/`. If it doesn't exist on the target machine, rsync it:
```bash
rsync -avz --progress -e "ssh -p 31239 -i /home/ubuntu/.thunder/keys/o36jty3g -o StrictHostKeyChecking=no" \
    ubuntu@185.216.20.240:/home/ubuntu/look_ahead/look_ahead/data_owt/ \
    /home/ubuntu/look_ahead/look_ahead/data_owt/
```

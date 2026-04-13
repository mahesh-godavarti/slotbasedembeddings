# Handoff: Resume Experiments on A100

## This machine

- **Instance**: `6g9fu64p` on ThunderCompute
- **GPU**: 1x NVIDIA A100 80GB PCIe
- **vCPUs**: 18
- **RAM**: 90GB
- **Disk**: 400GB
- **Mode**: Production (persistent)

The A100 is ~2-3x faster than the A6000 we were using. With BF16, expect ~4-5 it/s for the 163M param model (vs ~2 it/s on A6000).

The data and code need to be rsynced from the source machine (`lnbocddd`, A6000). If not already done:

```bash
# Run FROM the source machine (lnbocddd):
rsync -avz --progress -e "ssh -p 32265 -i ~/.thunder/keys/6g9fu64p -o StrictHostKeyChecking=no" \
    /home/ubuntu/pos_agnostic/ ubuntu@38.128.232.34:/home/ubuntu/pos_agnostic/ --exclude='__pycache__'
rsync -avz --progress -e "ssh -p 32265 -i ~/.thunder/keys/6g9fu64p -o StrictHostKeyChecking=no" \
    /home/ubuntu/look_ahead/look_ahead/data_owt/ ubuntu@38.128.232.34:/home/ubuntu/look_ahead/look_ahead/data_owt/
rsync -avz --progress -e "ssh -p 32265 -i ~/.thunder/keys/6g9fu64p -o StrictHostKeyChecking=no" \
    /home/ubuntu/exp8/venv/ ubuntu@38.128.232.34:/home/ubuntu/exp8/venv/
rsync -avz --progress -e "ssh -p 32265 -i ~/.thunder/keys/6g9fu64p -o StrictHostKeyChecking=no" \
    /home/ubuntu/look_ahead6/check_progress.sh ubuntu@38.128.232.34:/home/ubuntu/look_ahead6/check_progress.sh
```

## What this project is

Comparing attention mechanisms for length generalization in transformers. Key question: can JoFormer (data-dependent angles + cumsum + V rotation) beat standard RoPE?

## Code location

All code is in `/home/ubuntu/pos_agnostic/`:
- `models.py` — all attention types (rope, nope, joformer, datadep, etc.)
- `train.py` — training script
- `continue_training.py` — resume from checkpoint at new lr
- `eval_all.py` — evaluate checkpoints (200-iter clean eval, handles all model types)
- `RESULTS.md` — full experiment write-up with all results so far

## Data

- **OWT (OpenWebText)**: `/home/ubuntu/look_ahead/look_ahead/data_owt/` — 9.1B tokens, vocab=32K
- **Wiki**: `/home/ubuntu/look_ahead/look_ahead/data_full/` — 983M tokens, vocab=16K
- Both are preprocessed memmap files (wiki_tokens.bin, wiki_tokens.meta)

## Venv

`/home/ubuntu/exp8/venv/` — has torch, numpy, tqdm, tokenizers

## What needs to run

### Experiment: Scale-up comparison — RoPE vs JoFormer v2 (163M vs 193M params)

**RoPE is DONE** (100K total iters). Checkpoint at:
`/home/ubuntu/pos_agnostic/checkpoints/scale_up_full/rope_best.pt`

RoPE results (n_embed=768, n_layers=16, n_heads=8, block_size=512, unwindowed, OWT):
- Val PPL: 32.40
- Extrap: 512:31.51, 1024:43.76, 2048:88.86, 4096:154.82
- (Degrades at longer lengths — expected for unwindowed RoPE)

**JoFormer v2 needs to run** (100K iters from scratch):

```bash
nohup /home/ubuntu/exp8/venv/bin/python /home/ubuntu/pos_agnostic/train.py \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --models joformer2 \
    --n_embed 768 --n_layers 16 --n_heads 8 \
    --block_size 512 --window_size 999999 \
    --batch_size 32 --max_iters 100000 \
    --eval_interval 5000 --extrap_interval 25000 \
    --eval_lengths 512,1024,2048,4096 \
    --lr 5e-4 --bf16 \
    --checkpoint_dir /home/ubuntu/pos_agnostic/checkpoints/scale_up_full \
    > /home/ubuntu/pafl_scale_up_joformer2_bf16.log 2>&1 &
```

### After JoFormer v2 finishes (100K iters)

1. **Run 200-iteration clean eval on both checkpoints:**

```bash
/home/ubuntu/exp8/venv/bin/python /home/ubuntu/pos_agnostic/eval_all.py \
    --checkpoint_dir /home/ubuntu/pos_agnostic/checkpoints/scale_up_full/ \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --eval_iters 200 --eval_lengths 512,1024,2048,4096
```

Note: For the rope checkpoint, eval_all.py will reconstruct the model correctly from the saved config. For joformer2, it auto-detects the v2 embedding size (n_embed in checkpoint is C+C/2, eval_all.py computes correct content dim as n_embed*2//3).

2. **Continue both at lr=2e-4 for 50K more iters:**

```bash
# RoPE continuation
nohup /home/ubuntu/exp8/venv/bin/python /home/ubuntu/pos_agnostic/continue_training.py \
    --checkpoint /home/ubuntu/pos_agnostic/checkpoints/scale_up_full/rope_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 \
    --eval_interval 5000 --extrap_interval 25000 \
    --eval_lengths 512,1024,2048,4096 \
    --checkpoint_dir /home/ubuntu/pos_agnostic/checkpoints/scale_up_continue \
    --bf16 \
    > /home/ubuntu/pafl_scale_up_rope_continue.log 2>&1 &

# JoFormer v2 continuation
nohup /home/ubuntu/exp8/venv/bin/python /home/ubuntu/pos_agnostic/continue_training.py \
    --checkpoint /home/ubuntu/pos_agnostic/checkpoints/scale_up_full/joformer2_best.pt \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --lr 2e-4 --max_iters 50000 \
    --eval_interval 5000 --extrap_interval 25000 \
    --eval_lengths 512,1024,2048,4096 \
    --checkpoint_dir /home/ubuntu/pos_agnostic/checkpoints/scale_up_continue \
    --bf16 \
    > /home/ubuntu/pafl_scale_up_joformer2_continue.log 2>&1 &
```

3. **Final 200-iter clean eval on continued checkpoints:**

```bash
/home/ubuntu/exp8/venv/bin/python /home/ubuntu/pos_agnostic/eval_all.py \
    --checkpoint_dir /home/ubuntu/pos_agnostic/checkpoints/scale_up_continue/ \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --eval_iters 200 --eval_lengths 512,1024,2048,4096
```

## Monitoring progress

```bash
bash /home/ubuntu/look_ahead6/check_progress.sh /path/to/logfile.log
```

This handles tqdm's \r output correctly.

## Model types reference

| Name | Angles | Cumsum | Rotate V | Description |
|------|--------|--------|----------|-------------|
| `rope` | Fixed (position) | N/A | No | Standard RoPE |
| `joformer_fixed` | Fixed (position) | N/A | Yes | RoPE + V rotation |
| `nope` | None | N/A | No | No position encoding |
| `datadep` | Data-dependent | No | No | Content angles, Q/K only |
| `datadep3` | Data-dependent | No | No | Same but MLP angle_proj |
| `monoidal` | Data-dependent | Yes | No | + cumsum |
| `joformer` | Data-dependent | Yes | Yes | + V rotation (v1, angle_proj) |
| `joformer2` | Data-dependent | Yes | Yes | + V rotation (v2, angle flow) |

Hybrid configs: `hybrid_K` = (L-K) RoPE windowed + K NoPE full. Similarly `joformer_hybrid_K`, `joformer2_hybrid_K`, etc.

## Key results so far (small scale, OWT, 9.4-11.6M params)

After 100K + 50K continuation, 200-iter clean eval:

| Length | hybrid_1 (RoPE+NoPE) | joformer_fixed+NoPE | joformer_v1+NoPE | joformer_v2+NoPE |
|--------|---------------------|---------------------|------------------|------------------|
| 512 | 70.30 | 70.32 | 68.74 | **67.25** |
| 1024 | 65.79 | 66.39 | 65.77 | **64.13** |
| 2048 | 65.21 | 67.17 | 64.91 | **62.89** |
| 4096 | 67.22 | 68.45 | 66.07 | **65.91** |

JoFormer v2 wins by ~2 PPL over RoPE at small scale. The scale-up experiment tests if this advantage holds or grows at 163M/193M params.

## Known issues

- Thundercloud GPU drops frequently. Checkpoints save every eval_interval (5K iters).
- Checkpoint config saves attn_config as a list. `eval_all.py` and `continue_training.py` handle this via `detect_attn_config()` which maps lists back to string configs.
- For datadep2/joformer2 models, checkpoint saves n_embed as embedding dim (C+C/2), not content dim (C). `detect_n_embed()` handles this.
- Use `--bf16` flag for ~2x speedup on A100 (and A6000).
- Eval functions use fixed seeds (torch.manual_seed(42)) for reproducible results.

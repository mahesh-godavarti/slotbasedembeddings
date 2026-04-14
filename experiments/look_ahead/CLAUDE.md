# Look-Ahead Architecture — Project Handoff

## Quick Start

```bash
# Activate venv (create one if needed with torch, numpy, tqdm, tokenizers)
source /path/to/venv/bin/activate

# Run an experiment (example: D=8 C=768 on OWT)
cd ~/look_ahead6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_wiki_streaming.py train \
  --models block_head_corr_ffn_add --n_embed 768 --n_layers 40 --block_size 256 --batch_size 64 \
  --lr 2e-4 --softmax --convergence_weight 0.1 --k_min 0 --d_block 8 \
  --max_iters 100000 --eval_interval 5000 \
  --data_dir ~/look_ahead/look_ahead/data_owt \
  --amp 2>&1 | tee logs/my_experiment.log
```

## What This Project Is

A shared-weight transformer architecture that achieves competitive perplexity with **fewer inference FLOPs** than standard transformers (roformer).

**Core idea**: A D-block unit (D transformer layers with separate weights) is iterated K times during training. At inference, sequential autoregressive generation means K=1 is sufficient — each position naturally sees corrected representations from all previous positions. The model is a D-layer transformer at inference but was trained with K×D effective depth.

**Key result**: D=8 (104C² inference FLOPs) achieves 39.10 PPL on OWT, only 0.36 PPL behind roformer N=11 (132C²) — **21% fewer inference FLOPs**.

## Architecture Details

Read `ARCHITECTURE.md` for full mathematical description. Key points:

- **Non-cumulative corrections**: `processed_x = tok_emb + shift(correction)` — resets to tok_emb each iteration
- **Past-only shift**: position t gets correction from t-1 (causal)
- **corr_ffn_add** (best variant): `correction = corr_ffn(ln(shift(z) + tok_emb))`
- **FLOPs**: (12D + 8)C² per token at inference
- **Sequential K=1 = Parallel K=N** at convergence (proven, experimentally confirmed)

## Key Files

| File | Purpose |
|------|---------|
| `train_wiki_streaming.py` | Main training script (preprocessing + training) |
| `models.py` | All model definitions (block_head variants, roformer, etc.) |
| `results_summary.md` | **Master results document** — all experiments, training curves, analysis |
| `ARCHITECTURE.md` | Full architecture writeup with math |
| `check_progress.sh` | Check running experiment progress: `bash check_progress.sh logfile.log` |
| `blocks.py` | Block building blocks |
| `../joformer/train_wiki.py` | Imported for RoFormerBlock, RoFormerAttention classes |

## Data

- **OpenWebText (OWT)**: Preprocessed at `~/look_ahead/look_ahead/data_owt/` (~34GB). Vocab=32000.
- **Wiki**: Preprocessed at `~/look_ahead/look_ahead/data/` and `data_full/`. Vocab=16000.
- If data_owt doesn't exist, preprocess with:
  ```bash
  python train_wiki_streaming.py preprocess --data_dir ~/look_ahead/look_ahead/data_owt --vocab_size 32000
  ```
  (Requires raw OWT data)

## Important CLI Arguments

| Arg | Description |
|-----|-------------|
| `--models` | Model name(s): `block_head_corr_ffn_add`, `roformer`, etc. |
| `--n_embed` | Embedding dimension (C) |
| `--n_layers` | Total layers = D × K |
| `--d_block` | D (blocks per unit). Default 1. |
| `--block_size` | Context window. Must be >= K. |
| `--batch_size` | Batch size |
| `--lr` | Learning rate (use 2e-4 for C=768+) |
| `--softmax` | Use softmax attention (always use this) |
| `--convergence_weight` | Aux convergence loss weight (use 0.1) |
| `--k_min` | Random K training. 0=disabled. With k_min=2, samples K~Uniform(k_min, n_iters). |
| `--amp` | Mixed precision (bfloat16). Always use. |
| `--compile` | torch.compile. Only works with fixed K (--k_min 0). |
| `--k_schedule` | K curriculum: `"0:1,50000:2,90000:2-5"` |
| `--eval_interval` | Eval frequency (use 5000 for 100K runs) |
| `--data_dir` | Path to preprocessed data directory |

## Model Zoo

### Look-ahead variants (use `block_head_corr_ffn_add`)
- `block_head_corr_ffn_add`: **Best variant.** correction = corr_ffn(ln(shift(z) + tok_emb)). FLOPs = (12D+8)C².
- `block_head_corr_ffn_concat`: correction = corr_ffn(ln(concat(shift(z), tok_emb))). FLOPs = (12D+16)C². No benefit over add at D>=5.
- Other variants exist in models.py but are inferior.

### Baselines
- `roformer`: Standard RoPE transformer. FLOPs = 12N × C² where N = n_layers.

## Key Experimental Results

### Best head-to-head (all C=768, OWT, 100K iters)

| Model | FLOPs | Final PPL |
|-------|-------|-----------|
| D=8 add (K=5) | 104C² | 39.10 |
| Roformer N=11 | 132C² | 38.74 |
| Roformer N=12 | 144C² | 37.83 |

D=8 is only 0.36 PPL behind N=11 with 21% fewer inference FLOPs.

### D scaling vs N scaling (Wiki, C=446)
- Roformer gains 1.5× more PPL per additional layer than corr_ffn_add per additional D block
- FLOP-matched advantage shrinks: 1.03 → 0.97 → 0.89 PPL per step
- Sweet spot: D=3–8. Beyond D≈14, roformer catches up.

### Concat vs add at high D
- D=8 concat (112C²): 39.22 PPL
- D=8 add (104C²): 39.10 PPL
- **Concat provides no benefit at D>=5. Use add.**

### Training speedup experiments (D=8 C=768 OWT)

| Run | Final PPL | Seq K=1 | Wall time | Speedup |
|-----|-----------|---------|-----------|---------|
| K=5 baseline | 39.10 | 39.10 (0.00) | ~17.4h | 1.0× |
| K=2 compiled | 40.17 | 40.59 (+0.43) | 10.5h | 1.7× |
| K-schedule (1→2→2-5) | 41.63 | 41.62 (0.00) | 9.8h | 1.8× |

- K=2 costs only 1.07 PPL for 1.7× speedup, but has 0.43 seq K=1 penalty
- K-schedule restores zero seq K=1 penalty but 2.5 PPL behind (K=1 phase too long)
- K-schedule needs tuning: shorter K=1 phase, or start at K=2

## Things That Work

- `--amp` (bfloat16) — always use
- `--softmax` — always use
- `--convergence_weight 0.1` — helps convergence
- `--k_min 2` with K=5 — makes model robust to variable depth
- `--lr 2e-4` for C=768+
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents OOM fragmentation

## torch.compile Setup

If torch.compile fails with Triton/GCC errors:
```bash
sudo apt-get install -y python3.12-dev  # or whatever python version
sudo ldconfig
```

Only use `--compile` with `--k_min 0` (fixed K). Random K breaks the static graph.

## Running Experiments

### Long runs (hours)
Always use `nohup` or `tee` to capture output:
```bash
nohup bash run_script.sh > /dev/null 2>&1 &
# or
bash run_script.sh  # (already has tee in the script)
```

### Checking progress
```bash
bash check_progress.sh path/to/logfile.log
```
Or manually:
```bash
tail -c 500 logfile.log | tr '\r' '\n' | tail -3
```
(The `tr` is needed because tqdm uses `\r` not `\n`)

### GPU monitoring
```bash
nvidia-smi
```

## Critical Rules

1. **Sequential K=1 is the ONLY valid inference metric.** Never report parallel K=1.
2. **Look-ahead saves inference FLOPs, NOT training FLOPs.** Training is more expensive than roformer.
3. **block_size must be >= K** during training.
4. **Don't use concat at D>=5** — add is better (fewer FLOPs, same PPL).
5. **k_min=2 is important** for robustness when using K>=3 training.
6. **All results go in `results_summary.md`** — this is the master document.

## Next Steps / Ideas

1. **Better K-schedule**: The 50K K=1 phase was too long. Try `"0:2,80000:2-5"` (K=2 throughout, brief random phase at end) — should get close to K=5 quality with K=2 speed.

2. **Mid-FFN insertion**: Insert an extra FFN between blocks 4 and 5 in D=8. Hypothesis: breaks the shared-weight monotony, adds capacity cheaply.

3. **Multi-token prediction (MTP)**: Use correction vectors to predict t+1, t+2, t+3. Enables speculative decoding without a draft model.

4. **Scale up**: With H100s, can train larger models. GPT-2 XL scale (1.5B params) comparison would be very compelling.

5. **Multi-GPU**: With 2× H100, can either run two experiments in parallel (one per GPU) or use DDP for faster single-experiment training.

## Hardware Notes

- Previous machine: AWS L40S (46GB VRAM)
- D=8 C=768 batch=64 block_size=256 fits on L40S with AMP
- With H100 (80GB), can increase batch_size, C, or block_size significantly

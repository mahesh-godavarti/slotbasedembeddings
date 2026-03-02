# JoFormer — Wiki-Scale RoFormer vs JoFormer Comparison

## Overview

Compares RoFormer (standard RoPE) against three JoFormer variants on Wikipedia text with BPE tokenization. Pure language modeling (no KG). Measures val PPL to compare positional encoding strategies at scale.

## Models

| Model | Angle type | Description |
|-------|-----------|-------------|
| `roformer` | Standard RoPE | Baseline — fixed sinusoidal rotation angles |
| `joformer_fixed` | Fixed cumsum angles | Flip → cumsum → flip on fixed angles |
| `joformer_learned` | Learned cumsum angles | Per-layer learned angles, flip → cumsum → flip |
| `joformer_projected` | Projected angles | Per-layer MLP projects angles from residual stream |

All models use GELU activation, causal attention, and BPE subword tokenization.

## Data

Uses Wikipedia text from `../exp8/data/wiki.en.txt` (auto-detected relative to script). Train/val split: 90/10.

## Setup

```bash
cd ~/experiments
source venv/bin/activate
pip install torch numpy tokenizers tqdm
```

## Running Experiments

### Smoke test
```bash
python joformer/train_wiki.py --smoke
```

### Full runs
```bash
# All 4 models, default settings
python joformer/train_wiki.py --models roformer joformer_fixed joformer_learned joformer_projected \
  --max_iters 10000 --n_embed 128 --n_layers 4 --vocab_size 8000

# Large-scale comparison
python joformer/train_wiki.py --models roformer joformer_fixed joformer_learned joformer_projected \
  --max_iters 200000 --n_embed 500 --n_layers 2 --vocab_size 16000 --wiki_lines 1000000
```

### Long-running experiments (hours/days)

**CRITICAL**: The Bash tool has a 10-minute hard timeout. Long training runs MUST use `nohup`:
```bash
nohup python joformer/train_wiki.py --models roformer joformer_fixed joformer_learned joformer_projected \
  --max_iters 200000 --n_embed 500 --n_layers 2 --vocab_size 16000 \
  > joformer_run.log 2>&1 &
```

Then check progress with:
```bash
tail -c 500 joformer_run.log 2>/dev/null | tr '\r' '\n' | tail -1
```

**DO NOT** use `run_in_background=true` for runs longer than a few minutes — it will timeout and kill the process.

### Key CLI flags
- `--models roformer joformer_fixed ...` — which models to train
- `--n_embed N` — embedding dimension (default 128)
- `--n_layers N` — number of transformer layers (default 4)
- `--block_size N` — context window size (default 64)
- `--batch_size N` — batch size (default 32)
- `--lr FLOAT` — learning rate (default 5e-4)
- `--max_iters N` — training iterations (default 10000)
- `--vocab_size N` — BPE vocabulary size (default 8000)
- `--wiki_lines N` — how many wiki lines to load (default 100000)
- `--eval_interval N` — eval frequency in iterations (default 500)
- `--dropout FLOAT` — dropout rate (default 0.2)
- `--checkpoint_dir DIR` — checkpoint directory (default: `joformer/checkpoints`)
- `--smoke` — quick test (50 iters, 1000 lines, vocab 2000, 2 layers, n_embed 64)
- `--seed N` — random seed (default 42)
- `--generate_len N` — generation sample length (default 200)

## Results

Results are auto-saved as `joformer_results_YYYYMMDD_HHMMSS.json` and `joformer_results_latest.json`.

Each result file contains:
- `config` — hyperparameters
- `results` — per-model `val_loss`, `val_ppl`, and `ppl_curve` (iter/train_ppl/val_ppl arrays)
- `timestamp`

PPL curves allow comparing how fast each model's PPL drops during training.

### NEVER do this
- Never use `sed`/`grep`/`awk` to extract results from log files
- Never use `TaskOutput` to check background experiment progress — it dumps the entire buffer and can block indefinitely
- Never use `run_in_background=true` for training runs longer than a few minutes — the 10-minute timeout will kill the process

### Checking background experiment progress
Always use the Bash tool directly (NOT `run_in_background`, NOT `TaskOutput`) with this exact command:
```bash
tail -c 500 joformer_run.log 2>/dev/null | tr '\r' '\n' | tail -1
```

**Why this exact command**: tqdm progress bars use `\r` (carriage return) to overwrite the same line. The entire progress bar history is technically one enormous line with no `\n` newlines. If you try to Read the file or use `tail` without `tr '\r' '\n'`, you end up trying to process a single line that's hundreds of thousands of characters long — which causes the tool to hang or dump a massive buffer. The `tr '\r' '\n'` converts carriage returns to newlines so `tail -1` can grab just the last update.

**CRITICAL**: Do NOT use `run_in_background=true` for the tail check. Do NOT use `TaskOutput` after running tail. Just run the tail command directly with the Bash tool — it returns instantly.

## Key Rules

- **Always stay available for chat**: Never block on long-running processes. Use `nohup ... &` for training runs and remain responsive.
- **NEVER use TaskOutput to check on anything**: TaskOutput with `block=true` (the default) will hang indefinitely waiting for completion. This has caused Claude Code to get stuck multiple times. Just don't use it.
- **NEVER use `run_in_background=true` for experiment commands**: The 10-minute timeout will kill long-running processes. Always use `nohup ... > logfile 2>&1 &` instead.
- **When checking log files, use Bash tool directly**: Run `tail -c 500 logfile ...` as a regular Bash command (not background). It returns instantly.
- **Use Bash tool directly for experiments, NOT Task agents**: Task agents can't get user permission approval.
- **Format PPL consistently**: Use fixed decimal format (21.61, not "6.4M"). Keep 2 decimal places.
- **NumPy warning is harmless**: The NumPy 2.x compatibility warning when importing torch is safe to ignore.

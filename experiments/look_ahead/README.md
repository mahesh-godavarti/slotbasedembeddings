# Look-Ahead Architecture Experiments

Experiment code for the look-ahead paper. Compares look-ahead models (non-cumulative corrections + past-only contextualization) against shared-weight baselines using joformer block types.

## Files

- **`models.py`** — Model definitions. Wraps joformer block classes (`RoFormerBlock`, `JoFormerFixedBlock`, `JoFormerLearnedBlock`, `JoFormerProjectedBlock`) with look-ahead architecture: shared weights, configurable residual strategy (non-cumulative vs cumulative), and contextualization scope (past-only vs self-inclusive).
- **`train_wiki_streaming.py`** — Training script. BPE tokenization, memory-mapped data loading, training loop with convergence diagnostics, depth sweeps, and self-speculative evaluation.

## Models

All models use a single shared-weight block applied N times (`n_layers` = iteration count).

| Model | Residual | Contextualization | Role |
|---|---|---|---|
| `*_look_ahead` | Non-cumulative (`x_0 + correction`) | Past-only (position shift) | Model A |
| `*_baseline` | Cumulative (`x_k + correction`) | Self-inclusive | Model B |
| `*_noncum_only` | Non-cumulative | Self-inclusive | Ablation |
| `*_pastonly_only` | Cumulative | Past-only | Ablation |

Each variant is available for all four block types: `roformer`, `joformer_fixed`, `joformer_learned`, `joformer_projected`. The original joformer models (separate blocks, not shared) are also available for reference.

Default comparison: `joformer_fixed_look_ahead` vs `joformer_fixed_baseline`.

## Key Methods

- **`forward(idx, targets)`** — Standard training forward pass (N iterations).
- **`generate(idx, max_new_tokens)`** — Full-depth autoregressive generation. O(N) block evaluations per token.
- **`generate2(idx, max_new_tokens)`** — Single-step warm-started generation (look-ahead only). Reuses previous step's correction as warm start. O(1) block evaluations per token after bootstrap.
- **`forward_at_depth(idx, K, targets)`** — Evaluate at inference depth K instead of N (adaptive depth experiment).
- **`forward_with_diagnostics(idx, targets)`** — Returns convergence diagnostics: correction norms and empirical contraction ratios per iteration.
- **`generate_speculative(idx, max_new_tokens, draft_length)`** — Self-speculative multi-token generation. `generate2` drafts tokens, `generate` verifies.

## Usage

```bash
# Preprocess wiki data (BPE tokenization to memmap binary)
python train_wiki_streaming.py preprocess --wiki_path /path/to/wiki.en.txt --vocab_size 8000

# Train look-ahead vs baseline
python train_wiki_streaming.py train --data_dir look_ahead/data \
    --models joformer_fixed_look_ahead joformer_fixed_baseline \
    --n_embed 200 --n_layers 10 --block_size 128 --batch_size 64

# Or do both in one command
python train_wiki_streaming.py auto --wiki_path /path/to/wiki.en.txt \
    --models joformer_fixed_look_ahead joformer_fixed_baseline --smoke

# Run all four ablation variants
python train_wiki_streaming.py train --data_dir look_ahead/data \
    --models joformer_fixed_look_ahead joformer_fixed_baseline \
            joformer_fixed_noncum_only joformer_fixed_pastonly_only
```

The `--smoke` flag runs a quick test (50 iterations, small model).

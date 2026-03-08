# Test Plan: Look-Ahead Architecture Experiments

This document is a step-by-step test plan for running all experiments on an AWS GPU server. Each phase has exact commands, expected outputs, and verification criteria. Phases are ordered by dependency — complete each before moving to the next.

## Prerequisites

### Environment Setup

```bash
# 1. Clone/sync the repo to the AWS server
# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install torch numpy tqdm tokenizers
```

### Data

The experiments use `wiki.en.txt` (English Wikipedia text dump). The default path is `../exp8/data/wiki.en.txt` relative to the `look_ahead/` directory. If your data is elsewhere, pass `--wiki_path /path/to/wiki.en.txt` to the commands below.

### Directory Structure

All commands assume you are in the `experiments/` directory:

```
experiments/
├── joformer/          # Joformer model definitions (imported by look_ahead)
│   └── train_wiki.py
├── look_ahead/        # This experiment
│   ├── models.py
│   ├── train_wiki_streaming.py
│   └── README.md
├── exp8/data/         # Wiki data (default location)
│   └── wiki.en.txt
└── venv/              # Virtual environment
```

---

## Phase 0: Smoke Test (5 minutes)

**Goal:** Verify the code runs end-to-end before committing GPU hours.

```bash
cd experiments/look_ahead

python train_wiki_streaming.py auto \
    --wiki_path ../exp8/data/wiki.en.txt \
    --data_dir look_ahead/data \
    --models joformer_fixed_look_ahead joformer_fixed_baseline \
    --smoke
```

**Expected output:**
- BPE tokenizer trains (if first run)
- Both models train for 50 iterations
- Comparison table printed at the end with val loss and PPL for both models
- Results JSON saved to `look_ahead_results_*.json`

**Verification:**
- No crashes or NaN losses
- Both models produce generation samples (may be gibberish — that's fine for 50 iters)
- The look-ahead model reports `empirical L` values (contraction ratios)
- The look-ahead model reports depth sweep results (K=1,2,3,5,...)

**If this fails:** Check that `joformer/train_wiki.py` is importable from `look_ahead/`. The `models.py` file does `sys.path.insert(0, .../joformer/)` to find it.

---

## Phase 1: Preprocessing (10-30 minutes, CPU only)

**Goal:** Tokenize the full wiki corpus to a memory-mapped binary file.

```bash
cd experiments/look_ahead

python train_wiki_streaming.py preprocess \
    --wiki_path ../exp8/data/wiki.en.txt \
    --vocab_size 8000 \
    --data_dir look_ahead/data
```

**Expected output:**
- `look_ahead/data/wiki_tokenizer.json` — trained BPE tokenizer
- `look_ahead/data/wiki_tokens.bin` — binary token file (several GB for full wiki)
- `look_ahead/data/wiki_tokens.meta` — JSON metadata with `total_tokens`, `vocab_size`

**Verification:**
- Check `wiki_tokens.meta`: `total_tokens` should be in the tens of millions for full wiki
- `vocab_size` should be close to 8000 (actual may differ slightly)

**Note:** This only needs to run once. All subsequent training runs reuse the preprocessed data. If you want a smaller dataset for faster iteration, use `--wiki_lines 100000`.

---

## Phase 2: Main Result — Look-Ahead vs Baseline (Section 4.2)

**Goal:** Train Model A (look-ahead) and Model B (baseline) head-to-head. This is the paper's central experiment.

```bash
cd experiments/look_ahead

python train_wiki_streaming.py train \
    --data_dir look_ahead/data \
    --models joformer_fixed_look_ahead joformer_fixed_baseline \
    --n_embed 200 \
    --n_layers 10 \
    --block_size 128 \
    --batch_size 64 \
    --lr 5e-4 \
    --max_iters 10000 \
    --eval_interval 500 \
    --cosine_decay \
    --checkpoint_dir look_ahead/checkpoints \
    --seed 42
```

**Expected output per model:**
- PPL curves logged every 500 iterations
- Convergence diagnostics (empirical L, correction norms) at each eval
- Generation samples (full-depth `generate` and single-step `generate2`) periodically
- Depth sweep at end: PPL at K=1,2,3,5,10,20
- Self-speculative evaluation at end (look-ahead model only): acceptance rates for draft k=2,4,8
- Final comparison table

**Key results to record:**

| Metric | Look-Ahead (Model A) | Baseline (Model B) |
|---|---|---|
| Final val PPL (full depth, K=N) | | |
| Final val PPL (single step, K=1) | | |
| Parameter count | | |
| Empirical contraction constant L | | |

**Verification criteria:**
- Both models should reach val PPL well below the initial (~vocab_size)
- **Critical:** Look-ahead model's K=1 PPL should be close to its K=N PPL (this validates the single-step inference claim)
- Baseline's K=1 PPL should be significantly worse than its K=N PPL
- Look-ahead model's `generate2` samples should be comparable quality to `generate` samples
- Empirical L should be < 1 (ideally < 0.5) for the look-ahead model

**If look-ahead K=1 PPL is bad:** The contraction constant may be too large. Try increasing `n_layers` (more iterations during training forces smaller L) or decreasing `n_embed`.

---

## Phase 3: Ablations (Section 4.3)

**Goal:** Show that BOTH non-cumulative corrections AND past-only contextualization are necessary. Neither alone is sufficient.

```bash
cd experiments/look_ahead

python train_wiki_streaming.py train \
    --data_dir look_ahead/data \
    --models joformer_fixed_look_ahead joformer_fixed_baseline \
            joformer_fixed_noncum_only joformer_fixed_pastonly_only \
    --n_embed 200 \
    --n_layers 10 \
    --block_size 128 \
    --batch_size 64 \
    --lr 5e-4 \
    --max_iters 10000 \
    --eval_interval 500 \
    --cosine_decay \
    --checkpoint_dir look_ahead/checkpoints_ablation \
    --seed 42
```

**Key results to record:**

| Model | Non-Cum | Past-Only | Full-Depth PPL | Single-Step PPL |
|---|---|---|---|---|
| look_ahead | Yes | Yes | | |
| baseline | No | No | | |
| noncum_only | Yes | No | | |
| pastonly_only | No | Yes | | |

**Verification criteria:**
- `look_ahead` should have the best K=1 / K=N ratio (single-step works well)
- `noncum_only` (non-cumulative but self-inclusive): K=1 PPL should be worse than look_ahead's because without past-only, the single-step output still depends on depth
- `pastonly_only` (past-only but cumulative): K=1 PPL should be worse than look_ahead's because cumulative corrections make the output depth-dependent
- Both ablations should demonstrate that removing either insight degrades single-step performance

---

## Phase 4: Adaptive Inference Depth (Section 4.5)

**Goal:** Show that inference depth K is a runtime parameter. Train at N=10, evaluate at varying K.

This data comes from the depth sweep already produced in Phase 2. If you need a dedicated run:

```bash
cd experiments/look_ahead

python train_wiki_streaming.py train \
    --data_dir look_ahead/data \
    --models joformer_fixed_look_ahead \
    --n_embed 200 \
    --n_layers 10 \
    --block_size 128 \
    --batch_size 64 \
    --lr 5e-4 \
    --max_iters 10000 \
    --cosine_decay \
    --checkpoint_dir look_ahead/checkpoints_depth \
    --seed 42
```

**Key results to record:**

| Depth K | Val Loss | Val PPL |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |
| 5 | | |
| 10 (=N) | | |
| 20 (>N) | | |

**Verification criteria:**
- PPL should decrease monotonically (or near-monotonically) as K increases
- The gap between K=1 and K=N should be small for the look-ahead model
- K=20 (extrapolating beyond training depth) should not diverge — it should be close to or slightly better than K=N

---

## Phase 5: Effect of Training Depth N (Section 4.7)

**Goal:** Train with different N values, evaluate at K=1 (single-step) for each. Shows how training depth affects the learned contraction constant.

Run **5 separate training runs** with different `n_layers`:

```bash
cd experiments/look_ahead

for N in 1 2 5 10 20; do
    echo "===== Training with N=$N ====="
    python train_wiki_streaming.py train \
        --data_dir look_ahead/data \
        --models joformer_fixed_look_ahead \
        --n_embed 200 \
        --n_layers $N \
        --block_size 128 \
        --batch_size 64 \
        --lr 5e-4 \
        --max_iters 10000 \
        --cosine_decay \
        --checkpoint_dir "look_ahead/checkpoints_N${N}" \
        --seed 42
done
```

**Key results to record:**

| Training N | K=1 PPL | K=N PPL | Empirical L |
|---|---|---|---|
| 1 | | | N/A |
| 2 | | | |
| 5 | | | |
| 10 | | | |
| 20 | | | |

**Verification criteria:**
- K=1 PPL should generally improve (decrease) as N increases, because more training iterations force a smaller contraction constant
- Empirical L should decrease as N increases
- N=1 is a special case: no iteration, so K=1 = K=N (the model is effectively a single-block model)

---

## Phase 6: Cross-Block-Type Comparison (Optional)

**Goal:** Verify that the look-ahead architecture works across different block types, not just JoFormerFixed.

```bash
cd experiments/look_ahead

python train_wiki_streaming.py train \
    --data_dir look_ahead/data \
    --models roformer_look_ahead roformer_baseline \
            joformer_fixed_look_ahead joformer_fixed_baseline \
            joformer_projected_look_ahead joformer_projected_baseline \
    --n_embed 200 \
    --n_layers 10 \
    --block_size 128 \
    --batch_size 64 \
    --lr 5e-4 \
    --max_iters 10000 \
    --cosine_decay \
    --checkpoint_dir look_ahead/checkpoints_cross \
    --seed 42
```

**Note:** `joformer_learned_*` uses a different embedding scheme (half for tokens, half for angles) so parameter counts differ. Include it only if you want completeness:

```bash
    --models joformer_learned_look_ahead joformer_learned_baseline
```

**Key results to record:**

| Block Type | Look-Ahead K=1 PPL | Look-Ahead K=N PPL | Baseline K=N PPL |
|---|---|---|---|
| roformer | | | |
| joformer_fixed | | | |
| joformer_projected | | | |
| joformer_learned | | | |

**Verification criteria:**
- The look-ahead advantage (small K=1 vs K=N gap) should hold across all block types
- This demonstrates the architecture is mixer-agnostic

---

## Phase 7: Original Joformer Comparison (Optional)

**Goal:** Compare shared-weight look-ahead against original joformer models (separate blocks per layer).

```bash
cd experiments/look_ahead

python train_wiki_streaming.py train \
    --data_dir look_ahead/data \
    --models joformer_fixed_look_ahead joformer_fixed_baseline joformer_fixed \
    --n_embed 200 \
    --n_layers 10 \
    --block_size 128 \
    --batch_size 64 \
    --lr 5e-4 \
    --max_iters 10000 \
    --cosine_decay \
    --checkpoint_dir look_ahead/checkpoints_orig \
    --seed 42
```

**Note:** The original `joformer_fixed` has 10 separate blocks (10x the block parameters), so it will have significantly more parameters. This is NOT a fair parameter-count comparison — it shows the cost of weight sharing vs the benefit of look-ahead.

---

## Collecting Results

All training runs save results to `look_ahead_results_*.json` with timestamps. The latest run also saves to `look_ahead_results_latest.json`. Each results file contains:

```json
{
    "config": { "n_embed": 200, "n_layers": 10, ... },
    "results": {
        "joformer_fixed_look_ahead": {
            "val_loss": 3.45,
            "val_ppl": 31.5,
            "ppl_curve": { "iter": [...], "train_ppl": [...], "val_ppl": [...] },
            "diagnostics": [{ "iter": 0, "avg_correction_norms": [...], ... }],
            "depth_results": { "1": {"val_loss": ..., "val_ppl": ...}, ... },
            "speculative_results": { "2": {"acceptance_rate": ..., ...}, ... }
        }
    }
}
```

### Summary Table Template

After all phases, compile results into this table for the paper:

```
| Experiment | Model | K | Val PPL | Empirical L | Notes |
|---|---|---|---|---|---|
| Main (4.2) | look_ahead | N=10 | | | Full depth |
| Main (4.2) | look_ahead | K=1 | | | Single step |
| Main (4.2) | baseline | N=10 | | | Full depth |
| Main (4.2) | baseline | K=1 | | | Single step |
| Ablation (4.3) | noncum_only | K=1 | | | |
| Ablation (4.3) | pastonly_only | K=1 | | | |
| Depth (4.5) | look_ahead | K=1..20 | curve | | |
| Training N (4.7) | look_ahead | K=1 | per N | per N | |
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'train_wiki'`**
- The `models.py` file imports from `joformer/train_wiki.py` via `sys.path` manipulation. Make sure the `joformer/` directory exists alongside `look_ahead/` in the `experiments/` folder.

**NaN loss during training**
- Try reducing learning rate (`--lr 1e-4`)
- Try reducing `n_layers` (fewer iterations = less gradient instability during early training)
- Gradient clipping is already enabled (max norm 1.0)

**Out of memory**
- Reduce `--batch_size` (try 32 or 16)
- Reduce `--block_size` (try 64)
- Reduce `--n_embed` (try 128)
- The preprocessing phase uses constant memory regardless of corpus size

**Slow training**
- Ensure you're using a GPU (`Device: cuda` should appear at start)
- The `--cosine_decay` flag helps convergence; use it for all serious runs
- If running multiple models sequentially takes too long, run them in separate processes

**Self-speculative acceptance rate is 0%**
- This is normal for early training or small models. The acceptance rate depends on the contraction constant L being small enough that single-step inference matches full-depth inference. Train longer or use larger N.

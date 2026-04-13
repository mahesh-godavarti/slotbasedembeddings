# KG+Text Experiment (Exp 7)

## Overview

This experiment compares different attention/positional encoding architectures for jointly learning knowledge graph (KG) triples and natural language text. The core question: can models transfer knowledge between modalities (KG-exclusive facts appearing in text predictions, and vice versa)?

## Models

All models are character-level transformers. Primed models (') include V rotation.

| Model | Angle type | KG format | KG training | Rotate V |
|-------|-----------|-----------|-------------|----------|
| A/A'  | RoPE + learned slot angles (shared) | Slotted (HEAD/REL/TAIL) | MLM | No/Yes |
| B/B'  | RoPE (standard) | Text only (linearized) | Causal | No/Yes |
| C/C'  | Learned per-token angles | Text only (linearized) | Causal | No/Yes |
| D/D'  | RoPE | Flat (rel as token) | MLM | No/Yes |
| E/E'  | Learned per-token cumsum + relation operator | Native (chars only) | Causal | No/Yes |
| F/F'  | Fixed RoPE | Flat (rel as token) | MLM | No/Yes |
| G/G'  | RoPE + learned slot angles (per-relation) | Slotted | MLM | No/Yes |
| H/H'  | Fixed cumsum + relation operator | Native (chars only) | Causal | No/Yes |
| I/I'  | Learned cumsum + shared relation operator | Native (chars only) | Causal | No/Yes |

## 7 Evaluation Tiers

| Tier | # chains | Description |
|------|----------|-------------|
| memorization | 60 | Seen in both KG and text during training |
| transfer | 15 | Base facts in both, derived facts only in KG |
| generalization | 15 | Base facts only, no derived |
| kg_excl_mem | 10 | KG only, never text — tests KG->Text cross-pollination |
| kg_excl_gen | 10 | KG only, no derived |
| text_excl_mem | 10 | Text only, never KG — tests Text->KG cross-pollination |
| text_excl_gen | 10 | Text only, no derived |

**IMPORTANT**: ALWAYS report ALL 7 tiers. Never omit kg_exclusive or text_exclusive. Always show BOTH h@5 AND PPL for every tier.

## kg_as_text Mode

New mode (`--kg_as_text` flag) that converts ALL KG triples to linearized text with relation tokens:
- Forward: `"Adam <son_of> Brian"` (where `<son_of>` is a single token)
- Inverse: `"Brian <inverse_son_of> Adam"`
- Only runs B/B'/C/C' models (no separate KG modality)
- KG-exclusive chains become linearized text, so B/C can learn from them
- Two evaluations: natural language text eval + linearized KG eval (12 prompts per chain)

## Running Experiments

### Setup (first time on new machine)
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
```

### Activate venv (every time)
```bash
source venv/bin/activate
```

### Smoke test
```bash
python kg_text_experiment.py --models B "B'" --smoke --exp 7a
python kg_text_experiment.py --models B "B'" --smoke --kg_as_text --exp 7a
```

### Full runs
```bash
# Standard run with all models
python kg_text_experiment.py --models A "A'" B "B'" C "C'" D "D'" E "E'" F "F'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000 --exp 7a

# KG-as-text mode (B/C only)
python kg_text_experiment.py --models B "B'" C "C'" --kg_as_text --seeds 3 --n_embed 100 --n_layers 20 --iters 10000 --exp 7a

# Causal KG training (for E/H/I)
python kg_text_experiment.py --models E "E'" H "H'" I "I'" --causal_kg --seeds 1 --n_embed 100 --n_layers 20 --iters 10000 --exp 7a
```

### Key CLI flags
- `--models A "A'" B ...` — which models to run
- `--exp 7a` — generous linearization (use this)
- `--n_embed N` — embedding dimension
- `--n_layers N` — number of transformer layers
- `--iters N` — training iterations
- `--seeds N` — number of random seeds to average
- `--smoke` — quick test (50 iters, 1 seed)
- `--kg_as_text` — linearized KG mode for B/C models
- `--causal_kg` — causal KG training for E/H/I models
- `--inverse_kg` — add inverse KG triples
- `--checkpoint_dir DIR` — save/load checkpoints
- `--load_checkpoints` — skip training, load saved models
- `--resume_training` — continue from checkpoint

### Splitting runs across parallel processes
Long runs can be split. Example — run KG models and text models in parallel:
```bash
# Terminal 1: KG-capable models
python kg_text_experiment.py --models A "A'" D "D'" F "F'" --exp 7a --n_embed 100 --n_layers 20 --iters 10000

# Terminal 2: Text-only models
python kg_text_experiment.py --models B "B'" C "C'" --exp 7a --n_embed 100 --n_layers 20 --iters 10000

# Terminal 3: Causal KG models
python kg_text_experiment.py --models E "E'" H "H'" I "I'" --causal_kg --exp 7a --n_embed 100 --n_layers 20 --iters 10000
```

## Parsing and Formatting Results

### From log files
```bash
python parse_results.py <logfile>
python format_results.py <logfile> [output.md]
```

### From JSON result files
Results are auto-saved as `exp7_results_YYYYMMDD_HHMMSS.json` and `exp7_results_latest.json`.

### NEVER do this
- Never use `sed`/`grep`/`awk` to extract results from log files — use `parse_results.py` and `format_results.py` instead
- Never use `TaskOutput` to check background experiment progress — it dumps the entire buffer

### Checking background experiment progress
Always use this exact command:
```bash
tail -c 500 <output_file> 2>/dev/null | tr '\r' '\n' | tail -1
```

## Key Rules

- **Always stay available for chat**: Run experiments with `run_in_background=true`. Never block on TaskOutput or sleep.
- **Use Bash tool directly for experiments, NOT Task agents**: Task agents can't get user permission approval.
- **Never use `&` inside background commands**: `run_in_background=true` already handles backgrounding.
- **Format PPL consistently**: Use fixed decimal format (21.61, not "6.4M"). Keep 2 decimal places.
- **PPL aggregation**: Uses geometric mean (exp of mean log), NOT arithmetic mean.
- **NumPy warning is harmless**: The NumPy 2.x compatibility warning when importing torch is safe to ignore.

## Current Best Results (130 chains, n100, l20, 10K iters)

### KG Champions
- **E'** dominates KG: .809/.806/.633 mem/trn/gen, .858 kg_excl_mem (learned cumsum + V rotation)
- V rotation is critical: E gets .323 KG mem without it vs E' .809 with it

### Text Champions
- **C'** dominates text at scale: .933 mem, .900 transfer (from n500/l2/10K run)
- At depth (l20): H'/E have best text PPL (~5.0 vs A/F ~5.5)

### Cross-Pollination
- Still weak but present: F kg_excl_mem on text = .183, E' text_excl_mem on text = .150
- text_excl on KG near zero for all models
- Zero cross-pollination at n500/l2/10K — KG-exclusive PPL on text is catastrophic for B/C

### Summary files
- `exp7_results_130ch_n100_l20_10Kiters_partial_summary.md` — latest (in progress)
- `exp7_results_130ch_n50_l20_1Kiters_comparison_summary.md` — all models at 1K iters
- `exp7_results_1000chains_n500_2layer_10000iters_summary.md` — 1000 chains, large models
- `exp7_results_n90_10000iters_summary.md` — original 130 chains results

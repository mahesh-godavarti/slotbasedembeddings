# KG+Text Experiment (Exp 8) — BPE Subword on Real Data

## Overview

Exp 8 extends Exp 7's architectural comparison to **real-world data** with **BPE subword tokenization**. Same core question: can models transfer knowledge between KG and text modalities?

Key differences from Exp 7:
- **BPE subword tokenization** (HuggingFace `tokenizers`) instead of character-level
- **Real data**: Wikipedia text, WordNet synonyms, FrameNet relations, BATS/Google/word analogies
- Vocab size is controlled exactly by `--vocab_size` (default 16000)
- Entities may be multi-token after BPE; KGDataset and eval handle variable-length head/tail
- Relation tokens (e.g. `<synonym_of>`) are BPE special tokens — never split

## First-Time Setup

```bash
cd ~/exp8
python3 -m venv venv
source venv/bin/activate
pip install torch numpy tokenizers tqdm
```

### Activate venv (every time)
```bash
cd ~/exp8
source venv/bin/activate
```

## Models

All models are subword-level transformers. Primed models (') include V rotation.

| Model | Angle type | KG format | KG training | Rotate V |
|-------|-----------|-----------|-------------|----------|
| A/A'  | RoPE + learned slot angles (shared) | Slotted (HEAD/REL/TAIL) | MLM | No/Yes |
| B/B'  | RoPE (standard) | Text only (linearized) | Causal | No/Yes |
| C/C'  | Learned per-token angles | Text only (linearized) | Causal | No/Yes |
| D/D'  | RoPE | Flat (rel as token) | MLM | No/Yes |
| E/E'  | Learned per-token cumsum + relation operator | Native (subwords only) | Causal | No/Yes |
| F/F'  | Fixed RoPE | Flat (rel as token) | MLM | No/Yes |
| G/G'  | RoPE + learned slot angles (per-relation) | Slotted | MLM | No/Yes |
| H/H'  | Fixed cumsum + relation operator | Native (subwords only) | Causal | No/Yes |
| I/I'  | Learned cumsum + shared relation operator | Native (subwords only) | Causal | No/Yes |

### Model categories
- **Text-only (linearized)**: B/B', C/C' — KG triples linearized as text
- **Slotted KG**: A/A', G/G' — HEAD/REL/TAIL slot structure
- **Flat KG**: D/D', F/F' — relation as a token in sequence
- **Native KG**: E/E', H/H', I/I' — no relation token, use angular operators

## 7 Evaluation Tiers

| Tier | Description |
|------|-------------|
| memorization | Seen in both KG and text during training |
| transfer | Base facts in both, derived facts only in KG |
| generalization | Base facts only, no derived |
| kg_exclusive_memorization | KG only, never text — tests KG->Text cross-pollination |
| kg_exclusive_generalization | KG only, no derived |
| text_exclusive_memorization | Text only, never KG — tests Text->KG cross-pollination |
| text_exclusive_generalization | Text only, no derived |

**IMPORTANT**: ALWAYS report ALL 7 tiers. Never omit kg_exclusive or text_exclusive. Always show BOTH h@5 AND PPL for every tier.

## Data

All data lives in `data/` (relative to the script). No path changes needed between local and AWS.

| File | Size | Description |
|------|------|-------------|
| `data/wiki.en.txt` | 3.0 GB | Wikipedia sentences (one per line) |
| `data/wordnet-synonyms.txt` | 5.1 MB | WordNet synonym pairs |
| `data/framenet.txt` | 3.4 MB | FrameNet relations |
| `data/BATS_3.0/` | 240 KB | BATS 3.0 analogies |
| `data/questions-words_for_training.txt` | 590 KB | Google word analogies |
| `data/wordanalogies.txt` | 49 KB | Word analogies |

## Running Experiments

### Step 1: Train the BPE tokenizer (once, before any experiments)
```bash
python word_experiment.py --train_tokenizer --vocab_size 16000
```
This trains on the full wiki text + all KG entity words and saves to `data/tokenizer.json`. All subsequent runs load it automatically — no retraining.

### Step 2: Smoke test
```bash
python word_experiment.py --models B "B'" --smoke --wiki_lines 1000 --vocab_size 4000
```

### Full runs
```bash
# Text-only models (B/C)
python word_experiment.py --models B "B'" C "C'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000 --vocab_size 16000

# Slotted KG models (A/G)
python word_experiment.py --models A "A'" G "G'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000

# Flat KG models (D/F)
python word_experiment.py --models D "D'" F "F'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000

# Native KG models with causal KG training (E/H/I)
python word_experiment.py --models E "E'" H "H'" I "I'" --causal_kg --seeds 3 --n_embed 100 --n_layers 20 --iters 10000

# KG-as-text mode (B/C only)
python word_experiment.py --models B "B'" C "C'" --kg_as_text --seeds 3 --n_embed 100 --n_layers 20 --iters 10000
```

### Key CLI flags
- `--models A "A'" B ...` — which models to run
- `--vocab_size N` — BPE vocabulary size (default 16000)
- `--n_embed N` — embedding dimension
- `--n_layers N` — number of transformer layers
- `--iters N` — training iterations
- `--seeds N` — number of random seeds to average
- `--smoke` — quick test (50 iters, 1 seed, vocab_size capped at 4000)
- `--wiki_lines N` — how many wiki lines to load (default 1000000)
- `--kg_as_text` — linearized KG mode for B/C models
- `--causal_kg` — causal KG training for E/H/I models
- `--inverse_kg` — add inverse KG triples
- `--checkpoint_dir DIR` — save/load checkpoints (default: `checkpoints/`)
- `--load_checkpoints` — skip training, load saved models
- `--resume_training` — continue from checkpoint

### Splitting runs across parallel terminals
```bash
# Terminal 1: Text-only models
python word_experiment.py --models B "B'" C "C'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000

# Terminal 2: Slotted + Flat KG models
python word_experiment.py --models A "A'" D "D'" F "F'" G "G'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000

# Terminal 3: Native KG models (causal)
python word_experiment.py --models E "E'" H "H'" I "I'" --causal_kg --seeds 3 --n_embed 100 --n_layers 20 --iters 10000
```

## Results

Results are auto-saved as `exp8_results_YYYYMMDD_HHMMSS.json` and `exp8_results_latest.json`.

### Parsing and formatting results from log files
```bash
python parse_results.py <logfile>
python format_results.py <logfile> [output.md]
```

### NEVER do this
- Never use `sed`/`grep`/`awk` to extract results from log files — use `parse_results.py` and `format_results.py` instead
- Never use `TaskOutput` to check background experiment progress — it dumps the entire buffer

## Key Rules

- **Always stay available for chat**: Run experiments with `run_in_background=true`. Never block on TaskOutput or sleep.
- **Use Bash tool directly for experiments, NOT Task agents**: Task agents can't get user permission approval.
- **Never use `&` inside background commands**: `run_in_background=true` already handles backgrounding.
- **Format PPL consistently**: Use fixed decimal format (21.61, not "6.4M"). Keep 2 decimal places.
- **PPL aggregation**: Uses geometric mean (exp of mean log), NOT arithmetic mean.
- **NumPy warning is harmless**: The NumPy 2.x compatibility warning when importing torch is safe to ignore.

### Checking background experiment progress
Always use this exact command:
```bash
tail -c 500 <output_file> 2>/dev/null | tr '\r' '\n' | tail -1
```

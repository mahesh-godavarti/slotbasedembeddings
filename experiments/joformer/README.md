# JoFormer Wiki Text Experiment

Compare 4 transformer architectures on Wikipedia text with BPE tokenization.
Single self-contained script: `train_wiki.py`.

## Setup

```bash
cd experiments
source venv/bin/activate
pip install torch numpy tokenizers tqdm
```

Wiki data must exist at `exp8/data/wiki.en.txt` (or pass `--wiki_path`).

## Quick Start

```bash
# Smoke test (all 4 models, ~30 seconds)
python joformer/train_wiki.py --smoke

# Full run (all 4 models, default settings)
python joformer/train_wiki.py

# Single model
python joformer/train_wiki.py --models joformer_projected --n_embed 256 --n_layers 6 --max_iters 10000

# Larger run
python joformer/train_wiki.py --n_embed 256 --n_layers 8 --block_size 128 --max_iters 20000 --vocab_size 16000
```

## 4 Model Architectures

All models use `log(exp(x)+1)` attention (not softmax), causal masking, and gradient clipping at 1.0.

### RoFormer (`roformer`)
- Standard RoPE: fixed angles from `outer(arange(T), arange(C/2))`, flipped along T
- Rotates K, Q only
- V and output left alone
- Full `n_embed` token embedding

### JoFormer-Fixed (`joformer_fixed`)
- Same fixed RoPE angles as RoFormer
- Rotates K, Q, **and V**
- Applies **inverse rotation** to attention output (de-rotation)
- Full `n_embed` token embedding

### JoFormer-Learned (`joformer_learned`)
- Per-token learned angles via `Embedding(vocab_size, n_embed//2)`
- Angles: flip along T -> cumsum -> flip back (accumulates token-dependent angles)
- Same angles used for all layers
- Rotates K, Q, V; inverse rotation on output
- Half-dim token embedding: `Embedding(vocab_size, n_embed//2)` + `Linear(n_embed//2, n_embed)` expander
- Separate `angle_embedding_table` produces raw angles per token

### JoFormer-Projected (`joformer_projected`)
- Angles **projected per-layer** from previous layer's output (not from a fixed embedding table)
- Each block has:
  - **Vector projection**: `Linear(C, C)` applied to block input
  - **Angle projection**: `Linear(C, 2C) -> ReLU -> Linear(2C, C//2)` (2-layer MLP)
- Angles: flip -> cumsum -> flip (same as JoFormer-Learned)
- Rotates K, Q, V; inverse rotation on output
- Full `n_embed` token embedding
- Slightly more parameters than others due to per-block projection MLPs

### Architecture Comparison Table

| Model | Angle source | Rotates V | De-rotates output | Token embed dim | Per-block angle params |
|-------|-------------|-----------|-------------------|-----------------|----------------------|
| `roformer` | Fixed RoPE | No | No | `n_embed` | None |
| `joformer_fixed` | Fixed RoPE | Yes | Yes | `n_embed` | None |
| `joformer_learned` | Learned embedding (cumsum) | Yes | Yes | `n_embed//2` + expander | None (shared angles) |
| `joformer_projected` | MLP from hidden state (cumsum) | Yes | Yes | `n_embed` | `Linear(C,2C) + Linear(2C,C//2)` |

## CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--wiki_path` | `../exp8/data/wiki.en.txt` | Path to wiki text file |
| `--wiki_lines` | 100000 | Max lines to load |
| `--vocab_size` | 8000 | BPE vocabulary size |
| `--models` | all 4 | Space-separated list: `roformer joformer_fixed joformer_learned joformer_projected` |
| `--n_embed` | 128 | Embedding dimension (must be even) |
| `--n_layers` | 4 | Number of transformer layers |
| `--block_size` | 64 | Context window size (tokens) |
| `--batch_size` | 32 | Batch size |
| `--lr` | 5e-4 | Learning rate (AdamW) |
| `--max_iters` | 10000 | Training iterations |
| `--dropout` | 0.2 | Dropout rate |
| `--eval_interval` | 500 | Evaluate every N iterations |
| `--checkpoint_dir` | `joformer/checkpoints` | Where to save checkpoints |
| `--smoke` | off | Quick test: 50 iters, 1000 lines, vocab 2000, n_embed 64, n_layers 2 |
| `--generate_len` | 200 | Tokens to generate in samples |
| `--seed` | 42 | Random seed (reset before each model for fair comparison) |

## Output

The script prints:
1. Per-model training progress with tqdm (train loss, val loss, val PPL)
2. Periodic text generation samples
3. Final comparison table with val loss and val PPL for all models

Checkpoints are saved to `--checkpoint_dir` at each eval interval and at the end:
- `{model_name}_iter{N}.pt` — periodic checkpoints
- `{model_name}_final.pt` — final checkpoint

## Data Pipeline

1. Load wiki text lines from file (skip blanks)
2. Train BPE tokenizer (HuggingFace `tokenizers` library) on the loaded text
3. Encode all text into a flat token-ID stream
4. 90/10 train/val split on the token stream
5. `get_batch()`: sample random `block_size`-length windows

## Key Implementation Details

- **Attention**: `log(exp(x)+1)` instead of softmax — preserves non-negative attention weights without the winner-take-all effect of softmax
- **Rotation**: Each C-dimensional vector is split into C/2 pairs, each pair rotated by a 2x2 rotation matrix built from cos/sin of the angle
- **Inverse rotation**: Transpose of the rotation matrix (orthogonal inverse) applied to attention output in JoFormer variants
- **Cumsum angles**: For learned/projected models, raw per-token angles are flip->cumsum->flip'd so that each position's rotation accumulates all subsequent token angles (enables relative position encoding from token identity)
- **Fair comparison**: `torch.manual_seed(seed)` is called before each model's initialization, so all models start from equivalent random states
- **n_embed must be even**: Required for the C/2 rotation pairs

## Recommended Runs

```bash
# Baseline comparison
python joformer/train_wiki.py --n_embed 128 --n_layers 4 --max_iters 10000

# Deeper
python joformer/train_wiki.py --n_embed 128 --n_layers 8 --max_iters 10000

# Wider
python joformer/train_wiki.py --n_embed 256 --n_layers 4 --max_iters 10000

# Longer context
python joformer/train_wiki.py --n_embed 128 --n_layers 4 --block_size 128 --max_iters 10000

# Just the projected model for a quick check
python joformer/train_wiki.py --models joformer_projected --n_embed 128 --n_layers 4 --max_iters 5000
```

## File Structure

```
joformer/
  README.md          # This file
  train_wiki.py      # Single self-contained training script
  checkpoints/       # Created at runtime
```

Source models adapted from `joformer_src/` (character-level originals):
- `roformer.py` -> `RoFormer`
- `journey_transformer_fixed_angles.py` -> `JoFormerFixed`
- `journey_transformer_per_token_angles.py` -> `JoFormerLearned`
- (new) `JoFormerProjected`

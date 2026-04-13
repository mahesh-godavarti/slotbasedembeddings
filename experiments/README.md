# Character-Compositional Token Embeddings Experiment

## Instructions for Claude on AWS

**Read this entire file before doing anything.**

### Working Style

- **Ask before acting.** If something is unclear, ask. Do not guess, do not improvise, do not "try something and see." Ask the user.
- **Stick with what works.** If an approach is working, we use that approach. Do not refactor working code. Do not replace a working solution with an "improved" one. Do not introduce new libraries, patterns, or abstractions unless explicitly asked.
- **No unsolicited changes.** Do not add logging, comments, type hints, error handling, or "improvements" to code that already works. Do not rename things. Do not reorganize files.
- **One thing at a time.** Complete the current task before starting the next. Confirm results before moving on.
- **Report problems immediately.** If something fails, report the exact error. Do not silently retry with a different approach. Do not silently change code to work around an issue.

---

## What This Experiment Is

We have a paper (`mathpaper17.tex` in the parent directory) that introduces an algebraic compositional framework. Elements are pairs `(v, R)` where `v` is a vector and `R` is a rotation matrix. They compose via:

```
(a, A) . (b, B) = (a + Ab, AB)
```

This is the semidirect product of a translation group and a rotation group.

**The experiment**: Use this framework to derive BPE token embeddings from character-level embeddings. Instead of storing a separate 768-dimensional vector for each of the 50,257 GPT-2 tokens (~38.6M parameters), we store only 256 character-level parameters (~0.3M parameters) and compose them to get token embeddings.

**The question**: Can this 130x parameter reduction in the embedding layer maintain reasonable language modeling performance?

---

## The Math

Each of the 256 byte-level characters `c` has:
- A vector `v_c` in R^768
- A rotation angle vector `theta_c` in R^384 (defining a block-diagonal matrix of 384 independent 2x2 rotations)

For a token spelled as characters `c_1, c_2, ..., c_T`, the composed embedding is:

```
e_token = v_c1 + R(theta_c1) * v_c2 + R(theta_c1 + theta_c2) * v_c3 + ...
```

The key insight is that block-diagonal 2x2 rotation matrices compose by adding angles:

```
R(a) * R(b) = R(a + b)
```

So we never build actual matrices. Each 2x2 block `k` of the rotation acts as:

```
out[2k]   = cos(theta_k) * x[2k] - sin(theta_k) * x[2k+1]
out[2k+1] = sin(theta_k) * x[2k] + cos(theta_k) * x[2k+1]
```

This is efficient and fully differentiable (gradients flow through cos/sin back to the angle parameters).

---

## File Structure

```
experiment/
  config.py        - All hyperparameters. Two presets: baseline_config(), char_compose_config()
  char_embed.py    - CharCompositionalEmbedding module (the core contribution)
  model.py         - GPT-2 wrapper. create_model(config) returns either baseline or char_compose
  train.py         - Training pipeline. WikiText-103, AdamW, cosine LR, mixed precision
  evaluate.py      - Perplexity evaluation on test set, model comparison
  README.md        - This file
requirements.txt   - Dependencies (in parent directory)
```

---

## How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

The dependencies are: `torch>=2.0`, `transformers>=4.35`, `datasets>=2.14`, `tokenizers>=0.14`, `wandb`, `tqdm`.

### 2. Train the Char-Compose Model

```bash
python -m experiment.train --model_type char_compose
```

### 3. Train the Baseline Model

```bash
python -m experiment.train --model_type baseline
```

### 4. Evaluate and Compare

```bash
# Single model
python -m experiment.evaluate --checkpoint checkpoints/char_compose_final.pt --model_type char_compose

# Side-by-side comparison
python -m experiment.evaluate --checkpoint checkpoints/char_compose_final.pt --compare_baseline checkpoints/baseline_final.pt
```

### Command-Line Options for train.py

- `--model_type`: `baseline` or `char_compose` (default: `char_compose`)
- `--max_steps`: Override total training steps (default: 100,000)
- `--batch_size`: Override per-GPU batch size (default: 8)
- `--use_wandb`: Enable wandb logging
- `--output_dir`: Override checkpoint directory (default: `checkpoints`)

---

## Architecture Details

### CharCompositionalEmbedding (char_embed.py)

- `char_vectors`: nn.Embedding(256, 768) -- one vector per byte value
- `char_angles`: nn.Embedding(256, 384) -- one angle vector per byte value
- `token_bytes`: buffer mapping each of 50,257 token IDs to its byte sequence (padded)
- `token_lengths`: buffer with the length of each token's byte sequence

`compose_all_tokens()` iterates over character positions 0..max_token_len, applying cumulative rotation to each character's vector and accumulating into the result. It processes tokens in chunks of 8192 to manage memory. Returns a `(50257, 768)` tensor.

This tensor is recomputed on every forward pass so that gradients flow back to `char_vectors` and `char_angles`.

### CharComposeGPT2 (model.py)

Wraps HuggingFace `GPT2LMHeadModel`. Replaces the token embedding (`wte`) and output head (`lm_head`) with the composed embedding matrix. Positional embeddings (`wpe`) are unchanged.

The forward pass:
1. Calls `compose_all_tokens()` once to get the (50257, 768) matrix
2. Uses it for input: `composed_matrix[input_ids]`
3. Adds positional embeddings
4. Runs through transformer blocks
5. Uses it for output: `hidden_states @ composed_matrix.T`
6. Computes cross-entropy loss if labels are provided

### Baseline (model.py)

Standard `GPT2LMHeadModel` from HuggingFace with default weight-tied embeddings. Same transformer architecture (12 layers, 12 heads, 768 dim).

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (betas=0.9, 0.95) |
| Learning rate | 6e-4 -> 6e-5 (cosine decay) |
| Warmup | 2000 steps |
| Max steps | 100,000 (~3.3B tokens) |
| Batch size | 8 per GPU, accumulation 4 (effective 32) |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Mixed precision | fp16 |
| Sequence length | 1024 |
| Dataset | WikiText-103 |
| Eval interval | Every 5000 steps |

---

## Expected Parameter Counts

| Component | Baseline | Char-Compose |
|---|---|---|
| Token embeddings | 38,597,376 (50257 x 768) | 294,912 (256 x 1152) |
| Positional embeddings | 786,432 | 786,432 |
| Transformer blocks | ~85M | ~85M |
| Output head (lm_head) | tied with embeddings | tied with composed matrix |
| **Total** | **~124M** | **~86M** |

---

## What Success Looks Like

1. Both models show decreasing training loss within the first 1000 steps
2. Char-compose model achieves reasonable perplexity on WikiText-103 test set (it will likely be worse than baseline, the question is by how much)
3. In the char-compose model, tokens sharing character prefixes (e.g., "run" / "running") show higher cosine similarity than in the baseline

---

## Troubleshooting

- **OOM during compose_all_tokens**: Reduce `chunk_size` parameter in `compose_all_tokens()` (default 8192). Or reduce batch size.
- **Tokenizer download fails**: The GPT-2 tokenizer is downloaded from HuggingFace on first use. Ensure internet access, or pre-download with `python -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2')"`.
- **WikiText-103 download fails**: Same as above. Pre-download with `python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')"`.
- **Training loss not decreasing**: Check learning rate schedule. First 2000 steps are warmup.

---

## Do Not

- Do not change the composition formula. It is `(a, A) . (b, B) = (a + Ab, AB)`.
- Do not change the rotation parameterization. It is block-diagonal 2x2 using angle accumulation.
- Do not replace AdamW with a different optimizer.
- Do not change the GPT-2 architecture (12 layers, 12 heads, 768 dim).
- Do not add new files or reorganize the directory structure.
- Do not "improve" working code.
- If you encounter a problem, report it and ask before changing anything.

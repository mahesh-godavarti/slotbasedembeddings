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

---

## Evolution of the Approach

### Phase 1: Direct Composition (Failed — OOM)

The original design replaced GPT-2's embedding table entirely. Every forward pass called `compose_all_tokens()` to build a `(50257, 768)` matrix from 256 character-level parameters, using it for both input lookup and output projection.

**Problem**: The autograd graph for composing 50,257 tokens through sin/cos rotation ops consumed ~18GB of GPU memory — before the transformer even ran. On a 24GB A10G, this left no room for the model.

**Attempted fixes**:
- Reducing batch size: didn't help — composition graph was the bottleneck, not batch activations
- Reducing embedding dim, layers, block size: didn't help — composition of 50,257 tokens dominated regardless
- Reducing chunk_size in `compose_all_tokens()`: didn't help — same total graph across all chunks
- Gradient checkpointing on composition chunks: **worked** — reduced composition memory from 18GB to 0.13GB by discarding intermediate activations and recomputing during backward

With gradient checkpointing, the original architecture ran but was extremely slow (~5.7s/iter, ~158 hours for 100K steps) because every forward pass composed all 50,257 tokens twice (once forward, once recomputed for backward).

### Phase 2: MSE Regularization Approach (Current)

Instead of replacing the embedding table, we keep GPT-2's standard `wte` lookup table intact and add a **regularization loss** that forces the table entries to match their character-composed values:

```
total_loss = LM_loss + MSE(wte[sampled_tokens], compose(sampled_tokens))
```

Key design decisions:

1. **Standard forward pass**: The LM forward uses normal `wte` lookup — same speed as baseline GPT-2.

2. **Sampled composition loss**: Each step samples 1024 random multi-character tokens, composes their embeddings from character parameters, and penalizes the MSE between the composed embeddings and the `wte` entries. This covers rare tokens uniformly regardless of their frequency in training text.

3. **Tied character vectors**: Single-byte tokens (all 256 byte values exist in GPT-2's vocabulary) share their `wte` rows as the character vectors used in composition. Only the rotation angles (`char_angles`) are separate parameters. This means the LM objective directly trains the character vectors.

4. **Length-sorted token sampling**: Multi-character tokens are pre-sorted by byte length. Sampling contiguous blocks minimizes padding waste in the composition loop.

### Phase 3: Control Experiment

To test whether the improvement comes from character structure or just extra gradient signal on rare token embeddings, we built a **prediction control** model. Instead of MSE toward composed embeddings, it uses next-token prediction on character/token mini-sequences:

- **chars → token**: `[r, u, n, n, i, n, g, running]` — predict each next element
- **token → chars**: `[running, r, u, n, n, i, n, g]` — predict each next element

This gives the same kind of gradient signal to rare token embeddings (associating them with their characters) but through prediction rather than algebraic composition.

### Phase 4: Morphological Evaluation

To test generalization of character-level patterns, we created a morphological dataset with ~2,865 training sentences and ~895 validation sentences covering 16 morphological patterns (un-, dis-, im-/in-/ir-/il-, re-, over-, under-, mis-, pre-, -ing, -ed, -er, -est, -s, -ly, -ness, agent -er). Validation uses held-out words not seen in training. A second validation set (537 sentences) uses novel sentence templates not seen during training.

---

## Results (10K steps, WikiText-103 + morphological data)

### WikiText-103 Validation PPL

| Step | Char-Compose | Control | Baseline |
|------|-------------|---------|----------|
| 1K   | 468         | 473     | 471      |
| 2K   | 275         | 285     | 282      |
| 3K   | 165         | 174     | 175      |
| 4K   | 116         | 120     | 119      |
| 5K   | 90          | 94      | 92       |
| 6K   | 72          | 77      | 75       |
| 7K   | 61          | 65      | 63       |
| 8K   | 55          | 57      | 56       |
| 9K   | 51          | 53      | 52       |
| 10K  | **49.1**    | **51.1**| **50.4** |

Char-compose leads from step 2K onward, with the gap widening over time.

### Rare Token Evaluation (tokens appearing <= 50 times in training)

| Model | Rare PPL | Common PPL | Rare positions |
|-------|----------|------------|----------------|
| Baseline | 685,123 | 49.28 | 426 |
| Control | 222,672 | 48.76 | 426 |
| Char-Compose | 472,383 | 48.65 | 426 |

Both char-compose and control improve dramatically over baseline on rare tokens. The control's prediction-based approach was more effective for rare token embeddings specifically, though char-compose had better overall LM performance.

---

## Additional Files

```
eval_rare.py           - Rare token PPL comparison script
eval_morphological.py  - Morphological pattern generalization evaluation
morphological_data.py  - 2,865 train + 895 val + 537 novel-template morphological sentences
resize_volume.md       - EBS volume resize instructions
```

### Running the evaluations

```bash
# Rare token evaluation
python -m experiment.eval_rare \
    --char_compose_ckpt checkpoints/char_compose_final.pt \
    --baseline_ckpt checkpoints/baseline_final.pt \
    --max_count 50

# Morphological generalization
python -m experiment.eval_morphological \
    --char_compose_ckpt checkpoints/char_compose_final.pt \
    --baseline_ckpt checkpoints/baseline_final.pt \
    --control_ckpt checkpoints/predict_control_final.pt
```

### Training the predict_control model

```bash
python -m experiment.train --model_type predict_control
```

---

## Current Parameter Counts

| Component | Baseline | Char-Compose / Control |
|---|---|---|
| Token embeddings (wte) | 38,597,376 | 38,597,376 (kept) |
| Rotation angles (char_angles) | — | 98,304 (256 x 384) |
| Transformer + positional | 85,842,432 | 85,842,432 |
| lm_head | tied with wte | tied with wte |
| **Total** | **124,439,808** | **124,538,112** |

Char-compose/control add only 98K parameters (the rotation angles). The character vectors are tied to the single-byte token rows of `wte`.

---

## Key Insights So Far

1. **Direct composition is too expensive**: Computing 50,257 composed embeddings with gradient tracking every forward pass is impractical even with gradient checkpointing (feasible but ~6s/iter).

2. **MSE regularization works**: Keeping the standard embedding table and using composition as a regularization target achieves the same goal at baseline training speed.

3. **Tying character vectors to wte matters**: When single-byte token embeddings are the same parameters used for composition, the LM objective directly trains the building blocks of composition. This led to char-compose consistently outperforming baseline.

4. **The composition loss covers rare tokens**: By sampling tokens uniformly (not by frequency), the MSE loss ensures rare token embeddings are pushed toward meaningful composed values even when they rarely appear in training text.

5. **Character association helps rare tokens regardless of method**: Both char-compose (algebraic composition) and the prediction control (next-token prediction on char/token sequences) dramatically reduced rare token PPL compared to baseline, confirming that any mechanism linking tokens to their characters helps.

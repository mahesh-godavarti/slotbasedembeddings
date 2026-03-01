# Mixed MLM + Causal Training

## Landscape: Projects That Mix MLM + Causal

A web search for repositories that have successfully combined MLM (masked language modeling) and causal/autoregressive training in a single model turned up 4 major projects:

### 1. UniLM (Microsoft, 2019) — Attention Mask Switching

- **Paper**: [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)
- **Code**: [github.com/microsoft/unilm](https://github.com/microsoft/unilm)
- **Approach**: Uses a single BERT-style transformer with **dynamic attention masks** to switch between bidirectional (MLM), unidirectional (causal), and seq2seq modes per training example. All weights are shared; only the attention mask and segment IDs change per example.
- **Key innovation**: No separate encoder/decoder — the same transformer layers serve all modes. The attention mask is the *only* mechanism that distinguishes objectives.
- **Why most relevant to our work**: Our Exp 7/8 already does something structurally similar — KG triples use MLM training, text uses causal training, same transformer weights. UniLM shows how to do this within the same batch rather than alternating batches, and adds segment embeddings to signal mode to the model.

### 2. GLM (Tsinghua/THUDM, 2022) — Autoregressive Blank Infilling

- **Paper**: [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://aclanthology.org/2022.acl-long.26/) (ACL 2022)
- **Code**: [github.com/THUDM/GLM](https://github.com/THUDM/GLM)
- **Approach**: Randomly blanks out contiguous spans from the input, then generates them **autoregressively** in an arbitrary order. Uses 2D positional encoding — one dimension for position in the original text, another for position within the generated span.
- **Key innovation**: A creative middle ground between MLM and causal LM. The model sees bidirectional context for the non-blanked tokens, then generates the blanked spans left-to-right. By varying span length, it can mimic BERT-style (short spans) or GPT-style (long spans) behavior.
- **Results**: Outperformed BERT, T5, and GPT at the same model size on NLU tasks. The ChatGLM family (ChatGLM-6B, GLM-130B) is built on this framework.

### 3. UL2 (Google, 2022) — Mixture-of-Denoisers

- **Paper**: [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)
- **Code**: Built on T5x (JAX/FLAX); [HuggingFace discussion](https://github.com/huggingface/transformers/issues/17207)
- **Approach**: Mixes 3 denoising objectives in a single model:
  - **R-denoiser**: Short spans, like MLM (BERT-style)
  - **S-denoiser**: Sequential/causal prefix-to-suffix (GPT-style)
  - **X-denoiser**: Long spans, extreme denoising
- **Key innovation**: Prepends a **mode token** (e.g., `[R]`, `[S]`, `[X]`) to each example so the model knows which objective is active. The model learns to condition its behavior on this token.
- **Relevance**: The mode-token approach is simple and effective — could be adapted for KG vs text signaling.

### 4. AntLM (Dec 2024) — Alternating Objectives

- **Paper**: [AntLM: Bridging Causal and Masked Language Models](https://arxiv.org/abs/2412.03275)
- **Code**: Check paper for links
- **Approach**: Alternates between MLM and causal objectives **across training steps** (not per-example like UniLM, but per-batch). Claims to be the first to truly unify the training *objectives* rather than just the architecture.
- **Key innovation**: Shows that simple alternation between objectives works — no need for complex per-example mixing. Inspired by how children learn through both cloze exercises and writing.

### Why UniLM Is Most Relevant

UniLM is the closest match to our Exp 7/8 setup because:
1. **Same architecture for both modes** — we already do this (same transformer for KG and text)
2. **Attention mask is the control mechanism** — our KG models (A, D, F, G) already use MLM-style masking while text models use causal masking
3. **Segment embeddings signal structure** — analogous to our slot embeddings (models A/G) or relation tokens
4. **Clean, readable PyTorch implementation** — the `unilm-v1/` codebase is straightforward to study

The rest of this document is a detailed study of UniLM's implementation.

---

## UniLM Deep Dive

Notes from studying [Microsoft UniLM](https://github.com/microsoft/unilm) (`unilm-v1/`).
All file references are relative to the cloned repo at `unilm_study/`.

### Core Idea

UniLM trains a **single transformer** on three objectives simultaneously by switching the **attention mask** per training example:

| Task | task_idx | Attention Pattern | Use Case |
|------|----------|-------------------|----------|
| MLM (bidirectional) | 0 | All-to-all | BERT-style pre-training |
| Causal L→R | 1 | Lower triangular | GPT-style pre-training |
| Causal R→L | 2 | Upper triangular | Reverse LM |
| Seq2Seq | 3 | Source=bidirectional, Target=causal | Encoder-decoder |

Everything else — weights, embeddings, FFN, loss function — is shared.

## Attention Mask Construction

### Data Preprocessing (3D mask per example)

**File**: `unilm-v1/src/biunilm/seq2seq_loader.py:119-120, 269-278`

During data loading, each example gets a `(max_len, max_len)` attention mask matrix:

```python
self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))

input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

if self.mode == "s2s":
    # Source (tokens_a) has bidirectional attention
    input_mask[:, :len(tokens_a)+2].fill_(1)
    # Target (tokens_b) has causal attention
    second_st, second_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3
    input_mask[second_st:second_end, second_st:second_end].copy_(
        self._tril_matrix[:second_end-second_st, :second_end-second_st])
else:  # l2r (causal) mode
    st, end = 0, len(tokens_a) + len(tokens_b) + 3
    input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])
```

### Dynamic Construction in Forward Pass

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:1213-1252`

When `token_type_ids` and `attention_mask` are not provided in the batch, the model builds 4 task-specific masks on-the-fly and selects based on `task_idx`:

```python
task_0 = (task_idx == 0)  # MLM
task_1 = (task_idx == 1)  # Causal L→R
task_2 = (task_idx == 2)  # Causal R→L
task_3 = (task_idx == 3)  # Seq2Seq

index_matrix = torch.arange(sequence_length).view(1, sequence_length)
index_matrix_t = index_matrix.view(1, sequence_length, 1)
tril = index_matrix <= index_matrix_t

# Task 0 (MLM): all tokens see all valid tokens
attention_mask_task_0 = (index_matrix < num_tokens) & (index_matrix_t < num_tokens)

# Task 1 (Causal): can only see current and past
attention_mask_task_1 = tril & attention_mask_task_0

# Task 2 (Reverse causal): can only see current and future
attention_mask_task_2 = torch.transpose(tril, dim0=-2, dim1=-1) & attention_mask_task_0

# Task 3 (Seq2Seq): source bidirectional + target causal
attention_mask_task_3 = (
    (index_matrix < num_tokens_a) | tril
) & attention_mask_task_0

# Select per-example based on task_idx
attention_mask = (
    (attention_mask_task_0 & task_0.view(-1, 1, 1)) |
    (attention_mask_task_1 & task_1.view(-1, 1, 1)) |
    (attention_mask_task_2 & task_2.view(-1, 1, 1)) |
    (attention_mask_task_3 & task_3.view(-1, 1, 1))
)
```

### Conversion to Numerical Mask

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:1006-1032`

The boolean mask is converted to floats for use in softmax:

```python
# Shape: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
extended_attention_mask = attention_mask.unsqueeze(1)

# 1 -> 0.0 (allow attention), 0 -> -10000.0 (mask out)
extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
```

The `-10000` gets added to attention scores before softmax, effectively zeroing out masked positions.

## Segment IDs (token_type_ids)

Segment IDs serve as a **soft structural signal** — they tell the model "what role does this token play" via learned embeddings. The hard structural enforcement (who can attend to whom) comes from the attention mask.

### Role 1: Input Embedding (always active)

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:238-248`

Segment IDs are looked up in a learned embedding table and added to every token's input representation:

```python
words_embeddings = self.word_embeddings(input_ids)
position_embeddings = self.position_embeddings(position_ids)
token_type_embeddings = self.token_type_embeddings(token_type_ids)

embeddings = words_embeddings + position_embeddings + token_type_embeddings
```

### Segment ID Values per Mode

**File**: `unilm-v1/src/biunilm/seq2seq_loader.py:158-173`

With `new_segment_ids=True`, UniLM uses 7 distinct segment IDs:

| Mode | Segment IDs | Meaning |
|------|------------|---------|
| **MLM** (task 0) | `0` for segment A, `1` for segment B | Standard BERT two-segment |
| **Causal L→R** (task 1) | All `2` | Uniform — no segment distinction |
| **Causal R→L** (task 2) | All `3` | Uniform — no segment distinction |
| **Seq2Seq** (task 3) | `4` for [CLS], `6` for source content, `5` for target content | 3 distinct roles |

Without `new_segment_ids` (original BERT style), it's just `0` for segment A and `1` for segment B.

### Initialization from BERT

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:793-803`

When loading from a BERT checkpoint (which only has 2 segment embeddings), the new segment embeddings are initialized by copying:

| New ID | Initialized from | Role |
|--------|-----------------|------|
| 0, 1 | Original BERT | MLM segment A/B |
| 2 | Copy of ID 0 | Causal L→R |
| 3 | Copy of ID 0 | Causal R→L |
| 4 | Copy of ID 0 | Seq2Seq [CLS] |
| 5 | Copy of ID 1 | Seq2Seq target |
| 6 | Copy of ID 1 | Seq2Seq source |

### Dynamic Construction in Forward Pass

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:1230-1233`

When not provided in the batch, segment IDs are computed from `task_idx`:

```python
token_type_ids = (task_idx + 1 + task_3) * base_mask
token_type_ids = token_type_ids - segment_a_mask * (task_0 | task_3)
```

This maps: task 0 → IDs `(0, 1)`, task 1 → all `2`s, task 2 → all `3`s, task 3 → IDs `(4, 5)`.

### Role 2: Attention Bias (optional, `seg_emb=True`)

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:287-294, 341-348`

When the `seg_emb` config is enabled, segment IDs also influence attention scores directly inside each attention layer:

```python
# In __init__:
self.b_q_s = nn.Parameter(torch.zeros(1, num_heads, 1, head_size))
self.seg_emb = nn.Embedding(type_vocab_size, all_head_size)

# In forward:
seg_rep = self.seg_emb(seg_ids)                           # (batch, pos, all_head_size)
seg_rep = seg_rep.view(B, T, n_heads, head_size)
qs = einsum('bnih,bjnh->bnij', query + self.b_q_s, seg_rep)
attention_scores = attention_scores + qs                   # added to QK^T scores
```

This computes `(query + bias) · seg_embedding[j]` for each key position `j`, adding a segment-dependent bias to the attention weights. Each head learns "how much should I modulate attention based on the segment type of the key position" — a **relative segment attention bias** on top of the mask.

## Training Loop: How Objectives Are Mixed

### Loss Computation

**File**: `unilm-v1/src/biunilm/run_seq2seq.py:429-448`

Both objectives are computed on **every example** — no alternation:

```python
input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, \
    masked_pos, masked_weights, is_next, task_idx = batch

loss_tuple = model(
    input_ids, segment_ids, input_mask, lm_label_ids, is_next,
    masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx,
    ...)

masked_lm_loss, next_sentence_loss = loss_tuple

# Simple addition — no weighting
loss = masked_lm_loss + next_sentence_loss
```

### MLM Loss with Position-Based Masking

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:1288-1307`

Only specific positions contribute to the loss, controlled by `masked_pos` and `masked_weights`:

```python
# Gather hidden states at masked positions
sequence_output_masked = gather_seq_out_by_pos(sequence_output, masked_pos)
prediction_scores_masked, seq_relationship_score = self.cls(
    sequence_output_masked, pooled_output, task_idx=task_idx)

# Cross-entropy at masked positions
masked_lm_loss = self.crit_mask_lm(
    prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)

# Normalize by mask weights (only count non-zero weights)
def loss_mask_and_normalize(loss, mask):
    mask = mask.type_as(loss)
    loss = loss * mask
    denominator = torch.sum(mask) + 1e-5
    return (loss / denominator).sum()

masked_lm_loss = loss_mask_and_normalize(masked_lm_loss.float(), masked_weights)
```

In seq2seq mode, only **target tokens** are masked (`seq2seq_loader.py:192-195`):

```python
for i, tk in enumerate(tokens):
    if (i >= len(tokens_a)+2) and (tk != '[CLS]'):  # Only target tokens
        cand_pos.append(i)
```

## Model Architecture: What's Shared

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:1175-1206`

```python
class BertForPreTrainingLossMask(PreTrainedBertModel):
    def __init__(self, config, ...):
        self.bert = BertModel(config)                    # SHARED backbone
        self.cls = BertPreTrainingHeads(config, ...)     # SHARED output head
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
```

| Component | Shared? | Notes |
|-----------|---------|-------|
| Transformer layers | Yes | All layers, all modes |
| Word embeddings | Yes | Same vocab |
| Position embeddings | Optionally per-task | `new_pos_ids` allows 4 separate position embeddings |
| Segment embeddings | Per-mode | 7 learned embeddings (IDs 0-6) |
| Output projection head | Yes | Same `BertPreTrainingHeads` |
| Loss functions | Yes | Same cross-entropy |

The key insight: **no separate encoder or decoder**. The same transformer weights serve all modes — only the attention mask and segment IDs change.

## Inference: Seq2Seq Decoding

**File**: `unilm-v1/src/pytorch_pretrained_bert/modeling.py:1370-1424`

For generation, UniLM uses incremental decoding with KV caching (`BertModelIncr`):

```python
class BertForSeq2SeqDecoder(PreTrainedBertModel):
    def __init__(self, config, mask_word_id=0, mode="s2s", ...):
        self.bert = BertModelIncr(config)  # Incremental (caching) decoder
        self.cls = BertPreTrainingHeads(config, ...)
```

The entire sequence (source + partial target) is input. Source positions attend bidirectionally, target positions have causal attention, and only the last position's prediction is used for the next token.

## Relevance to Exp 7/8 (KG + Text)

See also the [Landscape section](#landscape-projects-that-mix-mlm--causal) at the top for GLM, UL2, and AntLM.

Our setup already does something similar — KG triples use MLM, text uses causal, same transformer weights. Key differences:

| Aspect | UniLM | Exp 7/8 |
|--------|-------|---------|
| Mode switching | Per example (via attention mask) | Per data stream (KG vs text batches) |
| Loss combination | Both computed every step, summed | Alternating KG and text batches |
| Structural signal | Segment IDs (7 learned embeddings) | Slot embeddings (models A/G) or relation tokens |
| Architecture | Single BERT, same weights for all modes | Same — single transformer for both modalities |
| Position encoding | Standard learned + optional per-task | RoPE / learned angles depending on model variant |

Potential ideas to borrow:
1. **Per-example mode switching** instead of alternating batches — mix KG and text within the same batch
2. **Segment embeddings** to signal "this is KG data" vs "this is text data" to the model
3. **Attention-level segment bias** (`seg_emb`) to let the model learn cross-modal attention patterns

## Applying Both Objectives to Both Modalities

The key insight from UniLM/AntLM is that MLM and causal are applied to the **same data** — the two objectives extract different knowledge from the same samples. In Exp 7/8, we currently tie objective to modality (KG gets MLM, text gets causal). Could we apply both objectives to both modalities?

### MLM on Text — Straightforward

Apply random token masking to wiki/text sentences and train the model to predict the masked tokens. This is standard BERT-style training and requires no structural changes. The model already processes text; we just add a second loss on the same data.

### Causal on KG — The Problem

Slot-based KG (models A/G) uses `[HEAD] [REL] [TAIL]` with dedicated slot embeddings. There is no natural left-to-right ordering to impose causal masking on — the slots are semantic roles, not a sequence.

Standard autoregressive causal training doesn't apply: predicting REL after HEAD is not the same kind of sequential dependency as predicting the next word in a sentence.

### Structured Slot Masking — A KG-Native Alternative

Instead of forcing left-to-right causal order, cycle through different **structured masking patterns** on the same triple:

| Pattern | Visible | Predict | Analogy |
|---------|---------|---------|---------|
| 1. Predict TAIL | `HEAD REL [MASK]` | TAIL | "Who is Adam's son?" |
| 2. Predict HEAD | `[MASK] REL TAIL` | HEAD | "Who is Brian's father?" |
| 3. Predict REL | `HEAD [MASK] TAIL` | REL | "What is the relationship between Adam and Brian?" |
| 4. Random MLM | Random subset masked | Masked slots | Standard MLM |

Patterns 1-3 are the KG equivalent of causal prediction: "given these known facts, predict the missing piece." Each pattern forces the model to reason in a different direction about the same triple. Pattern 4 is standard MLM as currently used.

This is closer to **GLM's blank infilling** than to UniLM's approach — GLM blanks out spans and generates them autoregressively; here we blank out specific *slots* with semantic meaning.

### Linearized KG — Another Option

For linearized/flat KG formats (models B/C/D/F), both objectives apply naturally:

- **Causal**: `"Adam" → "<son_of>" → "Brian"` — standard left-to-right prediction
- **MLM**: `"Adam" "<son_of>" [MASK]` — random masking

The `--kg_as_text` mode already does the causal version of this. Adding MLM on top would mean the same linearized triple is trained with both objectives.

Both directions can also be used as separate causal examples:
- Forward: `"Adam <son_of> Brian"`
- Inverse: `"Brian <inverse_son_of> Adam"`

This is already supported via the `--inverse_kg` flag.

### Training Schedule

AntLM showed that the **alternation schedule barely matters** — per-epoch switching, per-batch switching, and rapid alternation all give similar results. This suggests the simplest approach is best:

- **Per-triple random pattern**: For each KG triple in a batch, randomly sample one of the 4 masking patterns. Over many batches, the model sees all patterns for all triples.
- **No careful scheduling needed**: AntLM's ablations show the model is robust to how you mix objectives.

### Summary: Both Objectives on Both Modalities

| Modality | MLM | Causal-like |
|----------|-----|-------------|
| **Text** | Random token masking (standard BERT) | Next-token prediction (already done) |
| **KG (slotted)** | Random slot masking (already done) | Structured slot masking: predict TAIL, HEAD, or REL with others visible |
| **KG (linearized)** | Random token masking | Left-to-right prediction (already done via `--kg_as_text`) |

# Tutorial: Measuring Cross-Pollination Between Knowledge Graphs and Text

## The Big Question

Can training a language model on knowledge graph (KG) data improve its text predictions, or vice versa? We train transformers on both Wikipedia text and structured KG triples like `(deception, frame_related, dissembler)`, then test whether knowledge transfers between the two domains.

This tutorial tells the story as it actually happened — a sequence of attempts, wrong assumptions, and corrections.

---

## Background

**Text data**: Full English Wikipedia (~2.5 billion tokens), character-level tokenizer (vocab ~14,500).

**KG data**: ~734,000 triples from WordNet, FrameNet, BATS, and Google word analogies. Each triple: `(head_entity, relation, tail_entity)`.

**Two ways to feed KG to the model**:
1. **Angle-gap**: A learned relation-specific angle is injected into rotary position encoding between head and tail. Structurally changes how attention works, but doesn't change the token stream.
2. **Text-linearized**: Triples become plain text (`"deception frame related dissembler ."`) and are trained with standard next-token prediction alongside wiki text.

---

## Attempt 1: Just compare PPL

The first thing we did was train three models for 100K iterations and compare test PPL:

| Model | KG Method | Test PPL |
|-------|-----------|----------|
| roformer | none | **7.86** |
| roformer_kg | angle-gap | 8.74 |
| roformer_text_kg | text-linearized | 8.43 |

Both KG methods **hurt** text PPL. The plain baseline wins.

At this point we asked: is the PPL comparison even meaningful for measuring cross-pollination? Test PPL averages over all tokens — mostly common words like "the", "of", "and". If KG training helps the model predict "dissembler" better but slightly hurts "the", the improvement is invisible in the average.

We needed a more targeted evaluation.

---

## Attempt 2: Check vocabulary overlap first

**Script**: `kg_vocab_overlap.py`

Before building anything elaborate, we checked a basic question: do KG entities and wiki text even share vocabulary?

```python
def load_wiki_words(wiki_path, max_lines=None):
    """Load wiki text and extract word-level vocabulary with counts."""
    word_counts = Counter()
    with open(wiki_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                words = re.findall(r'[a-zA-Z]+', line.lower())
                word_counts.update(words)
    return word_counts
```

Results:
```
KG words in wiki:      64,128 / 70,826 (90.6%)
Wiki token coverage by KG words: 85.4%
Wiki words NOT in KG: 1,102,271 / 1,170,211 (94.2%)
```

90% of KG words appear in wiki. Good — there's overlap to exploit. But 94% of wiki's vocabulary is NOT in KG. The KG covers common words densely, rare words sparsely.

---

## Attempt 3: Build cross-pollination eval sets

**Script**: `build_cross_eval.py`

We designed two evaluation sets:

### KG→Text: "Does KG training help predict rare KG words in wiki text?"

Find wiki sentences containing words that are in KG entities but rare in wiki. If KG training helps, the model should predict these rare words better.

```python
def find_rare_kg_words(kg_triples, wiki_word_counts, max_freq=5):
    """Find words that are in KG entities but rare (<=max_freq) in wiki."""
    kg_words = set()
    for h, r, t in kg_triples:
        kg_words.update(re.findall(r'[a-zA-Z]+', h.lower()))
        kg_words.update(re.findall(r'[a-zA-Z]+', t.lower()))

    rare = set()
    for w in kg_words:
        if wiki_word_counts.get(w, 0) <= max_freq:
            rare.add(w)
    return rare, kg_words


def extract_rare_kg_sentences(wiki_path, rare_words, max_lines=None,
                               max_sentences=50000):
    """Extract wiki lines containing at least one rare KG word."""
    sentences = []
    rare_set = set(w for w in rare_words if len(w) >= 3)

    with open(wiki_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(sentences) >= max_sentences:
                break
            stripped = line.strip()
            if not stripped or len(stripped) < 20:
                continue
            line_words = set(re.findall(r'[a-zA-Z]+', stripped.lower()))
            if line_words & rare_set:
                sentences.append(stripped)
    return sentences
```

This produced 50,000 sentences (17.7M tokens) saved as `cross_text_eval.bin`.

### Text→KG (v1): Programmatically generate novel triples

For the other direction, we wrote code to generate *new* morphological triples from wiki words:

```python
rules = [
    ('plural_of', '', 's', lambda w: len(w) > 3),
    ('verb_inf___ved', '', 'ed', lambda w: len(w) > 3),
    ('verb_inf___ving', '', 'ing', lambda w: len(w) > 3),
    ('adj___comparative', '', 'er', lambda w: len(w) > 3),
    # ...
]
```

For each rule, find word pairs in wiki vocabulary that match the pattern but aren't already in the KG. Example: if "compute" and "computing" are both common in wiki, and `(compute, verb_inf___ving, computing)` isn't in the KG, add it as a test triple.

This generated 38,817 novel triples. Felt clever. **But it was the wrong approach** — we'd realize this later.

---

## Attempt 4: Run the cross-eval

We added the cross-eval to `kg_text_experiment.py` and ran all three models. Results:

| Model | KG Method | Test PPL | Rare-KG PPL |
|-------|-----------|----------|-------------|
| roformer | none | **7.74** | **8.45** |
| roformer_kg | angle-gap | 8.86 | 9.03 |
| roformer_text_kg | text-linearized | 8.44 | 8.70 |

Still no evidence of cross-pollination. The baseline wins on Rare-KG PPL too.

**But wait** — Rare-KG PPL is computed over *entire sentences*. A sentence like "The city of Abbottabad is located in northern Pakistan" has one rare KG word ("Abbottabad") and dozens of common ones. The PPL is averaged over all tokens, so the common words dominate. The rare word's contribution is noise in the average.

The evaluation wasn't measuring what we wanted.

---

## Attempt 5: Filter existing KG test triples instead of generating new ones

Meanwhile, we also realized the programmatic Text→KG triples were a bad idea. The generated triples used morphological relations that might not match the KG's training relations. And we were testing pattern matching, not knowledge transfer.

Much simpler: just take the existing KG test set (already held out with a 90/10 split) and filter it for triples where the entity words are **common** in wiki text. If text training helps KG prediction, the model should do better on triples about words it saw often in wiki.

```python
def _filter_kg_triples_by_wiki_freq(kg_test_triples, tokenizer, wiki_word_freq,
                                     min_freq=100):
    """Filter KG test triples for those where entity words are common in wiki."""
    filtered = []
    for triple in kg_test_triples:
        head_words = _decode_entity(tokenizer, triple['head_ids'])
        tail_words = _decode_entity(tokenizer, triple['tail_ids'])
        all_words = head_words | tail_words
        if not all_words:
            continue
        if all(wiki_word_freq.get(w, 0) >= min_freq for w in all_words):
            filtered.append(triple)
    return filtered
```

No new data files needed. No programmatic generation. Just a filter on data we already had.

### Oh wait — the tokenizer is character-level

When we first wrote `_decode_entity`, it looked like this:

```python
def _decode_entity(tokenizer, token_ids):
    text = tokenizer.decode(token_ids)
    return set(re.findall(r'[a-zA-Z]+', text.lower()))
```

We ran it and got zero rare words. Everything was "common." That can't be right.

Debugging revealed the problem:

```python
>>> tok.decode(triple['head_ids'])
'd e c e p t i o n'    # spaces between every character!
```

The tokenizer is character-level. When you decode `[27, 28, 26, ...]` (the character IDs for d, e, c, ...), it inserts spaces between each character. So `re.findall(r'[a-zA-Z]+', ...)` extracts `["d", "e", "c", "e", "p", "t", "i", "o", "n"]` — individual characters. And single characters like "e" have wiki frequency of 425,054. So nothing looked rare.

Fix: strip spaces before extracting words.

```python
def _decode_entity(tokenizer, token_ids):
    text = tokenizer.decode(token_ids).replace(' ', '')  # <-- the fix
    return set(re.findall(r'[a-zA-Z]+', text.lower()))
```

Now `"d e c e p t i o n"` becomes `"deception"`, which correctly gets extracted as a single word.

---

## Attempt 6: Mask the loss to rare words only

Back to the main problem: Rare-KG PPL was averaging over whole sentences. We needed to measure loss **only on the rare KG word tokens**.

With a BPE tokenizer, you'd look for token IDs that map to rare words. But our tokenizer is character-level — "deception" is 9 separate tokens `[d, e, c, e, p, t, i, o, n]`. There are no "word tokens" to mask.

We needed to reconstruct words from the token stream. The wiki data looks like this when decoded:

```
goalsalexrestaurantdistrictsandstreetsinengland...
```

No spaces — just a continuous stream of characters. Word boundaries only appear where non-alphabetic characters show up (punctuation, digits, etc.).

The approach:
1. Build a mapping from token ID → character (ASCII a-z only)
2. Scan the target token sequence for runs of consecutive alphabetic characters
3. Reconstruct each word and check if it's in the rare KG word set
4. Mark those positions in a boolean mask

```python
def _build_word_mask(targets, id_to_char, rare_words):
    """Build a boolean mask: True at positions belonging to rare KG words."""
    B, T = targets.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=targets.device)
    targets_cpu = targets.cpu().numpy()

    for b in range(B):
        i = 0
        while i < T:
            tid = int(targets_cpu[b, i])
            if tid in id_to_char:
                # Start of a word — collect all alpha chars
                word_start = i
                chars = []
                while i < T and int(targets_cpu[b, i]) in id_to_char:
                    chars.append(id_to_char[int(targets_cpu[b, i])])
                    i += 1
                word = ''.join(chars)
                if word in rare_words:
                    mask[b, word_start:i] = True
            else:
                i += 1
    return mask
```

### Oh wait — `isalpha()` matches CJK characters

First version of `id_to_char`:

```python
for ch, tid in vocab.items():
    if len(ch) == 1 and ch.isalpha():
        id_to_char[tid] = ch.lower()
```

This matched 12,089 tokens — because the vocabulary includes thousands of CJK, Arabic, Cyrillic, etc. characters, and Python's `str.isalpha()` returns True for all of them. A Chinese character would be treated as part of an English word, corrupting the word reconstruction.

Fix: restrict to ASCII.

```python
for ch, tid in vocab.items():
    if len(ch) == 1 and ch.isascii() and ch.isalpha():
        id_to_char[tid] = ch.lower()
```

Now 26 entries — exactly a-z.

### The masked evaluation

With the mask built, the evaluation computes per-token loss and sums only the masked positions:

```python
@torch.no_grad()
def evaluate_cross(model, block_size, batch_size, device, eval_iters=50,
                   is_kg_model=False, cross_text_data=None,
                   cross_kg_dataset=None,
                   rare_kg_words=None, id_to_char=None):
    model.eval()
    out = {}

    if (cross_text_data is not None and len(cross_text_data) > block_size
            and rare_kg_words is not None and id_to_char is not None):
        total_loss = 0.0
        total_tokens = 0
        for k in range(eval_iters):
            X, Y = get_batch(cross_text_data, block_size, batch_size, device)
            if is_kg_model:
                logits, _ = model.forward_text(X, Y)
            else:
                logits, _ = model(X, Y)

            # Per-token loss (no reduction — this is the key)
            B, T, V = logits.shape
            per_token_loss = F.cross_entropy(
                logits.view(B * T, V), Y.view(B * T), reduction='none'
            ).view(B, T)

            # Only count rare KG word positions
            word_mask = _build_word_mask(Y, id_to_char, rare_kg_words)
            n_masked = word_mask.sum().item()
            if n_masked > 0:
                masked_loss = (per_token_loss * word_mask.float()).sum().item()
                total_loss += masked_loss
                total_tokens += n_masked

        if total_tokens > 0:
            out['cross_text'] = total_loss / total_tokens

    # Text→KG: held-out KG triples with common wiki words
    if is_kg_model and cross_kg_dataset is not None:
        kg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            tok, tgt, hlens, rels, neg = cross_kg_dataset.get_causal_batch(
                batch_size, device)
            _, loss = model.forward_kg_causal(tok, tgt, hlens, rels, neg)
            kg_losses[k] = loss.item()
        out['cross_kg'] = kg_losses.mean().item()

    model.train()
    return out
```

The crucial detail is `reduction='none'` in `F.cross_entropy`. Normally this function returns the mean loss over all positions. With `reduction='none'`, you get the loss at every single position, and then you choose which positions to average over.

---

## The model itself (for completeness)

**Script**: `kg_text_experiment.py`

The experiment needed a `RoFormerKG` model — a standard RoFormer (rotates Q,K only) extended with the angle-gap KG mechanism. The existing codebase had `JoFormerFixedKG` (rotates Q,K,V) but not a RoPE-only variant.

```python
class RoFormerKGAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        max_len = block_size * 2
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x, angles=None):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        if angles is None:
            angle1 = torch.arange(T, device=x.device)
            angle2 = torch.arange(C // 2, device=x.device)
            angle = torch.outer(angle1, angle2).unsqueeze(0)
            angle = torch.flip(angle, dims=(1,))
        else:
            angle = angles

        matrix = build_rotation_matrix(torch.cos(angle), torch.sin(angle))
        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)
        # V is NOT rotated — this is what makes it RoFormer, not JoFormer

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        out = self.proj(out)
        out = self.dropout(out)
        return out
```

The model has two forward paths:
- `forward_text(idx, targets)` — normal causal LM, `angles=None` triggers fixed RoPE
- `forward_kg_causal(token_ids, targets, head_lens, rel_names, negate_angles)` — computes custom angles with the relation gap injected

The angle-gap works by building a `(T+1)` length angle sequence — head positions get standard frequencies, then a learned relation angle, then tail positions — and doing `flip → cumsum → flip` over this extended sequence. The relation position is then removed, leaving `T` positions where the cumsum has "absorbed" the relation angle.

### Bug: tril buffer too small

First run crashed:
```
RuntimeError: The size of tensor a (64) must match the size of tensor b (72)
```

KG triples concatenate head + tail tokens. With block_size=64, some concatenated sequences were 72 tokens long. The causal mask `tril` was only `(64, 64)`. Fix: allocate `block_size * 2`.

### Text-linearized KG: just make it text

The text-linearized approach is much simpler. Convert triples to sentences and feed them through the normal model:

```python
def linearize_kg_to_text(triples):
    sentences = []
    for head, rel, tail in triples:
        rel_text = rel.replace('_', ' ')
        sentences.append(f"{head} {rel_text} {tail} .")
    return sentences
```

`(deception, frame_related, dissembler)` becomes `"deception frame related dissembler ."`. No special architecture needed. These are tokenized and written to a binary file, then sampled as random windows during training — exactly like wiki text. The training loss is: `text_loss + kg_weight * kg_text_loss`.

---

## Summary: what went wrong at each step

| Step | What we tried | What went wrong | The fix |
|------|--------------|-----------------|---------|
| 1 | Compare overall test PPL | Common words dominate; can't see cross-pollination signal | Need targeted eval |
| 2 | Build eval set of wiki sentences with rare KG words | Good idea, but... | See step 4 |
| 3 | Generate novel morphological triples programmatically | Artificial; relations might not match training set | Use real held-out KG triples instead |
| 4 | Compute PPL over full rare-KG-word sentences | Common words in those sentences still dominate (~95% of tokens) | Mask loss to rare word tokens only |
| 5 | Decode KG entities to extract words | Character-level tokenizer inserts spaces: `"d e c e p t i o n"` → extracts single chars | `.replace(' ', '')` before regex |
| 6 | Build id_to_char mapping with `ch.isalpha()` | Matches 12,089 CJK/Unicode chars, corrupts word reconstruction | Use `ch.isascii() and ch.isalpha()` → 26 chars |
| 7 | Filter KG test triples for common wiki words | Uses `_decode_entity` which had the same space bug | Same `.replace(' ', '')` fix |

---

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `kg_vocab_overlap.py` | Check word overlap between KG and wiki | Standalone analysis; run once |
| `build_cross_eval.py` | Build rare-KG-word sentence eval set | Generates `cross_text_eval.bin`; the programmatic triple generation in this script is no longer used |
| `kg_text_experiment.py` | Main experiment: models, training, all evaluation | Has the masked rare-word eval, filtered KG test triples, everything |

### Data flow

```
wiki.en.txt ──→ wiki_tokens.bin (2.5B tokens, char-level)
                    ├── 80% train
                    ├── 10% val
                    └── 10% test

KG sources ──→ kg_triples.pkl (734K encoded triples)
(WordNet,       ├── 90% train
 FrameNet,      └── 10% test ──→ KG test eval
 BATS,                        └──→ filter by wiki freq ──→ Text→KG cross-eval
 Google)
            ──→ kg_text_tokens.bin (20.9M tokens, linearized)
                    ├── 90% train
                    └── 10% test

wiki.en.txt ──→ cross_text_eval.bin (17.7M tokens)
  (sentences      → KG→Text cross-eval
   with rare       (loss masked to rare KG word tokens only)
   KG words)
```

### Usage
```bash
# Vocabulary overlap analysis
python kg_vocab_overlap.py

# Build the rare-KG-word sentence eval set
python build_cross_eval.py

# Run the experiment (smoke test)
python kg_text_experiment.py --smoke --models roformer roformer_kg roformer_text_kg

# Full run
nohup python kg_text_experiment.py \
  --models roformer roformer_kg roformer_text_kg \
  --n_embed 100 --n_layers 2 --max_iters 100000 \
  > kg_text_exp.log 2>&1 &
```

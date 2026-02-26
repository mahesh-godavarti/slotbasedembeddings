# Toy Experiments Plan

Based on the JoFormer codebase: https://github.com/mahesh-godavarti/joformer

Existing code:
- `roformer.py` — RoPE baseline
- `journey_transformer_fixed_angles.py` — toral/commutative journey operators
- `journey_transformer_per_token_angles.py` — content-dependent (per-token) operators

All experiments are CPU-friendly and runnable on a Mac.

---

## Exp 1: Group Language Recognition (validates thm:dfa + thm:comm-dfa)

**Theorem validated**: Journey aggregation simulates permutation DFAs; commutative aggregation cannot recognize order-sensitive languages.

**Task**: Classify strings over {a,b}:
- Task A: {w : #a mod 3 = 0} (order-insensitive, Parikh-closed)
- Task B: {w : first occurrence of "ab" precedes first "ba"} (order-sensitive)

**Models**:
- Journey: extend `journey_transformer_per_token_angles.py` with per-symbol operators and linear readout
- Commutative baseline: same but force operators to commute (diagonal matrices)

**Expected result**: Both solve Task A. Only journey solves Task B.

**Scale**: Strings length 10-50, ~5K training examples, d=16, single layer.

---

## Exp 2: RoPE vs Journey on Shuffled KB (validates thm:rope-impossibility)

**Theorem validated**: Slot-based RoPE positions are incompatible with permutation-invariant repositories.

**Task**: Key-value retrieval from a knowledge base. At each forward pass, KB entry order is randomly shuffled.

**Models**:
- RoPE: `roformer.py` with positional encoding on KB entries
- Journey: `journey_transformer_per_token_angles.py` with content-computed operators
- Position-free: no positional encoding (ablation)

**Expected result**: RoPE accuracy degrades with shuffling. Journey is permutation-stable. Position-free works but is less expressive (can't distinguish entries with identical content).

**Scale**: M=20-100 KB entries, d=32, single-layer attention.

---

## Exp 3: Commutative vs Non-Commutative Value Aggregation (validates thm:info-loss + lem:injectivity)

**Theorem validated**: Commutative aggregation carries 0 bits of order information; non-commutative carries Theta(N) bits.

**Task**: Given a sequence of colored tokens in specific positions, answer:
- "What color is at position k?" (order-sensitive)
- "How many red tokens?" (order-insensitive)

**Models**:
- Journey value aggregation: non-commutative rotational aggregation from `journey_transformer_fixed_angles.py`
- Commutative baseline: sum/mean pooling

**Expected result**: Both solve counting. Only journey solves positional retrieval. Empirically measure mutual information about ordering.

**Scale**: N=8-20 tokens, 2-4 colors, d=4, ~1K examples.

---

## Exp 4: Operator Recovery (validates thm:toral)

**Theorem validated**: Toral classification — learned operators converge to block-diagonal rotations under symmetry pressure.

**Task**: Train `journey_transformer_fixed_angles.py` on character-level LM (Tiny Shakespeare, already in the repo) but initialize R_s as unconstrained d x d matrices instead of block-diagonal rotations.

**Metrics** (after training):
- Off-diagonal 2x2 block Frobenius norms (should -> 0)
- Singular value spread within diagonal blocks (should -> 1, i.e., orthogonal)
- Angle extraction from learned blocks vs expected linear progression

**Expected result**: Operators converge to approximately block-diagonal rotations even without structural constraints.

**Scale**: d=8, vocab=65 (Tiny Shakespeare chars), seq_len=32, ~10K steps.

---

## Exp 5: SSM Equivalence (validates prop:ssm-bridge)

**Theorem validated**: Fixed-R journey aggregation equals a linear SSM recurrence up to R^{N-1}.

**Task**: No training. Fix a rotation matrix R. Run both:
1. Journey aggregation: sum_t alpha_t R^{-(t-1)} v_t
2. SSM recurrence: h_t = R h_{t-1} + alpha_t v_t

Compare h_N vs R^{N-1} * (journey output).

**Expected result**: Cosine similarity = 1.0 (exact algebraic equivalence).

**Scale**: d=4, sequences of length 20, purely computational.

---

## Exp 6: Single Journey Head vs d Additive Heads (validates thm:coupling + prop:coupling-rank)

**Theorem validated**: Cross-structure journey scores have rank d; any separable decomposition needs >= d terms.

**Task**: Cross-structure retrieval. Query at text position t asks for entity playing role s in KG triple e. Correct answer depends jointly on (t, s, e).

**Models**:
- 1 journey head: single head from `journey_transformer_per_token_angles.py`
- H additive heads: H separate heads (position-only + relation-only), scores summed. Vary H from 1 to d.

**Expected result**: 1 journey head matches d additive heads. Fewer than d additive heads show measurable accuracy drop.

**Scale**: d=8, 10 relations, T=20 positions, ~2K examples.

---

## Exp 7: Native KG+Text vs Linearized KG-as-Text (validates prop:lis-union + thm:coupling + thm:baked-key)

**Core claim**: Journey attention with native KG role operators outperforms the standard practice of linearizing KG triples into text sequences.

**Theorems validated**: prop:lis-union (heterogeneous union), thm:coupling (multiplicative coupling advantage), thm:baked-key (toral caching)

### Model Taxonomy

Six models organized by two axes:
1. **Angle computation**: RoPE positional (A/B) vs per-token cumsum (C)
2. **Value aggregation**: Commutative (unprimed) vs operator-based/non-commutative (primed)

| Model | Angle computation | KG handling | Value aggregation |
|-------|-------------------|-------------|-------------------|
| **A** | RoPE positional + learned slot angles | Native KG (HEAD/REL/TAIL slots, bidirectional MLM) | Commutative (rotate Q, K only) |
| **A'** | RoPE positional + learned slot angles | Native KG (HEAD/REL/TAIL slots, bidirectional MLM) | Operator-based (rotate Q, K, V + inverse on output) |
| **B** | RoPE positional | Linearized KG-as-text (causal LM) | Commutative (rotate Q, K only) |
| **B'** | RoPE positional | Linearized KG-as-text (causal LM) | Operator-based (rotate Q, K, V + inverse on output) |
| **C** | Per-token cumsum (journey) | Linearized KG-as-text (causal LM) | Commutative (rotate Q, K only) |
| **C'** | Per-token cumsum (journey) | Linearized KG-as-text (causal LM) | Operator-based (rotate Q, K, V + inverse on output) |

### Model A Architecture (Native KG)

**Same transformer** handles both text and KG. The difference is only in how angles are assigned and masking.

**Angle assignment for KG triples:**
- KG triple has 3 slots: HEAD, REL, TAIL
- HEAD = [A, d, a, m], REL = [<son_of>], TAIL = [B, r, i, a, n]
- Each token gets a RoPE-style positional angle based on position *within its slot*
- Each slot has a learned slot angle vector (R_HEAD, R_REL, R_TAIL)
- Effective angle at position i in slot s, dimension d: `i * d + θ_s[d]`
- Positions reset per slot (HEAD starts at 0, TAIL starts at 0)

**Journey operator between tokens:**
- Position i in slot s, position j in slot s': `R^i R_s R_{s'}^{-1} R^{-j}`
- Within same slot (s = s'): reduces to `R^{i-j}` (pure RoPE, slot cancels)
- Across slots: `R^{i-j}` plus slot difference `R_s R_{s'}^{-1}`

**KG training:**
- Bidirectional attention (no causal mask)
- MLM: predict ALL tokens (not just TAIL from HEAD+REL)

**Text training:**
- Same transformer, slot angles = 0 (R_s = I), reduces to pure RoPE
- Causal mask, next-token prediction

### Setup
- **Corpus**: Synthetic family trees (~30 males, 3-4 generations) + geography (~10 cities, ~5 countries)
- **KG**: ~91 triples with 8 relation types
- **Data split**: 60% teaching (all facts in text), 20% transfer (derived facts only in KG/extra text), 20% generalization (no derived facts anywhere)

### Sub-experiments
- **7a**: Generous linearization — Models B/C get ALL text variations of each KG fact
- **7b**: Realistic linearization — Models B/C get ONE text variation per KG fact
- Model A is identical in both (always sees structured KG triples)

### Evaluation
All models evaluated on identical natural English text completion prompts (cloze-style).
- **Tier 1 (Memorization)**: teaching chains — all models saw all facts
- **Tier 2 (Transfer)**: derived facts — all models saw them, format differs
- **Tier 3 (Generalization)**: derived facts — NO model saw them
- **Metrics**: hits@1, hits@5, perplexity, full entity accuracy

**Scale**: ~7K text tokens, ~91 KG triples, d=64, 2 layers, vocab ~64. ~30 min total on Mac CPU.

---

## Priority for Reviewer Response

| Priority | Experiment | Validates | Difficulty |
|----------|-----------|-----------|------------|
| **1** | **Exp 7** | **prop:lis-union + thm:coupling** | **Medium-Hard** |
| 2 | Exp 1 | thm:dfa + thm:comm-dfa | Medium |
| 3 | Exp 2 | thm:rope-impossibility | Medium |
| 4 | Exp 3 | thm:info-loss + lem:injectivity | Easy |
| 5 | Exp 5 | prop:ssm-bridge | Trivial |
| 6 | Exp 4 | thm:toral | Medium |
| 7 | Exp 6 | thm:coupling + prop:coupling-rank | Hard |

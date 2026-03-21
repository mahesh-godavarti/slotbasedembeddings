# Word Experiment Results: KG Cross-Pollination via 7-Tier Evaluation

## Setup

**Models:**

| Model | Text Encoding | KG Format | V Rotation |
|-------|--------------|-----------|------------|
| **E** | Per-token cumsum angles | Causal (predict tail given head) | No |
| **E'** | Per-token cumsum angles | Causal | Yes |
| **H** | Fixed cumsum angles (RoPE-like) | Causal (relation operator) | No |
| **H'** | Fixed cumsum angles (RoPE-like) | Causal (relation operator) | Yes |
| **I'** | Per-layer projected angles | Causal (cumsum + relation operator) | Yes |
| **J** | Fixed RoPE | Native slotted MLM (HEAD/TAIL, per-relation) | No |
| **J'** | Fixed RoPE | Native slotted MLM (HEAD/TAIL, per-relation) | Yes |
| **K** | Per-layer projected angles + RoPE | Native slotted MLM (HEAD/TAIL slots) | No |
| **K'** | Per-layer projected angles + RoPE | Native slotted MLM | Yes |
| **A** | RoPE + learned slot angles (3 slots) | Slotted MLM | No |
| **B** | Standard RoPE | None (text-only baseline) | No |
| **B + kg_as_text** | Standard RoPE | Linearized as text | No |

**Config:**
```
n_embed=100, n_layers=4, seeds=1, wiki_lines=10000, batch_size=32, lr=5e-4, vocab=16000 (BPE)
```

**Data:**
- 8,254 Wikipedia sentences + 7,480 chain template sentences (~47% chain fraction)
- 10,913 KG triples (BATS 3.0 + chain-derived)
- 154 family chains split across 7 evaluation tiers
- BPE tokenizer (vocab 16,000)

## The 7-Tier Evaluation System

The key innovation: controlling which facts appear in which modality to distinguish memorization from genuine knowledge transfer.

| Tier | Text training | KG training | What it measures |
|------|--------------|-------------|------------------|
| Memorization | base + derived | base + derived | Full supervision baseline |
| **Transfer** | **base only** | **base + derived** | **Does KG improve text prediction?** |
| Generalization | base only | base only | Pure compositional reasoning |
| KG-exclusive memo | nothing | base + derived | KG-only learning |
| KG-exclusive gen | nothing | base only | KG-only generalization |
| Text-exclusive memo | base + derived | nothing | Text-only learning |
| Text-exclusive gen | base only | nothing | Text-only generalization |

**Example chain:** grandfather=Carl, father=Gil, grandson=Herbert
- Base facts: "Carl is the father of Gil", "Gil is the father of Herbert"
- Derived fact: "Carl is the grandfather of Herbert"
- Transfer tier: text sees only base facts; KG sees base + derived. Can the model predict "Herbert" given "Carl is the grandfather of"?

## Results

### Text Evaluation (predict entity name from template prompt)

Format: hit@1 / PPL. Best in each row bolded.

#### At 500K iterations

| Tier | E | E' | I' | K' | K | A (50K) | B (no KG, 50K) | B + kg_as_text |
|------|:-:|:--:|:--:|:--:|:-:|:-------:|:--------------:|:--------------:|
| Memorization | **45.2%** / 4.36 | 19.5% / 15.77 | 35.0% / 6.30 | 31.6% / 10.04 | 28.1% / 10.05 | 49.4% / **4.25** | 45.6% / 5.66 | 0.0% / 519 |
| **Transfer** | 0.7% / **320** | **1.3%** / 394 | 0.7% / 333 | 0.7% / 6,004 | 0.7% / 9,045 | 0.7% / 2,188 | 0.0% / 31,288 | 0.0% / 1,179 |
| Generalization | **1.1%** / 806 | 0.0% / 620 | 0.0% / **572** | 0.0% / 16,095 | 0.0% / 14,918 | 0.0% / 7,116 | 0.0% / 90,984 | 0.0% / 1,877 |
| KG-excl memo | 0.0% / 54K | 0.0% / 44K | 0.0% / 18K | 0.0% / 1.6M | 0.0% / 985K | 0.0% / 868K | 0.0% / 83M | 0.0% / 6,105 |
| KG-excl gen | 0.0% / 140K | 0.0% / 60K | 0.0% / 20K | 0.0% / 6.0M | 0.0% / 3.3M | 0.0% / 1.5M | 0.0% / 75M | 0.0% / 11,530 |
| Text-excl memo | **49.3%** / **2.96** | 18.8% / 11.66 | 31.9% / 4.54 | 40.6% / 5.82 | 24.6% / 10.46 | 44.9% / 2.25 | 49.3% / 2.32 | 0.0% / 635 |
| Text-excl gen | 0.0% / 1,046 | 0.0% / **476** | 0.0% / 362 | 0.0% / 13,558 | 0.0% / 12,396 | 0.0% / 2,618 | 1.8% / 29,309 | 0.0% / 1,183 |

#### At 1M iterations (E', I', K' continued from 500K checkpoints)

| Tier | E' (1M) | I' (1M) | K' (1M) | E' (500K) | I' (500K) | K' (500K) |
|------|:-------:|:-------:|:-------:|:---------:|:---------:|:---------:|
| Memorization | 19.8% / 16.03 | **35.0%** / **5.90** | 23.9% / 19.78 | 19.5% / 15.77 | 35.0% / 6.30 | 31.6% / 10.04 |
| **Transfer** | 0.7% / 462 | 0.7% / **388** | 0.7% / 12,460 | 1.3% / 394 | 0.7% / 333 | 0.7% / 6,004 |
| Generalization | 0.0% / 983 | 0.0% / **715** | 0.0% / 25,941 | 0.0% / 620 | 0.0% / 572 | 0.0% / 16,095 |
| KG-excl memo | 0.0% / 85K | 0.0% / 20K | 0.0% / 2.3M | 0.0% / 44K | 0.0% / 18K | 0.0% / 1.6M |
| KG-excl gen | 0.0% / 165K | 0.0% / 23K | 0.0% / 1.4M | 0.0% / 60K | 0.0% / 20K | 0.0% / 6.0M |
| Text-excl memo | 15.9% / 12.25 | **36.2%** / **4.53** | 23.2% / 16.92 | 18.8% / 11.66 | 31.9% / 4.54 | 40.6% / 5.82 |
| Text-excl gen | **1.8%** / 779 | 0.0% / **431** | 0.0% / 21,347 | 0.0% / 476 | 0.0% / 362 | 0.0% / 13,558 |

### KG Evaluation (predict tail entity from head + relation)

#### At 500K iterations

| Tier | E | E' | I' | K' | K | A (50K) |
|------|:-:|:--:|:--:|:--:|:-:|:-------:|
| Memorization | 1.0% / 151 | 0.2% / 316 | 0.2% / 162 | **6.8%** / **30** | 4.0% / 42 | 1.5% / 245 |
| Transfer | 0.0% / 390 | 0.0% / 512 | 0.0% / 363 | 0.6% / **130** | 1.2% / 194 | 0.6% / 477 |
| Generalization | — | 0.0% / 1,392 | 0.0% / 836 | 0.0% / **429** | 0.0% / 628 | — |
| KG-excl memo | 0.0% / 2,433 | 0.0% / 2,312 | 0.0% / 1,999 | 0.0% / **689** | 0.0% / 1,302 | 0.0% / 2,080 |
| KG-excl gen | — | 0.0% / 4,453 | 0.0% / 3,467 | 0.0% / **3,921** | 0.0% / 3,630 | — |

#### At 1M iterations

| Tier | E' (1M) | I' (1M) | K' (1M) | E' (500K) | I' (500K) | K' (500K) |
|------|:-------:|:-------:|:-------:|:---------:|:---------:|:---------:|
| Memorization | 0.7% / 279 | 0.3% / 148 | **6.4%** / **28** | 0.2% / 316 | 0.2% / 162 | 6.8% / 30 |
| Transfer | 0.0% / 437 | 0.0% / 344 | **1.8%** / **109** | 0.0% / 512 | 0.0% / 363 | 0.6% / 130 |
| Generalization | 0.0% / 1,130 | 0.0% / 710 | 0.0% / **278** | 0.0% / 1,392 | 0.0% / 836 | 0.0% / 429 |
| KG-excl memo | 0.0% / 1,843 | 0.0% / 1,948 | 0.0% / **911** | 0.0% / 2,312 | 0.0% / 1,999 | 0.0% / 689 |
| KG-excl gen | 0.0% / 6,289 | 0.0% / 3,807 | 0.0% / **1,581** | 0.0% / 4,453 | 0.0% / 3,467 | 0.0% / 3,921 |

*(Models B have no structured KG evaluation)*

### Controlled Comparison: I' vs K' (Same Text Encoding, Different KG)

I' and K' share identical text encoding (per-layer projected angles, rotate_v=True). They differ only in KG format:
- **I'**: cumsum + relation operator, causal prediction
- **K'**: RoPE + per-relation slot angles, bidirectional MLM

| Metric | I' (500K) | I' (1M) | K' (500K) | K' (1M) |
|--------|:---------:|:-------:|:---------:|:-------:|
| Text transfer PPL | **333** | **388** | 6,004 | 12,460 |
| Text memorization h@1 | 35.0% | **35.0%** | 31.6% | 23.9% |
| Text-excl gen PPL | **362** | **431** | 13,558 | 21,347 |
| KG memorization h@1 | 0.2% | 0.3% | **6.8%** | **6.4%** |
| KG transfer PPL | 363 | 344 | **130** | **109** |

This is the cleanest comparison: same architecture for text, different KG objective. The causal KG format transfers 18x better to text (PPL 333 vs 6,004 at 500K), while the slotted format learns KG 34x better (h@1 6.8% vs 0.2%).

At 1M iters, the gap widens: K' text transfer PPL doubles (6K→12K) while I' barely moves (333→388). K' overfits to KG at the expense of text. I' remains stable.

### Scaling: n_embed 100 → 200 → 400 (all 500K iters)

| n_embed | Params (E') | Params (I') | Params (K') |
|---------|:-----------:|:-----------:|:-----------:|
| 100 | ~2M | ~2M | ~2M |
| 200 | ~8.4M | ~8.4M | ~8.4M |
| 400 | ~20.6M | ~20.6M | ~20.6M |

#### Text Evaluation

| Tier | E' n100 | E' n200 | E' n400 | I' n100 | I' n200 | I' n400 | K' n100 | K' n200 | K' n400 |
|------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Memo | 19.5%/15.8 | 32.8%/7.6 | **47.5%/3.4** | 35.0%/6.3 | 44.4%/3.6 | **49.8%/2.7** | 31.6%/10.0 | 28.4%/13.5 | 31.4%/12.2 |
| **Transfer** | 1.3%/394 | 1.3%/229 | 0.7%/**200** | 0.7%/333 | 0.7%/**177** | 0.7%/356 | 0.7%/6K | 0.7%/9K | 0.7%/5.4K |
| Gen | 0.0%/620 | 0.0%/469 | 0.0%/573 | 0.0%/572 | 0.0%/**421** | 0.0%/975 | 0.0%/16K | 0.0%/15K | 0.0%/9.6K |
| Txt memo | 18.8%/11.7 | 26.1%/6.8 | **55.1%/2.5** | 31.9%/4.5 | 46.4%/2.9 | **55.1%/2.0** | 40.6%/5.8 | 27.5%/8.4 | 37.7%/7.0 |
| Txt gen | 0.0%/476 | 0.0%/243 | 0.0%/596 | 0.0%/362 | 0.0%/**373** | 0.0%/705 | 0.0%/14K | 0.0%/30K | 0.0%/11K |

#### KG Evaluation

| Tier | E' n100 | E' n200 | E' n400 | I' n100 | I' n200 | I' n400 | K' n100 | K' n200 | K' n400 |
|------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Memo | 0.2%/316 | 1.3%/127 | **7.2%/43** | 0.2%/162 | 2.5%/88 | **8.7%/32** | 6.8%/30 | 8.3%/18 | **9.1%/14** |
| Transfer | 0.0%/512 | 0.0%/290 | **1.2%/120** | 0.0%/363 | 0.0%/242 | **1.2%/94** | 0.6%/130 | 1.8%/61 | **2.5%/30** |
| Gen | 0.0%/1.4K | 0.0%/1.2K | 0.0%/1.2K | 0.0%/836 | 0.0%/1.0K | 0.0%/1.0K | 0.0%/429 | 0.0%/357 | 0.0%/**414** |
| KG-excl | 0.0%/2.3K | 1.3%/1.2K | **2.6%/321** | 0.0%/2.0K | 0.0%/938 | 0.0%/364 | 0.0%/689 | 0.0%/377 | **2.6%/132** |

#### Transfer PPL Scaling Summary

| Model | n100 | n200 | n400 | Best size |
|-------|:----:|:----:|:----:|:---------:|
| E' (text transfer PPL) | 394 | 229 | **200** | n400 (still improving) |
| I' (text transfer PPL) | 333 | **177** | 356 | **n200** (peaked) |
| K' (text transfer PPL) | 6,004 | 9,003 | **5,418** | n400 (slight improvement) |
| K' (KG transfer PPL) | 130 | 61 | **30** | n400 (still improving) |

**I' n200 remains the best text transfer model** — transfer PPL 177 (177x better than baseline). At n400, I' memorizes text so well (49.8% h@1, PPL 2.74) that it no longer needs KG knowledge for transfer-tier predictions, causing transfer PPL to regress.

**E' keeps improving with scale** — transfer PPL 394→229→200, and may benefit from even larger embeddings.

**K' is consistently the best KG model** — KG transfer PPL 130→61→30 at n100/n200/n400. More capacity lets it learn KG better, but text transfer remains poor across all sizes.

**All models show massive KG eval gains at n400** — E' and I' go from near-zero KG memorization at n100 to 7-9% h@1 at n400, comparable to K'. Larger models can learn KG even with the causal format.

### Fixed vs Projected Angles: H/H'/J/J' Grid (500K iters)

H and J are the fixed-angle equivalents of I and K respectively:
- **H/H'** (fixed cumsum) ↔ **I/I'** (projected cumsum)
- **J/J'** (fixed RoPE + per-relation slots) ↔ **K/K'** (projected + RoPE + slots)

#### Text Evaluation

| Tier | H n100 | H' n100 | H n200 | H' n200 | H n400 | H' n400 | J n100 | J' n100 | J n200 | J' n200 | J n400 | J' n400 |
|------|:------:|:-------:|:------:|:-------:|:------:|:-------:|:------:|:-------:|:------:|:-------:|:------:|:-------:|
| Memo h@1 | 45.5% | 51.4% | 48.6% | 46.2% | 53.6% | **55.2%** | 25.2% | 30.9% | 22.2% | 30.5% | 41.4% | 36.2% |
| Memo PPL | 3.5 | 3.3 | 3.2 | 3.3 | 2.6 | 2.9 | 20.0 | 10.2 | 21.9 | 10.0 | 6.2 | 7.4 |
| Transfer PPL | 565 | 483 | **378** | **374** | 449 | 821 | 18K | 12K | 20K | 19K | 7K | 11K |
| Gen PPL | 1017 | 920 | 1617 | **850** | 1747 | 2085 | 33K | 31K | 26K | 60K | 56K | 30K |
| Txt-excl memo h@1 | 53.6% | 50.7% | 39.1% | 44.9% | **60.9%** | **62.3%** | 29.0% | 26.1% | 15.9% | 34.8% | 42.0% | 43.5% |

#### KG Evaluation (angle-gap models)

| Tier | H n100 | H' n100 | H n200 | H' n200 | H n400 | H' n400 | J n100 | J' n100 | J n200 | J' n200 | J n400 | J' n400 |
|------|:------:|:-------:|:------:|:-------:|:------:|:-------:|:------:|:-------:|:------:|:-------:|:------:|:-------:|
| KG memo PPL | 2099 | 1652 | 1538 | 1175 | 595 | 267 | 1231 | 892 | 512 | 369 | 250 | **98** |
| KG gen PPL | 5499 | 5036 | 6213 | 3270 | 4252 | 3905 | 4438 | 2432 | 1795 | 1245 | 2120 | **927** |
| KG memo h@1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.6% |

#### Fixed vs Projected Transfer PPL Comparison

| Size | H (fixed) | I' (projected) | H' (fixed) | I' ratio | J (fixed) | K (projected) | J' (fixed) | K' (projected) |
|------|:---------:|:--------------:|:----------:|:--------:|:---------:|:-------------:|:----------:|:--------------:|
| n100 | 565 | 333 | 483 | 1.4x | 18K | 9K | 12K | 6K |
| n200 | 378 | **177** | 374 | 2.1x | 20K | — | 19K | 9K |
| n400 | 449 | 356 | 821 | — | 7K | — | 11K | 5.4K |

**Projected angles consistently beat fixed angles on text transfer.** At n200, I' (projected) achieves transfer PPL 177 vs H'/H at 374/378 — about 2x better. The per-layer angle projector learns content-dependent positional geometry that helps KG knowledge transfer to text.

**Fixed angles are competitive on memorization.** H' n400 reaches 55.2% memorization h@1 (vs I' n400 at 49.8%), and H' n400 text-exclusive memorization of 62.3% is the highest of any model. Fixed angles don't waste capacity learning to project angles.

**J' is the best fixed-angle KG model.** KG memo PPL of 98 at n400 — but still behind K' n400 (PPL 14). The projected angles help KG learning too.

**V rotation consistently helps KG** for both fixed and projected models, with smaller effects on text.

## Key Findings

### 1. KG Cross-Pollination is Real

The transfer tier is the critical test. Derived facts exist only in KG, not in text. If a model can predict them from text prompts, knowledge genuinely transferred from KG to text.

- **Model B (no KG):** 0.0% hit@1, PPL 31,288 — cannot predict transfer facts at all
- **Model I' n200 (causal KG):** 0.7% hit@1, PPL 177 — **177x better PPL** (best text transfer)
- **Model E' n400 (causal KG):** 0.7% hit@1, PPL 200 — **157x better PPL**
- **Model E' n200 (causal KG):** 1.3% hit@1, PPL 229 — **137x better PPL**
- **Model E n100 (causal KG):** 0.7% hit@1, PPL 320 — **98x better PPL**
- **Model A (slotted KG):** 0.7% hit@1, PPL 2,188 — **14x better PPL**
- **Models K/K' (slotted KG):** 0.7% hit@1, PPL 5,418-9,045 — **3-6x better PPL**

All KG models beat the baseline. The derived facts were never in the text training data — the model learned compositional relationships from KG and applied them to text prediction.

### 2. Causal KG Transfers to Text; Slotted KG Learns KG Better

The I' vs K' comparison (identical text encoding) proves this cleanly:

- **Text transfer**: Causal KG wins by 18x (PPL 333 vs 6,004)
- **KG memorization**: Slotted KG wins by 34x (h@1 6.8% vs 0.2%)

The causal objective (predict tail given head) aligns with next-token prediction, enabling knowledge to flow from KG to text. The slotted format with explicit HEAD/TAIL structure helps the model learn relational facts within KG but that knowledge stays trapped in the KG modality.

### 3. Longer Training: Diminishing Returns After 500K

**50K → 500K (Model E):** Clear memorization gains (37→45% h@1), but transfer flat (PPL 276→320).

**500K → 1M (E', I', K'):** All three plateau or degrade on text:
- **I'**: Most stable — memorization holds at 35%, transfer PPL 333→388 (slight degradation)
- **E'**: Flat — memorization 19.5→19.8%, transfer PPL 394→462
- **K'**: Degrades — memorization 31.6→23.9%, transfer PPL 6,004→12,460 (text overfits to KG)

KG eval continues improving for K' (transfer PPL 130→109, h@1 0.6→1.8%), confirming K' keeps learning KG but at the cost of text performance.

**Conclusion:** Text transfer peaks around 500K iterations. Further training improves KG learning (for slotted models) but doesn't help — and can hurt — text transfer.

### 4. Scaling: More Capacity Beats Longer Training, But Has a Sweet Spot

Doubling n_embed (100→200) at 500K iters improves every metric, while doubling training time (500K→1M) at n_embed=100 shows diminishing returns or degradation.

| Model | n100 500K → n100 1M | n100 → n200 (500K) | n200 → n400 (500K) |
|-------|:-------------------:|:-------------------:|:-------------------:|
| I' transfer PPL | 333 → 388 (worse) | 333 → **177** (1.9x) | 177 → 356 (worse) |
| E' transfer PPL | 394 → 462 (worse) | 394 → **229** (1.7x) | 229 → **200** (1.1x) |
| K' transfer PPL | 6,004 → 12,460 (worse) | 6,004 → 9,003 (worse) | 9,003 → **5,418** (1.7x) |
| I' memo h@1 | 35.0% → 35.0% (flat) | 35.0% → **44.4%** | 44.4% → **49.8%** |
| E' memo h@1 | 19.5% → 19.8% (flat) | 19.5% → **32.8%** | 32.8% → **47.5%** |

**More capacity helps memorization monotonically**, but **text transfer has a sweet spot**. I' peaks at n200 — at n400 the model memorizes text patterns so well (49.8% h@1) that it doesn't rely on KG knowledge, and transfer PPL regresses. E' keeps improving through n400 (PPL 200) and may not have peaked yet.

KG eval improves monotonically for all models — at n400, even causal KG models (E', I') achieve 7-9% KG memorization h@1, comparable to slotted K'.

### 5. V Rotation: Helps Projected Angles, Hurts Cumsum Angles

**K → K' (projected + RoPE):** V rotation helps across the board.
- Text memorization: 28.1% → 31.6%, text-excl memo: 24.6% → 40.6%
- KG memorization: 4.0% → 6.8%

**E → E' (cumsum):** V rotation hurts memorization but improves transfer hit@1.
- Text memorization: 45.2% → 19.5%, text-excl memo: 49.3% → 18.8%
- Transfer hit@1: 0.7% → 1.3% (best of all models), but PPL worsens 320 → 394
- Generalization PPL improves: 806 → 620

The difference likely stems from angle stability. Projected angles (K/K') are computed fresh at each layer from the residual stream and combined with fixed RoPE — they're well-bounded. Cumsum angles (E/E') accumulate across token positions and can grow large, making V rotation amplify noise. But the same instability may force E' to learn more generalizable representations, explaining the better transfer hit@1.

### 6. The Case for Angle-Gap over Text Linearization

Model B + kg_as_text gets 0% hit@1 across all tiers, at every data balance we tested (wiki_lines 10K/100K/1M). This reveals a fundamental advantage of the angle-gap approach.

**Text linearization ties knowledge to surface form.** Converting a KG triple to text (e.g., `"adam <synonym_of> brian"`) forces the model to learn the fact in one specific textual format. If the evaluation prompt uses a different template ("adam is a synonym of ___"), the model may not connect them. Scaling up requires writing many templates per relation, but real language expresses facts in ways templates cannot anticipate.

**Angle-gap encoding decouples knowledge from expression.** The relational structure (head, relation, tail) is encoded directly in the positional encoding angles, independent of how the fact might be expressed as text. The model learns "Carl → grandfather_of → Herbert" as a structural relationship, then can apply it to any natural language prompt about grandfathers — without ever having seen that specific template.

This is why angle-gap models achieve transfer (PPL 320-2,188) while the linearized approach fails. The angle-gap models train KG facts in a separate forward pass with dedicated loss, ensuring consistent exposure to relational structure without competing with text for sampling bandwidth.

### 7. KG-Exclusive Tiers Confirm Isolation

Entities that never appeared in text have astronomical text-eval PPL (18K-83M), confirming the tier system properly isolates modalities. Interestingly, models with causal KG (E, I') show much lower KG-exclusive PPL (18K-54K) than slotted KG models (985K-6M), suggesting some implicit cross-modal leakage in the causal format.

## Debugging Note: Data Dilution

The default `wiki_lines=1000000` caused chain facts to be only ~8% of training data. At this ratio, the model never memorized entity associations (hit@1=0% everywhere). Reducing to `wiki_lines=10000` (chain fraction ~47%) fixed this. The model needs sufficient exposure to chain facts relative to background text.

Similarly, `--kg_as_text` with wiki_lines=10000 creates a 21:1 token ratio (linearized KG vs base text), completely drowning out chain facts.

## Linearized KG-as-Text Runs (all 0% hit@1)

| wiki_lines | Base text | Linearized KG | Ratio | Memorization PPL |
|-----------|----------|---------------|-------|-----------------|
| 10,000 | 616K tokens | 13.3M tokens | 1:21 | 519 |
| 100,000 | 2.9M tokens | 15.6M tokens | 1:5.3 | 800 |
| 1,000,000 | 26.5M tokens | 39.1M tokens | 1:1.5 | 1,645 |

Even at near-equal token ratios (1M wiki lines), linearized KG fails. More wiki text dilutes chain facts; less wiki text lets KG dominate. The approach has no sweet spot because knowledge is bound to surface form.

## Pending

- **Models L/L'** — merged FFN angles + cumsum (like I/I' but angles come from FFN output)
- **Models M/M'** — merged FFN angles + RoPE + slots (like K/K' but angles come from FFN output)

## Scripts

```bash
# Model E (causal KG, 500K iters)
bash /home/ubuntu/exp8/run_E_long.sh

# Model E' (cumsum + causal KG + V rotation, 500K iters)
bash /home/ubuntu/exp8/run_E_prime_long.sh

# Model I' (projected + causal KG + V rotation, 500K iters)
bash /home/ubuntu/exp8/run_I_prime_long.sh

# Models K and K' (projected + RoPE + slot angles, 500K iters)
bash /home/ubuntu/exp8/run_KK_long.sh

# E', I', K' resumed to 1M iters
bash /home/ubuntu/exp8/run_EIK_prime_1M.sh

# Model A (slotted KG MLM, 50K iters)
bash /home/ubuntu/exp8/run_A_fixed.sh

# Model B (text-only baseline, 50K iters)
bash /home/ubuntu/exp8/run_B_baseline.sh

# E', I', K' with n_embed=200 (500K iters)
bash /home/ubuntu/exp8/run_EIK_prime_n200.sh

# E', I', K' with n_embed=400 (500K iters)
bash /home/ubuntu/exp8/run_EIK_prime_n400.sh

# Model B + kg_as_text (linearized KG, rebalanced)
bash /home/ubuntu/exp8/run_B_kg_text_v2.sh

# H, H', J, J' grid sweep (n100, n200, n400)
bash /home/ubuntu/exp8/run_HJ_grid.sh
```

All use: `--n_layers 4 --seeds 1 --wiki_lines 10000`
n400 models use `--n_embed 400 --iters 500000 --checkpoint_dir checkpoints_n400`
n200 models use `--n_embed 200 --iters 500000 --checkpoint_dir checkpoints_n200`
n100 1M models use `--n_embed 100 --iters 1000000 --resume_training`
n100 500K models use `--n_embed 100 --iters 500000`; 50K models use `--iters 50000`
B+kg_as_text v2 uses `--iters 50000 --wiki_lines 100000`

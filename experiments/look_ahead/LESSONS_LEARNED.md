# Lessons Learned

## 1. Baseline architecture must be realistic

We spent most of our compute comparing against N=24 C=1024 (C/N=43). This ratio is far below what deployed models use:

| Model | C/N ratio |
|-------|-----------|
| Llama 7B | 128 |
| Llama 70B | 102 |
| GPT-3 175B | 128 |
| PaLM 540B | 156 |
| **N=24 C=1024 (ours)** | **43** |
| N=12 C=1408 (ours) | 117 |
| N=12 C=1536 (ours) | 128 |

N=24 C=1024 is unrealistically deep for its width. Nobody would deploy a model with C/N=43. Comparing against it flatters the look-ahead architecture because the baseline is overprovisioned in depth.

The evidence: N=12 C=1408 (29.92) is within 0.50 PPL of N=24 C=1024 (29.42) at fewer FLOPs. Doubling from 12 to 24 layers barely helps. The extra 12 layers are wasted.

**The right comparisons are at C/N=100-150**: D=12 C=1408 vs N=12 C=1536, D=6 C=2048 vs N=12 C=1536. These are the experiments that tell us whether look-ahead adds real value at realistic architectures.

**Cost of this mistake**: Thousands of dollars in GPU time on N=24 C=1024 baselines, D=23 C=1024 from scratch, D=24 fine-tuning, and related experiments that compare against an unrealistic baseline.

## 2. Never swap or rename source files

We destroyed blocks.py twice by using mv/ln to swap in flash attention variants. The trap cleanup in shell scripts fought with each other and deleted the original.

**Solution**: Use PYTHONPATH override (flash_override/ directory) or modify blocks.py directly. Never rename or move the original.

## 3. Verify FLOP calculations against actual code

Made a major error counting SA correction overhead as 12C^2 (full block) instead of 4C^2 (just Q,K,V,proj). The corr_ffn (8C^2) already existed in both variants.

**Solution**: Count the actual nn.Linear layers in the model class. Don't assume.

- corr_ffn_add D=x: (12D + 8)C^2
- SA D=x: (12D + 4 + 8)C^2 = (12D + 12)C^2
- SA overhead vs base = +4C^2 only

## 4. Check batch size consistency for eval

Different batch sizes produce different eval samples (the eval function seeds with 42 and draws batch_size samples). This caused a ~2 PPL discrepancy when comparing D=23 at batch=16 vs roformer at batch=32.

**Solution**: Always use the same batch size for eval. Or run a separate eval pass at a fixed batch size.

## 5. Don't emphasize training cost as a downside

The fine-tune recipe (train roformer, convert, fine-tune briefly at K>1) means you never need to train D from scratch in practice. Training cost is not the story.

## 6. C must be divisible by n_head, and head_dim must be even

RoPE requires even head_dim for the rotation pairs. C=1456 with n_head=16 gives head_dim=91 (odd), which crashes. Always verify C/n_head is even.

## 7. The real experimental comparisons needed

At ~300M FLOP budget, the realistic configurations are:

| Config | FLOPs | C/N | Notes |
|--------|-------|-----|-------|
| N=12 C=1536 | 340M | 128 | Realistic roformer baseline |
| N=12 C=1408 | 285M | 117 | Have this (29.92) |
| D=12 C=1408 | 301M | 117 | Have this (29.00) |
| D=6 C=2048 | 336M | 341 | Have this (29.04) |
| N=6 C=2048 | 302M | 341 | Have this (30.86) |
| N=6 C=2176 | 341M | 363 | Running on qmti92t1 |

The key comparisons:
1. D=12 C=1408 (29.00) vs N=12 C=1536 (running) -- does correction help at realistic C/N?
2. D=6 C=2048 (29.04) vs N=12 C=1536 (running) -- does wider+correction beat medium depth?
3. D=6 C=2048 (29.04) vs N=6 C=2176 (~31 projected) -- correction value at same depth

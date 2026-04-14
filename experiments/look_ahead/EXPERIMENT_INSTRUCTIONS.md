# look_ahead7: Attention-Based Correction Experiment

## What's new

`block_head_attn_corr_ffn_add` -- a new variant where the correction uses cross-attention over ALL previous z values instead of just z[t-1].

**Old (block_head_corr_ffn_add)**:
```
correction = corr_ffn(ln(z[t-1] + tok_emb[t]))    # sees only previous position
processed_x = tok_emb + correction
```

**New (block_head_attn_corr_ffn_add)**:
```
attn_out = cross_attention(Q=tok_emb[t], KV=z[0..t-1])   # sees ALL previous positions
correction = corr_ffn(ln(attn_out))
processed_x = tok_emb + correction
```

Inference FLOPs:
- Old: (12D + 8)C^2
- New: (12D + 20)C^2 (extra 12C^2 for the cross-attention layer)

## Files

| File | Purpose |
|------|---------|
| `models.py` | Contains `BlockHeadAttnCorrFFNAddModel` (new) and all existing models |
| `blocks.py` | Base building blocks (unchanged from look_ahead6) |
| `train_wiki_streaming.py` | Training script (unchanged from look_ahead6) |
| `check_progress.sh` | Progress checker |
| `convert_roformer_to_lookahead.py` | Checkpoint converter (works for old model only, NOT the new attn variant) |

## Experiment: Apples-to-apples comparison

We need to compare the new attention-based correction against the old FFN-only correction at matched FLOPs.

### The comparison

The new model at D=x has (12D + 20)C^2 FLOPs. The old model at D=x has (12D + 8)C^2 FLOPs. They're NOT FLOP-matched at the same D and C.

To FLOP-match, we compare:
- **New**: D=2, C=1024 -> (24 + 20) * 1024^2 = 44 * 1024^2 FLOPs
- **Old**: D=3, C=1024 -> (36 + 8) * 1024^2 = 44 * 1024^2 FLOPs (exact match!)

So D=2 new = D=3 old at C=1024. Both 44C^2.

Also compare at same D to see raw benefit of attention:
- **New**: D=2, C=1024 -> 44C^2
- **Old**: D=2, C=1024 -> 32C^2 (old is cheaper)

### Experiments to run

All experiments: C=1024, block_size=64, batch=256, n_head=16, OWT data, lr=2e-4, softmax, amp.
Token budget: 1,227M tokens (matching the scaling experiment).

**Experiment 1: New D=2 C=1024 from scratch**
```bash
cd /home/ubuntu/look_ahead7
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_attn_corr_ffn_add --n_embed 1024 --n_layers 10 --block_size 64 --batch_size 256 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 2 --n_head 16 --k_min 2 \
    --max_iters 74890 --eval_interval 1000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_attn_d2_c1024 \
    --gpu 0 \
    --amp 2>&1 | tee logs/attn_d2_c1024_scratch.log
```

**Experiment 2: Old D=3 C=1024 from scratch (FLOP-matched to new D=2)**

This already exists in look_ahead6 scaling experiment data. D=3 C=1024 at block_size=64 trained for 1,227M tokens. Final PPL after 3 cycles: 61.60 (N=3 roformer) and D=3 cont: 59.41. But the from-scratch D=3 at matched tokens needs to be checked.

If not available, run:
```bash
cd /home/ubuntu/look_ahead7
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 1024 --n_layers 15 --block_size 64 --batch_size 256 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 3 --n_head 16 --k_min 2 \
    --max_iters 74890 --eval_interval 1000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_old_d3_c1024 \
    --gpu 1 \
    --amp 2>&1 | tee logs/old_d3_c1024_scratch.log
```

**Experiment 3: New D=2 C=1024 vs Old D=2 C=1024 (same D, different FLOPs)**
Old D=2 C=1024 already exists from look_ahead6 width scaling. Final PPL: 60.84 at 1,227M tokens.

### Baselines (already available from look_ahead6)

| Model | FLOPs | PPL | Source |
|-------|-------|-----|--------|
| N=2 C=1024 | 24C^2 | 72.83 | look_ahead6 scaling experiment (3 cycles) |
| N=3 C=1024 | 36C^2 | 61.60 | look_ahead6 scaling experiment (3 cycles) |
| D=2 old C=1024 | 32C^2 | 60.84 | look_ahead6 width scaling from scratch |
| D=3 old C=1024 | 44C^2 | ? | Need from-scratch at 1,227M tokens |
| N=6 C=1024 | 72C^2 | 52.47 | look_ahead6 scaling experiment (3 cycles) |

### What we want to see

1. Does D=2 new (44C^2) beat D=3 old (44C^2) at FLOP parity? If yes, the attention-based correction is more efficient than adding another block.

2. Does D=2 new (44C^2) beat D=2 old (32C^2) despite higher FLOPs? Obviously it should since it has more compute. The question is by how much -- is the extra 12C^2 well spent on attention vs adding it as another block?

3. How does the iter-vs-PPL curve compare? Does the attention variant learn faster?

### Critical rules

1. **Run from look_ahead7 directory** -- `cd /home/ubuntu/look_ahead7` before launching
2. **Data is at** `/home/ubuntu/look_ahead/look_ahead/data_owt/` -- must exist on the machine
3. **Venv**: `/home/ubuntu/exp8/venv/`
4. **All C=1024 experiments use**: n_head=16, softmax, amp, lr=2e-4
5. **Checkpoint everything** -- use --checkpoint_dir
6. **Do NOT modify blocks.py or models.py in look_ahead6** -- look_ahead7 is isolated

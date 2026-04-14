# SA D=5 C=2048 — FLOP-matched to Roformer N=24 C=1024

## What this is

`block_head_sa_corr_ffn_add` is a variant of the look-ahead correction mechanism where the correction path uses self-attention over `(shifted_z + tok_emb)` before the correction FFN.

```
z = blocks(processed_x)                              # D=5 shared-weight blocks
shifted_z[t] = z[t-1]
corr_input = shifted_z + tok_emb
h = corr_input + self_attn(ln1(corr_input))          # causal self-attention + residual
correction = corr_ffn(ln2(h))
processed_x = tok_emb + correction                   # non-cumulative reset
head sees z
```

Inference FLOPs: (12D + 12)C² = 72 × 2048² = 288 × 1024² = same as Roformer N=24 C=1024.
5 sequential layers at inference vs 24 for roformer.

## Prerequisites

1. **Venv**: `/home/ubuntu/exp8/venv/` (needs torch, numpy, tqdm, tokenizers)
2. **Data**: `/home/ubuntu/look_ahead/look_ahead/data_owt/` (preprocessed OWT, ~34GB, vocab=32000)
   - If it doesn't exist, rsync from this machine (instance-qmti92t1):
     ```bash
     rsync -avz --progress -e "ssh -p PORT -i KEYFILE" \
         ubuntu@HOST:/home/ubuntu/look_ahead/look_ahead/data_owt/ \
         /home/ubuntu/look_ahead/look_ahead/data_owt/
     ```
3. **Code**: `/home/ubuntu/look_ahead7/` — rsync the entire directory:
   ```bash
   rsync -avz --progress -e "ssh -p PORT -i KEYFILE" \
       ubuntu@HOST:/home/ubuntu/look_ahead7/ \
       /home/ubuntu/look_ahead7/
   ```
   Key files: `models.py`, `train_wiki_streaming.py`, `blocks.py`, `check_progress.sh`

## Launch command

```bash
cd /home/ubuntu/look_ahead7

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_sa_corr_ffn_add \
    --n_embed 2048 --n_layers 25 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 \
    --d_block 5 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_sa_d5_c2048 \
    --gpu 0 --amp \
    >> logs/sa_d5_c2048.log 2>&1 &
```

### Parameter explanation
- `--n_embed 2048`: embedding dimension C
- `--n_layers 25`: D × K = 5 × 5 = 25
- `--d_block 5`: D=5 blocks per unit
- `--block_size 256`: context window
- `--batch_size 32`: batch size
- `--n_head 16`: attention heads (head_dim = 2048/16 = 128)
- `--k_min 2`: random K training, samples K from Uniform(2, 5)
- `--max_iters 200000`: total training iterations
- `--eval_interval 5000`: evaluate every 5K iters
- `--convergence_weight 0.1`: auxiliary convergence loss
- `--softmax`: use softmax attention (always)
- `--amp`: mixed precision bfloat16 (always)

## Check progress

```bash
bash /home/ubuntu/look_ahead7/check_progress.sh /home/ubuntu/look_ahead7/logs/sa_d5_c2048.log
```

Or manually:
```bash
tail -c 500 /home/ubuntu/look_ahead7/logs/sa_d5_c2048.log | tr '\r' '\n' | tail -3
```

## What to compare against

Roformer N=24 C=1024 at 200K iters on OWT: **29.42 PPL** (from look_ahead6).
Base (corr_ffn_add) D=24 C=1024 fine-tuned from 400K: **26.02 PPL**.

This SA D=5 C=2048 uses the same inference FLOPs as N=24 C=1024 (288 × 1024²) but with only 5 sequential layers instead of 24.

## Memory estimate

~460M params. With batch=32, block_size=256, K=5 iterations, should fit on H100 80GB with AMP. If OOM, try `--batch_size 16` and `--max_iters 400000` (same token budget).

## Critical rules

1. Run from `/home/ubuntu/look_ahead7/` directory
2. Data must be at `/home/ubuntu/look_ahead/look_ahead/data_owt/`
3. Always use `--amp` and `--softmax`
4. Do NOT modify `blocks.py` or `models.py`
5. Sequential K=1 is the only valid inference metric (reported automatically at end of training)

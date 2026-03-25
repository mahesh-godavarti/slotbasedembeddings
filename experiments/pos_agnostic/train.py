#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Any commercial use, including but not limited to use in a product, service,
# or for-profit research, is strictly prohibited without explicit written
# permission from the copyright holder.
#
# Patent Pending: Certain aspects of this software are the subject of a
# pending patent application.
#
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------
"""
Training script for PAFL length generalization experiments.

Trains transformer variants on Wikipedia text and evaluates length
extrapolation at multiple sequence lengths.

Usage:
  # Smoke test
  python train.py --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
      --models rope nope --smoke

  # Full experiment
  python train.py --data_dir /home/ubuntu/look_ahead/look_ahead/data_full \
      --models rope nope alibi hybrid_2 hybrid_4 \
      --n_embed 512 --n_layers 8 --n_heads 8 --block_size 512 \
      --max_iters 50000 --cosine_decay

Data: Uses preprocessed memmap files (wiki_tokens.bin, wiki_tokens.meta).
      Existing: /home/ubuntu/look_ahead/look_ahead/data_full/ (983M tokens, vocab=16000)
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import GPTModel


# ---------------------------------------------------------------------------
# Data loading (memmap — constant memory regardless of corpus size)
# ---------------------------------------------------------------------------

def load_memmap_data(data_dir):
    """Load preprocessed data as memory-mapped array.

    Returns: (train_data, val_data, meta_dict)
    """
    meta_path = os.path.join(data_dir, 'wiki_tokens.meta')
    bin_path = os.path.join(data_dir, 'wiki_tokens.bin')

    with open(meta_path) as f:
        meta = json.load(f)

    total = meta['total_tokens']
    data = np.memmap(bin_path, dtype=np.int32, mode='r', shape=(total,))
    n = int(total * 0.9)
    return data[:n], data[n:], meta


def get_batch(data, block_size, batch_size, device):
    """Random-access batch from memmap data."""
    n = len(data) - block_size
    ix = torch.randint(0, n, (batch_size,)).numpy()
    seqs = np.stack([data[i:i + block_size + 1] for i in ix])
    seqs = torch.from_numpy(seqs.astype(np.int64)).to(device)
    return seqs[:, :block_size], seqs[:, 1:block_size + 1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device,
                  eval_iters=20):
    """Estimate train/val loss at the training block_size. Uses fixed seed for reproducibility."""
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
        torch.random.set_rng_state(rng_state)
    model.train()
    return out


@torch.no_grad()
def eval_lengths(model, val_data, lengths, batch_size, device, eval_iters=10):
    """Evaluate at multiple sequence lengths for length extrapolation.

    Returns: dict of {length: {'loss': float, 'ppl': float}}
    """
    model.eval()
    results = {}
    rng_state = torch.random.get_rng_state()
    for length in lengths:
        if len(val_data) < length + 1:
            results[length] = {'loss': None, 'ppl': None, 'error': 'data too short'}
            continue
        try:
            torch.manual_seed(42 + length)  # deterministic per length
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(val_data, length, batch_size, device)
                _, loss = model(x, y)
                if not torch.isnan(loss):
                    losses.append(loss.item())
            if losses:
                avg = sum(losses) / len(losses)
                results[length] = {'loss': round(avg, 4), 'ppl': round(math.exp(avg), 2)}
            else:
                results[length] = {'loss': None, 'ppl': None, 'error': 'all NaN'}
        except RuntimeError as e:
            torch.cuda.empty_cache()
            if 'out of memory' in str(e).lower():
                results[length] = {'loss': None, 'ppl': None, 'error': 'OOM'}
            else:
                raise
    torch.random.set_rng_state(rng_state)
    model.train()
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_name, model, train_data, val_data, args, device):
    """Train one model, return results."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters)
                 if args.cosine_decay else None)

    model.to(device)
    model.train()
    if args.eval_topk > 0:
        model.set_eval_topk(args.eval_topk)
    use_bf16 = getattr(args, 'bf16', False) and torch.cuda.is_bf16_supported()
    if use_bf16:
        print(f"Using BF16 mixed precision")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {model_name}  ({n_params:,} params)")
    print(f"Layers: {model.layer_types}")
    if args.eval_topk > 0:
        print(f"Eval top-k: {args.eval_topk}")
    print(f"{'='*60}")

    ppl_log = {"iter": [], "train_ppl": [], "val_ppl": []}
    extrap_log = []
    eval_lengths_list = [int(x) for x in args.eval_lengths.split(',')]

    best_val_loss = float('inf')

    pbar = tqdm(range(args.max_iters), desc=model_name)
    for it in pbar:
        # --- Regular eval ---
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, args.batch_size, device)
            t_ppl = math.exp(losses['train'])
            v_ppl = math.exp(losses['val'])
            ppl_log["iter"].append(it)
            ppl_log["train_ppl"].append(round(t_ppl, 2))
            ppl_log["val_ppl"].append(round(v_ppl, 2))
            pbar.set_postfix(train_ppl=f"{t_ppl:.2f}", val_ppl=f"{v_ppl:.2f}")
            tqdm.write(f"  [{model_name}] iter {it}: "
                       f"train PPL {t_ppl:.2f}, val PPL {v_ppl:.2f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                # Save best checkpoint
                if args.checkpoint_dir:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pt")
                    torch.save({
                        'iter': it,
                        'model_state_dict': model.state_dict(),
                        'val_loss': losses['val'],
                        'config': {
                            'vocab_size': model.tok_emb.num_embeddings,
                            'n_embed': model.tok_emb.embedding_dim,
                            'n_layers': len(model.blocks),
                            'n_heads': model.blocks[0].attn.n_heads,
                            'block_size': model.block_size,
                            'window_size': model.window_size,
                            'attn_config': model.layer_types,
                        },
                    }, path)

        # --- Extrapolation eval ---
        if (args.extrap_interval > 0 and it > 0
                and (it % args.extrap_interval == 0 or it == args.max_iters - 1)):
            results = eval_lengths(model, val_data, eval_lengths_list,
                                   args.eval_batch_size, device)
            extrap_log.append({'iter': it, 'results': {str(k): v for k, v in results.items()}})
            parts = [f"{l}:{r['ppl']}" for l, r in sorted(results.items())
                     if r.get('ppl') is not None]
            tqdm.write(f"  [{model_name}] extrap iter {it}: {', '.join(parts)}")

        # --- Train step ---
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(x, y)

        if torch.isnan(loss):
            tqdm.write(f"  [{model_name}] NaN at iter {it}, stopping.")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

    # --- Final evaluation ---
    losses = estimate_loss(model, train_data, val_data,
                           args.block_size, args.batch_size, device)
    v_ppl = math.exp(losses['val'])
    print(f"\n  [{model_name}] final val PPL: {v_ppl:.2f}")

    # Final extrapolation eval
    final_extrap = eval_lengths(model, val_data, eval_lengths_list,
                                args.eval_batch_size, device)
    parts = [f"{l}:{r['ppl']}" for l, r in sorted(final_extrap.items())
             if r.get('ppl') is not None]
    print(f"  [{model_name}] final extrap: {', '.join(parts)}")

    # Save final checkpoint
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        path = os.path.join(args.checkpoint_dir, f"{model_name}_final.pt")
        torch.save({
            'iter': args.max_iters,
            'model_state_dict': model.state_dict(),
            'val_loss': losses['val'],
            'final_extrap': {str(k): v for k, v in final_extrap.items()},
        }, path)

    return {
        'val_loss': losses['val'],
        'val_ppl': round(v_ppl, 2),
        'params': n_params,
        'ppl_curve': ppl_log,
        'extrap_curve': extrap_log,
        'final_extrap': {str(k): v for k, v in final_extrap.items()},
        'best_val_loss': best_val_loss,
        'best_val_ppl': round(math.exp(best_val_loss), 2),
        'layer_types': model.layer_types,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PAFL Length Generalization Experiments")

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with preprocessed memmap data')
    parser.add_argument('--models', nargs='+', default=['rope'],
                        help='Model configs: rope, nope, alibi, hybrid_1, hybrid_2, ...')
    parser.add_argument('--n_embed', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=512,
                        help='Training context length')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size for extrapolation eval (smaller to avoid OOM)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--extrap_interval', type=int, default=5000,
                        help='Length extrapolation eval interval (0 to disable)')
    parser.add_argument('--eval_lengths', type=str, default='512,1024,2048,4096',
                        help='Comma-separated eval lengths')
    parser.add_argument('--window_size', type=int, default=256,
                        help='Sliding window attention size')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Checkpoint directory (empty to disable)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cosine_decay', action='store_true',
                        help='Use cosine annealing LR schedule')
    parser.add_argument('--softplus', action='store_true',
                        help='Use normalized softplus attention instead of softmax')
    parser.add_argument('--eval_topk', type=int, default=0,
                        help='At eval, use top-k attention over full history (0=disabled, use window)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 mixed precision training')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick test: small model, few iters')

    args = parser.parse_args()

    if args.smoke:
        args.max_iters = 100
        args.eval_interval = 50
        args.extrap_interval = 100
        args.n_embed = 128
        args.n_layers = 4
        args.n_heads = 4
        args.block_size = 64
        args.batch_size = 16
        args.eval_batch_size = 4
        args.eval_lengths = '64,128,256'
        args.window_size = 32

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_data, val_data, meta = load_memmap_data(args.data_dir)
    vocab_size = meta['vocab_size']
    print(f"  Tokens: {meta['total_tokens']:,}  (train {len(train_data):,}, val {len(val_data):,})")
    print(f"  Vocab: {vocab_size}")

    print(f"\nDevice: {device}")
    print(f"Config: n_embed={args.n_embed}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, block_size={args.block_size}, "
          f"window_size={args.window_size}")
    print(f"Training: lr={args.lr}, max_iters={args.max_iters}, "
          f"batch_size={args.batch_size}, dropout={args.dropout}")
    print(f"Eval lengths: {args.eval_lengths}")
    print(f"Models: {args.models}")

    # Train each model
    all_results = {}
    for model_name in args.models:
        torch.manual_seed(args.seed)
        model = GPTModel(vocab_size, args.n_embed, args.n_layers, args.n_heads,
                         args.block_size, args.dropout, attn_config=model_name,
                         window_size=args.window_size, use_softplus=args.softplus)
        result = train_model(model_name, model, train_data, val_data, args, device)
        all_results[model_name] = result

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # --- Print comparison table ---
    eval_lengths_list = [int(x) for x in args.eval_lengths.split(',')]

    print(f"\n{'='*80}")
    print("RESULTS — Length Generalization")
    print(f"{'='*80}")

    header = f"{'Model':<20} {'Params':>10} {'Val PPL':>10}"
    for l in eval_lengths_list:
        header += f" {'PPL@' + str(l):>10}"
    print(header)
    print('-' * len(header))

    for name in args.models:
        r = all_results[name]
        line = f"{name:<20} {r['params']:>10,} {r['val_ppl']:>10.2f}"
        fe = r.get('final_extrap', {})
        for l in eval_lengths_list:
            entry = fe.get(str(l), {})
            ppl = entry.get('ppl')
            if ppl is not None:
                line += f" {ppl:>10.2f}"
            else:
                err = entry.get('error', '-')
                line += f" {err:>10}"
        print(line)

    print(f"{'='*80}")

    # --- Save results ---
    save_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"results_{timestamp}.json")
    save_data = {
        'config': {k: v for k, v in vars(args).items()},
        'results': all_results,
        'timestamp': timestamp,
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    latest_file = os.path.join(save_dir, "results_latest.json")
    with open(latest_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")


if __name__ == '__main__':
    main()

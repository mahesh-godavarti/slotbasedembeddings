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
#
# train_wiki_streaming.py — Memory-efficient version of train_wiki.py
#
# Preprocesses wiki text to a memory-mapped binary file, then trains using
# random-access memmap for get_batch(). Supports the full 28.8M-line
# wiki.en.txt without OOM.
#
# Usage:
#   python train_wiki_streaming.py preprocess --wiki_path PATH --vocab_size 16000
#   python train_wiki_streaming.py train --data_dir joformer/data --n_embed 500 ...
#   python train_wiki_streaming.py auto --wiki_path PATH --vocab_size 16000 --n_embed 500 ...

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wiki import MODEL_CLASSES

# ---------------------------------------------------------------------------
# Phase 1: Preprocessing (constant memory)
# ---------------------------------------------------------------------------

def train_bpe_tokenizer_streaming(wiki_path, vocab_size, max_lines=None):
    """Train BPE tokenizer by streaming wiki file. Constant memory.

    Filters blank lines to a temp file (one line at a time),
    trains BPE on the temp file, deletes it.

    Returns: (tokenizer, actual_vocab_size, line_count)
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    # Stream-filter to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     encoding='utf-8') as f:
        tmp_path = f.name
        count = 0
        with open(wiki_path, 'r', encoding='utf-8') as src:
            for i, line in enumerate(src):
                if max_lines and i >= max_lines:
                    break
                stripped = line.strip()
                if stripped:
                    f.write(stripped + '\n')
                    count += 1

    print(f"  Filtered {count:,} non-empty lines to temp file for BPE training")

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>"],
    )
    tokenizer.train([tmp_path], trainer)
    os.unlink(tmp_path)

    actual_vocab_size = tokenizer.get_vocab_size()
    return tokenizer, actual_vocab_size, count


def tokenize_to_disk(tokenizer, wiki_path, output_bin_path, max_lines=None):
    """Tokenize wiki text line-by-line, writing int32 IDs to binary file.

    Memory: constant (one line + one encoding at a time).
    Returns: total_tokens written.
    """
    total_tokens = 0
    with open(wiki_path, 'r', encoding='utf-8') as src, \
         open(output_bin_path, 'wb') as dst:
        for i, line in enumerate(tqdm(src, desc="Tokenizing", unit=" lines")):
            if max_lines and i >= max_lines:
                break
            stripped = line.strip()
            if stripped:
                enc = tokenizer.encode(stripped)
                ids = enc.ids
                if ids:
                    chunk = np.array(ids, dtype=np.int32)
                    dst.write(chunk.tobytes())
                    total_tokens += len(ids)
    return total_tokens


def preprocess(args):
    """Run the full preprocessing pipeline: BPE training + tokenization to disk."""
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    bin_path = os.path.join(data_dir, 'wiki_tokens.bin')
    meta_path = os.path.join(data_dir, 'wiki_tokens.meta')
    tok_path = os.path.join(data_dir, 'wiki_tokenizer.json')

    print(f"Preprocessing wiki text: {args.wiki_path}")
    print(f"  max_lines={args.wiki_lines}, vocab_size={args.vocab_size}")
    print(f"  Output dir: {data_dir}")

    # Step 1: Train BPE tokenizer (streaming)
    print("\n[1/2] Training BPE tokenizer...")
    t0 = time.time()
    tokenizer, actual_vocab_size, line_count = train_bpe_tokenizer_streaming(
        args.wiki_path, args.vocab_size, args.wiki_lines
    )
    print(f"  Vocab size: {actual_vocab_size}, trained in {time.time()-t0:.1f}s")

    # Save tokenizer
    tokenizer.save(tok_path)
    print(f"  Tokenizer saved to {tok_path}")

    # Step 2: Tokenize to binary (streaming)
    print("\n[2/2] Tokenizing corpus to binary...")
    t0 = time.time()
    total_tokens = tokenize_to_disk(
        tokenizer, args.wiki_path, bin_path, args.wiki_lines
    )
    dt = time.time() - t0
    file_size_gb = os.path.getsize(bin_path) / (1024**3)
    print(f"  {total_tokens:,} tokens written in {dt:.1f}s")
    print(f"  Binary file: {bin_path} ({file_size_gb:.2f} GB)")

    # Save metadata
    meta = {
        'total_tokens': total_tokens,
        'vocab_size': actual_vocab_size,
        'source': os.path.abspath(args.wiki_path),
        'max_lines': args.wiki_lines,
        'line_count': line_count,
        'dtype': 'int32',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    print(f"\nPreprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# Phase 2: Training with memmap
# ---------------------------------------------------------------------------

def load_memmap_data(data_dir):
    """Load preprocessed data as memory-mapped numpy array.

    Returns: (train_data, val_data, tokenizer, meta)
      train_data, val_data are numpy memmap views (int32), no copy.
    """
    meta_path = os.path.join(data_dir, 'wiki_tokens.meta')
    bin_path = os.path.join(data_dir, 'wiki_tokens.bin')
    tok_path = os.path.join(data_dir, 'wiki_tokenizer.json')

    with open(meta_path) as f:
        meta = json.load(f)

    total_tokens = meta['total_tokens']
    data = np.memmap(bin_path, dtype=np.int32, mode='r', shape=(total_tokens,))

    # 90/10 train/val split (memmap views, no copy)
    n = int(total_tokens * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)

    return train_data, val_data, tokenizer, meta


def get_batch(train_data, val_data, split, block_size, batch_size, device):
    """Random-access batch from memory-mapped data."""
    data = train_data if split == "train" else val_data
    n = len(data) - block_size
    ix = torch.randint(0, n, (batch_size,)).numpy()

    # Each data[i:i+block_size+1] reads ~260 bytes from memmap (one page)
    sequences = np.stack([data[i:i + block_size + 1] for i in ix])
    sequences = torch.from_numpy(sequences.astype(np.int64)).to(device)

    x = sequences[:, :block_size].contiguous()
    y = sequences[:, 1:block_size + 1].contiguous()
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device,
                  eval_iters=20):
    """Estimate train/val loss using memmap data."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split,
                             block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(model_name, model, train_data, val_data, args, device, tokenizer):
    """Train a single model using memmap data. Returns (val_loss, val_ppl, ppl_log)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters)
                 if args.cosine_decay else None)
    model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {model_name}  ({n_params:,} parameters)")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    ppl_log = {"iter": [], "train_ppl": [], "val_ppl": []}

    pbar = tqdm(range(args.max_iters), desc=model_name)
    for it in pbar:
        # Eval
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, args.batch_size, device)
            train_ppl = math.exp(losses['train'])
            val_ppl = math.exp(losses['val'])
            ppl_log["iter"].append(it)
            ppl_log["train_ppl"].append(round(train_ppl, 2))
            ppl_log["val_ppl"].append(round(val_ppl, 2))
            pbar.set_postfix(train_loss=f"{losses['train']:.3f}",
                             val_loss=f"{losses['val']:.3f}",
                             val_ppl=f"{val_ppl:.2f}")
            tqdm.write(f"  [{model_name}] iter {it}: "
                       f"train loss {losses['train']:.4f} (PPL {train_ppl:.2f}), "
                       f"val loss {losses['val']:.4f} (PPL {val_ppl:.2f})")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

            # Checkpoint
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                path = os.path.join(args.checkpoint_dir, f"{model_name}_iter{it}.pt")
                torch.save({
                    'iter': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': losses['val'],
                }, path)

            # Generate sample
            if it > 0 and it % (args.eval_interval * 2) == 0:
                try:
                    model.eval()
                    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
                    generated = model.generate(prompt, args.generate_len)
                    text = tokenizer.decode(generated[0].cpu().tolist())
                    tqdm.write(f"  [{model_name}] sample: {text[:200]}")
                    model.train()
                except Exception as e:
                    tqdm.write(f"  [{model_name}] sample generation failed: {e}")
                    model.train()

        # Train step
        xb, yb = get_batch(train_data, val_data, "train",
                           args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)

        # NaN detection — stop early
        if torch.isnan(loss):
            tqdm.write(f"  [{model_name}] NaN loss detected at iter {it}, stopping early.")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

    # Final eval
    losses = estimate_loss(model, train_data, val_data,
                           args.block_size, args.batch_size, device)
    val_ppl = math.exp(losses['val'])

    # Final generation sample
    print(f"\n  [{model_name}] final val loss: {losses['val']:.4f} (PPL {val_ppl:.2f})")
    try:
        model.eval()
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(prompt, args.generate_len)
        text = tokenizer.decode(generated[0].cpu().tolist())
        print(f"  [{model_name}] sample: {text[:300]}")
    except Exception as e:
        print(f"  [{model_name}] final sample generation failed: {e}")

    # Save final checkpoint
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        path = os.path.join(args.checkpoint_dir, f"{model_name}_final.pt")
        torch.save({
            'iter': args.max_iters,
            'model_state_dict': model.state_dict(),
            'val_loss': losses['val'],
        }, path)

    return losses['val'], val_ppl, ppl_log


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def run_training(args):
    """Load memmap data and train all requested models."""
    if args.smoke:
        args.max_iters = 50
        args.eval_interval = 25
        args.n_layers = 2
        args.n_embed = 64
        args.generate_len = 50

    if args.n_embed % 2 != 0:
        args.n_embed += 1
        print(f"Adjusted n_embed to {args.n_embed} (must be even)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    attn_type = "softmax" if args.softmax else "normalized softplus"
    print(f"Config: n_embed={args.n_embed}, n_layers={args.n_layers}, "
          f"block_size={args.block_size}, batch_size={args.batch_size}, "
          f"lr={args.lr}, max_iters={args.max_iters}, attn={attn_type}")

    # Load memmap data
    print(f"\nLoading preprocessed data from {args.data_dir}...")
    train_data, val_data, tokenizer, meta = load_memmap_data(args.data_dir)
    actual_vocab_size = meta['vocab_size']
    print(f"Total tokens: {meta['total_tokens']:,}")
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
    print(f"Vocab size: {actual_vocab_size}")

    if len(val_data) < args.block_size + 1:
        print("WARNING: val data too small for block_size, reducing block_size")
        args.block_size = len(val_data) - 2

    # Train each model
    results = {}
    for model_name in args.models:
        torch.manual_seed(args.seed)  # same init for fair comparison
        cls = MODEL_CLASSES[model_name]
        model = cls(actual_vocab_size, args.n_embed, args.n_layers,
                    args.block_size, args.dropout, use_softmax=args.softmax)
        val_loss, val_ppl, ppl_log = train_model(
            model_name, model, train_data, val_data, args, device, tokenizer
        )
        results[model_name] = {'val_loss': val_loss, 'val_ppl': val_ppl,
                               'ppl_curve': ppl_log}

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Val Loss':>10} {'Val PPL':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10}")
    for name in args.models:
        r = results[name]
        print(f"{name:<25} {r['val_loss']:>10.4f} {r['val_ppl']:>10.2f}")
    print(f"{'='*60}")

    # Find best
    best_name = min(results, key=lambda k: results[k]['val_loss'])
    print(f"\nBest model: {best_name} (val PPL {results[best_name]['val_ppl']:.2f})")

    # Save results with PPL curves
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        "config": {
            "n_embed": args.n_embed, "n_layers": args.n_layers,
            "block_size": args.block_size, "batch_size": args.batch_size,
            "lr": args.lr, "max_iters": args.max_iters,
            "vocab_size": actual_vocab_size, "models": args.models,
            "data_dir": args.data_dir,
            "total_tokens": meta['total_tokens'],
        },
        "results": results,
        "timestamp": timestamp,
    }
    results_file = os.path.join(results_dir, f"joformer_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    latest_file = os.path.join(results_dir, "joformer_results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_training_args(parser):
    """Add training-specific arguments to a subparser."""
    parser.add_argument('--models', nargs='+',
                        default=['roformer', 'joformer_fixed', 'joformer_learned',
                                 'joformer_projected'],
                        choices=list(MODEL_CLASSES.keys()),
                        help='Which models to train')
    parser.add_argument('--n_embed', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--block_size', type=int, default=64,
                        help='Context window size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=10000,
                        help='Training iterations')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Eval frequency (iterations)')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Checkpoint directory (empty to disable)')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick test: 50 iters, small model')
    parser.add_argument('--generate_len', type=int, default=200,
                        help='Generation sample length in tokens')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--softmax', action='store_true',
                        help='Use softmax attention instead of normalized softplus')
    parser.add_argument('--cosine_decay', action='store_true',
                        help='Use cosine annealing LR schedule')


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient wiki training with memmap data"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # --- preprocess subcommand ---
    pp = subparsers.add_parser('preprocess', help='Tokenize wiki text to binary file')
    pp.add_argument('--wiki_path', type=str, default=None,
                    help='Path to wiki.en.txt')
    pp.add_argument('--wiki_lines', type=int, default=None,
                    help='Max lines to process (default: all)')
    pp.add_argument('--vocab_size', type=int, default=8000,
                    help='BPE vocabulary size')
    pp.add_argument('--data_dir', type=str, default='joformer/data',
                    help='Output directory for preprocessed files')

    # --- train subcommand ---
    tr = subparsers.add_parser('train', help='Train models from preprocessed data')
    tr.add_argument('--data_dir', type=str, default='joformer/data',
                    help='Directory containing preprocessed data')
    add_training_args(tr)

    # --- auto subcommand (preprocess if needed, then train) ---
    au = subparsers.add_parser('auto', help='Preprocess if needed, then train')
    au.add_argument('--wiki_path', type=str, default=None,
                    help='Path to wiki.en.txt')
    au.add_argument('--wiki_lines', type=int, default=None,
                    help='Max lines to process (default: all)')
    au.add_argument('--vocab_size', type=int, default=8000,
                    help='BPE vocabulary size')
    au.add_argument('--data_dir', type=str, default='joformer/data',
                    help='Directory for preprocessed data')
    add_training_args(au)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Default wiki path relative to this script
    if hasattr(args, 'wiki_path') and args.wiki_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.wiki_path = os.path.join(script_dir, '..', 'exp8', 'data', 'wiki.en.txt')

    if args.command == 'preprocess':
        preprocess(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'auto':
        bin_path = os.path.join(args.data_dir, 'wiki_tokens.bin')
        if not os.path.exists(bin_path):
            print("Preprocessed data not found, running preprocessing...")
            preprocess(args)
        else:
            print(f"Using existing preprocessed data in {args.data_dir}")
        run_training(args)


if __name__ == '__main__':
    main()

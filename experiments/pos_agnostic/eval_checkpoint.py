#!/usr/bin/env python3
"""Re-evaluate a saved checkpoint with different eval settings (e.g. topk)."""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from models import GPTModel
from train import load_memmap_data, get_batch


@torch.no_grad()
def eval_lengths(model, val_data, lengths, batch_size, device, eval_iters=20):
    model.eval()
    results = {}
    for length in lengths:
        if len(val_data) < length + 1:
            continue
        try:
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(val_data, length, batch_size, device)
                _, loss = model(x, y)
                if not torch.isnan(loss):
                    losses.append(loss.item())
            if losses:
                avg = sum(losses) / len(losses)
                results[length] = round(math.exp(avg), 2)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            results[length] = f"OOM"
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--eval_topk', type=int, default=0)
    parser.add_argument('--eval_lengths', type=str, default='512,1024,2048,4096')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_iters', type=int, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    print(f"Config: {cfg}")

    # Build model
    model = GPTModel(
        cfg['vocab_size'], cfg['n_embed'], cfg['n_layers'], cfg['n_heads'],
        cfg['block_size'], dropout=0.0, attn_config=cfg['attn_config'],
        window_size=cfg.get('window_size', 999999),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    if args.eval_topk > 0:
        model.set_eval_topk(args.eval_topk)
        print(f"Eval top-k: {args.eval_topk}")

    # Load data
    _, val_data, meta = load_memmap_data(args.data_dir)
    lengths = [int(x) for x in args.eval_lengths.split(',')]

    # Evaluate
    results = eval_lengths(model, val_data, lengths, args.batch_size, device, args.eval_iters)
    print(f"\nResults (topk={args.eval_topk}):")
    for l, ppl in sorted(results.items()):
        print(f"  {l}: {ppl}")


if __name__ == '__main__':
    main()

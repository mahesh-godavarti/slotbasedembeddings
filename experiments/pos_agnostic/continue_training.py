#!/usr/bin/env python3
"""Continue training from a checkpoint with a new learning rate."""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import GPTModel
from train import load_memmap_data, get_batch, estimate_loss, eval_lengths
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--extrap_interval', type=int, default=25000)
    parser.add_argument('--eval_lengths', type=str, default='512,1024,2048,4096')
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--cosine_decay', action='store_true')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 mixed precision training')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    print(f"Resuming from: {args.checkpoint}")
    print(f"Config: {cfg}")
    print(f"New lr: {args.lr}, iters: {args.max_iters}")

    # Reconstruct correct string config and n_embed from checkpoint
    from eval_all import detect_attn_config, detect_n_embed
    attn_config = detect_attn_config(cfg)
    n_embed = detect_n_embed(cfg)
    print(f"Detected attn_config: {attn_config}, n_embed: {n_embed}")

    # Build model
    model = GPTModel(
        cfg['vocab_size'], n_embed, cfg['n_layers'], cfg['n_heads'],
        cfg['block_size'], dropout=0.1, attn_config=attn_config,
        window_size=cfg.get('window_size', 32),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.train()

    model_name = os.path.basename(args.checkpoint).replace('_best.pt', '').replace('_final.pt', '')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params:,} params)")

    # Load data
    train_data, val_data, meta = load_memmap_data(args.data_dir)
    print(f"Data: {meta['total_tokens']:,} tokens")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters)
                 if args.cosine_decay else None)
    use_bf16 = args.bf16 and torch.cuda.is_bf16_supported()
    if use_bf16:
        print(f"Using BF16 mixed precision")
    scaler = torch.amp.GradScaler('cuda', enabled=not use_bf16)  # GradScaler for FP16 only, not BF16

    block_size = cfg['block_size']
    eval_lengths_list = [int(x) for x in args.eval_lengths.split(',')]
    ppl_log = {"iter": [], "train_ppl": [], "val_ppl": []}
    extrap_log = []
    best_val_loss = float('inf')

    pbar = tqdm(range(args.max_iters), desc=model_name)
    for it in pbar:
        # Eval
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   block_size, args.batch_size, device)
            t_ppl = math.exp(losses['train'])
            v_ppl = math.exp(losses['val'])
            ppl_log["iter"].append(it)
            ppl_log["train_ppl"].append(round(t_ppl, 2))
            ppl_log["val_ppl"].append(round(v_ppl, 2))
            pbar.set_postfix(train_ppl=f"{t_ppl:.2f}", val_ppl=f"{v_ppl:.2f}")
            tqdm.write(f"  [{model_name}] iter {it}: train PPL {t_ppl:.2f}, val PPL {v_ppl:.2f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if args.checkpoint_dir:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pt")
                    torch.save({
                        'iter': it,
                        'model_state_dict': model.state_dict(),
                        'val_loss': losses['val'],
                        'config': cfg,
                    }, path)

        # Extrap eval
        if (args.extrap_interval > 0 and it > 0
                and (it % args.extrap_interval == 0 or it == args.max_iters - 1)):
            results = eval_lengths(model, val_data, eval_lengths_list,
                                   args.eval_batch_size, device)
            extrap_log.append({'iter': it, 'results': {str(k): v for k, v in results.items()}})
            parts = [f"{l}:{r['ppl']}" for l, r in sorted(results.items())
                     if r.get('ppl') is not None]
            tqdm.write(f"  [{model_name}] extrap iter {it}: {', '.join(parts)}")

        # Train step
        x, y = get_batch(train_data, block_size, args.batch_size, device)
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

    # Final eval
    losses = estimate_loss(model, train_data, val_data, block_size, args.batch_size, device)
    v_ppl = math.exp(losses['val'])
    print(f"\n  [{model_name}] final val PPL: {v_ppl:.2f}")

    final_extrap = eval_lengths(model, val_data, eval_lengths_list,
                                args.eval_batch_size, device, eval_iters=20)
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
            'config': cfg,
        }, path)

    # Save results
    save_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"results_continue_{timestamp}.json")
    save_data = {
        'resumed_from': args.checkpoint,
        'config': cfg,
        'new_lr': args.lr,
        'max_iters': args.max_iters,
        'results': {
            'val_loss': losses['val'],
            'val_ppl': round(v_ppl, 2),
            'ppl_curve': ppl_log,
            'extrap_curve': extrap_log,
            'final_extrap': {str(k): v for k, v in final_extrap.items()},
        },
        'timestamp': timestamp,
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()

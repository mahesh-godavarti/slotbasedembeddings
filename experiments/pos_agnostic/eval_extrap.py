#!/usr/bin/env python3
"""Evaluate checkpoints at extended sequence lengths for length extrapolation."""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GPTModel
from train import load_memmap_data, get_batch
from eval_all import detect_attn_config, detect_n_embed


def eval_checkpoint_extrap(checkpoint_path, val_data, eval_lengths, batch_size=1,
                           eval_iters=20, device='cuda:0'):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    sd = ckpt['model_state_dict']
    attn_config = detect_attn_config(cfg)
    n_embed = detect_n_embed(cfg, sd)
    split_angles = 'angle_emb.weight' in sd

    model = GPTModel(
        cfg['vocab_size'], n_embed, cfg['n_layers'], cfg['n_heads'],
        cfg['block_size'], dropout=0.0, attn_config=attn_config,
        window_size=cfg.get('window_size', 999999), split_angles=split_angles,
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    name = os.path.basename(checkpoint_path).replace('_best.pt', '').replace('_final.pt', '')
    print(f"\n{name} (iter {ckpt.get('iter', '?')}, {n_params:,} params, val PPL {math.exp(ckpt.get('val_loss', 0)):.2f}):")

    results = {}
    with torch.no_grad():
        for length in eval_lengths:
            torch.manual_seed(42 + length)
            losses = []
            bs = batch_size
            # Reduce batch size for long sequences
            if length >= 16384:
                bs = min(bs, 1)
            elif length >= 8192:
                bs = min(bs, 2)
            try:
                for _ in range(eval_iters):
                    x, y = get_batch(val_data, length, bs, device)
                    _, loss = model(x, y)
                    if not torch.isnan(loss):
                        losses.append(loss.item())
                avg = sum(losses) / len(losses)
                ppl = round(math.exp(avg), 2)
                results[length] = ppl
                print(f"  {length}: {ppl}")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    results[length] = 'OOM'
                    print(f"  {length}: OOM")
                else:
                    raise
    del model
    torch.cuda.empty_cache()
    return {'name': name, 'iter': ckpt.get('iter', '?'), 'params': n_params, 'results': results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate length extrapolation")
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Checkpoint paths to evaluate')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--eval_lengths', type=str, default='512,1024,2048,4096,8192,16384,32768,65536')
    parser.add_argument('--eval_iters', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    eval_lengths = [int(x) for x in args.eval_lengths.split(',')]

    print(f"Loading data from {args.data_dir}...")
    _, val_data, meta = load_memmap_data(args.data_dir)
    print(f"  Val tokens: {len(val_data):,}, vocab: {meta['vocab_size']}")

    all_results = []
    for ckpt_path in args.checkpoints:
        result = eval_checkpoint_extrap(ckpt_path, val_data, eval_lengths,
                                         args.batch_size, args.eval_iters, device)
        all_results.append(result)

    # Print comparison table
    print(f"\n{'='*80}")
    header = f"{'Length':>8}"
    for r in all_results:
        header += f"  {r['name']:>20}"
    print(header)
    print('-' * len(header))
    for l in eval_lengths:
        line = f"{l:>8}"
        for r in all_results:
            val = r['results'].get(l, '-')
            if isinstance(val, (int, float)):
                line += f"  {val:>20.2f}"
            else:
                line += f"  {str(val):>20}"
        print(line)
    print(f"{'='*80}")

    # Save results
    from datetime import datetime
    save_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"eval_extrap_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()

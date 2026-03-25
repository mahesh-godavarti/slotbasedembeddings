#!/usr/bin/env python3
"""
Evaluate saved checkpoints with clean 200-iteration eval.

Usage:
  # Eval a single checkpoint
  python eval_all.py --checkpoint path/to/model_best.pt --data_dir path/to/data

  # Eval all checkpoints in a directory
  python eval_all.py --checkpoint_dir path/to/checkpoints/ --data_dir path/to/data

  # Custom eval iterations and lengths
  python eval_all.py --checkpoint path/to/model.pt --data_dir path/to/data \
      --eval_iters 200 --eval_lengths 512,1024,2048,4096 --batch_size 4

Model type detection:
  The checkpoint saves attn_config as a list (e.g., ['rope','rope',...,'nope']).
  This script maps list configs back to string configs for correct model construction.
  For datadep2/joformer2 models, it detects the larger embedding and uses n_embed
  derived from the actual content dimension (embedding_dim * 2 / 3).
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GPTModel
from train import load_memmap_data, get_batch


# ---------------------------------------------------------------------------
# Map saved list configs back to string configs
# ---------------------------------------------------------------------------

def detect_attn_config(cfg):
    """Detect the string attn_config from checkpoint config.

    Checkpoints save attn_config as a list of per-layer types.
    We need to reconstruct the string config for correct model construction,
    especially for window size assignment and datadep2/joformer2 embedding.
    """
    attn_config = cfg['attn_config']

    # If already a string, use it directly
    if isinstance(attn_config, str):
        return attn_config

    # It's a list — detect the pattern
    types = attn_config
    n = len(types)
    unique = set(types)

    # All same type
    if len(unique) == 1:
        return types[0]

    # Hybrid patterns: some type in early layers + nope at end
    if types[-1] == 'nope':
        # Count trailing nope layers
        k = 0
        for t in reversed(types):
            if t == 'nope':
                k += 1
            else:
                break
        base = types[0]  # The non-nope type

        # Check all non-nope layers are the same
        if all(t == base for t in types[:n - k]):
            if base == 'rope':
                return f'hybrid_{k}'
            elif base == 'joformer_fixed':
                return f'joformer_fixed_hybrid_{k}'
            elif base == 'joformer':
                return f'joformer_hybrid_{k}'
            elif base == 'joformer2':
                return f'joformer2_hybrid_{k}'
            elif base == 'monoidal':
                return f'monoidal_hybrid_{k}'
            elif base == 'monoidal2':
                return f'monoidal2_hybrid_{k}'
            elif base == 'datadep':
                return f'datadep_hybrid_{k}'
            elif base == 'datadep3':
                return f'datadep3_hybrid_{k}'

    # Cohere pattern: nope every 3rd layer
    if all(t in ('rope', 'nope') for t in types):
        nope_positions = [i for i, t in enumerate(types) if t == 'nope']
        if all((p + 1) % 3 == 0 for p in nope_positions):
            return 'cohere'

    # Alternating
    if all(t in ('rope', 'nope') for t in types):
        if all(types[i] == ('rope' if i % 2 == 0 else 'nope') for i in range(n)):
            return 'alternating'

    # Fallback: return the list (will use list-based construction)
    return attn_config


def detect_n_embed(cfg):
    """Detect the correct n_embed for model construction.

    For datadep2/joformer2 models, the checkpoint saves n_embed as the
    embedding dimension (C + C//2), but we need the content dimension C.
    """
    attn_config = cfg['attn_config']
    n_embed = cfg['n_embed']

    # Check if this is a v2 model (has larger embedding)
    is_v2 = False
    if isinstance(attn_config, list):
        is_v2 = any(t in ('datadep2', 'joformer2', 'monoidal2') for t in attn_config)
    elif isinstance(attn_config, str):
        is_v2 = any(attn_config.startswith(p) for p in ('datadep2', 'joformer2', 'monoidal2'))

    if is_v2:
        # n_embed in checkpoint is C + C//2 = 3C/2, so C = n_embed * 2 // 3
        content_dim = n_embed * 2 // 3
        return content_dim

    return n_embed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_checkpoint(checkpoint_path, val_data, eval_lengths, batch_size=4,
                    eval_iters=200, device='cuda'):
    """Load a checkpoint and evaluate at multiple lengths.

    Returns: dict with model info and PPL at each length.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    attn_config = detect_attn_config(cfg)
    n_embed = detect_n_embed(cfg)
    window_size = cfg.get('window_size', 32)

    model = GPTModel(
        cfg['vocab_size'], n_embed, cfg['n_layers'], cfg['n_heads'],
        cfg['block_size'], dropout=0.0, attn_config=attn_config,
        window_size=window_size,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    results = {
        'checkpoint': checkpoint_path,
        'attn_config': attn_config if isinstance(attn_config, str) else str(attn_config),
        'n_embed': n_embed,
        'n_params': n_params,
        'iter': ckpt.get('iter', '?'),
        'val_loss': ckpt.get('val_loss', None),
        'lengths': {},
    }

    rng_state = torch.random.get_rng_state()
    for length in eval_lengths:
        if len(val_data) < length + 1:
            results['lengths'][length] = None
            continue
        try:
            torch.manual_seed(42 + length)
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(val_data, length, batch_size, device)
                _, loss = model(x, y)
                if not torch.isnan(loss):
                    losses.append(loss.item())
            if losses:
                avg = sum(losses) / len(losses)
                results['lengths'][length] = round(math.exp(avg), 2)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                results['lengths'][length] = 'OOM'
            else:
                raise

    torch.random.set_rng_state(rng_state)
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to a single checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Directory containing checkpoints (evals all *_best.pt)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to preprocessed data')
    parser.add_argument('--eval_iters', type=int, default=200,
                        help='Number of eval iterations (default 200)')
    parser.add_argument('--eval_lengths', type=str, default='512,1024,2048,4096',
                        help='Comma-separated eval lengths')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Eval batch size')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_lengths = [int(x) for x in args.eval_lengths.split(',')]

    # Load data
    print(f"Loading data from {args.data_dir}...")
    _, val_data, meta = load_memmap_data(args.data_dir)
    print(f"  Val tokens: {len(val_data):,}, vocab: {meta['vocab_size']}")

    # Collect checkpoints
    checkpoints = []
    if args.checkpoint:
        checkpoints.append(args.checkpoint)
    if args.checkpoint_dir:
        for f in sorted(os.listdir(args.checkpoint_dir)):
            if f.endswith('_best.pt'):
                checkpoints.append(os.path.join(args.checkpoint_dir, f))

    if not checkpoints:
        print("No checkpoints found.")
        return

    # Evaluate each
    all_results = []
    for ckpt_path in checkpoints:
        name = os.path.basename(ckpt_path).replace('_best.pt', '').replace('_final.pt', '')
        print(f"\nEvaluating {name}...")
        result = eval_checkpoint(ckpt_path, val_data, eval_lengths,
                                  args.batch_size, args.eval_iters, device)
        all_results.append(result)

        # Print immediately
        ppls = [f"{l}:{result['lengths'].get(l, '?')}" for l in eval_lengths]
        print(f"  {name} (iter {result['iter']}, {result['n_params']:,} params): {', '.join(ppls)}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON (200-iteration eval)")
    print(f"{'='*80}")
    header = f"{'Model':<30} {'Params':>10} {'Iter':>8}"
    for l in eval_lengths:
        header += f" {'PPL@'+str(l):>10}"
    print(header)
    print('-' * len(header))

    for r in all_results:
        name = os.path.basename(r['checkpoint']).replace('_best.pt', '').replace('_final.pt', '')
        line = f"{name:<30} {r['n_params']:>10,} {str(r['iter']):>8}"
        for l in eval_lengths:
            ppl = r['lengths'].get(l, '?')
            if isinstance(ppl, (int, float)):
                line += f" {ppl:>10.2f}"
            else:
                line += f" {str(ppl):>10}"
        print(line)
    print(f"{'='*80}")

    # Save results
    save_dir = os.path.dirname(os.path.abspath(__file__))
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"eval_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()

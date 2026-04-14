#!/usr/bin/env python3
"""Convert a roformer N=D checkpoint to a block_head_corr_ffn_add D checkpoint.

Copies all shared parameters (token_embedding, blocks, ln_f) directly.
Maps lm_head -> head. Initializes corr_ffn and ln_corr to produce near-zero corrections.

Usage:
    python convert_roformer_to_lookahead.py \
        --roformer_ckpt checkpoints_n12/roformer_latest.pt \
        --output_ckpt checkpoints_d12_converted/block_head_corr_ffn_add_latest.pt \
        --n_embed 1024 --n_layers 60 --d_block 12 --n_head 16 \
        --block_size 256 --vocab_size 32000
"""

import argparse
import os
import torch
import torch.nn as nn


def convert(args):
    # Load roformer checkpoint
    print(f"Loading roformer checkpoint: {args.roformer_ckpt}")
    ckpt = torch.load(args.roformer_ckpt, map_location='cpu', weights_only=False)
    roformer_state = ckpt['model_state_dict']

    # Create look-ahead model to get the target state dict structure
    # Import here to avoid issues with PYTHONPATH
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MODEL_CLASSES

    la_model = MODEL_CLASSES['block_head_corr_ffn_add'](
        args.vocab_size, args.n_embed, args.n_layers, args.block_size, 0.0,
        use_softmax=True, n_head=args.n_head, convergence_weight=0.1,
        d_block=args.d_block, k_min=2
    )
    la_state = la_model.state_dict()

    # Copy matching parameters
    copied = []
    for name in la_state:
        if name in roformer_state:
            assert la_state[name].shape == roformer_state[name].shape, \
                f"Shape mismatch for {name}: {la_state[name].shape} vs {roformer_state[name].shape}"
            la_state[name] = roformer_state[name]
            copied.append(name)

    # Map lm_head -> head
    la_state['head.weight'] = roformer_state['lm_head.weight']
    la_state['head.bias'] = roformer_state['lm_head.bias']
    copied.extend(['head.weight (from lm_head.weight)', 'head.bias (from lm_head.bias)'])

    # Initialize corr_ffn to near-zero output
    # corr_ffn is FeedForward: Linear(C, 4C) -> GELU -> Linear(4C, C)
    # Zero the last linear layer so initial correction = 0
    nn.init.zeros_(la_state['corr_ffn.ffn.2.weight'])
    nn.init.zeros_(la_state['corr_ffn.ffn.2.bias'])
    # First layer can be random (output is zeroed anyway)
    print("  corr_ffn.ffn.2 initialized to zeros (correction starts as zero)")

    # ln_corr: initialize to standard (weight=1, bias=0) — already default
    print("  ln_corr initialized to default (weight=1, bias=0)")

    print(f"\nCopied {len(copied)} parameter groups from roformer")
    print(f"New parameters (initialized): corr_ffn.ffn.0, corr_ffn.ffn.2, ln_corr")

    # Build new checkpoint
    new_ckpt = {
        'model_state_dict': la_state,
        'optimizer_state_dict': None,  # Fresh optimizer
        'scheduler_state_dict': None,
        'scaler_state_dict': None,
        'val_loss': ckpt.get('val_loss', None),
        'val_ppl': ckpt.get('val_ppl', None),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
        'iter': 0,  # Start from iter 0
        'ppl_log': {'iter': [], 'train_ppl': [], 'val_ppl': []},
        'diagnostics_log': [],
    }

    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save(new_ckpt, args.output_ckpt)
    print(f"\nSaved converted checkpoint to: {args.output_ckpt}")
    print(f"  iter: 0 (fresh start with pretrained weights)")
    print(f"  optimizer: None (will be re-initialized)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roformer_ckpt', required=True)
    parser.add_argument('--output_ckpt', required=True)
    parser.add_argument('--n_embed', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=60, help='D * K')
    parser.add_argument('--d_block', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=32000)
    args = parser.parse_args()
    convert(args)

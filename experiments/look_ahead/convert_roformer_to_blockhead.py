#!/usr/bin/env python3
"""Convert a roformer N=D checkpoint to a block_head D checkpoint (no corr_ffn).

Copies all shared parameters (token_embedding, blocks, ln_f).
Maps lm_head -> head. No corr_ffn needed.

Usage:
    python convert_roformer_to_blockhead.py \
        --roformer_ckpt checkpoints/roformer_latest.pt \
        --output_ckpt checkpoints_blockhead_d24/block_head_latest.pt \
        --n_embed 1024 --n_layers 120 --d_block 24 --n_head 16 \
        --block_size 256 --vocab_size 32000
"""

import argparse
import os
import torch


def convert(args):
    print(f"Loading roformer checkpoint: {args.roformer_ckpt}")
    ckpt = torch.load(args.roformer_ckpt, map_location='cpu', weights_only=False)
    roformer_state = ckpt['model_state_dict']

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MODEL_CLASSES

    bh_model = MODEL_CLASSES['block_head'](
        args.vocab_size, args.n_embed, args.n_layers, args.block_size, 0.0,
        use_softmax=True, n_head=args.n_head, convergence_weight=0.1,
        d_block=args.d_block, k_min=2
    )
    bh_state = bh_model.state_dict()

    copied = []
    for name in bh_state:
        if name in roformer_state:
            assert bh_state[name].shape == roformer_state[name].shape, \
                f"Shape mismatch for {name}: {bh_state[name].shape} vs {roformer_state[name].shape}"
            bh_state[name] = roformer_state[name]
            copied.append(name)

    # Handle D=1: block_head uses 'block.X' but roformer uses 'blocks.0.X'
    if args.d_block == 1:
        for name in list(bh_state.keys()):
            if name.startswith('block.'):
                roformer_name = 'blocks.0.' + name[len('block.'):]
                if roformer_name in roformer_state:
                    assert bh_state[name].shape == roformer_state[roformer_name].shape
                    bh_state[name] = roformer_state[roformer_name]
                    copied.append(f"{name} (from {roformer_name})")

    # Map lm_head -> head
    bh_state['head.weight'] = roformer_state['lm_head.weight']
    bh_state['head.bias'] = roformer_state['lm_head.bias']
    copied.extend(['head.weight (from lm_head.weight)', 'head.bias (from lm_head.bias)'])

    print(f"Copied {len(copied)} parameter groups from roformer")

    new_ckpt = {
        'model_state_dict': bh_state,
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'scaler_state_dict': None,
        'val_loss': ckpt.get('val_loss', None),
        'val_ppl': ckpt.get('val_ppl', None),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
        'iter': 0,
        'ppl_log': {'iter': [], 'train_ppl': [], 'val_ppl': []},
        'diagnostics_log': [],
    }

    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save(new_ckpt, args.output_ckpt)
    print(f"Saved converted checkpoint to: {args.output_ckpt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roformer_ckpt', required=True)
    parser.add_argument('--output_ckpt', required=True)
    parser.add_argument('--n_embed', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=120, help='D * K')
    parser.add_argument('--d_block', type=int, default=24)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=32000)
    args = parser.parse_args()
    convert(args)

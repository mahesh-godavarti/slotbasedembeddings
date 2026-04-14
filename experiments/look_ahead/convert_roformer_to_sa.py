#!/usr/bin/env python3
"""Convert a roformer N=D+1 checkpoint to a SA D checkpoint.

Copies D blocks from the roformer (drops the last block).
Maps lm_head -> head. Zero-initializes corr_attn and corr_ffn output layers.

Usage:
    python convert_roformer_to_sa.py \
        --roformer_ckpt checkpoints_n3_c2656/roformer_latest.pt \
        --output_ckpt checkpoints_sa_d3_c2656_ft/block_head_sa_corr_ffn_add_latest.pt \
        --n_embed 2656 --n_layers 15 --d_block 3 --n_head 16 \
        --block_size 256 --vocab_size 32000
"""

import argparse
import os
import torch
import torch.nn as nn


def convert(args):
    print(f"Loading roformer checkpoint: {args.roformer_ckpt}")
    ckpt = torch.load(args.roformer_ckpt, map_location='cpu', weights_only=False)
    roformer_state = ckpt['model_state_dict']

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MODEL_CLASSES

    sa_model = MODEL_CLASSES['block_head_sa_corr_ffn_add'](
        args.vocab_size, args.n_embed, args.n_layers, args.block_size, 0.0,
        use_softmax=True, n_head=args.n_head, convergence_weight=0.1,
        d_block=args.d_block, k_min=2
    )
    sa_state = sa_model.state_dict()

    copied = []

    # Copy matching parameters (token_embedding, blocks, ln_f)
    for name in sa_state:
        if name in roformer_state:
            if sa_state[name].shape == roformer_state[name].shape:
                sa_state[name] = roformer_state[name]
                copied.append(name)

    # Handle D=1: SA model uses 'block.X' but roformer uses 'blocks.0.X'
    if args.d_block == 1:
        for name in list(sa_state.keys()):
            if name.startswith('block.'):
                roformer_name = 'blocks.0.' + name[len('block.'):]
                if roformer_name in roformer_state:
                    if sa_state[name].shape == roformer_state[roformer_name].shape:
                        sa_state[name] = roformer_state[roformer_name]
                        copied.append(f"{name} (from {roformer_name})")

    # Map lm_head -> head
    sa_state['head.weight'] = roformer_state['lm_head.weight']
    sa_state['head.bias'] = roformer_state['lm_head.bias']
    copied.extend(['head.weight (from lm_head.weight)', 'head.bias (from lm_head.bias)'])

    # Zero-initialize corr_attn output projection (attention starts as zero)
    nn.init.zeros_(sa_state['corr_attn.proj.weight'])
    nn.init.zeros_(sa_state['corr_attn.proj.bias'])
    print("  corr_attn.proj initialized to zeros (attention correction starts as zero)")

    # Zero-initialize corr_ffn output layer (correction starts as zero)
    nn.init.zeros_(sa_state['corr_ffn.ffn.2.weight'])
    nn.init.zeros_(sa_state['corr_ffn.ffn.2.bias'])
    print("  corr_ffn.ffn.2 initialized to zeros (FFN correction starts as zero)")

    # ln_q, ln_kv, ln_corr_ffn: default init (weight=1, bias=0)
    print("  ln_q, ln_kv, ln_corr_ffn initialized to default")

    print(f"\nCopied {len(copied)} parameter groups from roformer")

    new_ckpt = {
        'model_state_dict': sa_state,
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
    print(f"\nSaved converted checkpoint to: {args.output_ckpt}")
    print(f"  iter: 0 (fresh start with pretrained weights)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roformer_ckpt', required=True)
    parser.add_argument('--output_ckpt', required=True)
    parser.add_argument('--n_embed', type=int, default=2656)
    parser.add_argument('--n_layers', type=int, default=15, help='D * K')
    parser.add_argument('--d_block', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=32000)
    args = parser.parse_args()
    convert(args)

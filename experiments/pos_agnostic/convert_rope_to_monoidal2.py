#!/usr/bin/env python3
"""Convert a RoPE checkpoint to monoidal2 (split_angles) format.

The monoidal2 model starts as RoPE-equivalent (identity angle deviations)
by zeroing angle_emb and fc2_angles, and adding rope_base_angles buffer.
Same as convert_fixed_to_v2.py but targets monoidal2 (no V rotation).
"""

import argparse
import torch
from models import GPTModel


def convert(src_path, dst_path):
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    src_sd = ckpt['model_state_dict']

    # Build monoidal2 model with split_angles
    n_embed = cfg['n_embed']
    model = GPTModel(
        cfg['vocab_size'], n_embed, cfg['n_layers'], cfg['n_heads'],
        cfg['block_size'], dropout=0.1, attn_config='monoidal2',
        window_size=cfg.get('window_size', 999999),
        split_angles=True,
    )

    dst_sd = model.state_dict()

    # Map weights
    for key in dst_sd:
        if key == 'rope_base_angles':
            # Computed buffer, already set by model init
            continue
        elif key == 'angle_emb.weight':
            # Zero init (identity rotation deviation)
            dst_sd[key] = torch.zeros_like(dst_sd[key])
        elif '.ffn.fc2_angles.' in key:
            # Zero init
            dst_sd[key] = torch.zeros_like(dst_sd[key])
        elif '.ffn.fc1.' in key:
            # Map from ffn.net.0
            src_key = key.replace('.ffn.fc1.', '.ffn.net.0.')
            dst_sd[key] = src_sd[src_key]
        elif '.ffn.fc2_content.' in key:
            # Map from ffn.net.2
            src_key = key.replace('.ffn.fc2_content.', '.ffn.net.2.')
            dst_sd[key] = src_sd[src_key]
        elif key in src_sd:
            # Direct copy (tok_emb, attn.qkv, attn.out_proj, ln1, ln2, lm_head, ln_f)
            dst_sd[key] = src_sd[key]
        else:
            print(f"WARNING: no source for {key}, keeping random init")

    model.load_state_dict(dst_sd)

    # Save with monoidal2 config
    new_cfg = dict(cfg)
    new_cfg['attn_config'] = ['monoidal2'] * cfg['n_layers']
    torch.save({
        'iter': ckpt.get('iter', 0),
        'model_state_dict': dst_sd,
        'val_loss': ckpt.get('val_loss', float('inf')),
        'config': new_cfg,
    }, dst_path)
    print(f"Converted {src_path} -> {dst_path}")
    print(f"  Source config: {cfg['attn_config'][:3]}... ({cfg['n_layers']} layers)")
    print(f"  Dest config: monoidal2 x {cfg['n_layers']}, split_angles=True")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='RoPE checkpoint')
    parser.add_argument('--dst', type=str, required=True, help='output monoidal2 checkpoint')
    args = parser.parse_args()
    convert(args.src, args.dst)

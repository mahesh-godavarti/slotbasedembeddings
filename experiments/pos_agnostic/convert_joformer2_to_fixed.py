#!/usr/bin/env python3
"""Convert a joformer2 (split_angles, iter=0) checkpoint back to joformer_fixed format.

Drops angle_emb, rope_base_angles, fc2_angles.
Remaps ffn.fc1 → ffn.net.0, ffn.fc2_content → ffn.net.2.
"""

import argparse
import torch
from models import GPTModel


def convert(src_path, dst_path):
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    src_sd = ckpt['model_state_dict']

    n_embed = cfg['n_embed']
    model = GPTModel(
        cfg['vocab_size'], n_embed, cfg['n_layers'], cfg['n_heads'],
        cfg['block_size'], dropout=0.1, attn_config='joformer_fixed',
        window_size=cfg.get('window_size', 999999),
    )

    dst_sd = model.state_dict()

    for key in dst_sd:
        if '.ffn.net.0.' in key:
            src_key = key.replace('.ffn.net.0.', '.ffn.fc1.')
            dst_sd[key] = src_sd[src_key]
        elif '.ffn.net.2.' in key:
            src_key = key.replace('.ffn.net.2.', '.ffn.fc2_content.')
            dst_sd[key] = src_sd[src_key]
        elif key in src_sd:
            dst_sd[key] = src_sd[key]
        else:
            print(f"WARNING: no source for {key}")

    model.load_state_dict(dst_sd)

    new_cfg = dict(cfg)
    new_cfg['attn_config'] = ['joformer_fixed'] * cfg['n_layers']
    torch.save({
        'iter': ckpt.get('iter', 0),
        'model_state_dict': dst_sd,
        'val_loss': ckpt.get('val_loss', float('inf')),
        'config': new_cfg,
    }, dst_path)
    print(f"Converted {src_path} -> {dst_path}")
    print(f"  Dropped: angle_emb, rope_base_angles, fc2_angles")
    print(f"  Remapped: fc1 -> net.0, fc2_content -> net.2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    args = parser.parse_args()
    convert(args.src, args.dst)

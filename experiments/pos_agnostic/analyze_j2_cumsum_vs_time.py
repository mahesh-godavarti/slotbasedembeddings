#!/usr/bin/env python3
"""Analyze joformer2 cumsum growth as a function of position for long sequences."""

import torch
import sys
import json
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel
from train import load_memmap_data, get_batch

device = 'cuda:3'
torch.cuda.set_device(3)

ckpt_path = 'checkpoints/joformer2_angle_200k/joformer2_best.pt'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
cfg = ckpt['config']

model = GPTModel(
    cfg['vocab_size'], 768, cfg['n_layers'], cfg['n_heads'],
    cfg['block_size'], dropout=0.0, attn_config='joformer2',
    window_size=cfg.get('window_size', 999999),
    split_angles=True,
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(device)
model.eval()

train_data, val_data, meta = load_memmap_data('/home/ubuntu/look_ahead/look_ahead/data_owt')
torch.manual_seed(42)

# Get a long sequence (8192 tokens), batch size 1
x, y = get_batch(val_data, 8192, 1, device)
print(f"Input shape: {x.shape}")

rope_base = model.rope_base_angles.data

# Pick a few layers to analyze
analyze_layers = [0, 1, 2, 5, 10, 15]
# Pick a few dimensions (high freq, mid freq, low freq)
analyze_dims = [0, 5, 48, 96, 192, 383]

with torch.no_grad():
    initial_angles = torch.tanh(model.angle_emb(x)) * 3.14159 + rope_base
    x_hidden = model.tok_emb(x)
    angles = initial_angles

    for layer_idx, block in enumerate(model.blocks):
        if layer_idx in analyze_layers:
            # angles shape: (1, 8192, 384)
            a = angles[0]  # (8192, 384)

            # Cumsum (flip-cumsum-flip)
            flipped = torch.flip(a, dims=(0,))
            cs = torch.cumsum(flipped, dim=0)
            cs = torch.flip(cs, dims=(0,))

            # Pure rope cumsum
            rope_a = rope_base.unsqueeze(0).expand_as(a)
            rope_cs = torch.flip(torch.cumsum(torch.flip(rope_a, dims=(0,)), dim=0), dims=(0,))

            cs_diff = cs - rope_cs

            print(f"\n{'='*60}")
            print(f"LAYER {layer_idx} — angles into attention")
            print(f"{'='*60}")

            # Running mean of angles as function of position
            running_sum = torch.cumsum(a, dim=0)  # (T, 384)
            positions = torch.arange(1, 8193, device=device, dtype=torch.float32).unsqueeze(1)
            running_mean = running_sum / positions  # (T, 384)

            # Deviation of running mean from rope_base
            running_mean_dev = running_mean - rope_base.unsqueeze(0)

            for dim in analyze_dims:
                freq = rope_base[dim].abs().item()
                rm = running_mean_dev[:, dim].cpu()

                # Sample at positions 64, 128, 256, 512, 1024, 2048, 4096, 8192
                positions_to_show = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
                vals = [f"{rm[p-1].item():.4f}" for p in positions_to_show]

                # Also show cumsum diff at those positions
                cd = cs_diff[:, dim].cpu()
                cs_vals = [f"{cd[p-1].item():.1f}" for p in positions_to_show]

                print(f"\n  Dim {dim} (freq={freq:.6f}):")
                print(f"    Running mean dev from base at positions {positions_to_show}:")
                print(f"    {vals}")
                print(f"    Cumsum diff from RoPE at same positions:")
                print(f"    {cs_vals}")

        # Forward through block
        x_hidden = x_hidden + block.attn(block.ln1(x_hidden), angles)
        content, new_angles = block.ffn(block.ln2(x_hidden))
        new_angles = new_angles + rope_base
        x_hidden = x_hidden + content
        angles = new_angles

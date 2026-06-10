#!/usr/bin/env python3
"""Analyze joformer2 angles per layer."""

import torch
import sys
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
x, y = get_batch(val_data, 512, 4, device)

rope_base = model.rope_base_angles.data

with torch.no_grad():
    # Initial angles
    initial_angles = torch.tanh(model.angle_emb(x)) * 3.14159 + rope_base

    x_hidden = model.tok_emb(x)
    angles = initial_angles  # these go INTO layer 0's attention

    for i, block in enumerate(model.blocks):
        # angles is what THIS layer's attention uses
        # Compute cumsum (flip-cumsum-flip) for this layer
        flipped = torch.flip(angles, dims=(1,))
        cs = torch.cumsum(flipped, dim=1)
        cs = torch.flip(cs, dims=(1,))

        # Also compute pure rope cumsum
        rope_angles = rope_base.unsqueeze(0).unsqueeze(0).expand_as(angles)
        rope_flipped = torch.flip(rope_angles, dims=(1,))
        rope_cs = torch.cumsum(rope_flipped, dim=1)
        rope_cs = torch.flip(rope_cs, dims=(1,))

        # Deviation of angles from rope_base
        dev = angles - rope_base
        dev_mean_across_pos = dev[0].mean(dim=0)  # batch 0, mean across T

        # Cumsum difference
        cs_diff = cs - rope_cs

        print(f"Layer {i}:")
        print(f"  Angles into attention — abs mean: {angles[0].abs().mean().item():.4f}, "
              f"deviation from base abs mean: {dev[0].abs().mean().item():.4f}")
        print(f"  Deviation mean across positions — abs mean: {dev_mean_across_pos.abs().mean().item():.6f}")
        print(f"  Cumsum diff from RoPE — std: {cs_diff[0].std().item():.4f}, "
              f"abs mean: {cs_diff[0].abs().mean().item():.4f}")

        # Check a few dimensions
        for dim in [0, 50, 200]:
            d = cs_diff[0, :, dim]
            print(f"    Dim {dim} (freq={rope_base[dim].abs().item():.4f}): "
                  f"cumsum diff range=[{d.min().item():.2f}, {d.max().item():.2f}], std={d.std().item():.2f}")

        # Forward through the block
        x_hidden = x_hidden + block.attn(block.ln1(x_hidden), angles)
        content, new_angles = block.ffn(block.ln2(x_hidden))
        new_angles = new_angles + rope_base
        x_hidden = x_hidden + content
        angles = new_angles

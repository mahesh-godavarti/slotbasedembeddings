#!/usr/bin/env python3
"""Analyze joformer2 angle behavior from checkpoint."""

import torch
import numpy as np
import sys
import json
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel
from train import load_memmap_data, get_batch

device = 'cuda:3'
torch.cuda.set_device(3)

# Load checkpoint
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

# Load data for test sequences
train_data, val_data, meta = load_memmap_data('/home/ubuntu/look_ahead/look_ahead/data_owt')
torch.manual_seed(42)
x, y = get_batch(val_data, 512, 4, device)

print("=" * 60)
print("1. ANGLE EMBEDDING ANALYSIS")
print("=" * 60)

# Check angle_emb — how far from zero?
angle_emb_w = model.angle_emb.weight.data  # (vocab, C//2)
print(f"angle_emb shape: {angle_emb_w.shape}")
print(f"angle_emb abs mean: {angle_emb_w.abs().mean().item():.6f}")
print(f"angle_emb abs max: {angle_emb_w.abs().max().item():.6f}")
print(f"angle_emb std: {angle_emb_w.std().item():.6f}")
print(f"After tanh*pi — abs mean: {(torch.tanh(angle_emb_w) * 3.14159).abs().mean().item():.6f}")
print(f"After tanh*pi — abs max: {(torch.tanh(angle_emb_w) * 3.14159).abs().max().item():.6f}")

# Compare to rope_base
rope_base = model.rope_base_angles.data  # (C//2,)
print(f"\nrope_base abs mean: {rope_base.abs().mean().item():.6f}")
print(f"rope_base abs max: {rope_base.abs().max().item():.6f}")
print(f"rope_base abs min: {rope_base.abs().min().item():.6f}")
print(f"rope_base[:10]: {rope_base[:10].tolist()}")

# Ratio of learned deviation to base
learned_dev = (torch.tanh(angle_emb_w) * 3.14159).abs().mean(dim=0)  # per-dim mean
base_mag = rope_base.abs()
ratio = learned_dev / (base_mag + 1e-8)
print(f"\nLearned/base ratio per dim — mean: {ratio.mean().item():.4f}, max: {ratio.max().item():.4f}, min: {ratio.min().item():.4f}")

print("\n" + "=" * 60)
print("2. FC2_ANGLES WEIGHT ANALYSIS")
print("=" * 60)

for i, block in enumerate(model.blocks):
    w = block.ffn.fc2_angles.weight.data
    b = block.ffn.fc2_angles.bias.data
    w_norm = w.norm().item()
    b_norm = b.norm().item()
    w_abs_mean = w.abs().mean().item()
    if i < 3 or i == 15:
        print(f"Layer {i}: fc2_angles weight norm={w_norm:.4f}, abs_mean={w_abs_mean:.6f}, bias norm={b_norm:.4f}")

print("\n" + "=" * 60)
print("3. ACTUAL ANGLE VALUES ON REAL DATA")
print("=" * 60)

with torch.no_grad():
    # Get initial angles
    initial_angles = torch.tanh(model.angle_emb(x)) * 3.14159 + model.rope_base_angles
    print(f"Initial angles shape: {initial_angles.shape}")
    print(f"Initial angles abs mean: {initial_angles.abs().mean().item():.6f}")
    print(f"Initial angles mean (should be ~rope_base if learned is small): {initial_angles.mean().item():.6f}")

    # Trace through layers to get per-layer angles
    x_hidden = model.tok_emb(x)
    angles = initial_angles

    for i, block in enumerate(model.blocks):
        x_hidden = x_hidden + block.attn(block.ln1(x_hidden), angles)
        content, new_angles = block.ffn(block.ln2(x_hidden))
        new_angles = new_angles + model.rope_base_angles
        x_hidden = x_hidden + content

        if i < 3 or i == 15:
            # Deviation from rope_base
            dev = new_angles - model.rope_base_angles
            print(f"Layer {i}: angle abs mean={new_angles.abs().mean().item():.4f}, "
                  f"deviation abs mean={dev.abs().mean().item():.4f}, "
                  f"deviation/base ratio={dev.abs().mean().item() / rope_base.abs().mean().item():.4f}")
        angles = new_angles

print("\n" + "=" * 60)
print("4. CUMSUM PATTERNS")
print("=" * 60)

with torch.no_grad():
    # Get angles at final layer for one batch element
    x_hidden = model.tok_emb(x)
    angles = initial_angles
    for block in model.blocks:
        x_hidden = x_hidden + block.attn(block.ln1(x_hidden), angles)
        content, new_angles = block.ffn(block.ln2(x_hidden))
        new_angles = new_angles + model.rope_base_angles
        x_hidden = x_hidden + content
        angles = new_angles

    # Last layer angles for batch 0
    final_angles = angles[0]  # (T, C//2)

    # Cumsum (flip-cumsum-flip like the model does)
    flipped = torch.flip(final_angles, dims=(0,))
    cs = torch.cumsum(flipped, dim=0)
    cs = torch.flip(cs, dims=(0,))

    # Also compute pure rope cumsum for comparison
    rope_angles = model.rope_base_angles.unsqueeze(0).expand(512, -1)
    rope_flipped = torch.flip(rope_angles, dims=(0,))
    rope_cs = torch.cumsum(rope_flipped, dim=0)
    rope_cs = torch.flip(rope_cs, dims=(0,))

    # Compare cumsums at various dimensions
    for dim in [0, 10, 50, 100, 200, 383]:
        j2_vals = cs[:, dim]
        rope_vals = rope_cs[:, dim]
        diff = (j2_vals - rope_vals)
        print(f"Dim {dim} (freq={rope_base[dim].abs().item():.6f}):")
        print(f"  j2 cumsum range: [{j2_vals.min().item():.2f}, {j2_vals.max().item():.2f}], std={j2_vals.std().item():.2f}")
        print(f"  rope cumsum range: [{rope_vals.min().item():.2f}, {rope_vals.max().item():.2f}], std={rope_vals.std().item():.2f}")
        print(f"  difference std: {diff.std().item():.2f}, mean: {diff.mean().item():.2f}")

print("\n" + "=" * 60)
print("5. ANGLE MEAN ACROSS POSITIONS (drift analysis)")
print("=" * 60)

with torch.no_grad():
    # Check if angles are zero-mean across positions
    # Use final layer angles, batch 0
    pos_mean = final_angles.mean(dim=0)  # mean across T for each dim
    print(f"Per-dim mean across positions — abs mean: {pos_mean.abs().mean().item():.6f}")
    print(f"Per-dim mean across positions — max: {pos_mean.max().item():.6f}, min: {pos_mean.min().item():.6f}")
    print(f"rope_base mean: {rope_base.mean().item():.6f}")
    print(f"Expected if angles ≈ rope_base: mean should be ≈ rope_base")

    # Compare: how much does the mean deviate from rope_base?
    dev_from_base = pos_mean - rope_base
    print(f"Mean deviation from rope_base — abs mean: {dev_from_base.abs().mean().item():.6f}")
    print(f"This is the 'learned drift' per step on average")

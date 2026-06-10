#!/usr/bin/env python3
"""Analyze attention patterns in joformer2 — near vs far token contributions."""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel, apply_rotary_emb, build_attn_mask
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
x, y = get_batch(val_data, 8192, 1, device)

NEAR_THRESHOLD = 256
QUERY_START = 8092  # last 100 positions
QUERY_END = 8192

rope_base = model.rope_base_angles.data

print(f"Analyzing positions {QUERY_START}-{QUERY_END-1} (last 100 tokens)")
print(f"Near: within {NEAR_THRESHOLD} tokens, Far: beyond {NEAR_THRESHOLD} tokens")
print()

with torch.no_grad():
    initial_angles = torch.tanh(model.angle_emb(x)) * 3.14159 + rope_base
    x_hidden = model.tok_emb(x)
    angles = initial_angles

    for layer_idx, block in enumerate(model.blocks):
        B, T, C = x_hidden.shape
        h, d = block.attn.n_heads, block.attn.head_dim

        # Compute attention manually
        x_ln = block.ln1(x_hidden)

        # Cumsum on angles
        a = torch.flip(angles, dims=(1,))
        a = torch.cumsum(a, dim=1)
        a = torch.flip(a, dims=(1,))
        a = a.view(B, T, h, d // 2).transpose(1, 2)
        cos, sin = torch.cos(a), torch.sin(a)

        # QKV
        qkv = block.attn.qkv(x_ln).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply rotary
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # V rotation
        if block.attn.rotate_v:
            v_rot = apply_rotary_emb(v, cos, sin)
        else:
            v_rot = v

        # Attention scores — use float32 for stability
        q_f = q.float()
        k_f = k.float()
        scores = (q_f @ k_f.transpose(-1, -2)) * (d ** -0.5)

        # Causal mask
        mask = build_attn_mask(T, 999999, device)
        scores = scores.masked_fill(mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Analyze last 100 query positions
        # attn shape: (B, h, T, T)
        attn_queries = attn[0, :, QUERY_START:QUERY_END, :]  # (h, 100, T)

        # For each query position, compute near vs far attention
        near_attn_total = 0.0
        far_attn_total = 0.0
        near_v_norm_total = 0.0
        far_v_norm_total = 0.0
        n_queries = QUERY_END - QUERY_START

        for qi in range(n_queries):
            pos = QUERY_START + qi
            attn_row = attn_queries[:, qi, :pos+1]  # (h, pos+1) — causal

            # Near: positions within NEAR_THRESHOLD
            near_start = max(0, pos - NEAR_THRESHOLD + 1)
            near_mask = torch.zeros(pos + 1, device=device)
            near_mask[near_start:pos+1] = 1.0
            far_mask = 1.0 - near_mask

            # Attention mass
            near_attn = (attn_row * near_mask.unsqueeze(0)).sum(dim=-1)  # (h,)
            far_attn = (attn_row * far_mask.unsqueeze(0)).sum(dim=-1)  # (h,)

            near_attn_total += near_attn.mean().item()
            far_attn_total += far_attn.mean().item()

            # V contribution norms
            # attn_row: (h, pos+1), v_rot: (B, h, T, d)
            v_for_pos = v_rot[0, :, :pos+1, :]  # (h, pos+1, d)
            weighted_v = attn_row.unsqueeze(-1) * v_for_pos  # (h, pos+1, d)

            near_v = (weighted_v * near_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (h, d)
            far_v = (weighted_v * far_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (h, d)

            near_v_norm_total += near_v.norm(dim=-1).mean().item()
            far_v_norm_total += far_v.norm(dim=-1).mean().item()

        near_attn_avg = near_attn_total / n_queries
        far_attn_avg = far_attn_total / n_queries
        near_v_avg = near_v_norm_total / n_queries
        far_v_avg = far_v_norm_total / n_queries

        print(f"Layer {layer_idx:2d}: "
              f"attn near={near_attn_avg:.4f} far={far_attn_avg:.4f} "
              f"(near%={near_attn_avg/(near_attn_avg+far_attn_avg)*100:.1f}%) | "
              f"V_norm near={near_v_avg:.4f} far={far_v_avg:.4f} "
              f"(near%={near_v_avg/(near_v_avg+far_v_avg)*100:.1f}%)")

        # Forward through block normally
        x_hidden = x_hidden + block.attn(block.ln1(x_hidden), angles)
        content, new_angles = block.ffn(block.ln2(x_hidden))
        new_angles = new_angles + rope_base
        x_hidden = x_hidden + content
        angles = new_angles

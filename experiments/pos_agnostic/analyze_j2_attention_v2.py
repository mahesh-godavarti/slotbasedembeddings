#!/usr/bin/env python3
"""Analyze attention patterns using hooks."""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel, apply_rotary_emb, build_attn_mask, apply_inverse_rotary_emb
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
QUERY_START = 8092
QUERY_END = 8192

# Store attention weights per layer
attn_weights = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # Recompute attention weights inside the hook
        x_input = input[0]  # (B, T, C)
        angles = input[1]  # (B, T, C//2)
        B, T, C = x_input.shape
        h, d = module.n_heads, module.head_dim

        with torch.no_grad():
            qkv = module.qkv(x_input).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # Cumsum
            a = torch.flip(angles, dims=(1,))
            a = torch.cumsum(a, dim=1)
            a = torch.flip(a, dims=(1,))
            a = a.view(B, T, h, d // 2).transpose(1, 2)
            cos, sin = torch.cos(a), torch.sin(a)

            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

            if module.rotate_v:
                v = apply_rotary_emb(v, cos, sin)

            # Attention scores — only for query positions we care about
            # q_subset: (B, h, 100, d)
            q_sub = q[:, :, QUERY_START:QUERY_END, :].float()
            k_f = k.float()
            v_f = v.float()

            scores = (q_sub @ k_f.transpose(-1, -2)) * (d ** -0.5)  # (B, h, 100, T)

            # Causal mask for these query positions
            for qi in range(QUERY_END - QUERY_START):
                pos = QUERY_START + qi
                scores[:, :, qi, pos+1:] = float('-inf')

            attn = F.softmax(scores, dim=-1)  # (B, h, 100, T)

            # Compute near/far stats
            near_attn_sum = 0.0
            far_attn_sum = 0.0
            near_v_norm_sum = 0.0
            far_v_norm_sum = 0.0
            n_q = QUERY_END - QUERY_START

            for qi in range(n_q):
                pos = QUERY_START + qi
                attn_row = attn[0, :, qi, :pos+1]  # (h, pos+1)

                near_start = max(0, pos - NEAR_THRESHOLD + 1)
                near_mask = torch.zeros(pos + 1, device=device)
                near_mask[near_start:pos+1] = 1.0
                far_mask = 1.0 - near_mask

                near_a = (attn_row * near_mask.unsqueeze(0)).sum(dim=-1).mean().item()
                far_a = (attn_row * far_mask.unsqueeze(0)).sum(dim=-1).mean().item()
                near_attn_sum += near_a
                far_attn_sum += far_a

                # V contribution
                v_pos = v_f[0, :, :pos+1, :]  # (h, pos+1, d)
                wv = attn_row.unsqueeze(-1) * v_pos
                near_v = (wv * near_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                far_v = (wv * far_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                near_v_norm_sum += near_v.norm(dim=-1).mean().item()
                far_v_norm_sum += far_v.norm(dim=-1).mean().item()

            attn_weights[layer_idx] = {
                'near_attn': near_attn_sum / n_q,
                'far_attn': far_attn_sum / n_q,
                'near_v_norm': near_v_norm_sum / n_q,
                'far_v_norm': far_v_norm_sum / n_q,
            }
    return hook_fn

# Register hooks
hooks = []
for i, block in enumerate(model.blocks):
    h = block.attn.register_forward_hook(make_hook(i))
    hooks.append(h)

# Run forward
with torch.no_grad():
    logits, loss = model(x, y)

# Remove hooks
for h in hooks:
    h.remove()

# Print results
print(f"Analyzing positions {QUERY_START}-{QUERY_END-1} (last 100 tokens)")
print(f"Near: within {NEAR_THRESHOLD} tokens, Far: beyond {NEAR_THRESHOLD} tokens")
print(f"Loss at 8192: {loss.item():.4f}")
print()

for i in range(16):
    d = attn_weights[i]
    na, fa = d['near_attn'], d['far_attn']
    nv, fv = d['near_v_norm'], d['far_v_norm']
    near_pct_a = na / (na + fa) * 100
    near_pct_v = nv / (nv + fv) * 100
    print(f"Layer {i:2d}: "
          f"attn near={na:.4f} far={fa:.4f} (near%={near_pct_a:.1f}%) | "
          f"V_norm near={nv:.4f} far={fv:.4f} (near%={near_pct_v:.1f}%)")

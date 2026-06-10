import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_lfbf_qk',
                 window_size=999999, angle_hidden_mult=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_lfbf_qk: {n_params:,} params")
print(f"use_base_freq_learned: {model.shared_angle_mlp.use_base_freq_learned}")
print(f"base_freq shape: {model.shared_angle_mlp.base_freq.shape}")
print(f"base_freq[:5]: {model.shared_angle_mlp.base_freq[:5]}")
print(f"use_output_ln: {model.shared_angle_mlp.use_output_ln}")

# Quick forward pass
x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

# Also test qkv
model2 = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_lfbf_qkv',
                  window_size=999999, angle_hidden_mult=1)
x = torch.randint(0, 32000, (2, 64))
logits, loss = model2(x, x)
print(f"shared_lfbf_qkv forward pass OK, loss={loss.item():.4f}")

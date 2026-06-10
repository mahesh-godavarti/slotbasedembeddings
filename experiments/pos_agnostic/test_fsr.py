import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_fsr_qk',
                 window_size=999999, angle_hidden_mult=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_fsr_qk: {n_params:,} params")
print(f"random_freq_scales: {model.shared_angle_mlp.random_freq_scales}")
print(f"use_freq_scales: {model.shared_angle_mlp.use_freq_scales}")
print(f"use_output_ln: {model.shared_angle_mlp.use_output_ln}")
print(f"freq_scales[:5]: {model.shared_angle_mlp.freq_scales[:5]}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

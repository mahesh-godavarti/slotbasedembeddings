import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='joformer2',
                 window_size=999999, angle_activation='tanh_freq',
                 angle_hidden_mult=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"joformer2 tanh_freq h1: {n_params:,} params")
print(f"angle_freq_scales[:5]: {model.blocks[0].ffn.angle_freq_scales[:5]}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

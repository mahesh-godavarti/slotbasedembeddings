import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_fssx_qk',
                 window_size=999999)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_fssx_qk: {n_params:,} params")
print(f"has shared_angle_mlp: {hasattr(model, 'shared_angle_mlp')}")
print(f"has _fssx_ln: {hasattr(model, '_fssx_ln')}")
print(f"_fssx_freq_scales[:5]: {model._fssx_freq_scales[:5]}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

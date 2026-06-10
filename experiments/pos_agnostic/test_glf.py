import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='random_glf_qk',
                 window_size=999999)
n_params = sum(p.numel() for p in model.parameters())
print(f"random_glf_qk: {n_params:,} params")
print(f"random_angle_scales is Parameter: {isinstance(model.random_angle_scales, torch.nn.Parameter)}")
print(f"random_angle_scales[:5]: {model.random_angle_scales.data[:5]}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

# Check angle_params returns the scales
ap = model.angle_params()
print(f"angle_params count: {len(ap)}")

import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='joformer2',
                 window_size=999999, split_angles=True, angle_activation='ln_tanh_freq')
model._use_random_angle_emb = True
model.angle_emb.weight.requires_grad = False

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}")
print(f"_angle_emb_random_init shape: {model._angle_emb_random_init.shape}")
print(f"_angle_emb_random_init[0,:5]: {model._angle_emb_random_init[0,:5]}")
print(f"angle_emb frozen: {not model.angle_emb.weight.requires_grad}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

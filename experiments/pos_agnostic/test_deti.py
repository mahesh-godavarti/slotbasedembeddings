import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_deti_qk',
                 window_size=999999, angle_hidden_mult=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_deti_qk: {n_params:,} params")
print(f"sign_emb_0 shape: {model.sign_emb_0.shape}")
print(f"sign_emb_0[0,:5]: {model.sign_emb_0[0,:5]}")
print(f"sign_emb_1[0,:5]: {model.sign_emb_1[0,:5]}")
print(f"Different per layer: {(model.sign_emb_0 != model.sign_emb_1).any().item()}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

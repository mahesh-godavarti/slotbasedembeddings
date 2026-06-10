import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_pmlp_qk',
                 window_size=999999)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_pmlp_qk: {n_params:,} params")
print(f"base embs: {len(model.layer_angle_embs)}")
print(f"correction MLPs: {len(model.layer_correction_mlps)}")
print(f"MLP[0] layers: {model.layer_correction_mlps[0]}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

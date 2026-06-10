import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_pemb_qk',
                 window_size=999999)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_pemb_qk: {n_params:,} params")
print(f"layer_angle_embs: {len(model.layer_angle_embs)} layers")
print(f"each emb shape: {model.layer_angle_embs[0].weight.shape}")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

# K=4 (using angle_hidden_mult as K)
model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_cb_qk',
                 window_size=999999, angle_hidden_mult=4)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_cb_qk K=4: {n_params:,} params")
print(f"angle_codebook shape: {model.angle_codebook.shape}")
print(f"score_projs: {len(model.cb_score_projs)} layers, each {model.cb_score_projs[0].weight.shape}")

x = torch.randint(0, 32000, (2, 64))
model.train()
logits, loss = model(x, x)
loss.backward()
print(f"Forward pass OK, loss={loss.item():.4f}")
print(f"codebook grad norm: {model.angle_codebook.grad.norm().item():.6f}")
print(f"score_proj[0] grad norm: {model.cb_score_projs[0].weight.grad.norm().item():.6f}")

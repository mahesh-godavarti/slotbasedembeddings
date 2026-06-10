import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_fssd_qk',
                 window_size=999999, angle_hidden_mult=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_fssd_qk: {n_params:,} params")

x = torch.randint(0, 32000, (2, 64))
model.train()
logits, loss = model(x, x)
loss.backward()
print(f"Forward pass OK, loss={loss.item():.4f}")

# Check angle MLP gets gradients (through STE)
grad_norm = model.shared_angle_mlp.fc2.weight.grad.norm().item()
print(f"angle MLP fc2 grad norm: {grad_norm:.6f}")

# Check that main model also gets gradients (through the non-angle path)
grad_norm_emb = model.tok_emb.weight.grad.norm().item()
print(f"tok_emb grad norm: {grad_norm_emb:.6f}")

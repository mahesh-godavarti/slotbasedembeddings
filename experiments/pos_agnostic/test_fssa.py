import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_fssa_qk',
                 window_size=999999)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_fssa_qk: {n_params:,} params")

x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")

# Run twice — should be deterministic (same input, same argsort)
logits2, loss2 = model(x, x)
print(f"Second pass loss={loss2.item():.4f} (should be same — deterministic)")

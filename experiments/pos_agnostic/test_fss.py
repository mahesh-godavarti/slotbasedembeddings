import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_fss_qk',
                 window_size=999999, angle_hidden_mult=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_fss_qk: {n_params:,} params")
print(f"sign_freq_scales: {model.shared_angle_mlp.sign_freq_scales}")
print(f"freq_scales[:5]: {model.shared_angle_mlp.freq_scales[:5]}")

# Forward pass with gradient check
x = torch.randint(0, 32000, (2, 64))
model.train()
logits, loss = model(x, x)
loss.backward()
print(f"Forward pass OK, loss={loss.item():.4f}")

# Check gradients flow to MLP
grad_norm = model.shared_angle_mlp.fc2.weight.grad.norm().item()
print(f"fc2 grad norm: {grad_norm:.6f} (should be >0)")

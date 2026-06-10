import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='shared_pcb_qk',
                 window_size=999999, angle_hidden_mult=4)
n_params = sum(p.numel() for p in model.parameters())
print(f"shared_pcb_qk K=4: {n_params:,} params")
print(f"base embs: {len(model.layer_angle_embs)}")
print(f"corrections: {len(model.layer_corrections)}, shape: {model.layer_corrections[0].shape}")
print(f"projs: {len(model.pcb_projs)}")

# Compare param count
pemb_params = 162591488 + 16 * 32000 * 384  # base + 16 embeddings
print(f"pemb params: {pemb_params:,}")
print(f"pcb overhead vs pemb: {n_params - pemb_params:,}")

x = torch.randint(0, 32000, (2, 64))
model.train()
logits, loss = model(x, x)
loss.backward()
print(f"Forward pass OK, loss={loss.item():.4f}")

import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

model = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='joformer2',
                 window_size=999999, split_angles=True)

# Check angle_emb
print(f"angle_emb weight sum: {model.angle_emb.weight.sum().item()}")
print(f"angle_emb weight max: {model.angle_emb.weight.abs().max().item()}")

# Check fc2_angles in each block
for i, block in enumerate(model.blocks):
    if hasattr(block, 'ffn') and hasattr(block.ffn, 'fc2_angles'):
        w = block.ffn.fc2_angles.weight
        b = block.ffn.fc2_angles.bias
        print(f"Block {i} fc2_angles weight sum: {w.sum().item():.6f}, bias sum: {b.sum().item():.6f}")
        if i >= 2:
            print("  ...")
            break

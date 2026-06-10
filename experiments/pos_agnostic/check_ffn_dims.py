import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel

# joformer2 with split_angles
m1 = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='joformer2',
              window_size=999999, split_angles=True)

# joformer_fixed
m2 = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='joformer_fixed',
              window_size=999999)

# RoPE
m3 = GPTModel(32000, 768, 16, 8, 512, 0.1, attn_config='rope',
              window_size=999999)

for name, m in [('joformer2 split', m1), ('joformer_fixed', m2), ('rope', m3)]:
    print(f"\n=== {name} ===")
    print(f"Total params: {sum(p.numel() for p in m.parameters()):,}")
    b = m.blocks[0]
    print(f"Block type: {type(b).__name__}")
    print(f"FFN type: {type(b.ffn).__name__}")
    for pname, p in b.ffn.named_parameters():
        print(f"  ffn.{pname}: {p.shape}")
    for pname, p in b.attn.named_parameters():
        print(f"  attn.{pname}: {p.shape}")

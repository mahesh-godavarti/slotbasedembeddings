#!/usr/bin/env python3
"""Check where NaN appears in joformer2 forward pass."""

import torch
import sys
sys.path.insert(0, '/home/ubuntu/pos_agnostic')
from models import GPTModel
from train import load_memmap_data, get_batch

device = 'cuda:3'
torch.cuda.set_device(3)

ckpt_path = 'checkpoints/joformer2_angle_200k/joformer2_best.pt'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
cfg = ckpt['config']

model = GPTModel(
    cfg['vocab_size'], 768, cfg['n_layers'], cfg['n_heads'],
    cfg['block_size'], dropout=0.0, attn_config='joformer2',
    window_size=cfg.get('window_size', 999999),
    split_angles=True,
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(device)
model.eval()

train_data, val_data, meta = load_memmap_data('/home/ubuntu/look_ahead/look_ahead/data_owt')

for seq_len in [512, 1024, 2048, 4096, 8192]:
    torch.manual_seed(42)
    x, y = get_batch(val_data, seq_len, 1, device)
    with torch.no_grad():
        logits, loss = model(x, y)
    print(f"seq_len={seq_len}: loss={loss.item():.4f}, logits has nan={torch.isnan(logits).any().item()}")

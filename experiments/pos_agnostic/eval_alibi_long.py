#!/usr/bin/env python3
"""Evaluate ALiBi at long sequences using chunked attention to avoid OOM."""

import argparse
import math
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from train import load_memmap_data, get_batch
from models import GPTModel, ALiBiAttention
from eval_all import detect_attn_config, detect_n_embed
import types


def alibi_chunked_forward(self, x, chunk_size=4096):
    """ALiBi attention with chunked computation to avoid T×T memory."""
    B, T, C = x.shape
    h, d = self.n_heads, self.head_dim

    qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each (B, h, T, d)

    # Process queries in chunks, attend to all keys
    out_chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        q_chunk = q[:, :, start:end, :]  # (B, h, chunk, d)

        # Only attend to keys up to current position (causal)
        k_causal = k[:, :, :end, :]  # (B, h, end, d)
        v_causal = v[:, :, :end, :]

        scores = (q_chunk @ k_causal.transpose(-1, -2)) * (d ** -0.5)  # (B, h, chunk, end)

        # ALiBi bias for this chunk
        q_pos = torch.arange(start, end, device=x.device, dtype=torch.float32)
        k_pos = torch.arange(0, end, device=x.device, dtype=torch.float32)
        dist = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)  # (chunk, end)
        bias = -self.slopes.view(1, h, 1, 1) * dist.abs().unsqueeze(0).unsqueeze(0)
        scores = scores + bias

        # Causal mask: mask future positions
        causal_mask = dist < 0  # (chunk, end), True where key is after query
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out_chunk = attn @ v_causal  # (B, h, chunk, d)
        out_chunks.append(out_chunk)

    out = torch.cat(out_chunks, dim=2)  # (B, h, T, d)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_drop(self.out_proj(out))


def eval_at_length(model, val_data, length, batch_size, device, eval_iters=20):
    model.eval()
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    torch.manual_seed(42 + length)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + length)

    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, length, batch_size, device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        if loss is not None and not torch.isnan(loss):
            losses.append(loss.item())

    torch.random.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    if losses:
        avg = sum(losses) / len(losses)
        return math.exp(avg)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--eval_lengths', type=str, default='512,1024,2048,4096,8192,16384,32768,65536')
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--eval_iters', type=int, default=20)
    parser.add_argument('--chunk_size', type=int, default=4096)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _, val_data, _ = load_memmap_data(args.data_dir)
    lengths = [int(x) for x in args.eval_lengths.split(',')]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']

    model = GPTModel(cfg['vocab_size'], cfg['n_embed'], cfg['n_layers'], cfg['n_heads'],
                     cfg['block_size'], dropout=0.0, attn_config='alibi', window_size=999999)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Patch ALiBi attention with chunked forward
    for module in model.modules():
        if isinstance(module, ALiBiAttention):
            module.forward = types.MethodType(
                lambda self, x, cs=args.chunk_size: alibi_chunked_forward(self, x, cs), module)

    print(f'ALiBi eval (chunk_size={args.chunk_size})')
    for length in lengths:
        bs = args.eval_batch_size
        if length >= 32768:
            bs = 1
        ppl = eval_at_length(model, val_data, length, bs, device, args.eval_iters)
        if ppl:
            print(f'  {length}: PPL={ppl:.2f}')
        else:
            print(f'  {length}: failed')

    base = None
    print('\nRatios to 512:')
    for length in lengths:
        # rerun to get ratios (or store above)
        pass


if __name__ == '__main__':
    main()

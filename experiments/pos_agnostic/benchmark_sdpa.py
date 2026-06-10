#!/usr/bin/env python3
"""Benchmark SDPA vs manual attention for RoPE and jfixed models."""

import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '.')
from models import GPTModel, DataDep2Attention, RoPEAttention, JoFormerFixedAttention
from models import apply_rotary_emb, apply_inverse_rotary_emb, build_rope_angles
import types


def patch_sdpa(model):
    """Monkey-patch attention to use SDPA."""
    def _rope_forward_sdpa(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        angles = build_rope_angles(T, d, x.device)
        cos, sin = torch.cos(angles), torch.sin(angles)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

    def _jfixed_forward_sdpa(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        angles = build_rope_angles(T, d, x.device)
        cos, sin = torch.cos(angles), torch.sin(angles)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        v = apply_rotary_emb(v, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = apply_inverse_rotary_emb(out, cos, sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

    for module in model.modules():
        if isinstance(module, RoPEAttention):
            module.forward = types.MethodType(_rope_forward_sdpa, module)
        elif isinstance(module, JoFormerFixedAttention):
            module.forward = types.MethodType(_jfixed_forward_sdpa, module)


def benchmark(model, batch_size, seq_len, device, n_warmup=5, n_iter=20):
    """Time forward pass."""
    model.eval()
    idx = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(n_warmup):
            model(idx, targets)
    torch.cuda.synchronize()

    # Timed
    start = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(n_iter):
            model(idx, targets)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_iter
    return elapsed


def patch_sdpa_batched(model):
    """SDPA with batched rotary for jfixed - rotate QKV together."""
    def _jfixed_forward_batched(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        angles = build_rope_angles(T, d, x.device)
        cos, sin = torch.cos(angles), torch.sin(angles)
        # Batch rotate Q, K, V in one stacked operation
        stacked = torch.stack([q, k, v], dim=0)  # (3, B, h, T, d)
        d2 = d // 2
        s1, s2 = stacked[..., :d2], stacked[..., d2:]
        stacked_rot = torch.cat([s1 * cos - s2 * sin, s1 * sin + s2 * cos], dim=-1)
        q, k, v = stacked_rot.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Inverse rotation
        o1, o2 = out[..., :d2], out[..., d2:]
        out = torch.cat([o1 * cos + o2 * sin, -o1 * sin + o2 * cos], dim=-1)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

    for module in model.modules():
        if isinstance(module, JoFormerFixedAttention):
            module.forward = types.MethodType(_jfixed_forward_batched, module)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seq_lengths = [512, 1024, 2048, 4096, 8192]
    batch_size = 4

    for model_name in ['rope', 'joformer_fixed', 'joformer_fixed_batched', 'rope_compiled', 'joformer_fixed_compiled']:
        print(f'\n{"="*60}')
        print(f'Model: {model_name}')
        print(f'{"Seq Len":>8} | {"Time (ms)":>12}')
        print('-' * 25)

        for seq_len in seq_lengths:
            actual_model = model_name.replace('_batched', '').replace('_compiled', '')
            use_batched = '_batched' in model_name
            use_compiled = '_compiled' in model_name

            torch.manual_seed(42)
            model = GPTModel(32000, 768, 16, 8, seq_len, dropout=0.0,
                           attn_config=actual_model, window_size=999999)
            model.to(device)
            if use_batched:
                patch_sdpa_batched(model)
            else:
                patch_sdpa(model)
            if use_compiled:
                model = torch.compile(model)
            try:
                t = benchmark(model, batch_size, seq_len, device, n_warmup=10 if use_compiled else 5) * 1000
            except RuntimeError as e:
                print(f'{seq_len:>8} | OOM or error: {e}')
                del model
                torch.cuda.empty_cache()
                continue

            print(f'{seq_len:>8} | {t:>10.1f}ms')
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

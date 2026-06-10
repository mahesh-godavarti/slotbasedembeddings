#!/usr/bin/env python3
"""Standalone extrapolation eval. Same data, same seeds, high eval_iters for low variance."""

import argparse
import math
import sys
import torch
import torch.nn.functional as F
from train import load_memmap_data, get_batch
from models import GPTModel, DataDep2Attention, RoPEAttention, JoFormerFixedAttention, apply_rotary_emb, apply_inverse_rotary_emb, build_rope_angles, build_attn_mask
from eval_all import detect_attn_config, detect_n_embed


def patch_sdpa(model):
    """Monkey-patch attention classes to use F.scaled_dot_product_attention for memory efficiency."""

    def _datadep2_forward_sdpa(self, x, angles, v_angles=None):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_cumsum:
            angles = torch.flip(angles, dims=(1,))
            angles = torch.cumsum(angles, dim=1)
            angles = torch.flip(angles, dims=(1,))
        a = angles.view(B, T, h, d // 2).transpose(1, 2)
        cos, sin = torch.cos(a), torch.sin(a)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.rotate_v:
            if self.rope_v:
                rope_angles = build_rope_angles(T, d, x.device)
                v_cos, v_sin = torch.cos(rope_angles), torch.sin(rope_angles)
            elif v_angles is not None:
                if self.use_cumsum:
                    v_angles = torch.flip(v_angles, dims=(1,))
                    v_angles = torch.cumsum(v_angles, dim=1)
                    v_angles = torch.flip(v_angles, dims=(1,))
                va = v_angles.view(B, T, h, d // 2).transpose(1, 2)
                v_cos, v_sin = torch.cos(va), torch.sin(va)
            else:
                v_cos, v_sin = cos, sin
            if self.detach_v:
                v_cos, v_sin = v_cos.detach(), v_sin.detach()
            v = apply_rotary_emb(v, v_cos, v_sin)

        # Use SDPA instead of manual attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if self.rotate_v:
            out = apply_inverse_rotary_emb(out, v_cos, v_sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

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

    import types
    for module in model.modules():
        if isinstance(module, DataDep2Attention):
            module.forward = types.MethodType(_datadep2_forward_sdpa, module)
        elif isinstance(module, RoPEAttention):
            module.forward = types.MethodType(_rope_forward_sdpa, module)
        elif isinstance(module, JoFormerFixedAttention):
            module.forward = types.MethodType(_jfixed_forward_sdpa, module)


def eval_at_length(model, val_data, length, batch_size, device, eval_iters=100):
    """Evaluate model at a specific sequence length."""
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
        return {'loss': avg, 'ppl': math.exp(avg)}
    return {'loss': None, 'ppl': None, 'error': 'no valid losses'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--eval_lengths', type=str, default='512,1024,2048,4096,8192,16384,32768')
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        torch.cuda.set_device(args.gpu)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _, val_data, meta = load_memmap_data(args.data_dir)
    lengths = [int(x) for x in args.eval_lengths.split(',')]

    for ckpt_path in args.checkpoints:
        print(f'\n{"="*60}')
        print(f'Checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt['config']

        attn_config = detect_attn_config(cfg)
        split_angles = 'angle_emb.weight' in ckpt['model_state_dict']
        n_embed = cfg['n_embed'] if split_angles else detect_n_embed(cfg)

        # Detect angle_hidden_mult
        sd = ckpt['model_state_dict']
        angle_hidden_mult = 4
        if 'angle_codebook' in sd:
            angle_hidden_mult = sd['angle_codebook'].shape[1]
        else:
            for key in sd:
                if 'angle_mlp.fc1.weight' in key or 'shared_angle_mlp.fc1.weight' in key:
                    hidden_dim = sd[key].shape[0]
                    angle_hidden_mult = hidden_dim // n_embed
                    break
                if 'blocks.0.ffn.fc1.weight' in key and 'angle_mlp' not in key:
                    hidden_dim = sd[key].shape[0]
                    angle_hidden_mult = hidden_dim // n_embed
                    break

        # Detect angle_activation
        angle_activation = 'tanh'
        if any('angle_ln' in k or 'shared_angle_mlp.ln' in k for k in sd):
            angle_activation = 'ln'

        model = GPTModel(
            cfg['vocab_size'], n_embed, cfg['n_layers'], cfg['n_heads'],
            cfg['block_size'], dropout=0.0, attn_config=attn_config,
            window_size=cfg.get('window_size', 32),
            split_angles=split_angles,
            angle_hidden_mult=angle_hidden_mult,
            angle_activation=angle_activation,
        )
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        patch_sdpa(model)

        val_ppl = math.exp(ckpt['val_loss']) if 'val_loss' in ckpt else None
        print(f'Val PPL (from checkpoint): {val_ppl:.2f}' if val_ppl else 'Val PPL: unknown')
        print(f'Eval: batch_size={args.eval_batch_size}, eval_iters={args.eval_iters}')

        results = {}
        for length in lengths:
            if len(val_data) < length + 1:
                print(f'  {length}: skipped (data too short)')
                continue
            try:
                bs = args.eval_batch_size
                if length >= 16384:
                    bs = max(1, bs // 2)
                if length >= 32768:
                    bs = 1
                r = eval_at_length(model, val_data, length, bs, device, args.eval_iters)
                results[length] = r
                if r['ppl'] is not None:
                    print(f'  {length}: PPL={r["ppl"]:.2f}')
                else:
                    print(f'  {length}: {r}')
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f'  {length}: OOM')
                    torch.cuda.empty_cache()
                else:
                    print(f'  {length}: ERROR {e}')

        if 512 in results and results[512]['ppl']:
            base = results[512]['ppl']
            print(f'\nRatios to 512:')
            for length in sorted(results.keys()):
                if results[length]['ppl']:
                    ratio = results[length]['ppl'] / base
                    print(f'  {length}: {ratio:.2f}x')

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

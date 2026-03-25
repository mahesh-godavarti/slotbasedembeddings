#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Any commercial use, including but not limited to use in a product, service,
# or for-profit research, is strictly prohibited without explicit written
# permission from the copyright holder.
#
# Patent Pending: Certain aspects of this software are the subject of a
# pending patent application.
#
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------
"""
Position-Agnostic Final Layer (PAFL) — Length Generalization Models

Four attention types:
  - RoPE:   Rotary Position Embeddings (standard, from Su et al. 2021)
  - NoPE:   No positional encoding
  - ALiBi:  Attention with Linear Biases (Press et al. 2022)
  - Hybrid: RoPE in early layers, NoPE in late layers (PAFL — this paper)

All models use multi-head causal self-attention with sliding window,
pre-LayerNorm, GELU FFN. No absolute position embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def build_rope_angles(seq_len, head_dim, device):
    """Standard RoPE: theta_i = 1 / 10000^(2i/d).

    Returns (seq_len, head_dim // 2) angles.
    """
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (dim / head_dim))  # (head_dim//2,)
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (T, head_dim//2)
    return angles


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings.  x: (..., T, d), cos/sin: (T, d//2)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_inverse_rotary_emb(x, cos, sin):
    """Apply inverse (transpose) rotary embeddings."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Sliding window causal mask
# ---------------------------------------------------------------------------

def apply_attn_weights(scores, mask, use_softplus=False, topk=0):
    """Apply masking and compute attention weights.

    softmax:  exp(x_i) / sum(exp(x_j)) — standard, sharp
    softplus: log(1+exp(x_i)) / sum(log(1+exp(x_j))) — smoother, no winner-take-all
    topk:     if > 0, keep only top-k scores per query position (applied after causal mask)
    """
    scores.masked_fill_(mask, float('-inf') if not use_softplus else -1e9)

    # Top-k: keep only the k highest scores per query position
    if topk > 0:
        # Clamp topk to actual number of valid (non-masked) positions
        k = min(topk, scores.shape[-1])
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        topk_mask = torch.ones_like(scores, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_idx, False)
        scores.masked_fill_(topk_mask, float('-inf') if not use_softplus else -1e9)

    if use_softplus:
        wei = torch.log(torch.exp(scores) + 1)
        wei.masked_fill_(scores < -1e8, 0.0)
        wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
    else:
        wei = F.softmax(scores, dim=-1)
    return wei


def build_attn_mask(T, window_size, device):
    """Build causal sliding-window mask.

    Returns bool mask of shape (T, T) where True = masked (don't attend).
    Position i can attend to position j iff j <= i and i - j < window_size.
    """
    # Causal: mask future (j > i)
    row = torch.arange(T, device=device)
    col = torch.arange(T, device=device)
    dist = row.unsqueeze(1) - col.unsqueeze(0)  # (T, T), [i,j] = i - j

    # Mask where j > i (future) OR i - j >= window_size (too far past)
    mask = (dist < 0) | (dist >= window_size)
    return mask


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class RoPEAttention(nn.Module):
    """Multi-head causal attention with Rotary Position Embeddings."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256, use_softplus=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.eval_topk = 0
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, h, T, d)

        # RoPE
        angles = build_rope_angles(T, d, x.device)
        cos, sin = torch.cos(angles), torch.sin(angles)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Attention — at eval with topk, use full causal mask (topk selects the positions)
        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class JoFormerFixedAttention(nn.Module):
    """Like RoPE but rotates Q, K, AND V, with inverse rotation on output."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256, use_softplus=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.eval_topk = 0
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Fixed RoPE angles (same as RoPE)
        angles = build_rope_angles(T, d, x.device)
        cos, sin = torch.cos(angles), torch.sin(angles)

        # Rotate Q, K, V
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        v = apply_rotary_emb(v, cos, sin)

        # Attention
        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))

        # Inverse rotation on output
        out = attn @ v
        out = apply_inverse_rotary_emb(out, cos, sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class NoPEAttention(nn.Module):
    """Multi-head causal attention without any positional encoding."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256, use_softplus=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.eval_topk = 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class ALiBiAttention(nn.Module):
    """Multi-head causal attention with ALiBi (Press et al. 2022)."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256, use_softplus=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.eval_topk = 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # ALiBi slopes: geometric series 2^(-8/n * k) for k = 1..n
        slopes = 2.0 ** (-8.0 / n_heads * torch.arange(1, n_heads + 1,
                                                         dtype=torch.float32))
        self.register_buffer('slopes', slopes)  # (n_heads,)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)

        # ALiBi bias: -slope * |i - j| for each head
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)  # (T, T), [i,j] = i - j
        bias = -self.slopes.view(1, h, 1, 1) * dist.abs().unsqueeze(0).unsqueeze(0)
        scores = scores + bias

        # Sliding window causal mask
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class DataDepAttention(nn.Module):
    """Multi-head causal attention with data-dependent rotation angles.

    angles = angle_proj(x) per position.
    - use_cumsum=False (datadep/monoidal): purely content-dependent
    - use_cumsum=True: angles accumulated via flip-cumsum-flip
    - rotate_v=False (monoidal): rotate Q and K only
    - rotate_v=True (joformer): rotate Q, K, V and inverse-rotate output
    - mlp_angles=True (v3): angle_proj is MLP instead of linear
    """

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_softplus=False, use_cumsum=False, rotate_v=False,
                 mlp_angles=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.use_cumsum = use_cumsum
        self.rotate_v = rotate_v
        self.mlp_angles = mlp_angles
        self.eval_topk = 0
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Data-dependent angle projection: x -> per-head angles
        if mlp_angles:
            self.angle_proj = nn.Sequential(
                nn.Linear(n_embed, 2 * n_embed),
                nn.GELU(),
                nn.Linear(2 * n_embed, n_embed // 2),
            )
        else:
            self.angle_proj = nn.Linear(n_embed, n_embed // 2)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, h, T, d)

        # Data-dependent angles
        angles = self.angle_proj(x)  # (B, T, C//2)
        if self.use_cumsum:
            angles = torch.flip(angles, dims=(1,))
            angles = torch.cumsum(angles, dim=1)
            angles = torch.flip(angles, dims=(1,))
        angles = angles.view(B, T, h, d // 2).transpose(1, 2)  # (B, h, T, d//2)
        cos, sin = torch.cos(angles), torch.sin(angles)

        # Rotate Q, K (and optionally V)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.rotate_v:
            v = apply_rotary_emb(v, cos, sin)

        # Attention with sliding window
        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))

        out = attn @ v
        if self.rotate_v:
            out = apply_inverse_rotary_emb(out, cos, sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# DataDep v2: angles flow through the network (embedding → attn → FFN → attn)
# ---------------------------------------------------------------------------

class DataDep2Attention(nn.Module):
    """Multi-head causal attention with externally-provided data-dependent angles."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_softplus=False, use_cumsum=False, rotate_v=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.use_cumsum = use_cumsum
        self.rotate_v = rotate_v
        self.eval_topk = 0
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, angles):
        """x: (B, T, C), angles: (B, T, C//2) — rotation angles from embedding or prev FFN."""
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Optional cumsum on angles
        if self.use_cumsum:
            angles = torch.flip(angles, dims=(1,))
            angles = torch.cumsum(angles, dim=1)
            angles = torch.flip(angles, dims=(1,))
        # Reshape angles to per-head: (B, T, C//2) -> (B, h, T, d//2)
        a = angles.view(B, T, h, d // 2).transpose(1, 2)
        cos, sin = torch.cos(a), torch.sin(a)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.rotate_v:
            v = apply_rotary_emb(v, cos, sin)

        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))

        out = attn @ v
        if self.rotate_v:
            out = apply_inverse_rotary_emb(out, cos, sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForwardWithAngles(nn.Module):
    """FFN that outputs C content dims + C//2 angle dims."""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.n_embed = n_embed
        self.fc1 = nn.Linear(n_embed, 4 * n_embed)
        self.fc2 = nn.Linear(4 * n_embed, n_embed + n_embed // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(self.fc2(F.gelu(self.fc1(x))))
        content = out[..., :self.n_embed]
        angles = out[..., self.n_embed:]
        return content, angles


class DataDep2Block(nn.Module):
    """Transformer block for datadep2: receives angles, produces new angles."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_softplus=False, use_cumsum=False, rotate_v=False):
        super().__init__()
        self.attn = DataDep2Attention(n_embed, n_heads, dropout, window_size,
                                      use_softplus, use_cumsum, rotate_v)
        self.ffn = FeedForwardWithAngles(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles):
        x = x + self.attn(self.ln1(x), angles)
        content, new_angles = self.ffn(self.ln2(x))
        x = x + content
        return x, new_angles


# ---------------------------------------------------------------------------
# Transformer block and full model
# ---------------------------------------------------------------------------

def _make_datadep3_attn(n_embed, n_heads, dropout, window_size=256, use_softplus=False):
    return DataDepAttention(n_embed, n_heads, dropout, window_size, use_softplus,
                            mlp_angles=True)

def _make_monoidal_attn(n_embed, n_heads, dropout, window_size=256, use_softplus=False):
    return DataDepAttention(n_embed, n_heads, dropout, window_size, use_softplus,
                            use_cumsum=True, rotate_v=False)

def _make_monoidal3_attn(n_embed, n_heads, dropout, window_size=256, use_softplus=False):
    return DataDepAttention(n_embed, n_heads, dropout, window_size, use_softplus,
                            use_cumsum=True, rotate_v=False, mlp_angles=True)

def _make_joformer_attn(n_embed, n_heads, dropout, window_size=256, use_softplus=False):
    return DataDepAttention(n_embed, n_heads, dropout, window_size, use_softplus,
                            use_cumsum=True, rotate_v=True)

def _make_joformer3_attn(n_embed, n_heads, dropout, window_size=256, use_softplus=False):
    return DataDepAttention(n_embed, n_heads, dropout, window_size, use_softplus,
                            use_cumsum=True, rotate_v=True, mlp_angles=True)

ATTN_CLS = {
    'rope': RoPEAttention,
    'joformer_fixed': JoFormerFixedAttention,
    'nope': NoPEAttention,
    'alibi': ALiBiAttention,
    'datadep': DataDepAttention,
    'datadep3': _make_datadep3_attn,
    'monoidal': _make_monoidal_attn,
    'monoidal3': _make_monoidal3_attn,
    'joformer': _make_joformer_attn,
    'joformer3': _make_joformer3_attn,
}


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with configurable attention type."""

    def __init__(self, n_embed, n_heads, dropout, attn_type='rope',
                 window_size=256, use_softplus=False):
        super().__init__()
        self.attn = ATTN_CLS[attn_type](n_embed, n_heads, dropout, window_size, use_softplus)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """GPT-style transformer with per-layer positional encoding config.

    attn_config options:
      - 'rope':      RoPE in all layers
      - 'nope':      No position encoding in any layer
      - 'alibi':     ALiBi in all layers
      - 'hybrid_K':  RoPE in first (L-K) layers, NoPE in last K layers
      - list:        explicit per-layer types, e.g. ['rope','rope','nope','nope']
    """

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, block_size,
                 dropout, attn_config='rope', window_size=256, use_softplus=False):
        super().__init__()
        self.block_size = block_size
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.is_datadep2 = isinstance(attn_config, str) and (
            attn_config.startswith('datadep2') or attn_config.startswith('monoidal2')
            or attn_config.startswith('joformer2'))

        if self.is_datadep2:
            # v2 angle flow: embedding outputs C + C//2, angles flow through network
            self.tok_emb = nn.Embedding(vocab_size, n_embed + n_embed // 2)
            self.n_embed = n_embed

            # Determine cumsum and rotate_v from config name
            base = attn_config.split('_')[0]  # datadep2, monoidal2, or joformer2
            use_cumsum = base in ('monoidal2', 'joformer2')
            rotate_v = base == 'joformer2'

            def _make_v2_block(ws):
                return DataDep2Block(n_embed, n_heads, dropout, ws,
                                     use_softplus, use_cumsum, rotate_v)

            if '_hybrid_' in attn_config:
                # v2 windowed early + NoPE full late
                k = int(attn_config.split('_')[-1])
                assert 0 < k < n_layers, f"{attn_config} invalid for {n_layers} layers"
                self.layer_types = [base] * (n_layers - k) + ['nope'] * k
                blocks = [_make_v2_block(window_size) for _ in range(n_layers - k)]
                blocks += [TransformerBlock(n_embed, n_heads, dropout, 'nope',
                                            999999, use_softplus) for _ in range(k)]
                self.blocks = nn.ModuleList(blocks)
            elif attn_config in ('datadep2', 'monoidal2', 'joformer2'):
                # All layers same type
                self.layer_types = [attn_config] * n_layers
                self.blocks = nn.ModuleList([_make_v2_block(window_size)
                                             for _ in range(n_layers)])
            else:
                # datadep2_K / monoidal2_K / joformer2_K
                k = int(attn_config.split('_')[1])
                assert 0 < k < n_layers, f"{attn_config} invalid for {n_layers} layers"
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
                self.layer_types = [base] * n_layers
                self.blocks = nn.ModuleList([_make_v2_block(ws)
                                             for ws in layer_windows])
        else:
            # Standard models: rope, nope, alibi, hybrid_K, datadep, datadep_K
            self.tok_emb = nn.Embedding(vocab_size, n_embed)
            self.n_embed = n_embed

            # Parse attn_config into per-layer (type, window_size) pairs
            if isinstance(attn_config, list):
                layer_types = attn_config
                layer_windows = [window_size if lt == 'rope' else 999999
                                 for lt in layer_types]
            elif attn_config.startswith('hybrid_'):
                k = int(attn_config.split('_')[1])
                assert 0 < k < n_layers, f"hybrid_{k} invalid for {n_layers} layers"
                layer_types = ['rope'] * (n_layers - k) + ['nope'] * k
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config == 'alternating':
                # Every other layer: RoPE windowed, NoPE full
                layer_types = ['rope' if i % 2 == 0 else 'nope' for i in range(n_layers)]
                layer_windows = [window_size if lt == 'rope' else 999999 for lt in layer_types]
            elif attn_config == 'cohere':
                # Cohere RNoPE-SWA style: NoPE every 3rd layer (1:2 ratio)
                layer_types = ['nope' if (i + 1) % 3 == 0 else 'rope' for i in range(n_layers)]
                layer_windows = [window_size if lt == 'rope' else 999999 for lt in layer_types]
            elif attn_config.startswith('ropefull_'):
                # RoPE windowed early + RoPE full late
                k = int(attn_config.split('_')[1])
                assert 0 < k < n_layers, f"ropefull_{k} invalid for {n_layers} layers"
                layer_types = ['rope'] * n_layers
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config.startswith('datadep3full_'):
                # RoPE windowed early + datadep3 (MLP angles) full late
                k = int(attn_config.split('_')[1])
                assert 0 < k < n_layers, f"datadep3full_{k} invalid for {n_layers} layers"
                layer_types = ['rope'] * (n_layers - k) + ['datadep3'] * k
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config.startswith('datadepfull_'):
                # RoPE windowed early + datadep full late
                k = int(attn_config.split('_')[1])
                assert 0 < k < n_layers, f"datadepfull_{k} invalid for {n_layers} layers"
                layer_types = ['rope'] * (n_layers - k) + ['datadep'] * k
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config.startswith('joformerfull_'):
                # RoPE windowed early + joformer (datadep+cumsum) full late
                k = int(attn_config.split('_')[1])
                assert 0 < k < n_layers, f"joformerfull_{k} invalid for {n_layers} layers"
                layer_types = ['rope'] * (n_layers - k) + ['joformer'] * k
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config.startswith('monoidal_hybrid_'):
                # Monoidal (cumsum, Q/K only) windowed early + NoPE full late
                k = int(attn_config.split('_')[2])
                assert 0 < k < n_layers, f"monoidal_hybrid_{k} invalid for {n_layers} layers"
                layer_types = ['monoidal'] * (n_layers - k) + ['nope'] * k
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config.startswith('joformer_fixed_hybrid_'):
                # JoFormer-fixed (fixed angles + V rotation) windowed early + NoPE full late
                k = int(attn_config.split('_')[3])
                assert 0 < k < n_layers, f"joformer_fixed_hybrid_{k} invalid for {n_layers} layers"
                layer_types = ['joformer_fixed'] * (n_layers - k) + ['nope'] * k
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config.startswith('joformer_hybrid_'):
                # JoFormer (cumsum, Q/K/V + inverse) windowed early + NoPE full late
                k = int(attn_config.split('_')[2])
                assert 0 < k < n_layers, f"joformer_hybrid_{k} invalid for {n_layers} layers"
                layer_types = ['joformer'] * (n_layers - k) + ['nope'] * k
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            elif attn_config.startswith('datadep_'):
                k = int(attn_config.split('_')[1])
                assert 0 < k < n_layers, f"datadep_{k} invalid for {n_layers} layers"
                layer_types = ['datadep'] * n_layers
                layer_windows = [window_size] * (n_layers - k) + [999999] * k
            else:
                assert attn_config in ATTN_CLS, f"Unknown attn_config: {attn_config}"
                layer_types = [attn_config] * n_layers
                WINDOWED_TYPES = ('rope', 'joformer_fixed', 'datadep', 'datadep3',
                                  'monoidal', 'monoidal3', 'joformer', 'joformer3')
                layer_windows = [window_size if attn_config in WINDOWED_TYPES else 999999] * n_layers

            assert len(layer_types) == n_layers
            self.layer_types = layer_types

            self.blocks = nn.ModuleList([
                TransformerBlock(n_embed, n_heads, dropout, lt, ws, use_softplus)
                for lt, ws in zip(layer_types, layer_windows)
            ])

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def set_eval_topk(self, topk):
        """Set top-k attention for eval mode on all layers."""
        for block in self.blocks:
            attn = block.attn if hasattr(block, 'attn') else None
            if attn is not None:
                attn.eval_topk = topk

    def forward(self, idx, targets=None):
        if self.is_datadep2:
            emb = self.tok_emb(idx)  # (B, T, C + C//2)
            C = self.n_embed
            x = emb[..., :C]          # content
            angles = emb[..., C:]      # initial angles (C//2)
            for block in self.blocks:
                if isinstance(block, DataDep2Block):
                    x, angles = block(x, angles)
                else:
                    x = block(x)
        else:
            x = self.tok_emb(idx)
            for block in self.blocks:
                x = block(x)

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

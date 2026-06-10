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


class RoPELearnedFreqAttention(nn.Module):
    """RoPE with learned frequencies. Initialized from standard RoPE frequencies."""

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
        # Learned frequencies, initialized from RoPE
        d = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        self.learned_freqs = nn.Parameter(freqs)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        angles = pos.unsqueeze(1) * self.learned_freqs.unsqueeze(0)
        cos, sin = torch.cos(angles), torch.sin(angles)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class JoFormerFixedLearnedFreqAttention(nn.Module):
    """JoFormer-fixed with learned frequencies. Q/K/V rotation + inverse."""

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
        # Learned frequencies, initialized from RoPE
        d = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        self.learned_freqs = nn.Parameter(freqs)

    def forward(self, x):
        B, T, C = x.shape
        h, d = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        angles = pos.unsqueeze(1) * self.learned_freqs.unsqueeze(0)
        cos, sin = torch.cos(angles), torch.sin(angles)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        v = apply_rotary_emb(v, cos, sin)
        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))
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
                 use_softplus=False, use_cumsum=False, rotate_v=False,
                 detach_v=False, rope_v=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.use_cumsum = use_cumsum
        self.rotate_v = rotate_v
        self.detach_v = detach_v
        self.rope_v = rope_v
        self.eval_topk = 0
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, angles, v_angles=None):
        """x: (B, T, C), angles: (B, T, C//2) — rotation angles for Q/K.
        v_angles: optional (B, T, C//2) — separate angles for V rotation.
        If v_angles is None and rotate_v is True, uses same angles for V."""
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
            if self.rope_v:
                # Use fixed RoPE position-indexed angles for V (predictable rotation)
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

        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        topk = self.eval_topk if not self.training and self.eval_topk > 0 else 0
        ws = 999999 if topk > 0 else self.window_size
        mask = build_attn_mask(T, ws, x.device)
        attn = self.attn_drop(apply_attn_weights(scores, mask, self.use_softplus, topk))

        out = attn @ v
        if self.rotate_v:
            out = apply_inverse_rotary_emb(out, v_cos, v_sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class SharedAngleMLP(nn.Module):
    """Shared MLP that produces per-token rotation angles from hidden states.

    Flow: Linear → GELU → Linear → LayerNorm → [abs].
    use_abs=True: positive angles only (shared_pos / perlayer_pos).
    use_abs=False: angles can be negative (shared_ln / perlayer_ln).
    """

    def __init__(self, n_embed, use_abs=True, use_output_ln=True, use_rms=False,
                 hidden_mult=4, angle_dropout=0.0, freq_scales=None, use_tanh=True,
                 learned_freq=False, base_freq_learned=None):
        super().__init__()
        self.use_abs = use_abs
        self.use_output_ln = use_output_ln
        self.use_rms = use_rms
        self.use_base_freq_learned = base_freq_learned is not None
        self.random_freq_scales = False  # set by model config
        self.sign_freq_scales = False  # set by model config
        self.normalize_weights = False  # set by model config
        if self.use_base_freq_learned:
            self.register_buffer('base_freq', base_freq_learned)
        self.use_freq_scales = freq_scales is not None
        self.use_tanh = use_tanh
        self.learned_freq = learned_freq
        self.bernoulli = False  # set by model config
        self._fixed_signs = None  # set externally per forward pass
        hidden = hidden_mult * n_embed
        self.fc1 = nn.Linear(n_embed, hidden)
        self.fc2 = nn.Linear(hidden, n_embed // 2)
        if self.use_base_freq_learned:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        if use_output_ln or self.use_freq_scales or (learned_freq and use_output_ln):
            self.ln = nn.LayerNorm(n_embed // 2)
        if use_rms:
            self.rms_weight = nn.Parameter(torch.ones(n_embed // 2))
        if self.use_freq_scales:
            self.register_buffer('freq_scales', freq_scales)
        self.angle_drop = nn.Dropout(angle_dropout) if angle_dropout > 0 else None

    def normalize_weights_post_step(self):
        """Zero out biases and normalize weight rows to unit norm. Call after optimizer.step()."""
        with torch.no_grad():
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()
            self.fc1.weight.div_(self.fc1.weight.norm(dim=1, keepdim=True).clamp(min=1e-8))
            self.fc2.weight.div_(self.fc2.weight.norm(dim=1, keepdim=True).clamp(min=1e-8))

    def forward(self, x):
        """x: (B, T, C) → angles: (B, T, C//2)."""
        h = F.gelu(self.fc1(x))
        angles = self.fc2(h)
        if self.use_base_freq_learned:
            # Uniform(-abs(tanh(learned)*π + base_freq), abs(tanh(learned)*π + base_freq))
            freq = torch.abs(torch.tanh(angles) * math.pi + self.base_freq)
            noise = 2.0 * torch.rand_like(angles) - 1.0
            angles = noise * freq
        elif self.learned_freq:
            if self.use_output_ln:
                freq = torch.abs(self.ln(angles))  # LN → abs
            else:
                freq = torch.abs(angles)  # just abs, no LN
            if self._fixed_signs is not None:
                angles = self._fixed_signs * freq
            elif self.bernoulli:
                noise = 2.0 * torch.bernoulli(torch.full_like(angles, 0.5)) - 1.0  # {-1, +1}
                angles = noise * freq
            else:
                noise = 2.0 * torch.rand_like(angles) - 1.0  # Uniform(-1, 1)
                angles = noise * freq
        elif self.use_freq_scales:
            angles = self.ln(angles)
            if self.sign_freq_scales:
                # LN → sign (straight-through) → × freq_scales
                hard = torch.sign(angles)
                angles = (hard - angles).detach() + angles  # forward=hard, backward=soft
                angles = angles * self.freq_scales
            elif self.random_freq_scales:
                # LN → × freq_scales → abs → × Uniform(-1, 1)
                freq = torch.abs(angles * self.freq_scales)
                noise = 2.0 * torch.rand_like(angles) - 1.0
                angles = noise * freq
            elif self.use_tanh:
                angles = torch.tanh(angles)
                angles = angles * self.freq_scales
            else:
                angles = angles * self.freq_scales
        elif self.use_rms:
            rms = torch.sqrt(torch.mean(angles ** 2, dim=-1, keepdim=True) + 1e-8)
            angles = angles / rms * self.rms_weight
        elif self.use_output_ln:
            angles = self.ln(angles)
        if self.use_abs:
            angles = torch.abs(angles)
        if self.angle_drop is not None:
            angles = self.angle_drop(angles)
        return angles


class SharedAngleMLPSplit(nn.Module):
    """Two separate shared MLPs for Q/K and V angles."""

    def __init__(self, n_embed, use_abs=True, use_output_ln=True, hidden_mult=4, angle_dropout=0.0):
        super().__init__()
        self.qk_mlp = SharedAngleMLP(n_embed, use_abs=use_abs, use_output_ln=use_output_ln, hidden_mult=hidden_mult, angle_dropout=angle_dropout)
        self.v_mlp = SharedAngleMLP(n_embed, use_abs=use_abs, use_output_ln=use_output_ln, hidden_mult=hidden_mult, angle_dropout=angle_dropout)

    def forward(self, x):
        """x: (B, T, C) → (qk_angles, v_angles), each (B, T, C//2)."""
        return self.qk_mlp(x), self.v_mlp(x)


class ExternalAngleBlock(nn.Module):
    """Transformer block that receives angles externally (no angle-producing FFN).

    Used by shared_pos and random_pos models where angles come from outside the block.
    """

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_softplus=False, rotate_v=False, detach_v=False, rope_v=False):
        super().__init__()
        self.attn = DataDep2Attention(n_embed, n_heads, dropout, window_size,
                                      use_softplus, use_cumsum=True, rotate_v=rotate_v,
                                      detach_v=detach_v, rope_v=rope_v)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles, v_angles=None):
        x = x + self.attn(self.ln1(x), angles, v_angles=v_angles)
        x = x + self.ffn(self.ln2(x))
        return x


class PerLayerAngleBlock(nn.Module):
    """Transformer block with its own angle MLP (LayerNorm → [abs]).

    Each layer has a separate angle MLP: Linear → GELU → Linear → LayerNorm → [abs].
    The angle MLP takes the block's input and produces angles for that layer's attention.
    """

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_softplus=False, rotate_v=False, use_abs=True,
                 split=False, ln_input=False, use_output_ln=True,
                 use_rms=False, hidden_mult=4, angle_dropout=0.0, freq_scales=None, use_tanh=True,
                 learned_freq=False, detach_v=False, rope_v=False):
        super().__init__()
        self.split = split
        self.ln_input = ln_input
        self.attn = DataDep2Attention(n_embed, n_heads, dropout, window_size,
                                      use_softplus, use_cumsum=True, rotate_v=rotate_v,
                                      detach_v=detach_v, rope_v=rope_v)
        self.ffn = FeedForward(n_embed, dropout)
        self.angle_mlp = SharedAngleMLP(n_embed, use_abs=use_abs, use_output_ln=use_output_ln, use_rms=use_rms, hidden_mult=hidden_mult, angle_dropout=angle_dropout, freq_scales=freq_scales, use_tanh=use_tanh, learned_freq=learned_freq)
        if split:
            self.v_angle_mlp = SharedAngleMLP(n_embed, use_abs=use_abs, use_output_ln=use_output_ln, use_rms=use_rms, hidden_mult=hidden_mult, angle_dropout=angle_dropout, freq_scales=freq_scales, use_tanh=use_tanh, learned_freq=learned_freq)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x_ln = self.ln1(x)
        angle_input = x_ln if self.ln_input else x
        angles = self.angle_mlp(angle_input)
        v_angles = self.v_angle_mlp(angle_input) if self.split else None
        x = x + self.attn(x_ln, angles, v_angles=v_angles)
        x = x + self.ffn(self.ln2(x))
        return x


class FeedForwardWithAngles(nn.Module):
    """FFN that outputs C content dims + C//2 angle dims.

    split_angles=True uses separate projections for content and angles
    (allows separate lr). split_angles=False uses a single fc2 (legacy).
    """

    def __init__(self, n_embed, dropout, split_angles=False, hidden_mult=4,
                 angle_activation='tanh', shared_angle_ln=None,
                 angle_dropout=0.0, n_heads=8):
        super().__init__()
        self.n_embed = n_embed
        self.split_angles = split_angles
        self.angle_activation = angle_activation
        hidden = hidden_mult * n_embed
        self.fc1 = nn.Linear(n_embed, hidden)
        if split_angles:
            self.fc2_content = nn.Linear(hidden, n_embed)
            self.fc2_angles = nn.Linear(hidden, n_embed // 2)
            nn.init.zeros_(self.fc2_angles.weight)
            nn.init.zeros_(self.fc2_angles.bias)
        else:
            self.fc2 = nn.Linear(hidden, n_embed + n_embed // 2)
        if angle_activation in ('ln', 'ln_tanh_freq'):
            if shared_angle_ln is not None:
                self.angle_ln = shared_angle_ln  # shared across layers
            else:
                self.angle_ln = nn.LayerNorm(n_embed // 2)
        if angle_activation in ('tanh_freq', 'ln_tanh_freq'):
            head_dim = n_embed // n_heads
            freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            self.register_buffer('angle_freq_scales', freqs.repeat(n_heads))
        if angle_activation == 'tanh_lfreq':
            head_dim = n_embed // n_heads
            freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            self.angle_freq_scales = nn.Parameter(freqs.repeat(n_heads))
        self.dropout = nn.Dropout(dropout)
        self.angle_drop = nn.Dropout(angle_dropout) if angle_dropout > 0 else None

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        if self.split_angles:
            content = self.dropout(self.fc2_content(h))
            raw_angles = self.fc2_angles(h)
        else:
            out = self.dropout(self.fc2(h))
            content = out[..., :self.n_embed]
            raw_angles = out[..., self.n_embed:]
        if self.angle_activation == 'tanh':
            angles = torch.tanh(raw_angles) * math.pi
        elif self.angle_activation in ('tanh_freq', 'tanh_lfreq'):
            angles = torch.tanh(raw_angles) * torch.abs(self.angle_freq_scales)
        elif self.angle_activation == 'ln_tanh_freq':
            angles = torch.tanh(self.angle_ln(raw_angles)) * self.angle_freq_scales
        elif self.angle_activation == 'ln':
            angles = self.angle_ln(raw_angles)
        else:
            angles = raw_angles
        if self.angle_drop is not None:
            angles = self.angle_drop(angles)
        return content, angles


class DataDep2Block(nn.Module):
    """Transformer block for datadep2: receives angles, produces new angles."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_softplus=False, use_cumsum=False, rotate_v=False,
                 split_angles=False, hidden_mult=4, angle_activation='tanh',
                 shared_angle_ln=None, angle_dropout=0.0):
        super().__init__()
        self.attn = DataDep2Attention(n_embed, n_heads, dropout, window_size,
                                      use_softplus, use_cumsum, rotate_v)
        self.ffn = FeedForwardWithAngles(n_embed, dropout, split_angles=split_angles,
                                         hidden_mult=hidden_mult, angle_activation=angle_activation,
                                         shared_angle_ln=shared_angle_ln,
                                         angle_dropout=angle_dropout, n_heads=n_heads)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles, rope_base=None):
        x = x + self.attn(self.ln1(x), angles)
        content, new_angles = self.ffn(self.ln2(x))
        if rope_base is not None:
            new_angles = new_angles + rope_base
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
    'rope_lf': RoPELearnedFreqAttention,
    'joformer_fixed': JoFormerFixedAttention,
    'joformer_fixed_lf': JoFormerFixedLearnedFreqAttention,
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
      - 'shared_pos_qk':   shared angle MLP (LN→abs), cumsum, Q/K only
      - 'shared_pos_qkv':  shared angle MLP (LN→abs), cumsum, Q/K/V + inverse
      - 'random_pos_qk':   random positive angles (log-spaced), cumsum, Q/K only
      - 'random_pos_qkv':  random positive angles (log-spaced), cumsum, Q/K/V + inverse
      - list:        explicit per-layer types, e.g. ['rope','rope','nope','nope']
    """

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, block_size,
                 dropout, attn_config='rope', window_size=256, use_softplus=False,
                 split_angles=False, angle_hidden_mult=4, angle_activation='tanh',
                 angle_dropout=0.0, uniform_random_freq=0.0, detach_v=False, rope_v=False):
        super().__init__()
        self.block_size = block_size
        self.window_size = window_size
        self.use_softplus = use_softplus
        self.split_angles = split_angles
        self._angle_hidden_mult = angle_hidden_mult
        self._angle_activation = angle_activation
        self._angle_dropout = angle_dropout
        self._uniform_random_freq = uniform_random_freq
        self._detach_v = detach_v
        self._rope_v = rope_v
        self.is_datadep2 = isinstance(attn_config, str) and (
            attn_config.startswith('datadep2') or attn_config.startswith('monoidal2')
            or attn_config.startswith('joformer2'))

        # Detect new-style model types from attn_config string
        # Naming: {scope}_{sign}[_split]_{rotation}
        #   scope: shared, perlayer, random
        #   sign: pos (LN→abs), ln (LN only, allows negative)
        #   split: optional, separate angles for Q/K vs V
        #   rotation: qk or qkv
        self._new_style = False
        self._scope = self._sign = self._split = self._rotation = None
        self._ln_input = False
        if isinstance(attn_config, str):
            # ln_ prefix: LN on input to angle MLP
            # ln_perlayer_qkv: input LN, no output LN
            # ln_perlayer_ln_qkv: input LN + output LN
            # ln_perlayer_pos_qkv: input LN + output LN + abs
            if attn_config.startswith('ln_'):
                rest = attn_config[3:]  # strip 'ln_' prefix
                # Parse rest as {scope}[_{sign}]_{rotation}
                for scope in ('perlayer', 'shared'):
                    for sign in ('pos', 'ln', 'cc', 'rms', 'fs', 'fsr', 'fss', 'fssd', 'fssx', 'fssr', 'fssa', 'fsnt', 'lf', 'lfnl', 'lfds', 'lfb', 'lfbf', 'glf', 'cb', 'cbd', 'pcb', 'pmlp', 'pmlp2', 'rpemb4', 'rpemb3', 'rpemb', 'pemb2', 'pemb', 'det', 'deti', 'detb', ''):
                        for suffix in ('split_qkv', 'qk', 'qkv'):
                            if sign:
                                name = f'{scope}_{sign}_{suffix}'
                            else:
                                name = f'{scope}_{suffix}'
                            if rest == name:
                                self._new_style = True
                                self._scope = scope
                                self._sign = sign if sign else 'none'
                                self._ln_input = True
                                self._indep = False
                                self._rotation = 'qkv' if suffix.endswith('qkv') else 'qk'
                                self._split = 'split' in suffix
                                break
                        if self._new_style:
                            break
                    if self._new_style:
                        break

            if not self._new_style:
                for scope in ('shared', 'perlayer', 'random'):
                    for sign in ('pos', 'ln', 'cc', 'rms', 'fs', 'fsr', 'fss', 'fssd', 'fssx', 'fssr', 'fssa', 'fsnt', 'lf', 'lfnl', 'lfds', 'lfb', 'lfbf', 'glf', 'cb', 'cbd', 'pcb', 'pmlp', 'pmlp2', 'rpemb4', 'rpemb3', 'rpemb', 'pemb2', 'pemb', 'det', 'deti', 'detb'):
                        for suffix in ('indep_qk', 'indep_qkv', 'split_qkv', 'qk', 'qkv'):
                            name = f'{scope}_{sign}_{suffix}'
                            if attn_config == name:
                                self._new_style = True
                                self._scope = scope
                                self._sign = sign
                                self._indep = 'indep' in suffix
                                self._rotation = 'qkv' if suffix.endswith('qkv') else 'qk'
                                self._split = 'split' in suffix or (self._indep and self._rotation == 'qkv')
                                break
                        if self._new_style:
                            break
                    if self._new_style:
                        break

        if self._new_style:
            self.tok_emb = nn.Embedding(vocab_size, n_embed)
            self.n_embed = n_embed
            rotate_v = self._rotation == 'qkv'
            use_abs = self._sign == 'pos'

            # Fixed random ±1 sign embedding for lfds/det
            if self._sign in ('lfds', 'det'):
                sign_emb = 2 * torch.randint(0, 2, (vocab_size, n_embed // 2)).float() - 1
                self.register_buffer('sign_emb', sign_emb)

            # Per-layer independent sign embeddings for deti
            if self._sign == 'deti':
                for layer_i in range(n_layers):
                    sign_emb = 2 * torch.randint(0, 2, (vocab_size, n_embed // 2)).float() - 1
                    self.register_buffer(f'sign_emb_{layer_i}', sign_emb)

            # Fixed RoPE freq for det/detb/deti
            if self._sign in ('det', 'detb', 'deti'):
                head_dim = n_embed // n_heads
                freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                self.register_buffer('det_freq', freqs.repeat(n_heads))

            # Compute base_freq for 'lfbf' sign (learned freq + base freq)
            _base_freq_learned = None
            if self._sign == 'lfbf':
                head_dim = n_embed // n_heads
                freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                _base_freq_learned = freqs.repeat(n_heads)

            # Compute freq_scales for 'fs'/'fsnt' sign
            _freq_scales = None
            _use_tanh = self._sign == 'fs'
            _learned_freq = self._sign in ('lf', 'lfnl', 'lfds', 'lfb')
            if self._sign in ('fssx', 'fssr', 'fssa'):
                head_dim = n_embed // n_heads
                freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                self.register_buffer('_fssx_freq_scales', freqs.repeat(n_heads))
                self._fssx_ln = nn.LayerNorm(n_embed // 2)

            if self._sign in ('fs', 'fsr', 'fss', 'fssd', 'fsnt'):
                head_dim = n_embed // n_heads
                freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                _freq_scales = freqs.repeat(n_heads)  # (C//2,)

            if self._scope == 'perlayer':
                use_output_ln = self._sign in ('pos', 'ln')
                use_rms = self._sign == 'rms'
                self.layer_types = [attn_config] * n_layers
                self.blocks = nn.ModuleList([
                    PerLayerAngleBlock(n_embed, n_heads, dropout, window_size,
                                       use_softplus, rotate_v, use_abs=use_abs,
                                       split=self._split, ln_input=self._ln_input,
                                       use_rms=use_rms,
                                       use_output_ln=use_output_ln,
                                       hidden_mult=self._angle_hidden_mult,
                                       angle_dropout=self._angle_dropout,
                                       freq_scales=_freq_scales,
                                       use_tanh=_use_tanh,
                                       learned_freq=_learned_freq,
                                       detach_v=self._detach_v,
                                       rope_v=self._rope_v)
                    for _ in range(n_layers)
                ])
                if self._sign == 'lfb':
                    for block in self.blocks:
                        block.angle_mlp.bernoulli = True
            else:
                # shared or random — use ExternalAngleBlock
                if self._scope == 'shared' and self._sign == 'pmlp':
                    # Per-layer base embeddings + MLP correction from x
                    embs = []
                    for _ in range(n_layers):
                        emb = nn.Embedding(vocab_size, n_embed // 2)
                        nn.init.zeros_(emb.weight)
                        embs.append(emb)
                    self.layer_angle_embs = nn.ModuleList(embs)
                    # Per-layer correction MLPs
                    self.layer_correction_mlps = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(n_embed, n_embed),
                            nn.GELU(),
                            nn.Linear(n_embed, n_embed // 2),
                        ) for _ in range(n_layers)
                    ])
                    # MLP output keeps random init; scale=0 ensures correction starts at 0
                    # Learnable scale per layer, initialized to 0
                    self.layer_correction_scales = nn.ParameterList([
                        nn.Parameter(torch.zeros(1)) for _ in range(n_layers)
                    ])
                    # rope_base
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_pemb_rope_base', -freqs.repeat(n_heads))
                elif self._scope == 'shared' and self._sign == 'pmlp2':
                    # Stacked corrections: one shared base embedding, per-layer MLP corrections accumulate
                    base_emb = nn.Embedding(vocab_size, n_embed // 2)
                    nn.init.zeros_(base_emb.weight)
                    self.angle_base_emb = base_emb
                    # Per-layer correction MLPs
                    self.layer_correction_mlps = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(n_embed, n_embed),
                            nn.GELU(),
                            nn.Linear(n_embed, n_embed // 2),
                        ) for _ in range(n_layers)
                    ])
                    # Learnable scale per layer, initialized to 0
                    self.layer_correction_scales = nn.ParameterList([
                        nn.Parameter(torch.zeros(1)) for _ in range(n_layers)
                    ])
                    # rope_base
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_pemb_rope_base', -freqs.repeat(n_heads))
                elif self._scope == 'shared' and self._sign == 'rpemb4':
                    # Random-pemb v4: freq = (1 + tanh(LN(emb))) * rope_base, angle = noise * freq
                    embs = []
                    for _ in range(n_layers):
                        emb = nn.Embedding(vocab_size, n_embed // 2)
                        nn.init.zeros_(emb.weight)
                        embs.append(emb)
                    self.layer_angle_embs = nn.ModuleList(embs)
                    self._rpemb_ln = nn.LayerNorm(n_embed // 2)
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_pemb_rope_base', freqs.repeat(n_heads))
                    self._angle_rng = None
                elif self._scope == 'shared' and self._sign == 'rpemb3':
                    # Random-pemb v3: freq = abs(tanh(LN(emb))), angle = noise * freq
                    embs = []
                    for _ in range(n_layers):
                        emb = nn.Embedding(vocab_size, n_embed // 2)
                        nn.init.zeros_(emb.weight)
                        embs.append(emb)
                    self.layer_angle_embs = nn.ModuleList(embs)
                    self._rpemb_ln = nn.LayerNorm(n_embed // 2)
                    self._angle_rng = None
                elif self._scope == 'shared' and self._sign == 'rpemb':
                    # Random-pemb v2: freq = abs(LN(tanh(emb)*π + rope_base)), angle = noise * freq
                    embs = []
                    for _ in range(n_layers):
                        emb = nn.Embedding(vocab_size, n_embed // 2)
                        nn.init.zeros_(emb.weight)
                        embs.append(emb)
                    self.layer_angle_embs = nn.ModuleList(embs)
                    self._rpemb_ln = nn.LayerNorm(n_embed // 2)
                    # rope_base used as initial frequency scale
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_pemb_rope_base', freqs.repeat(n_heads))
                    # RNG for random sampling
                    self._angle_rng = None
                elif self._scope == 'shared' and self._sign == 'pemb2':
                    # pemb v2: tanh(emb) * rope_base, emb init to 1
                    embs = []
                    for _ in range(n_layers):
                        emb = nn.Embedding(vocab_size, n_embed // 2)
                        nn.init.ones_(emb.weight)
                        embs.append(emb)
                    self.layer_angle_embs = nn.ModuleList(embs)
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_pemb_rope_base', -freqs.repeat(n_heads))
                elif self._scope == 'shared' and self._sign == 'pemb':
                    # Per-layer learned angle embeddings, zero-init (starts as rope_base)
                    embs = []
                    for _ in range(n_layers):
                        emb = nn.Embedding(vocab_size, n_embed // 2)
                        nn.init.zeros_(emb.weight)
                        embs.append(emb)
                    self.layer_angle_embs = nn.ModuleList(embs)
                    # rope_base for additive structure
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_pemb_rope_base', -freqs.repeat(n_heads))
                elif self._scope == 'shared' and self._sign == 'pcb':
                    # Factored: per-token base embedding + K shared corrections
                    K = self._angle_hidden_mult
                    self._pcb_K = K
                    # Per-layer base embeddings (like pemb, zero-init)
                    embs = []
                    for _ in range(n_layers):
                        emb = nn.Embedding(vocab_size, n_embed // 2)
                        nn.init.zeros_(emb.weight)
                        embs.append(emb)
                    self.layer_angle_embs = nn.ModuleList(embs)
                    # Per-layer shared correction codebooks (K × C//2, small init)
                    self.layer_corrections = nn.ParameterList([
                        nn.Parameter(torch.randn(K, n_embed // 2) * 0.01) for _ in range(n_layers)
                    ])
                    # Per-layer projections for scoring
                    self.pcb_projs = nn.ModuleList([
                        nn.Linear(n_embed, n_embed // 2, bias=False) for _ in range(n_layers)
                    ])
                    # rope_base
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_pemb_rope_base', -freqs.repeat(n_heads))
                elif self._scope == 'shared' and self._sign == 'cbd':
                    # Angle codebook with dot-product selection
                    K = self._angle_hidden_mult  # reuse hidden_mult as K
                    self._cb_K = K
                    # K angle templates per token, near-zero init (starts close to rope_base)
                    self.angle_codebook = nn.Parameter(torch.randn(vocab_size, K, n_embed // 2) * 0.01)
                    # rope_base for additive structure
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    self.register_buffer('_cbd_rope_base', -freqs.repeat(n_heads))
                    # Per-layer projections from x to angle space
                    self.cbd_projs = nn.ModuleList([
                        nn.Linear(n_embed, n_embed // 2, bias=False) for _ in range(n_layers)
                    ])
                elif self._scope == 'shared' and self._sign == 'cb':
                    # Angle codebook: K angle templates per token
                    K = self._angle_hidden_mult  # reuse hidden_mult as K
                    self._cb_K = K
                    head_dim = n_embed // n_heads
                    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    rope_freq = freqs.repeat(n_heads)  # (C//2,)
                    # Initialize codebook: random signs × rope_freq per entry per token
                    cb_init = (2 * torch.randint(0, 2, (vocab_size, K, n_embed // 2)).float() - 1) * rope_freq
                    self.angle_codebook = nn.Parameter(cb_init)
                    # Per-layer score projections
                    self.cb_score_projs = nn.ModuleList([
                        nn.Linear(n_embed, K) for _ in range(n_layers)
                    ])
                elif self._scope == 'shared' and self._sign in ('fssx', 'fssr', 'fssa'):
                    pass  # No MLP needed, LN and freq_scales already set up above
                elif self._scope == 'shared':
                    use_output_ln = False if self._sign in ('lfnl', 'lfbf') else not self._ln_input
                    if self._split:
                        self.shared_angle_mlp = SharedAngleMLPSplit(n_embed, use_abs=use_abs, use_output_ln=use_output_ln, hidden_mult=self._angle_hidden_mult, angle_dropout=self._angle_dropout)
                    else:
                        self.shared_angle_mlp = SharedAngleMLP(n_embed, use_abs=use_abs, use_output_ln=use_output_ln, hidden_mult=self._angle_hidden_mult, angle_dropout=self._angle_dropout, freq_scales=_freq_scales, use_tanh=_use_tanh, learned_freq=_learned_freq, base_freq_learned=_base_freq_learned)
                    if hasattr(self, 'shared_angle_mlp') and self._sign == 'lfb':
                        self.shared_angle_mlp.bernoulli = True
                    if hasattr(self, 'shared_angle_mlp') and self._sign == 'fsr':
                        self.shared_angle_mlp.random_freq_scales = True
                    if hasattr(self, 'shared_angle_mlp') and self._sign in ('fss', 'fssd'):
                        self.shared_angle_mlp.sign_freq_scales = True
                        self.shared_angle_mlp.normalize_weights = True
                    if self._ln_input:
                        self.angle_input_ln = nn.LayerNorm(n_embed)

                if self._scope == 'random':
                    # Separate RNG so random angles don't pollute global RNG during eval
                    # Created lazily in forward() to match device
                    self._angle_rng = None
                    head_dim = n_embed // n_heads
                    if self._uniform_random_freq > 0:
                        # All dimensions use the same frequency
                        freqs = torch.full((head_dim // 2,), self._uniform_random_freq)
                    else:
                        # Log-spaced frequencies matching RoPE
                        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    if use_abs:
                        # Uniform(0, 2*freq) — positive only, mean = freq
                        random_scales = (2.0 * freqs).repeat(n_heads)
                    else:
                        # Uniform(-freq, freq) — signed, mean magnitude = freq/2
                        random_scales = freqs.repeat(n_heads)
                    if self._sign == 'glf':
                        self.random_angle_scales = nn.Parameter(random_scales)
                    else:
                        self.register_buffer('random_angle_scales', random_scales)
                    self.register_buffer('_random_signed', torch.tensor(not use_abs))

                self.layer_types = [attn_config] * n_layers
                self.blocks = nn.ModuleList([
                    ExternalAngleBlock(n_embed, n_heads, dropout, window_size,
                                       use_softplus, rotate_v, detach_v=self._detach_v,
                                       rope_v=self._rope_v)
                    for _ in range(n_layers)
                ])

        elif self.is_datadep2:
            if split_angles:
                # Separate content and angle embeddings (allows separate lr)
                self.tok_emb = nn.Embedding(vocab_size, n_embed)
                self.angle_emb = nn.Embedding(vocab_size, n_embed // 2)
                nn.init.zeros_(self.angle_emb.weight)  # start as pure RoPE
                # RoPE-equivalent base angles: constant vector added at every position
                # Cumsum is flip-cumsum-flip (suffix sum), giving (T-t)*δ at position t.
                # Negate δ so we get (t-T)*δ, which has the same relative differences
                # as RoPE's t*δ (just shifted by a global constant T*δ, which cancels
                # in Q·K because both Q and K get the same shift).
                head_dim = n_embed // n_heads
                freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                # Negate so suffix-sum matches RoPE direction
                rope_base = -freqs.repeat(n_heads)
                self.register_buffer('rope_base_angles', rope_base)
                # Random ±1 × rope_freq initialization (for freeze_angle_emb mode)
                signs = 2 * torch.randint(0, 2, (vocab_size, n_embed // 2)).float() - 1
                self.register_buffer('_angle_emb_random_init', signs * freqs.repeat(n_heads))
            else:
                # Legacy: single combined embedding
                self.tok_emb = nn.Embedding(vocab_size, n_embed + n_embed // 2)
            self.n_embed = n_embed

            # Determine cumsum and rotate_v from config name
            base = attn_config.split('_')[0]  # datadep2, monoidal2, or joformer2
            use_cumsum = base in ('monoidal2', 'joformer2')
            rotate_v = base == 'joformer2'

            # Shared angle LayerNorm across all layers (if using LN activation)
            _shared_angle_ln = (nn.LayerNorm(n_embed // 2)
                                if self._angle_activation == 'ln' else None)
            # Embedding angle LN (for consistent LN on initial angles)
            if self._angle_activation == 'ln' and not split_angles:
                self._emb_angle_ln = nn.LayerNorm(n_embed // 2)

            def _make_v2_block(ws):
                return DataDep2Block(n_embed, n_heads, dropout, ws,
                                     use_softplus, use_cumsum, rotate_v,
                                     split_angles=split_angles,
                                     hidden_mult=self._angle_hidden_mult,
                                     angle_activation=self._angle_activation,
                                     shared_angle_ln=_shared_angle_ln,
                                     angle_dropout=self._angle_dropout)

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

        # Zero-init angle parameters so model starts with identity rotations
        if self.is_datadep2 and split_angles:
            with torch.no_grad():
                for p in self.angle_params():
                    p.zero_()

    def angle_params(self):
        """Return parameters that produce rotation angles (for separate lr)."""
        params = []
        # New-style models: perlayer or shared angle MLPs
        if self._new_style and self._scope in ('perlayer', 'shared'):
            if self._scope == 'perlayer':
                for block in self.blocks:
                    if isinstance(block, PerLayerAngleBlock):
                        params.extend(block.angle_mlp.parameters())
                        if block.split:
                            params.extend(block.v_angle_mlp.parameters())
            elif self._scope == 'shared':
                if hasattr(self, 'shared_angle_mlp'):
                    params.extend(self.shared_angle_mlp.parameters())
            if hasattr(self, 'angle_input_ln'):
                params.extend(self.angle_input_ln.parameters())
            return params
        # Legacy datadep2 models
        if not self.is_datadep2 or not self.split_angles:
            return []
        params = list(self.angle_emb.parameters())
        for block in self.blocks:
            if isinstance(block, DataDep2Block) and block.ffn.split_angles:
                params.extend(block.ffn.fc2_angles.parameters())
        return params

    def non_angle_params(self):
        """Return all parameters except angle-producing ones."""
        angle_param_ids = {id(p) for p in self.angle_params()}
        return [p for p in self.parameters() if id(p) not in angle_param_ids]

    def set_eval_topk(self, topk):
        """Set top-k attention for eval mode on all layers."""
        for block in self.blocks:
            attn = block.attn if hasattr(block, 'attn') else None
            if attn is not None:
                attn.eval_topk = topk

    def forward(self, idx, targets=None):
        if self._new_style:
            x = self.tok_emb(idx)  # (B, T, C)
            B, T, C = x.shape

            # Look up fixed signs for lfds/det
            _signs = self.sign_emb[idx] if self._sign in ('lfds', 'det') else None

            # det/detb: fixed rope freq, no MLP
            if self._sign == 'det':
                angles = _signs * self.det_freq  # (B, T, C//2)
                for block in self.blocks:
                    x = block(x, angles)
            elif self._sign == 'pcb':
                # Factored: per-token base + shared correction codebook
                for i, block in enumerate(self.blocks):
                    # Base angle from per-token embedding
                    raw = self.layer_angle_embs[i](idx)  # (B, T, C//2)
                    base = torch.tanh(raw) * math.pi + self._pemb_rope_base
                    # Build K candidates: base + each correction
                    codebook = self.layer_corrections[i]  # (K, C//2)
                    candidates = base.unsqueeze(2) + codebook  # (B, T, K, C//2)
                    # Score candidates via dot product with projected x
                    x_proj = self.pcb_projs[i](x)  # (B, T, C//2)
                    scores = (x_proj.unsqueeze(2) * candidates).sum(dim=-1)  # (B, T, K)
                    # Soft selection
                    soft_weights = F.softmax(scores, dim=-1)  # (B, T, K)
                    soft_selected = (soft_weights.unsqueeze(-1) * candidates).sum(dim=2)  # (B, T, C//2)
                    # Hard selection (argmax)
                    hard_idx = scores.argmax(dim=-1)  # (B, T)
                    hard_selected = candidates[
                        torch.arange(B, device=x.device).unsqueeze(1).expand(B, T),
                        torch.arange(T, device=x.device).unsqueeze(0).expand(B, T),
                        hard_idx
                    ]  # (B, T, C//2)
                    # STE
                    angles = (hard_selected - soft_selected).detach() + soft_selected
                    x = block(x, angles)
            elif self._sign == 'pmlp':
                # Per-layer base embeddings + bounded MLP(x) correction
                for i, block in enumerate(self.blocks):
                    raw = self.layer_angle_embs[i](idx)  # (B, T, C//2)
                    base = torch.tanh(raw) * math.pi + self._pemb_rope_base
                    correction = self.layer_correction_scales[i] * math.pi * torch.tanh(self.layer_correction_mlps[i](x))
                    angles = base + correction
                    x = block(x, angles)
            elif self._sign == 'pmlp2':
                # Stacked corrections: one shared base, per-layer MLP corrections accumulate
                angles = torch.tanh(self.angle_base_emb(idx)) * math.pi + self._pemb_rope_base
                for i, block in enumerate(self.blocks):
                    correction = self.layer_correction_scales[i] * math.pi * torch.tanh(self.layer_correction_mlps[i](x))
                    angles = angles + correction
                    x = block(x, angles)
            elif self._sign == 'rpemb4':
                # Random-pemb v4: freq = (1 + tanh(LN(emb))) * rope_base
                if self._angle_rng is None or self._angle_rng.device != x.device:
                    self._angle_rng = torch.Generator(device=x.device)
                    self._angle_rng.manual_seed(12345)
                g = self._angle_rng
                for i, block in enumerate(self.blocks):
                    raw = self.layer_angle_embs[i](idx)  # (B, T, C//2)
                    freq = (1.0 + torch.tanh(self._rpemb_ln(raw))) * self._pemb_rope_base
                    noise = 2.0 * torch.rand(B, T, C // 2, device=x.device, dtype=x.dtype, generator=g) - 1.0
                    angles = noise * freq
                    x = block(x, angles)
            elif self._sign == 'rpemb3':
                # Random-pemb v3: freq = abs(LN(tanh(emb) * π)), angle = noise * freq
                if self._angle_rng is None or self._angle_rng.device != x.device:
                    self._angle_rng = torch.Generator(device=x.device)
                    self._angle_rng.manual_seed(12345)
                g = self._angle_rng
                for i, block in enumerate(self.blocks):
                    raw = self.layer_angle_embs[i](idx)  # (B, T, C//2)
                    freq = torch.abs(self._rpemb_ln(torch.tanh(raw) * math.pi))
                    noise = 2.0 * torch.rand(B, T, C // 2, device=x.device, dtype=x.dtype, generator=g) - 1.0
                    angles = noise * freq
                    x = block(x, angles)
            elif self._sign == 'rpemb':
                # Random-pemb v2: learned per-token frequencies with LN, random sampling
                if self._angle_rng is None or self._angle_rng.device != x.device:
                    self._angle_rng = torch.Generator(device=x.device)
                    self._angle_rng.manual_seed(12345)
                g = self._angle_rng
                for i, block in enumerate(self.blocks):
                    raw = self.layer_angle_embs[i](idx)  # (B, T, C//2)
                    freq = torch.abs(self._rpemb_ln(torch.tanh(raw) * math.pi + self._pemb_rope_base))
                    noise = 2.0 * torch.rand(B, T, C // 2, device=x.device, dtype=x.dtype, generator=g) - 1.0
                    angles = noise * freq
                    x = block(x, angles)
            elif self._sign == 'pemb2':
                # pemb v2: tanh(emb) * rope_base
                for i, block in enumerate(self.blocks):
                    raw = self.layer_angle_embs[i](idx)  # (B, T, C//2)
                    angles = torch.tanh(raw) * self._pemb_rope_base
                    x = block(x, angles)
            elif self._sign == 'pemb':
                # Per-layer learned angle embeddings — no MLP, no hidden state
                # tanh * π + rope_base (same structure as monoidal2 angle_emb)
                for i, block in enumerate(self.blocks):
                    raw = self.layer_angle_embs[i](idx)  # (B, T, C//2)
                    angles = torch.tanh(raw) * math.pi + self._pemb_rope_base
                    x = block(x, angles)
            elif self._sign == 'deti':
                for i, block in enumerate(self.blocks):
                    layer_signs = getattr(self, f'sign_emb_{i}')[idx]  # (B, T, C//2)
                    angles = layer_signs * self.det_freq
                    x = block(x, angles)
            elif self._sign == 'detb':
                noise = 2.0 * torch.bernoulli(torch.full(
                    (B, T, C // 2), 0.5, device=x.device, dtype=x.dtype)) - 1.0
                angles = noise * self.det_freq
                for block in self.blocks:
                    x = block(x, angles)
            elif self._scope == 'perlayer':
                for block in self.blocks:
                    if _signs is not None:
                        block.angle_mlp._fixed_signs = _signs
                    x = block(x)
            elif self._scope == 'shared' and self._sign == 'cbd':
                # Codebook with dot-product selection
                # Look up K raw templates per token, apply tanh*π + rope_base
                raw_templates = self.angle_codebook[idx]  # (B, T, K, C//2)
                token_angles = torch.tanh(raw_templates) * math.pi + self._cbd_rope_base  # (B, T, K, C//2)
                for i, block in enumerate(self.blocks):
                    # Project x to angle space
                    x_proj = self.cbd_projs[i](x)  # (B, T, C//2)
                    # Dot product scores: similarity between projection and each template
                    scores = (x_proj.unsqueeze(2) * token_angles).sum(dim=-1)  # (B, T, K)
                    # Soft selection
                    soft_weights = F.softmax(scores, dim=-1)  # (B, T, K)
                    soft_selected = (soft_weights.unsqueeze(-1) * token_angles).sum(dim=2)  # (B, T, C//2)
                    # Hard selection (argmax)
                    hard_idx = scores.argmax(dim=-1)  # (B, T)
                    hard_selected = token_angles[
                        torch.arange(B, device=x.device).unsqueeze(1).expand(B, T),
                        torch.arange(T, device=x.device).unsqueeze(0).expand(B, T),
                        hard_idx
                    ]  # (B, T, C//2)
                    # STE: forward=hard, backward=soft
                    result = (hard_selected - soft_selected).detach() + soft_selected
                    x = block(x, result)
            elif self._scope == 'shared' and self._sign == 'cb':
                # Codebook selection: per-layer argmax+STE from K per-token angles
                token_angles = self.angle_codebook[idx]  # (B, T, K, C//2)
                for i, block in enumerate(self.blocks):
                    scores = self.cb_score_projs[i](x)  # (B, T, K)
                    # Soft selection
                    soft_weights = F.softmax(scores, dim=-1)  # (B, T, K)
                    soft_selected = (soft_weights.unsqueeze(-1) * token_angles).sum(dim=2)  # (B, T, C//2)
                    # Hard selection (argmax)
                    hard_idx = scores.argmax(dim=-1)  # (B, T)
                    hard_selected = token_angles[
                        torch.arange(B, device=x.device).unsqueeze(1).expand(B, T),
                        torch.arange(T, device=x.device).unsqueeze(0).expand(B, T),
                        hard_idx
                    ]  # (B, T, C//2)
                    # STE: forward=hard, backward=soft
                    result = (hard_selected - soft_selected).detach() + soft_selected
                    x = block(x, result)
            elif self._scope == 'shared':
                for block in self.blocks:
                    if self._sign == 'fssa':
                        # Argsort subset: pick C//2 dims with smallest values, LN, sign × freq
                        xd = x.detach()
                        selected = xd.argsort(dim=-1)[..., :C // 2]  # (B, T, C//2)
                        x_subset = torch.gather(xd, -1, selected)  # (B, T, C//2)
                        hard = torch.sign(self._fssx_ln(x_subset))
                        result = hard * self._fssx_freq_scales
                    elif self._sign == 'fssr':
                        # Random subset: pick C//2 dims from C per position, LN, sign × freq
                        xd = x.detach()
                        # Random permutation per (batch, position)
                        perm = torch.rand(B, T, C, device=x.device).argsort(dim=-1)
                        selected = perm[..., :C // 2]  # (B, T, C//2)
                        x_subset = torch.gather(xd, -1, selected)  # (B, T, C//2)
                        hard = torch.sign(self._fssx_ln(x_subset))
                        result = hard * self._fssx_freq_scales
                    elif self._sign == 'fssx':
                        # sign(LN(x.detach()[:C//2])) × freq_scales — no MLP
                        x_half = x.detach()[..., :C // 2]
                        hard = torch.sign(self._fssx_ln(x_half))
                        result = hard * self._fssx_freq_scales
                    else:
                        angle_input = self.angle_input_ln(x) if self._ln_input else x
                        if self._sign == 'fssd':
                            angle_input = angle_input.detach()
                        if _signs is not None:
                            self.shared_angle_mlp._fixed_signs = _signs
                        result = self.shared_angle_mlp(angle_input)
                    if self._sign == 'cc':
                        # Causal centering: subtract running mean across time
                        cs = torch.cumsum(result, dim=1)
                        positions = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
                        result = result - cs / positions
                    if self._split:
                        qk_angles, v_angles = result
                        x = block(x, qk_angles, v_angles=v_angles)
                    else:
                        x = block(x, result)
            elif self._scope == 'random':
                # Lazy-init generator on correct device
                if self._angle_rng is None or self._angle_rng.device != x.device:
                    self._angle_rng = torch.Generator(device=x.device)
                    self._angle_rng.manual_seed(12345)
                g = self._angle_rng
                signed = self._random_signed.item()

                _scales = torch.abs(self.random_angle_scales) if self._sign == 'glf' else self.random_angle_scales

                def _sample_angles():
                    if signed:
                        return (2.0 * torch.rand(B, T, C // 2, device=x.device, dtype=x.dtype, generator=g) - 1.0) * _scales
                    else:
                        return torch.rand(B, T, C // 2, device=x.device, dtype=x.dtype, generator=g) * _scales

                if self._indep:
                    # Per-layer independent angles
                    for block in self.blocks:
                        v_ang = _sample_angles() if self._split else None
                        x = block(x, _sample_angles(), v_angles=v_ang)
                elif self._split:
                    angles = _sample_angles()
                    v_angles = _sample_angles()
                    for block in self.blocks:
                        x = block(x, angles, v_angles=v_angles)
                else:
                    angles = _sample_angles()
                    for block in self.blocks:
                        x = block(x, angles)
        elif self.is_datadep2:
            if self.split_angles:
                x = self.tok_emb(idx)          # (B, T, C)
                if hasattr(self, '_use_random_angle_emb') and self._use_random_angle_emb:
                    # Frozen random ±1 × rope_freq per token, no tanh/base
                    angles = self._angle_emb_random_init[idx]
                else:
                    # Learned deviation (zero-init) + constant RoPE base
                    angles = torch.tanh(self.angle_emb(idx)) * math.pi + self.rope_base_angles
            else:
                emb = self.tok_emb(idx)  # (B, T, C + C//2)
                C = self.n_embed
                x = emb[..., :C]
                if hasattr(self, '_emb_angle_ln'):
                    angles = self._emb_angle_ln(emb[..., C:])
                else:
                    angles = torch.tanh(emb[..., C:]) * math.pi
            if hasattr(self, '_use_random_angle_emb') and self._use_random_angle_emb and not hasattr(self, '_keep_rope_base'):
                rope_base = None  # FFN produces complete angles, no additive base
            else:
                rope_base = self.rope_base_angles if self.split_angles else None
            for block in self.blocks:
                if isinstance(block, DataDep2Block):
                    x, angles = block(x, angles, rope_base=rope_base)
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

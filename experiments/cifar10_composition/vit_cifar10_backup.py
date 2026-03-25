#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Patent Pending.
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------
#
# vit_cifar10.py — Vision Transformer for CIFAR-10 with multiple
# positional encoding schemes:
#   1. learned   — standard learned positional embeddings (baseline)
#   2. rope2d    — 2D Rotary Position Embeddings (fixed frequencies)
#   3. monoidal  — Learnable per-axis rotation angles (our framework)
#
# The monoidal PE generalizes RoPE: same rotation mechanism but with
# learnable frequencies instead of fixed 10000^(-2d/D).

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =====================================================================
# Positional Encoding Modules
# =====================================================================

class LearnedPE(nn.Module):
    """Standard learned positional embeddings. Added to patch embeddings."""

    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)

    def forward(self, x):
        # x: (B, N+1, D) where N+1 includes CLS token
        return x + self.pos_embed[:, :x.size(1)]

    def apply_to_attention(self, q, k, patch_positions=None):
        # No-op: learned PE is additive, already applied to embeddings
        return q, k


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings with fixed frequencies.

    Splits embedding into x-dims and y-dims. Each gets 1D RoPE
    with standard frequencies theta_d = 10000^(-2d/D).
    Applied to Q and K in attention, not added to embeddings.
    """

    def __init__(self, embed_dim, grid_h, grid_w, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Split head_dim in half: first half for x-axis, second for y-axis
        half = self.head_dim // 2
        assert half % 2 == 0, "head_dim // 2 must be even for RoPE pairs"
        self.rope_dim = half

        # Fixed frequencies: theta_d = 1 / 10000^(2d/rope_dim)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('freqs', freqs)  # (rope_dim // 2,)

        # Precompute position-dependent angles
        self._precompute_angles()

    def _precompute_angles(self):
        pos_y = torch.arange(self.grid_h).float()
        pos_x = torch.arange(self.grid_w).float()

        # Angles for y-axis: (H, rope_dim//2)
        angles_y = pos_y[:, None] * self.freqs[None, :]
        # Angles for x-axis: (W, rope_dim//2)
        angles_x = pos_x[:, None] * self.freqs[None, :]

        # Expand to full grid: (H*W, rope_dim//2) for each axis
        # For each position (i,j): y_angles from row i, x_angles from col j
        ii, jj = torch.meshgrid(torch.arange(self.grid_h),
                                torch.arange(self.grid_w), indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        grid_angles_y = angles_y[ii]  # (H*W, rope_dim//2)
        grid_angles_x = angles_x[jj]  # (H*W, rope_dim//2)

        # Concatenate: first half = y angles, second half = x angles
        # Each is rope_dim//2 pairs, total = rope_dim = head_dim//2
        self.register_buffer('cos_y', torch.cos(grid_angles_y))
        self.register_buffer('sin_y', torch.sin(grid_angles_y))
        self.register_buffer('cos_x', torch.cos(grid_angles_x))
        self.register_buffer('sin_x', torch.sin(grid_angles_x))

    def _rotate(self, x, cos, sin):
        """Apply rotary embedding to paired dimensions."""
        # x: (..., rope_dim) where rope_dim is even
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)

    def forward(self, x):
        # No additive PE for RoPE
        return x

    def apply_to_attention(self, q, k, patch_positions=None):
        """Apply 2D RoPE to Q and K.

        Args:
            q, k: (B, n_heads, N, head_dim) — N includes CLS at position 0
        """
        B, H, N, D = q.shape
        half = D // 2
        rope_pairs = half // 2  # number of (cos, sin) pairs per axis

        # Split head_dim: first half for y-axis rope, second half for x-axis rope
        q_y, q_x = q[..., :half], q[..., half:]
        k_y, k_x = k[..., :half], k[..., half:]

        # CLS token (position 0) gets no rotation — skip it
        # Patch tokens start at position 1
        if N > self.grid_h * self.grid_w:
            # Has CLS token
            q_y_cls, q_y_patches = q_y[:, :, :1], q_y[:, :, 1:]
            k_y_cls, k_y_patches = k_y[:, :, :1], k_y[:, :, 1:]
            q_x_cls, q_x_patches = q_x[:, :, :1], q_x[:, :, 1:]
            k_x_cls, k_x_patches = k_x[:, :, :1], k_x[:, :, 1:]

            q_y_rot = self._rotate(q_y_patches, self.cos_y, self.sin_y)
            k_y_rot = self._rotate(k_y_patches, self.cos_y, self.sin_y)
            q_x_rot = self._rotate(q_x_patches, self.cos_x, self.sin_x)
            k_x_rot = self._rotate(k_x_patches, self.cos_x, self.sin_x)

            q = torch.cat([
                torch.cat([q_y_cls, q_y_rot], dim=2),
                torch.cat([q_x_cls, q_x_rot], dim=2)
            ], dim=-1)
            k = torch.cat([
                torch.cat([k_y_cls, k_y_rot], dim=2),
                torch.cat([k_x_cls, k_x_rot], dim=2)
            ], dim=-1)
        else:
            q = torch.cat([
                self._rotate(q_y, self.cos_y, self.sin_y),
                self._rotate(q_x, self.cos_x, self.sin_x)
            ], dim=-1)
            k = torch.cat([
                self._rotate(k_y, self.cos_y, self.sin_y),
                self._rotate(k_x, self.cos_x, self.sin_x)
            ], dim=-1)

        return q, k


class JoFormerOldPE(RoPE2D):
    """Old RoPE2D (per-head, split dimensions) + V rotation + inverse.
    The only difference from RoPE2D is V rotation and inversion."""

    def _inverse_rotate(self, x, cos, sin):
        """Inverse rotation: use cos, -sin."""
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out0 = x0 * cos + x1 * sin
        out1 = -x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)

    def apply_to_attention(self, q, k, patch_positions=None):
        """Apply 2D RoPE to Q, K, AND V. Store state for inverse."""
        # This gets called from the RoPE path in Attention.forward
        # But we need V too — so we override to signal we need the full path
        raise RuntimeError("JoFormerOldPE should use apply_to_attention_with_v")

    def apply_to_attention_with_v(self, q, k, v):
        """Apply 2D RoPE to Q, K, V. Returns rotated Q, K, V."""
        B, H, N, D = q.shape
        half = D // 2

        q_y, q_x = q[..., :half], q[..., half:]
        k_y, k_x = k[..., :half], k[..., half:]
        v_y, v_x = v[..., :half], v[..., half:]

        has_cls = N > self.grid_h * self.grid_w
        self._has_cls = has_cls

        if has_cls:
            q_y_cls, q_y_p = q_y[:, :, :1], q_y[:, :, 1:]
            k_y_cls, k_y_p = k_y[:, :, :1], k_y[:, :, 1:]
            v_y_cls, v_y_p = v_y[:, :, :1], v_y[:, :, 1:]
            q_x_cls, q_x_p = q_x[:, :, :1], q_x[:, :, 1:]
            k_x_cls, k_x_p = k_x[:, :, :1], k_x[:, :, 1:]
            v_x_cls, v_x_p = v_x[:, :, :1], v_x[:, :, 1:]

            q = torch.cat([
                torch.cat([q_y_cls, self._rotate(q_y_p, self.cos_y, self.sin_y)], dim=2),
                torch.cat([q_x_cls, self._rotate(q_x_p, self.cos_x, self.sin_x)], dim=2)
            ], dim=-1)
            k = torch.cat([
                torch.cat([k_y_cls, self._rotate(k_y_p, self.cos_y, self.sin_y)], dim=2),
                torch.cat([k_x_cls, self._rotate(k_x_p, self.cos_x, self.sin_x)], dim=2)
            ], dim=-1)
            v = torch.cat([
                torch.cat([v_y_cls, self._rotate(v_y_p, self.cos_y, self.sin_y)], dim=2),
                torch.cat([v_x_cls, self._rotate(v_x_p, self.cos_x, self.sin_x)], dim=2)
            ], dim=-1)
        else:
            q = torch.cat([self._rotate(q_y, self.cos_y, self.sin_y),
                           self._rotate(q_x, self.cos_x, self.sin_x)], dim=-1)
            k = torch.cat([self._rotate(k_y, self.cos_y, self.sin_y),
                           self._rotate(k_x, self.cos_x, self.sin_x)], dim=-1)
            v = torch.cat([self._rotate(v_y, self.cos_y, self.sin_y),
                           self._rotate(v_x, self.cos_x, self.sin_x)], dim=-1)

        return q, k, v

    def inverse_rotate_output_perhead(self, out):
        """Inverse rotate attention output per-head."""
        B, H, N, D = out.shape
        half = D // 2
        out_y, out_x = out[..., :half], out[..., half:]

        if self._has_cls:
            out_y_cls, out_y_p = out_y[:, :, :1], out_y[:, :, 1:]
            out_x_cls, out_x_p = out_x[:, :, :1], out_x[:, :, 1:]
            out = torch.cat([
                torch.cat([out_y_cls, self._inverse_rotate(out_y_p, self.cos_y, self.sin_y)], dim=2),
                torch.cat([out_x_cls, self._inverse_rotate(out_x_p, self.cos_x, self.sin_x)], dim=2)
            ], dim=-1)
        else:
            out = torch.cat([self._inverse_rotate(out_y, self.cos_y, self.sin_y),
                             self._inverse_rotate(out_x, self.cos_x, self.sin_x)], dim=-1)
        return out


class MonoidalAxialPE(nn.Module):
    """Learnable axial RoPE: same split-dimension structure as RoPE2D
    but with learnable frequencies instead of fixed. Per-head, per-axis."""

    def __init__(self, embed_dim, grid_h, grid_w, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.grid_h = grid_h
        self.grid_w = grid_w

        half = self.head_dim // 2
        assert half % 2 == 0, "head_dim // 2 must be even"
        self.rope_dim = half
        n_freq = half // 2

        # Learnable frequencies per axis (randomly initialized)
        self.freqs_y = nn.Parameter(torch.randn(n_freq) * 0.02)
        self.freqs_x = nn.Parameter(torch.randn(n_freq) * 0.02)

        # Grid positions
        ii, jj = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        self.register_buffer('pos_y', ii.reshape(-1).float())
        self.register_buffer('pos_x', jj.reshape(-1).float())

    def _get_cos_sin(self):
        """Compute cos/sin from learnable frequencies."""
        angles_y = self.pos_y[:, None] * self.freqs_y[None, :]  # (H*W, n_freq)
        angles_x = self.pos_x[:, None] * self.freqs_x[None, :]
        return torch.cos(angles_y), torch.sin(angles_y), torch.cos(angles_x), torch.sin(angles_x)

    def _rotate(self, x, cos, sin):
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)

    def forward(self, x):
        return x

    def apply_to_attention(self, q, k, patch_positions=None):
        """Apply learnable 2D axial rotation to Q and K (per-head)."""
        B, H, N, D = q.shape
        half = D // 2
        cos_y, sin_y, cos_x, sin_x = self._get_cos_sin()

        q_y, q_x = q[..., :half], q[..., half:]
        k_y, k_x = k[..., :half], k[..., half:]

        if N > self.grid_h * self.grid_w:
            q_y_cls, q_y_p = q_y[:, :, :1], q_y[:, :, 1:]
            k_y_cls, k_y_p = k_y[:, :, :1], k_y[:, :, 1:]
            q_x_cls, q_x_p = q_x[:, :, :1], q_x[:, :, 1:]
            k_x_cls, k_x_p = k_x[:, :, :1], k_x[:, :, 1:]

            q = torch.cat([
                torch.cat([q_y_cls, self._rotate(q_y_p, cos_y, sin_y)], dim=2),
                torch.cat([q_x_cls, self._rotate(q_x_p, cos_x, sin_x)], dim=2)
            ], dim=-1)
            k = torch.cat([
                torch.cat([k_y_cls, self._rotate(k_y_p, cos_y, sin_y)], dim=2),
                torch.cat([k_x_cls, self._rotate(k_x_p, cos_x, sin_x)], dim=2)
            ], dim=-1)
        else:
            q = torch.cat([self._rotate(q_y, cos_y, sin_y),
                           self._rotate(q_x, cos_x, sin_x)], dim=-1)
            k = torch.cat([self._rotate(k_y, cos_y, sin_y),
                           self._rotate(k_x, cos_x, sin_x)], dim=-1)

        return q, k


class JoFormerAxialPE(MonoidalAxialPE):
    """MonoidalAxialPE + V rotation + inverse rotation on output.
    Learnable axial frequencies, applied to Q, K, AND V."""

    def _inverse_rotate(self, x, cos, sin):
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out0 = x0 * cos + x1 * sin
        out1 = -x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)

    def apply_to_attention_with_v(self, q, k, v):
        """Apply learnable 2D axial rotation to Q, K, V (per-head)."""
        B, H, N, D = q.shape
        half = D // 2
        self._cos_sin = self._get_cos_sin()
        cos_y, sin_y, cos_x, sin_x = self._cos_sin

        q_y, q_x = q[..., :half], q[..., half:]
        k_y, k_x = k[..., :half], k[..., half:]
        v_y, v_x = v[..., :half], v[..., half:]

        has_cls = N > self.grid_h * self.grid_w
        self._has_cls = has_cls

        if has_cls:
            q_y_cls, q_y_p = q_y[:, :, :1], q_y[:, :, 1:]
            k_y_cls, k_y_p = k_y[:, :, :1], k_y[:, :, 1:]
            v_y_cls, v_y_p = v_y[:, :, :1], v_y[:, :, 1:]
            q_x_cls, q_x_p = q_x[:, :, :1], q_x[:, :, 1:]
            k_x_cls, k_x_p = k_x[:, :, :1], k_x[:, :, 1:]
            v_x_cls, v_x_p = v_x[:, :, :1], v_x[:, :, 1:]

            q = torch.cat([
                torch.cat([q_y_cls, self._rotate(q_y_p, cos_y, sin_y)], dim=2),
                torch.cat([q_x_cls, self._rotate(q_x_p, cos_x, sin_x)], dim=2)
            ], dim=-1)
            k = torch.cat([
                torch.cat([k_y_cls, self._rotate(k_y_p, cos_y, sin_y)], dim=2),
                torch.cat([k_x_cls, self._rotate(k_x_p, cos_x, sin_x)], dim=2)
            ], dim=-1)
            v = torch.cat([
                torch.cat([v_y_cls, self._rotate(v_y_p, cos_y, sin_y)], dim=2),
                torch.cat([v_x_cls, self._rotate(v_x_p, cos_x, sin_x)], dim=2)
            ], dim=-1)
        else:
            q = torch.cat([self._rotate(q_y, cos_y, sin_y),
                           self._rotate(q_x, cos_x, sin_x)], dim=-1)
            k = torch.cat([self._rotate(k_y, cos_y, sin_y),
                           self._rotate(k_x, cos_x, sin_x)], dim=-1)
            v = torch.cat([self._rotate(v_y, cos_y, sin_y),
                           self._rotate(v_x, cos_x, sin_x)], dim=-1)

        return q, k, v

    def inverse_rotate_output_perhead(self, out):
        """Inverse rotate attention output per-head."""
        B, H, N, D = out.shape
        half = D // 2
        cos_y, sin_y, cos_x, sin_x = self._cos_sin
        out_y, out_x = out[..., :half], out[..., half:]

        if self._has_cls:
            out_y_cls, out_y_p = out_y[:, :, :1], out_y[:, :, 1:]
            out_x_cls, out_x_p = out_x[:, :, :1], out_x[:, :, 1:]
            out = torch.cat([
                torch.cat([out_y_cls, self._inverse_rotate(out_y_p, cos_y, sin_y)], dim=2),
                torch.cat([out_x_cls, self._inverse_rotate(out_x_p, cos_x, sin_x)], dim=2)
            ], dim=-1)
        else:
            out = torch.cat([self._inverse_rotate(out_y, cos_y, sin_y),
                             self._inverse_rotate(out_x, cos_x, sin_x)], dim=-1)
        return out


class RoPE2Dv2(nn.Module):
    """Fixed-frequency 2D rotary position encoding.

    Same structure as MonoidalPE but with fixed (non-learnable) frequencies.
    D/2 frequencies per axis, applied to all D/2 rotation pairs.
    Each pair gets angle = pos_y * freq_y[d] + pos_x * freq_x[d].
    """

    def __init__(self, embed_dim, grid_h, grid_w, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_pairs = embed_dim // 2

        # Fixed frequencies: standard RoPE formula, D/2 per axis
        # Zero out upper half for x, lower half for y (matches split-dimension approach)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        half = len(freqs) // 2
        freqs_y = freqs.clone()
        freqs_y[half:] = 0  # lower half of pairs: y only
        freqs_x = freqs.clone()
        freqs_x[:half] = 0  # upper half of pairs: x only
        self.register_buffer('freqs_y', freqs_y)
        self.register_buffer('freqs_x', freqs_x)

        ii, jj = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        self.register_buffer('pos_y', ii.reshape(-1).float())
        self.register_buffer('pos_x', jj.reshape(-1).float())

    def _rotate(self, x, angles):
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out0 = x0 * cos_a - x1 * sin_a
        out1 = x0 * sin_a + x1 * cos_a
        out = torch.empty_like(x)
        out[..., 0::2] = out0
        out[..., 1::2] = out1
        return out

    def _get_angles(self):
        return self.pos_y[:, None] * self.freqs_y[None, :] + \
               self.pos_x[:, None] * self.freqs_x[None, :]

    def forward(self, x):
        return x

    def apply_to_qkv(self, q, k, v):
        """Apply fixed rotation to Q and K. V unchanged."""
        angles = self._get_angles()
        N = q.shape[1]
        has_cls = N > self.grid_h * self.grid_w

        if has_cls:
            q = torch.cat([q[:, :1], self._rotate(q[:, 1:], angles)], dim=1)
            k = torch.cat([k[:, :1], self._rotate(k[:, 1:], angles)], dim=1)
        else:
            q = self._rotate(q, angles)
            k = self._rotate(k, angles)

        return q, k, v


class MonoidalPE(nn.Module):
    """Learnable per-axis rotation angles (our framework).

    Same rotation mechanism as RoPE but with learnable frequencies
    instead of fixed 10000^(-2d/D). D/2 frequencies per axis, applied
    to the full embedding dimension. Initialized to DFT-like frequencies.
    """

    def __init__(self, embed_dim, grid_h, grid_w, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_pairs = embed_dim // 2  # D/2 rotation pairs

        # LEARNABLE per-axis frequencies: D/2 per axis, random initialization
        self.freqs_y = nn.Parameter(torch.randn(self.n_pairs) * 0.02)
        self.freqs_x = nn.Parameter(torch.randn(self.n_pairs) * 0.02)

        ii, jj = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        self.register_buffer('pos_y', ii.reshape(-1).float())
        self.register_buffer('pos_x', jj.reshape(-1).float())

    def _rotate(self, x, angles):
        """Apply block-diagonal 2x2 rotation to x using angles.
        x: (..., D), angles: (N, D/2) -> rotated (..., D)"""
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out0 = x0 * cos_a - x1 * sin_a
        out1 = x0 * sin_a + x1 * cos_a
        out = torch.empty_like(x)
        out[..., 0::2] = out0
        out[..., 1::2] = out1
        return out

    def _get_angles(self):
        """Compute position-dependent angles: theta_y * i + theta_x * j for each position."""
        # (H*W, D/2)
        return self.pos_y[:, None] * self.freqs_y[None, :] + \
               self.pos_x[:, None] * self.freqs_x[None, :]

    def forward(self, x):
        return x

    def apply_to_qkv(self, q, k, v):
        """Apply rotation to Q and K. V unchanged."""
        # q, k, v: (B, N, D) — full embedding, before head split
        angles = self._get_angles()  # (H*W, D/2)
        N = q.shape[1]
        has_cls = N > self.grid_h * self.grid_w

        if has_cls:
            q_cls, q_p = q[:, :1], q[:, 1:]
            k_cls, k_p = k[:, :1], k[:, 1:]
            q = torch.cat([q_cls, self._rotate(q_p, angles)], dim=1)
            k = torch.cat([k_cls, self._rotate(k_p, angles)], dim=1)
        else:
            q = self._rotate(q, angles)
            k = self._rotate(k, angles)

        return q, k, v


class JoFormerPE(MonoidalPE):
    """MonoidalPE + V rotation + inverse rotation on output.
    Learnable frequencies, applied to Q, K, AND V."""

    def apply_to_qkv(self, q, k, v):
        """Apply same rotation to Q, K, and V."""
        self._cached_angles = self._get_angles()
        N = q.shape[1]
        self._has_cls = N > self.grid_h * self.grid_w

        if self._has_cls:
            q = torch.cat([q[:, :1], self._rotate(q[:, 1:], self._cached_angles)], dim=1)
            k = torch.cat([k[:, :1], self._rotate(k[:, 1:], self._cached_angles)], dim=1)
            v = torch.cat([v[:, :1], self._rotate(v[:, 1:], self._cached_angles)], dim=1)
        else:
            q = self._rotate(q, self._cached_angles)
            k = self._rotate(k, self._cached_angles)
            v = self._rotate(v, self._cached_angles)

        return q, k, v

    def inverse_rotate_output(self, out):
        """Apply inverse rotation to attention output."""
        if self._has_cls:
            return torch.cat([out[:, :1],
                              self._rotate(out[:, 1:], -self._cached_angles)], dim=1)
        else:
            return self._rotate(out, -self._cached_angles)


class JoFormerFixedPE(RoPE2Dv2):
    """RoPE2Dv2 + V rotation + inverse rotation on output.
    Fixed frequencies, applied to Q, K, AND V."""

    def apply_to_qkv(self, q, k, v):
        """Apply fixed rotation to Q, K, and V."""
        self._cached_angles = self._get_angles()
        N = q.shape[1]
        self._has_cls = N > self.grid_h * self.grid_w

        if self._has_cls:
            q = torch.cat([q[:, :1], self._rotate(q[:, 1:], self._cached_angles)], dim=1)
            k = torch.cat([k[:, :1], self._rotate(k[:, 1:], self._cached_angles)], dim=1)
            v = torch.cat([v[:, :1], self._rotate(v[:, 1:], self._cached_angles)], dim=1)
        else:
            q = self._rotate(q, self._cached_angles)
            k = self._rotate(k, self._cached_angles)
            v = self._rotate(v, self._cached_angles)

        return q, k, v

    def inverse_rotate_output(self, out):
        """Apply inverse rotation to attention output."""
        if self._has_cls:
            return torch.cat([out[:, :1],
                              self._rotate(out[:, 1:], -self._cached_angles)], dim=1)
        else:
            return self._rotate(out, -self._cached_angles)


# =====================================================================
# Vision Transformer
# =====================================================================

class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, pe_module):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.pe = pe_module

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, D)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]  # each (B, N, D)

        if hasattr(self.pe, 'apply_to_qkv'):
            # Monoidal / JoFormer: apply rotation on full D before head split
            q, k, v = self.pe.apply_to_qkv(q, k, v)
        elif hasattr(self.pe, 'apply_to_attention_with_v'):
            # JoFormerOld: per-head rotation of Q, K, V + inverse on output
            q = q.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            q, k, v = self.pe.apply_to_attention_with_v(q, k, v)
            attn = (k @ q.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = attn @ v
            x = self.pe.inverse_rotate_output_perhead(x)
            x = x.transpose(1, 2).reshape(B, N, D)
            return self.proj(x)
        elif hasattr(self.pe, 'apply_to_attention'):
            # RoPE: apply per-head rotation after head split
            q = q.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            q, k = self.pe.apply_to_attention(q, k)
            v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            attn = (k @ q.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, D)
            return self.proj(x)

        # Split into heads after rotation
        q = q.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (k @ q.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Inverse rotation on output (JoFormer)
        if hasattr(self.pe, 'inverse_rotate_output'):
            x = self.pe.inverse_rotate_output(x)

        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, pe_module, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, n_heads, pe_module)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=256, n_layers=6, n_heads=8,
                 pe_type="learned", n_classes=10):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        n_patches = self.grid_h * self.grid_w

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                      kernel_size=patch_size, stride=patch_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding
        self.pe_type = pe_type
        if pe_type == "learned":
            self.pe = LearnedPE(n_patches, embed_dim)
        elif pe_type == "rope2d":
            self.pe = RoPE2D(embed_dim, self.grid_h, self.grid_w, n_heads)
        elif pe_type == "joformer_old":
            self.pe = JoFormerOldPE(embed_dim, self.grid_h, self.grid_w, n_heads)
        elif pe_type == "monoidal_axial":
            self.pe = MonoidalAxialPE(embed_dim, self.grid_h, self.grid_w, n_heads)
        elif pe_type == "joformer_axial":
            self.pe = JoFormerAxialPE(embed_dim, self.grid_h, self.grid_w, n_heads)
        elif pe_type == "rope2dv2":
            self.pe = RoPE2Dv2(embed_dim, self.grid_h, self.grid_w, n_heads)
        elif pe_type == "monoidal":
            self.pe = MonoidalPE(embed_dim, self.grid_h, self.grid_w, n_heads)
        elif pe_type == "joformer":
            self.pe = JoFormerPE(embed_dim, self.grid_h, self.grid_w, n_heads)
        elif pe_type == "joformer_fixed":
            self.pe = JoFormerFixedPE(embed_dim, self.grid_h, self.grid_w, n_heads)
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

        # Transformer blocks (all share the same PE module for attention)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, self.pe)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embed: (B, C, H, W) -> (B, D, grid_h, grid_w) -> (B, N, D)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Apply additive PE (learned) or no-op (rope/monoidal apply in attention)
        x = self.pe(x)

        # Transformer
        for block in self.blocks:
            x = block(x)

        # Classify from CLS token
        x = self.norm(x[:, 0])
        return self.head(x)


# =====================================================================
# Training
# =====================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total += x.size(0)
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--pe_type", choices=["learned", "rope2d", "joformer_old", "monoidal_axial", "joformer_axial", "rope2dv2", "monoidal", "joformer", "joformer_fixed"],
                        default="monoidal")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset_class = datasets.CIFAR100 if args.dataset == "cifar100" else datasets.CIFAR10
    n_classes = 100 if args.dataset == "cifar100" else 10
    train_data = dataset_class('./data', train=True, download=True, transform=transform_train)
    test_data = dataset_class('./data', train=False, download=True, transform=transform_test)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=g)
    test_loader = DataLoader(test_data, batch_size=256)

    model = ViT(
        img_size=32, patch_size=args.patch_size, in_channels=3,
        embed_dim=args.embed_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, pe_type=args.pe_type, n_classes=n_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    pe_params = sum(p.numel() for p in model.pe.parameters())
    print(f"ViT with {args.pe_type} PE")
    print(f"  embed_dim={args.embed_dim}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"  patch_size={args.patch_size}, grid={32//args.patch_size}x{32//args.patch_size}")
    print(f"  Total params: {n_params:,}")
    print(f"  PE params: {pe_params:,}")
    print(f"  Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
                  f"test_acc={test_acc:.2f}%, best={best_acc:.2f}%")

    print(f"\nFinal: {args.pe_type} PE, Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

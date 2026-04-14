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
Core transformer building blocks: RoFormer attention, blocks, and feed-forward.

Originally from joformer/train_wiki.py, extracted here for standalone use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared modules
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


def build_rotation_matrix(cos_a, sin_a):
    """Build 2x2 rotation matrices from cos/sin tensors.

    cos_a, sin_a: (..., C//2) shaped tensors
    Returns: (..., C//2, 2, 2) rotation matrices

    NOTE: Only used by RoFormerAttention. JoFormer uses the fast
    element-wise apply_rotation_fast / apply_inverse_rotation_fast instead.
    """
    cos_a = cos_a.unsqueeze(-1)  # (..., C//2, 1)
    sin_a = sin_a.unsqueeze(-1)
    top = torch.cat((cos_a, sin_a), dim=-1)        # (..., C//2, 2)
    bot = torch.cat((-sin_a, cos_a), dim=-1)
    top = top.unsqueeze(-1)                          # (..., C//2, 2, 1)
    bot = bot.unsqueeze(-1)
    return torch.cat((top, bot), dim=-1)             # (..., C//2, 2, 2)


def apply_rotation(x, matrix):
    """Apply rotation matrices to x. x: (B,T,C), matrix: (1 or B, T, C//2, 2, 2)."""
    B, T, C = x.shape
    x = x.reshape(B, T, C // 2, 2, 1)
    x = torch.matmul(matrix, x)
    return x.reshape(B, T, C)


def apply_inverse_rotation(x, matrix):
    """Apply transpose (inverse) rotation."""
    B, T, C = x.shape
    x = x.reshape(B, T, C // 2, 2, 1)
    x = torch.matmul(matrix.transpose(-1, -2), x)
    return x.reshape(B, T, C)


def apply_rotation_fast(x, cos_a, sin_a):
    """Apply rotation using element-wise ops (fast). Same math as apply_rotation.

    x: (..., C), cos_a/sin_a: (..., C//2)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos_a - x_odd * sin_a
    out_odd = x_even * sin_a + x_odd * cos_a
    return torch.stack([out_even, out_odd], dim=-1).reshape(x.shape)


def apply_inverse_rotation_fast(x, cos_a, sin_a):
    """Apply inverse rotation using element-wise ops (fast). Same math as apply_inverse_rotation.

    x: (..., C), cos_a/sin_a: (..., C//2)
    Inverse rotation: transpose of rotation matrix = use -sin.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos_a + x_odd * sin_a
    out_odd = -x_even * sin_a + x_odd * cos_a
    return torch.stack([out_even, out_odd], dim=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# RoFormer — fixed RoPE, rotates K & Q only
# ---------------------------------------------------------------------------

class RoFormerAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        assert n_embed % n_head == 0, f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        D = self.head_dim

        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        if H > 1:
            # Reshape to (B, T, H, D) -> (B*H, T, D) for per-head RoPE and attention
            k = k.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
            q = q.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
            v = v.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)

        # Fixed RoPE angles: outer(pos, dim), flipped along T
        angle1 = torch.arange(T, device=x.device)
        angle2 = torch.arange(D // 2, device=x.device)
        angle = torch.outer(angle1, angle2).unsqueeze(0)  # (1, T, D//2)
        angle = torch.flip(angle, dims=(1,))
        matrix = build_rotation_matrix(torch.cos(angle), torch.sin(angle))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)
        # V not rotated

        if False and self.use_softmax and H > 1 and T > 64:
            # Flash attention: memory-efficient, avoids materializing T×T matrix
            k = k.view(B, H, T, D)
            q = q.view(B, H, T, D)
            v = v.view(B, H, T, D).contiguous()
            drop_p = self.dropout.p if self.training else 0.0
            # Original code does k @ q^T, so pass k as query and q as key
            out = F.scaled_dot_product_attention(k, q, v, is_causal=True, dropout_p=drop_p)
            out = out.transpose(1, 2).reshape(B, T, C)
        else:
            wei = k @ q.transpose(-1, -2) * D ** (-0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            if self.use_softmax:
                wei = F.softmax(wei, dim=-1)
            else:
                wei = torch.log(torch.exp(wei) + 1)
                wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
            wei = self.dropout(wei)
            out = wei @ v

            if H > 1:
                out = out.view(B, H, T, D).transpose(1, 2).reshape(B, T, C)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class RoFormerBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        self.sa_head = RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=n_head)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# RoFormer — standard transformer with separate weights per layer
# ---------------------------------------------------------------------------

class RoFormer(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False, n_head=1, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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


# ---------------------------------------------------------------------------
# JoFormer-Fixed — fixed RoPE, rotates K, Q, V; inverse on output
# ---------------------------------------------------------------------------

class JoFormerFixedAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        assert n_embed % n_head == 0, f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        D = self.head_dim

        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        if H > 1:
            k = k.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
            q = q.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
            v = v.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)

        angle1 = torch.arange(T, device=x.device)
        angle2 = torch.arange(D // 2, device=x.device)
        angle = torch.outer(angle1, angle2).unsqueeze(0)
        angle = torch.flip(angle, dims=(1,))
        matrix = build_rotation_matrix(torch.cos(angle), torch.sin(angle))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)
        v = apply_rotation(v, matrix)

        wei = k @ q.transpose(-1, -2) * D ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v

        out = apply_inverse_rotation(out, matrix)

        if H > 1:
            out = out.view(B, H, T, D).transpose(1, 2).reshape(B, T, C)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class JoFormerFixedBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        self.sa_head = JoFormerFixedAttention(n_embed, block_size, dropout, use_softmax, n_head=n_head)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class JoFormerFixed(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False, n_head=1, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerFixedBlock(n_embed, block_size, dropout, use_softmax, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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


# ---------------------------------------------------------------------------
# JoFormer-Learned — per-token learned angles (cumsum), rotates K, Q, V
# ---------------------------------------------------------------------------

class JoFormerLearnedAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        assert n_embed % n_head == 0, f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, angles):
        """x: (B,T,C), angles: (B,T,C//2) — already cumsum'd."""
        B, T, C = x.shape
        H = self.n_head
        D = self.head_dim

        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        if H > 1:
            k = k.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
            q = q.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
            v = v.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
            # Expand angles for each head: (B, T, C//2) -> per-head (B*H, T, D//2)
            angles = angles.view(B, T, H, D // 2).transpose(1, 2).reshape(B * H, T, D // 2)

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        k = apply_rotation_fast(k, cos_a, sin_a)
        q = apply_rotation_fast(q, cos_a, sin_a)
        v = apply_rotation_fast(v, cos_a, sin_a)

        wei = k @ q.transpose(-1, -2) * D ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v

        out = apply_inverse_rotation_fast(out, cos_a, sin_a)

        if H > 1:
            out = out.view(B, H, T, D).transpose(1, 2).reshape(B, T, C)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class JoFormerLearnedBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        self.sa_head = JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax, n_head=n_head)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles):
        x = x + self.sa_head(self.ln1(x), angles)
        x = x + self.ffn(self.ln2(x))
        return x


class JoFormerLearned(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False, n_head=1, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerLearnedBlock(n_embed, block_size, dropout, use_softmax, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.expander(self.token_embedding_table(idx))
        raw_angles = self.angle_embedding_table(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))

        for block in self.blocks:
            x = block(x, angles)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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


# ---------------------------------------------------------------------------
# JoFormer-Projected — angles projected from residual stream per block
# ---------------------------------------------------------------------------

class JoFormerProjectedBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        self.sa_head = JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax, n_head=n_head)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.vector_proj = nn.Linear(n_embed, n_embed)
        self.angle_proj = nn.Sequential(
            nn.Linear(n_embed, 2 * n_embed),
            nn.GELU(),
            nn.Linear(2 * n_embed, n_embed // 2),
        )

    def forward(self, x):
        x_proj = self.vector_proj(x)
        raw_angles = self.angle_proj(x)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        x_proj = x_proj + self.sa_head(self.ln1(x_proj), angles)
        x_proj = x_proj + self.ffn(self.ln2(x_proj))
        return x_proj


class JoFormerProjected(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False, n_head=1, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerProjectedBlock(n_embed, block_size, dropout, use_softmax, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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

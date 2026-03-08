"""
Windowed (sliding window) causal attention variants of RoFormer/JoFormer blocks.

Identical to the originals in joformer/train_wiki.py except the causal mask
is restricted to the last `window_size` positions. This prevents train/test
length mismatch when sequential inference generates sequences longer than
the training block_size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'joformer'))
from train_wiki import (
    FeedForward,
    build_rotation_matrix, apply_rotation, apply_inverse_rotation,
)


def _build_windowed_causal_mask(block_size, window_size):
    """Causal mask attending to at most the last `window_size` positions."""
    mask = torch.tril(torch.ones(block_size, block_size))
    if window_size < block_size:
        for i in range(window_size, block_size):
            mask[i, :i - window_size + 1] = 0
    return mask


# ---------------------------------------------------------------------------
# RoFormer (fixed RoPE, no value rotation)
# ---------------------------------------------------------------------------

class WindowedRoFormerAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, window_size=64):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        self.register_buffer('mask', _build_windowed_causal_mask(block_size, window_size))

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        angle1 = torch.arange(T, device=x.device)
        angle2 = torch.arange(C // 2, device=x.device)
        angle = torch.outer(angle1, angle2).unsqueeze(0)
        angle = torch.flip(angle, dims=(1,))
        matrix = build_rotation_matrix(torch.cos(angle), torch.sin(angle))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v

        out = self.proj(out)
        out = self.dropout(out)
        return out


class WindowedRoFormerBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, window_size=64):
        super().__init__()
        self.sa_head = WindowedRoFormerAttention(n_embed, block_size, dropout, use_softmax, window_size)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# JoFormer Learned/Projected (input-dependent angles, rotates K/Q/V)
# ---------------------------------------------------------------------------

class WindowedJoFormerLearnedAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, window_size=64):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        self.register_buffer('mask', _build_windowed_causal_mask(block_size, window_size))

    def forward(self, x, angles):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        matrix = build_rotation_matrix(torch.cos(angles), torch.sin(angles))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)
        v = apply_rotation(v, matrix)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v

        out = apply_inverse_rotation(out, matrix)

        out = self.proj(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Projected block with causal angle shift + windowed attention
# ---------------------------------------------------------------------------

class WindowedJoFormerProjectedBlockCausal(nn.Module):
    """JoFormerProjectedBlockCausal with sliding window attention."""
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, window_size=64):
        super().__init__()
        self.sa_head = WindowedJoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax, window_size)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.vector_proj = nn.Linear(n_embed, n_embed)
        self.angle_proj = nn.Sequential(
            nn.Linear(n_embed, 2 * n_embed),
            nn.GELU(),
            nn.Linear(2 * n_embed, n_embed // 2),
        )

    def forward(self, x, return_raw_angles=False):
        x_proj = self.vector_proj(x)
        raw_angles = self.angle_proj(x)
        zero = torch.zeros_like(raw_angles[:, :1, :])
        shifted_angles = torch.cat([zero, raw_angles[:, :-1, :]], dim=1)
        angles = torch.flip(shifted_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        x_proj = x_proj + self.sa_head(self.ln1(x_proj), angles)
        x_proj = x_proj + self.ffn(self.ln2(x_proj))
        if return_raw_angles:
            return x_proj, raw_angles
        return x_proj


# ---------------------------------------------------------------------------
# Standalone windowed RoFormer (separate weights per layer, for baseline)
# ---------------------------------------------------------------------------

class WindowedRoFormer(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, window_size=64, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            WindowedRoFormerBlock(n_embed, block_size, dropout, use_softmax, window_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
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
        if return_raw_angles:
            return x_proj, raw_angles
        return x_proj

"""Thin override: imports everything from the real blocks.py, then overrides
RoFormerAttention and RoFormerBlock with flash attention versions."""

import sys
import os

# Temporarily remove this directory from sys.path so we can import the real blocks
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.remove(_this_dir)

# Import everything from the real blocks.py
from blocks import *  # noqa: F401,F403
from blocks import build_rotation_matrix, apply_rotation, FeedForward

# Restore this directory to sys.path
sys.path.insert(0, _this_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoFormerAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False, n_head=1):
        super().__init__()
        assert use_softmax, "Flash Attention requires use_softmax=True"
        assert n_embed % n_head == 0
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
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

        k = k.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)
        q = q.view(B, T, H, D).transpose(1, 2).reshape(B * H, T, D)

        angle1 = torch.arange(T, device=x.device)
        angle2 = torch.arange(D // 2, device=x.device)
        angle = torch.outer(angle1, angle2).unsqueeze(0)
        angle = torch.flip(angle, dims=(1,))
        matrix = build_rotation_matrix(torch.cos(angle), torch.sin(angle))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)

        k = k.view(B, H, T, D)
        q = q.view(B, H, T, D)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Original code does k @ q^T, so pass k as query and q as key
        drop_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(k, q, v, is_causal=True, dropout_p=drop_p)

        out = out.transpose(1, 2).reshape(B, T, C)
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

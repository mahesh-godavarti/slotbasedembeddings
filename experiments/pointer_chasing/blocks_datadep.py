"""
Data-dependent positional encoding blocks for pointer chasing experiments.

Variants (all from pos_agnostic/models.py):
  - datadep:    Data-dependent angles, no cumsum, Q/K only
  - monoidal:   Data-dependent angles + cumsum, Q/K only
  - joformer:   Data-dependent angles + cumsum, Q/K/V + inverse on output
  - datadep3:   Like datadep but MLP for angles
  - monoidal3:  Like monoidal but MLP for angles
  - joformer3:  Like joformer but MLP for angles
  - datadep2:   Angles flow through layers (v2), no cumsum
  - monoidal2:  Angles flow through layers + cumsum
  - joformer2:  Angles flow through layers + cumsum + rotate V

Each can be used as a drop-in replacement for RoFormerBlock in pointer_chasing.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary embedding utilities
# ---------------------------------------------------------------------------

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings. x: (..., T, d), cos/sin: (..., T, d//2)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_inverse_rotary_emb(x, cos, sin):
    """Apply inverse (transpose) rotary embeddings."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Causal sliding-window mask
# ---------------------------------------------------------------------------

def build_attn_mask(T, window_size, device):
    """Build causal sliding-window mask. True = masked (don't attend)."""
    row = torch.arange(T, device=device)
    col = torch.arange(T, device=device)
    dist = row.unsqueeze(1) - col.unsqueeze(0)
    mask = (dist < 0) | (dist >= window_size)
    return mask


# ---------------------------------------------------------------------------
# DataDepAttention (v1): angles computed within attention layer
# ---------------------------------------------------------------------------

class DataDepAttention(nn.Module):
    """Multi-head causal attention with data-dependent rotation angles.

    angles = angle_proj(x) per position.
    - use_cumsum=False: purely content-dependent angles
    - use_cumsum=True: angles accumulated via flip-cumsum-flip
    - rotate_v=False: rotate Q and K only (monoidal)
    - rotate_v=True: rotate Q, K, V and inverse-rotate output (joformer)
    - mlp_angles=True: angle_proj is MLP instead of linear
    """

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_cumsum=False, rotate_v=False, mlp_angles=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_cumsum = use_cumsum
        self.rotate_v = rotate_v
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

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
        q, k, v = qkv.unbind(0)

        angles = self.angle_proj(x)
        if self.use_cumsum:
            angles = torch.flip(angles, dims=(1,))
            angles = torch.cumsum(angles, dim=1)
            angles = torch.flip(angles, dims=(1,))
        angles = angles.view(B, T, h, d // 2).transpose(1, 2)
        cos, sin = torch.cos(angles), torch.sin(angles)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.rotate_v:
            v = apply_rotary_emb(v, cos, sin)

        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        mask = build_attn_mask(T, self.window_size, x.device)
        scores.masked_fill_(mask, float('-inf'))
        attn = self.attn_drop(F.softmax(scores, dim=-1))

        out = attn @ v
        if self.rotate_v:
            out = apply_inverse_rotary_emb(out, cos, sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# DataDep v1 Block
# ---------------------------------------------------------------------------

class DataDepBlock(nn.Module):
    """Transformer block with DataDepAttention."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_cumsum=False, rotate_v=False, mlp_angles=False):
        super().__init__()
        self.attn = DataDepAttention(n_embed, n_heads, dropout, window_size,
                                     use_cumsum, rotate_v, mlp_angles)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# DataDep2Attention (v2): angles flow through the network
# ---------------------------------------------------------------------------

class DataDep2Attention(nn.Module):
    """Multi-head causal attention with externally-provided data-dependent angles."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_cumsum=False, rotate_v=False):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.window_size = window_size
        self.use_cumsum = use_cumsum
        self.rotate_v = rotate_v
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, angles):
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
            v = apply_rotary_emb(v, cos, sin)

        scores = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        mask = build_attn_mask(T, self.window_size, x.device)
        scores.masked_fill_(mask, float('-inf'))
        attn = self.attn_drop(F.softmax(scores, dim=-1))

        out = attn @ v
        if self.rotate_v:
            out = apply_inverse_rotary_emb(out, cos, sin)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# FFN with angle output (for v2)
# ---------------------------------------------------------------------------

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


class FeedForwardWithAngles(nn.Module):
    """FFN that outputs C content dims + C//2 angle dims."""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.n_embed = n_embed
        self.fc1 = nn.Linear(n_embed, 4 * n_embed)
        self.fc2_content = nn.Linear(4 * n_embed, n_embed)
        self.fc2_angles = nn.Linear(4 * n_embed, n_embed // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        content = self.dropout(self.fc2_content(h))
        angles = torch.tanh(self.fc2_angles(h)) * math.pi
        return content, angles


# ---------------------------------------------------------------------------
# DataDep v2 Block: angles flow through layers
# ---------------------------------------------------------------------------

class DataDep2Block(nn.Module):
    """Transformer block for datadep2: receives angles, produces new angles."""

    def __init__(self, n_embed, n_heads, dropout, window_size=256,
                 use_cumsum=False, rotate_v=False):
        super().__init__()
        self.attn = DataDep2Attention(n_embed, n_heads, dropout, window_size,
                                      use_cumsum, rotate_v)
        self.ffn = FeedForwardWithAngles(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles):
        x = x + self.attn(self.ln1(x), angles)
        content, new_angles = self.ffn(self.ln2(x))
        x = x + content
        return x, new_angles


# ---------------------------------------------------------------------------
# Full models for pointer chasing
# ---------------------------------------------------------------------------

class DataDepTransformer(nn.Module):
    """N-layer transformer with DataDep v1 attention."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_heads=4,
                 dropout=0.0, window_size=256,
                 use_cumsum=False, rotate_v=False, mlp_angles=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([
            DataDepBlock(n_embed, n_heads, dropout, window_size,
                        use_cumsum, rotate_v, mlp_angles)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


class DataDep2Transformer(nn.Module):
    """N-layer transformer with DataDep v2 attention (angles flow through layers)."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_heads=4,
                 dropout=0.0, window_size=256,
                 use_cumsum=False, rotate_v=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.initial_angle_proj = nn.Linear(n_embed, n_embed // 2)
        self.blocks = nn.ModuleList([
            DataDep2Block(n_embed, n_heads, dropout, window_size,
                         use_cumsum, rotate_v)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        x = self.token_embedding(idx)
        angles = torch.tanh(self.initial_angle_proj(x)) * math.pi
        for block in self.blocks:
            x, angles = block(x, angles)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Factory: create model by name
# ---------------------------------------------------------------------------

DATADEP_VARIANTS = {
    'datadep':   dict(use_cumsum=False, rotate_v=False, mlp_angles=False),
    'datadepv':  dict(use_cumsum=False, rotate_v=True,  mlp_angles=False),
    'datadep3':  dict(use_cumsum=False, rotate_v=False, mlp_angles=True),
    'datadep3v': dict(use_cumsum=False, rotate_v=True,  mlp_angles=True),
    'monoidal':  dict(use_cumsum=True,  rotate_v=False, mlp_angles=False),
    'monoidal3': dict(use_cumsum=True,  rotate_v=False, mlp_angles=True),
    'joformer':  dict(use_cumsum=True,  rotate_v=True,  mlp_angles=False),
    'joformer3': dict(use_cumsum=True,  rotate_v=True,  mlp_angles=True),
}

DATADEP2_VARIANTS = {
    'datadep2':  dict(use_cumsum=False, rotate_v=False),
    'monoidal2': dict(use_cumsum=True,  rotate_v=False),
    'joformer2': dict(use_cumsum=True,  rotate_v=True),
}


def make_datadep_model(variant, vocab_size, n_embed, n_layers, block_size,
                       n_heads=4, dropout=0.0, window_size=256):
    """Create a DataDep model by variant name."""
    if variant in DATADEP_VARIANTS:
        kwargs = DATADEP_VARIANTS[variant]
        return DataDepTransformer(vocab_size, n_embed, n_layers, block_size,
                                  n_heads, dropout, window_size, **kwargs)
    elif variant in DATADEP2_VARIANTS:
        kwargs = DATADEP2_VARIANTS[variant]
        return DataDep2Transformer(vocab_size, n_embed, n_layers, block_size,
                                   n_heads, dropout, window_size, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Available: {list(DATADEP_VARIANTS) + list(DATADEP2_VARIANTS)}")

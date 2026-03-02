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
#
# train_wiki.py — Compare RoFormer, JoFormer-Fixed, JoFormer-Learned on wiki text
# with BPE tokenization. Adapted from joformer_src/ character-level models.

import argparse
import json
import math
import os
import tempfile
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

def train_bpe_tokenizer(texts, vocab_size):
    """Train a BPE tokenizer on a list of text strings. Returns (tokenizer, vocab_size)."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>"],
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     encoding='utf-8') as f:
        tmp_path = f.name
        for line in texts:
            f.write(line + "\n")
    tokenizer.train([tmp_path], trainer)
    os.unlink(tmp_path)
    actual_vocab_size = tokenizer.get_vocab_size()
    return tokenizer, actual_vocab_size


def load_wiki_text(wiki_path, max_lines):
    """Load wiki text lines, stripping blank lines."""
    lines = []
    with open(wiki_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def encode_text(tokenizer, lines):
    """Encode lines into a single flat token-ID tensor."""
    ids = []
    for line in lines:
        enc = tokenizer.encode(line)
        ids.extend(enc.ids)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(train_data, val_data, split, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device,
                  eval_iters=20):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

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

# ---------------------------------------------------------------------------
# RoFormer — fixed RoPE, rotates K & Q only, log(exp+1) attention
# ---------------------------------------------------------------------------

class RoFormerAttention(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        # Fixed RoPE angles: outer(pos, dim), flipped along T
        angle1 = torch.arange(T, device=x.device)
        angle2 = torch.arange(C // 2, device=x.device)
        angle = torch.outer(angle1, angle2).unsqueeze(0)  # (1, T, C//2)
        angle = torch.flip(angle, dims=(1,))
        matrix = build_rotation_matrix(torch.cos(angle), torch.sin(angle))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)
        # V not rotated

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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


class RoFormerBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.sa_head = RoFormerAttention(n_embed, block_size, dropout, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class RoFormer(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [RoFormerBlock(n_embed, block_size, dropout, use_softmax) for _ in range(n_layers)]
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
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

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
        v = apply_rotation(v, matrix)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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


class JoFormerFixedBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.sa_head = JoFormerFixedAttention(n_embed, block_size, dropout, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class JoFormerFixed(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerFixedBlock(n_embed, block_size, dropout, use_softmax) for _ in range(n_layers)]
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
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, angles):
        """x: (B,T,C), angles: (B,T,C//2) — already cumsum'd."""
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        matrix = build_rotation_matrix(torch.cos(angles), torch.sin(angles))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)
        v = apply_rotation(v, matrix)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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


class JoFormerLearnedBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.sa_head = JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles):
        x = x + self.sa_head(self.ln1(x), angles)
        x = x + self.ffn(self.ln2(x))
        return x


class JoFormerLearned(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerLearnedBlock(n_embed, block_size, dropout, use_softmax) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.expander(self.token_embedding_table(idx))
        raw_angles = self.angle_embedding_table(idx)  # (B, T, C//2)
        # flip -> cumsum -> flip (matches original per-token angle computation)
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
# JoFormer-Projected — angles projected from previous layer output per block
# Each block: vector = Linear(C, C)(x), angle = Linear(C, 2C) -> GELU -> Linear(2C, C//2)
# Then flip -> cumsum -> flip on angles, same attention as JoFormerLearned
# ---------------------------------------------------------------------------

class JoFormerProjectedBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.sa_head = JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
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
        raw_angles = self.angle_proj(x)  # (B, T, C//2)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        x_proj = x_proj + self.sa_head(self.ln1(x_proj), angles)
        x_proj = x_proj + self.ffn(self.ln2(x_proj))
        return x_proj


class JoFormerProjected(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerProjectedBlock(n_embed, block_size, dropout, use_softmax) for _ in range(n_layers)]
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
# Model registry
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    'roformer': RoFormer,
    'joformer_fixed': JoFormerFixed,
    'joformer_learned': JoFormerLearned,
    'joformer_projected': JoFormerProjected,
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_name, model, train_data, val_data, args, device, tokenizer):
    """Train a single model and return final val loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {model_name}  ({n_params:,} parameters)")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    ppl_log = {"iter": [], "train_ppl": [], "val_ppl": []}

    pbar = tqdm(range(args.max_iters), desc=model_name)
    for it in pbar:
        # Eval
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, args.batch_size, device)
            train_ppl = math.exp(losses['train'])
            val_ppl = math.exp(losses['val'])
            ppl_log["iter"].append(it)
            ppl_log["train_ppl"].append(round(train_ppl, 2))
            ppl_log["val_ppl"].append(round(val_ppl, 2))
            pbar.set_postfix(train_loss=f"{losses['train']:.3f}",
                             val_loss=f"{losses['val']:.3f}",
                             val_ppl=f"{val_ppl:.2f}")
            tqdm.write(f"  [{model_name}] iter {it}: "
                       f"train loss {losses['train']:.4f} (PPL {train_ppl:.2f}), "
                       f"val loss {losses['val']:.4f} (PPL {val_ppl:.2f})")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

            # Checkpoint
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                path = os.path.join(args.checkpoint_dir, f"{model_name}_iter{it}.pt")
                torch.save({
                    'iter': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': losses['val'],
                }, path)

            # Generate sample
            if it > 0 and it % (args.eval_interval * 2) == 0:
                model.eval()
                prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(prompt, args.generate_len)
                text = tokenizer.decode(generated[0].cpu().tolist())
                tqdm.write(f"  [{model_name}] sample: {text[:200]}")
                model.train()

        # Train step
        xb, yb = get_batch(train_data, val_data, "train",
                           args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Final eval
    losses = estimate_loss(model, train_data, val_data,
                           args.block_size, args.batch_size, device)
    val_ppl = math.exp(losses['val'])

    # Final generation sample
    model.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(prompt, args.generate_len)
    text = tokenizer.decode(generated[0].cpu().tolist())
    print(f"\n  [{model_name}] final val loss: {losses['val']:.4f} (PPL {val_ppl:.2f})")
    print(f"  [{model_name}] sample: {text[:300]}")

    # Save final checkpoint
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        path = os.path.join(args.checkpoint_dir, f"{model_name}_final.pt")
        torch.save({
            'iter': args.max_iters,
            'model_state_dict': model.state_dict(),
            'val_loss': losses['val'],
        }, path)

    return losses['val'], val_ppl, ppl_log

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare RoFormer vs JoFormer on wiki text")
    parser.add_argument('--wiki_path', type=str, default=None,
                        help='Path to wiki.en.txt (default: ../exp8/data/wiki.en.txt relative to script)')
    parser.add_argument('--wiki_lines', type=int, default=100000,
                        help='Max lines to load from wiki text')
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='BPE vocabulary size')
    parser.add_argument('--models', nargs='+',
                        default=['roformer', 'joformer_fixed', 'joformer_learned',
                                 'joformer_projected'],
                        choices=['roformer', 'joformer_fixed', 'joformer_learned',
                                 'joformer_projected'],
                        help='Which models to train')
    parser.add_argument('--n_embed', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--block_size', type=int, default=64,
                        help='Context window size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=10000,
                        help='Training iterations')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Eval frequency (iterations)')
    parser.add_argument('--checkpoint_dir', type=str, default='joformer/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick test: 50 iters, 1000 lines, vocab 2000')
    parser.add_argument('--generate_len', type=int, default=200,
                        help='Generation sample length in tokens')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--softmax', action='store_true',
                        help='Use softmax attention instead of normalized softplus')
    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.max_iters = 50
        args.wiki_lines = 1000
        args.vocab_size = 2000
        args.eval_interval = 25
        args.n_layers = 2
        args.n_embed = 64
        args.generate_len = 50

    # Default wiki path relative to this script
    if args.wiki_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.wiki_path = os.path.join(script_dir, '..', 'exp8', 'data', 'wiki.en.txt')

    # Ensure n_embed is even (needed for rotation pairs)
    if args.n_embed % 2 != 0:
        args.n_embed += 1
        print(f"Adjusted n_embed to {args.n_embed} (must be even)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    attn_type = "softmax" if args.softmax else "normalized softplus"
    print(f"Config: n_embed={args.n_embed}, n_layers={args.n_layers}, "
          f"block_size={args.block_size}, batch_size={args.batch_size}, "
          f"lr={args.lr}, max_iters={args.max_iters}, attn={attn_type}")

    # Load wiki text
    print(f"\nLoading wiki text from {args.wiki_path} (max {args.wiki_lines} lines)...")
    lines = load_wiki_text(args.wiki_path, args.wiki_lines)
    print(f"Loaded {len(lines)} non-empty lines")

    # Train BPE tokenizer
    print(f"Training BPE tokenizer (target vocab_size={args.vocab_size})...")
    tokenizer, actual_vocab_size = train_bpe_tokenizer(lines, args.vocab_size)
    print(f"Actual vocab size: {actual_vocab_size}")

    # Encode all text
    print("Encoding text...")
    data = encode_text(tokenizer, lines)
    print(f"Total tokens: {len(data):,}")

    # Train/val split
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    if len(val_data) < args.block_size + 1:
        print("WARNING: val data too small for block_size, reducing block_size")
        args.block_size = len(val_data) - 2

    # Train each model
    results = {}
    for model_name in args.models:
        torch.manual_seed(args.seed)  # same init for fair comparison
        cls = MODEL_CLASSES[model_name]
        model = cls(actual_vocab_size, args.n_embed, args.n_layers,
                    args.block_size, args.dropout, use_softmax=args.softmax)
        val_loss, val_ppl, ppl_log = train_model(model_name, model, train_data, val_data,
                                                  args, device, tokenizer)
        results[model_name] = {'val_loss': val_loss, 'val_ppl': val_ppl,
                               'ppl_curve': ppl_log}

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Val Loss':>10} {'Val PPL':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10}")
    for name in args.models:
        r = results[name]
        print(f"{name:<25} {r['val_loss']:>10.4f} {r['val_ppl']:>10.2f}")
    print(f"{'='*60}")

    # Find best
    best_name = min(results, key=lambda k: results[k]['val_loss'])
    print(f"\nBest model: {best_name} (val PPL {results[best_name]['val_ppl']:.2f})")

    # Save results with PPL curves
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        "config": {
            "n_embed": args.n_embed, "n_layers": args.n_layers,
            "block_size": args.block_size, "batch_size": args.batch_size,
            "lr": args.lr, "max_iters": args.max_iters,
            "vocab_size": args.vocab_size, "models": args.models,
        },
        "results": results,
        "timestamp": timestamp,
    }
    results_file = os.path.join(results_dir, f"joformer_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    latest_file = os.path.join(results_dir, "joformer_results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")


if __name__ == '__main__':
    main()

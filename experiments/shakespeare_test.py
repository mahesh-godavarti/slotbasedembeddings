# -----------------------------------------------------------------------------
# Shakespeare sanity check: B, B', C, C'
# Verifies that primed (operator-based) models beat unprimed on standard text.
# Based on joformer_src/roformer.py (B/B') and journey_transformer_per_token_angles.py (C/C')
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import math
import argparse

# Hyperparams
block_size = 20
batch_size = 32
lr = 5e-4
n_embed = 90
n_layers = 1
dropout = 0.2
max_iters = 10000
eval_iters = 200
eval_interval = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Attention: RoPE (Models B/B')
# ---------------------------------------------------------------------------
class RoPEAttention(nn.Module):
    def __init__(self, n_embed, primed=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.primed = primed
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def _build_rotation(self, T, C):
        """Build RoPE rotation matrices. Shape: (1, T, C//2, 2, 2)"""
        pos = torch.arange(T, device=device)
        dim = torch.arange(C // 2, device=device)
        angle = torch.outer(pos, dim)
        angle = angle.unsqueeze(0)  # (1, T, C//2)
        angle = torch.flip(angle, dims=(1,))  # right-cumsum convention
        cos_a = torch.cos(angle).unsqueeze(3)  # (1, T, C//2, 1)
        sin_a = torch.sin(angle).unsqueeze(3)
        top = torch.cat((cos_a, sin_a), dim=3).unsqueeze(4)      # (1,T,C//2,2,1)
        bot = torch.cat((-sin_a, cos_a), dim=3).unsqueeze(4)     # (1,T,C//2,2,1)
        return torch.cat((top, bot), dim=4)  # (1, T, C//2, 2, 2)

    def _rotate(self, x, matrix):
        B, T, C = x.shape
        x = x.reshape(B, T, C // 2, 2, 1)
        x = torch.matmul(matrix, x)
        return x.reshape(B, T, C)

    def _inv_rotate(self, x, matrix):
        B, T, C = x.shape
        x = x.reshape(B, T, C // 2, 2, 1)
        x = torch.matmul(matrix.transpose(3, 4), x)
        return x.reshape(B, T, C)

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        matrix = self._build_rotation(T, C)
        k = self._rotate(k, matrix)
        q = self._rotate(q, matrix)

        if self.primed:
            v = self._rotate(v, matrix)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.log(torch.exp(wei) + 1)  # softplus
        wei = self.dropout(wei)
        out = wei @ v

        if self.primed:
            out = self._inv_rotate(out, matrix)

        out = self.proj(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Attention: Cumsum / Journey (Models C/C')
# ---------------------------------------------------------------------------
class CumsumAttention(nn.Module):
    def __init__(self, n_embed, angle_embedding, primed=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.primed = primed
        self.angle_embedding = angle_embedding
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def _build_rotation(self, idx):
        """Build rotation matrices from per-token angle cumsum."""
        angle = self.angle_embedding(idx)  # (B, T, C//2)
        angle = torch.flip(angle, dims=(1,))
        angle = torch.cumsum(angle, dim=1)
        angle = torch.flip(angle, dims=(1,))  # right-cumsum
        cos_a = torch.cos(angle).unsqueeze(3)
        sin_a = torch.sin(angle).unsqueeze(3)
        top = torch.cat((cos_a, sin_a), dim=3).unsqueeze(4)
        bot = torch.cat((-sin_a, cos_a), dim=3).unsqueeze(4)
        return torch.cat((top, bot), dim=4)

    def _rotate(self, x, matrix):
        B, T, C = x.shape
        x = x.reshape(B, T, C // 2, 2, 1)
        x = torch.matmul(matrix, x)
        return x.reshape(B, T, C)

    def _inv_rotate(self, x, matrix):
        B, T, C = x.shape
        x = x.reshape(B, T, C // 2, 2, 1)
        x = torch.matmul(matrix.transpose(3, 4), x)
        return x.reshape(B, T, C)

    def forward(self, x, idx):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        matrix = self._build_rotation(idx)
        k = self._rotate(k, matrix)
        q = self._rotate(q, matrix)

        if self.primed:
            v = self._rotate(v, matrix)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.log(torch.exp(wei) + 1)  # softplus
        wei = self.dropout(wei)
        out = wei @ v

        if self.primed:
            out = self._inv_rotate(out, matrix)

        out = self.proj(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class RoPEBlock(nn.Module):
    def __init__(self, n_embed, primed=False):
        super().__init__()
        self.sa = RoPEAttention(n_embed, primed=primed)
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CumsumBlock(nn.Module):
    def __init__(self, n_embed, angle_embedding, primed=False):
        super().__init__()
        self.sa = CumsumAttention(n_embed, angle_embedding, primed=primed)
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, idx):
        x = x + self.sa(self.ln1(x), idx)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ModelB(nn.Module):
    """RoPE, commutative (Q/K only)"""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([RoPEBlock(n_embed, primed=False) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


class ModelBPrime(nn.Module):
    """RoPE, operator-based (Q/K/V + inverse)"""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([RoPEBlock(n_embed, primed=True) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


class ModelC(nn.Module):
    """Per-token cumsum, commutative (Q/K only)"""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([CumsumBlock(n_embed, self.angle_emb, primed=False) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.expander(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, idx)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


class ModelCPrime(nn.Module):
    """Per-token cumsum, operator-based (Q/K/V + inverse)"""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([CumsumBlock(n_embed, self.angle_emb, primed=True) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.expander(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, idx)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(name, model):
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {name}: {n_params:,} params")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for it in tqdm(range(max_iters), desc=name):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            ppl_train = math.exp(losses['train'].item())
            ppl_val = math.exp(losses['val'].item())
            print(f"  Step {it}: train loss={losses['train']:.4f} (ppl={ppl_train:.2f}), "
                  f"val loss={losses['val']:.4f} (ppl={ppl_val:.2f})")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Final eval
    losses = estimate_loss(model)
    ppl_train = math.exp(losses['train'].item())
    ppl_val = math.exp(losses['val'].item())
    print(f"\n  FINAL: train ppl={ppl_train:.2f}, val ppl={ppl_val:.2f}")

    # Generate sample
    model.eval()
    with torch.no_grad():
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(200):
            ctx = start[:, -block_size:]
            logits, _ = model(ctx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            start = torch.cat((start, nxt), dim=1)
        print(f"\n  Sample: {decode(start[0].tolist())[:200]}")
    model.train()

    return ppl_train, ppl_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Config: n_embed={n_embed}, n_layers={n_layers}, block_size={block_size}, "
          f"max_iters={max_iters}, device={device}")
    print(f"Data: {len(data)} chars, vocab={vocab_size}")

    results = {}
    for name, model_cls in [("B", ModelB), ("B'", ModelBPrime),
                             ("C", ModelC), ("C'", ModelCPrime)]:
        model = model_cls()
        ppl_train, ppl_val = train_model(name, model)
        results[name] = {'train_ppl': ppl_train, 'val_ppl': ppl_val}

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<8} {'Train PPL':>12} {'Val PPL':>12}")
    print(f"{'-'*32}")
    for name in ["B", "B'", "C", "C'"]:
        r = results[name]
        print(f"{name:<8} {r['train_ppl']:>12.2f} {r['val_ppl']:>12.2f}")

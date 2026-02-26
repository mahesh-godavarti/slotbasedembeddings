# -----------------------------------------------------------------------------
# Test: Run the EXACT Shakespeare-validated model code on KG data.
#
# Purpose: verify whether B/B'/C/C' models work on the KG dataset.
# If they do, the bug is in kg_text_experiment.py's model implementations.
# If they don't, the issue is with the data/batching.
#
# Model code is copied verbatim from shakespeare_test.py.
# Data pipeline is imported from kg_text_experiment.py.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import math
import numpy as np
from collections import defaultdict

# Import KG data pipeline
from kg_text_experiment import (
    Config, cfg, prepare_data, build_eval_prompts,
    Vocabulary, TextDataset, KGDataset,
)

# Override config for this test
cfg.n_embed = 24
cfg.n_layers = 1
cfg.max_iters = 10000
cfg.eval_interval = 500

n_embed = cfg.n_embed
n_layers = cfg.n_layers
# block_size will be set dynamically after data loading (max sentence length)
block_size = None  # placeholder, set in main
batch_size = cfg.batch_size  # 32
lr = cfg.lr
dropout = cfg.dropout
device = cfg.device
USE_PAD_MASK = True  # toggled by --no-pad-mask


# ============================================================================
# Model code below is IDENTICAL to shakespeare_test.py
# (only change: vocab_size is passed as constructor arg instead of global)
# ============================================================================

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

    def forward(self, x, pad_mask=None):
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
        if pad_mask is not None:
            wei = wei.masked_fill(~pad_mask.unsqueeze(1), float('-inf'))
        wei = torch.log(torch.exp(wei) + 1)  # softplus
        wei = self.dropout(wei)
        out = wei @ v

        if self.primed:
            out = self._inv_rotate(out, matrix)

        out = self.proj(out)
        out = self.dropout(out)
        return out


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

    def forward(self, x, idx, pad_mask=None):
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
        if pad_mask is not None:
            wei = wei.masked_fill(~pad_mask.unsqueeze(1), float('-inf'))
        wei = torch.log(torch.exp(wei) + 1)  # softplus
        wei = self.dropout(wei)
        out = wei @ v

        if self.primed:
            out = self._inv_rotate(out, matrix)

        out = self.proj(out)
        out = self.dropout(out)
        return out


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

    def forward(self, x, pad_mask=None):
        x = x + self.sa(self.ln1(x), pad_mask=pad_mask)
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

    def forward(self, x, idx, pad_mask=None):
        x = x + self.sa(self.ln1(x), idx, pad_mask=pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class ModelB(nn.Module):
    """RoPE, commutative (Q/K only)"""
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([RoPEBlock(n_embed, primed=False) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        pad_mask = (idx != 0) if USE_PAD_MASK else None
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


class ModelBPrime(nn.Module):
    """RoPE, operator-based (Q/K/V + inverse)"""
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([RoPEBlock(n_embed, primed=True) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        pad_mask = (idx != 0) if USE_PAD_MASK else None
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


class ModelC(nn.Module):
    """Per-token cumsum, commutative (Q/K only)"""
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([CumsumBlock(n_embed, self.angle_emb, primed=False) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        pad_mask = (idx != 0) if USE_PAD_MASK else None
        x = self.expander(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, idx, pad_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


class ModelCPrime(nn.Module):
    """Per-token cumsum, operator-based (Q/K/V + inverse)"""
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_emb = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([CumsumBlock(n_embed, self.angle_emb, primed=True) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        pad_mask = (idx != 0) if USE_PAD_MASK else None
        x = self.expander(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, idx, pad_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


# ============================================================================
# Training (uses KG experiment's per-sentence batching)
# ============================================================================

def train_model(name, model, text_dataset):
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Training Model {name} ---")
    print(f"Model {name}: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for it in tqdm(range(cfg.max_iters), desc=f"Model {name}"):
        x, y = text_dataset.get_batch(batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % cfg.eval_interval == 0:
            print(f"  [{name}] iter {it}, loss: {loss.item():.4f}")

    return model


# ============================================================================
# Evaluation (same cloze-style as KG experiment)
# ============================================================================

def evaluate_model(model, eval_prompts, vocab, model_name="?"):
    model.eval()
    model.to(device)

    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for p in eval_prompts:
            tier = p["tier"]
            relation = p["relation"]
            prompt_tokens = p["prompt_tokens"]
            target_tokens = p["target_tokens"]

            if len(prompt_tokens) > block_size:
                prompt_tokens = prompt_tokens[-block_size:]

            x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

            logits = model.predict_text(x)
            last_logits = logits[0, -1, :]

            top1 = torch.argmax(last_logits).item()
            hit1 = 1 if top1 == target_tokens[0] else 0

            top5 = torch.topk(last_logits, k=min(5, last_logits.shape[0])).indices.tolist()
            hit5 = 1 if target_tokens[0] in top5 else 0

            log_probs = F.log_softmax(last_logits, dim=0)
            target_log_prob = log_probs[target_tokens[0]].item()
            ppl = np.exp(-target_log_prob)

            full_correct = True
            current_tokens = prompt_tokens.copy()
            for t_idx, t_tok in enumerate(target_tokens):
                x = torch.tensor([current_tokens[-block_size:]],
                                 dtype=torch.long, device=device)
                logits = model.predict_text(x)
                pred = torch.argmax(logits[0, -1, :]).item()
                if pred != t_tok:
                    full_correct = False
                    break
                current_tokens.append(pred)

            results[tier][relation].append({
                "hit1": hit1,
                "hit5": hit5,
                "ppl": ppl,
                "full_correct": 1 if full_correct else 0,
            })

    summary = {}
    for tier in results:
        tier_results = {"hit1": [], "hit5": [], "ppl": [], "full_correct": []}
        for rel in results[tier]:
            for r in results[tier][rel]:
                tier_results["hit1"].append(r["hit1"])
                tier_results["hit5"].append(r["hit5"])
                tier_results["ppl"].append(r["ppl"])
                tier_results["full_correct"].append(r["full_correct"])

        summary[tier] = {
            "hit1": np.mean(tier_results["hit1"]),
            "hit5": np.mean(tier_results["hit5"]),
            "ppl": np.mean(tier_results["ppl"]),
            "full_correct": np.mean(tier_results["full_correct"]),
            "n": len(tier_results["hit1"]),
        }

    print(f"\n  Evaluation: {model_name}")
    for tier in ["memorization", "transfer", "generalization"]:
        if tier in summary:
            s = summary[tier]
            print(f"  {tier:>15s}: hit@1={s['hit1']:.3f}  hit@5={s['hit5']:.3f}  "
                  f"ppl={s['ppl']:.2f}  full_acc={s['full_correct']:.3f}  (n={s['n']})")

    model.train()
    return summary


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pad-mask", action="store_true", help="Disable pad masking")
    args = parser.parse_args()
    USE_PAD_MASK = not args.no_pad_mask

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Generate KG data (generous linearization = exp 7a)
    vocab, text_base, text_linearized, kg_dataset, eval_prompts = prepare_data(
        generous_linearization=True)

    # Set block_size to max sentence length in the dataset
    max_sent_len = max(len(s) for s in text_linearized.encoded)
    block_size = max_sent_len
    cfg.block_size = block_size

    print(f"Config: n_embed={n_embed}, n_layers={n_layers}, block_size={block_size}, "
          f"max_iters={cfg.max_iters}, lr={lr}, device={device}")
    print(f"Vocabulary size: {vocab.size}")
    print(f"Text dataset (linearized): {len(text_linearized.data)} tokens")
    print(f"Max sentence length: {max_sent_len}")
    print(f"Eval prompts: {len(eval_prompts)}")

    all_results = {}
    for name, model_cls in [("B", ModelB), ("B'", ModelBPrime),
                             ("C", ModelC), ("C'", ModelCPrime)]:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        model = model_cls(vocab.size)
        model = train_model(name, model, text_linearized)
        summary = evaluate_model(model, eval_prompts, vocab, name)
        all_results[name] = summary
        del model

    # Print comparison
    print(f"\n{'='*80}")
    pad_label = "pad_mask=ON" if USE_PAD_MASK else "pad_mask=OFF"
    print(f"  COMPARISON (no BOS/EOS, block_size={block_size}, {pad_label})")
    print(f"{'='*80}")
    for tier in ["memorization", "transfer", "generalization"]:
        print(f"\n  {tier.upper()}")
        print(f"  {'Model':<8} {'hit@1':>8} {'hit@5':>8} {'ppl':>10} {'full_acc':>10}")
        print(f"  {'-'*46}")
        for name in ["B", "B'", "C", "C'"]:
            if tier in all_results[name]:
                s = all_results[name][tier]
                print(f"  {name:<8} {s['hit1']:>8.3f} {s['hit5']:>8.3f} "
                      f"{s['ppl']:>10.2f} {s['full_correct']:>10.3f}")

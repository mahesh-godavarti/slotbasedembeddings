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
Split-block look-ahead models: separate attention and FFN pathways.

Three variants that address the corrhead variable-scaling problem by ensuring
the head always sees a signal with stable token identity (coefficient 1 from
the attention residual).

Variant 1 (attn_corr_ffn): Attention-only iteration, FFN generates corrections
    y[t] = processed_x[t] + attn(ln1(processed_x))[t]
    correction[t] = corr_ffn(ln2(y))[t]
    processed_x[t] = tok_emb[t] + shift(correction)[t]
    head_input[t] = y[t]

Variant 2 (attn_head_ffn): Attention-only iteration, FFN at head
    y[t] = processed_x[t] + attn(ln1(processed_x))[t]
    correction[t] = y[t] - processed_x[t]
    processed_x[t] = tok_emb[t] + shift(correction)[t]
    head_input[t] = head_ffn(ln2(y[t]))

Variant 3 (block_head_ffn): Standard block iteration, extra FFN at head
    y[t] = processed_x[t] + attn(ln1(processed_x))[t]
    z[t] = y[t] + ffn(ln2(y))[t]
    correction[t] = z[t] - processed_x[t]
    processed_x[t] = tok_emb[t] + shift(correction)[t]
    head_input[t] = head_ffn(ln3(z[t]))
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import joformer building blocks
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'joformer'))
from train_wiki import (
    RoFormerBlock, RoFormerAttention, FeedForward,
    RoFormer, JoFormerFixed, JoFormerLearned, JoFormerProjected,
)

# Import look_ahead3 models for backwards compatibility
_look_ahead3_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'look_ahead3')
sys.path.insert(0, _look_ahead3_dir)
import importlib.util
_look_ahead3_spec = importlib.util.spec_from_file_location(
    "look_ahead3_models",
    os.path.join(_look_ahead3_dir, 'models.py')
)
_look_ahead3_mod = importlib.util.module_from_spec(_look_ahead3_spec)
_look_ahead3_spec.loader.exec_module(_look_ahead3_mod)
LOOK_AHEAD3_MODELS = _look_ahead3_mod.MODEL_CLASSES


# ---------------------------------------------------------------------------
# Base class for split-block look-ahead models
# ---------------------------------------------------------------------------

class SplitBlockLookAhead(nn.Module):
    """Base class for split-block look-ahead variants.

    Subclasses implement _iteration_step() to define what happens each iteration
    and _get_head_input() to define what the head sees.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_iters = n_layers
        self.block_size = block_size
        self.convergence_weight = convergence_weight

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

    def _iteration_step(self, processed_x):
        """Run one iteration. Returns (y, correction).

        y: the block output (what the head may see)
        correction: the delta to shift and add to tok_emb
        """
        raise NotImplementedError

    def _get_head_input(self, y):
        """Transform block output into head input."""
        raise NotImplementedError

    def _run_iterations(self, tok_emb, n_iters):
        """Run n_iters shared-weight iterations.

        Returns: (processed_x, y, aux_loss)
            processed_x: contextualized embeddings
            y: block output from last iteration (for head)
            aux_loss: convergence loss
        """
        B, T, C = tok_emb.shape
        processed_x = tok_emb
        prev_processed_x = None
        y = None
        total_conv_loss = 0.0

        for k in range(n_iters):
            prev_processed_x = processed_x

            y, correction = self._iteration_step(processed_x)

            # Past-only shift: position t gets correction from t-1
            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted = torch.cat([zero, correction[:, :-1, :]], dim=1)

            # Non-cumulative: reset to tok_emb
            processed_x = tok_emb + shifted

            # Convergence loss on last iteration
            if self.convergence_weight > 0 and self.training and k == n_iters - 1:
                total_conv_loss = F.mse_loss(
                    processed_x, prev_processed_x.detach()
                )

        return processed_x, y, total_conv_loss

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        processed_x, y, aux_loss = self._run_iterations(tok_emb, self.n_iters)
        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            if aux_loss > 0:
                loss = loss + self.convergence_weight * aux_loss
        return logits, loss

    def forward_at_depth(self, idx, K, targets=None):
        """Evaluate at inference depth K."""
        tok_emb = self.drop(self.token_embedding_table(idx))
        processed_x, y, _ = self._run_iterations(tok_emb, K)
        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None):
        """Sequential evaluation: process positions one at a time."""
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        y = None

        for t in range(T):
            y_full, correction_full = self._iteration_step(processed_x)

            # Save this position's y for head input later
            if y is None:
                y = torch.zeros_like(tok_emb)
            y[:, t, :] = y_full[:, t, :]

            # Past-only: correction at t contextualizes t+1
            if t < T - 1:
                processed_x[:, t+1, :] = tok_emb[:, t+1, :] + correction_full[:, t, :]

        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        """Forward pass with convergence diagnostics."""
        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb
        prev_correction = None
        prev_prev_correction = None
        y = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            y, correction = self._iteration_step(processed_x)

            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted = torch.cat([zero, correction[:, :-1, :]], dim=1)
            processed_x = tok_emb + shifted

            if prev_correction is not None:
                diff = (correction - prev_correction).norm(
                    p=float('inf'), dim=-1
                ).max().item()
                correction_norms.append(diff)

                if prev_prev_correction is not None:
                    prev_diff = (prev_correction - prev_prev_correction).norm(
                        p=float('inf'), dim=-1
                    ).max().item()
                    if prev_diff > 1e-10:
                        contraction_ratios.append(diff / prev_diff)

            prev_prev_correction = prev_correction
            prev_correction = correction.clone()

        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        diagnostics = {
            'correction_norms': correction_norms,
            'contraction_ratios': contraction_ratios,
        }
        return logits, loss, diagnostics

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    @torch.no_grad()
    def generate2(self, idx, max_new_tokens, prime_tokens=None):
        """Single-step warm-started generation."""
        if prime_tokens is None:
            prime_tokens = self.n_iters

        eff_corr = None

        for i in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            tok_emb = self.drop(self.token_embedding_table(idx_cond))
            B, T, C = tok_emb.shape
            zero = torch.zeros(B, 1, C, device=tok_emb.device)

            if i < prime_tokens:
                processed_x, y, _ = self._run_iterations(tok_emb, self.n_iters)
                eff_corr = processed_x - tok_emb
                head_input = self._get_head_input(y)
            else:
                ec = eff_corr
                if ec.shape[1] >= T:
                    ec = ec[:, -T:, :]
                else:
                    pad = torch.zeros(B, T - ec.shape[1], C, device=tok_emb.device)
                    ec = torch.cat([pad, ec], dim=1)

                processed_x = tok_emb + ec
                y, correction = self._iteration_step(processed_x)
                eff_corr = torch.cat([zero, correction[:, :-1, :]], dim=1)
                head_input = self._get_head_input(y)

            logits = self.head(self.ln_f(head_input))
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ---------------------------------------------------------------------------
# Variant 1: Attention + Correction FFN
# ---------------------------------------------------------------------------

class AttnCorrFFNModel(SplitBlockLookAhead):
    """Variant 1: Attention-only block, FFN generates corrections.

    y[t] = processed_x[t] + attn(ln1(processed_x))[t]
    correction[t] = corr_ffn(ln2(y))[t]
    processed_x[t] = tok_emb[t] + shift(correction)[t]
    head_input[t] = y[t]
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax, convergence_weight)

        # Shared attention (iterated K times)
        self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax)
        self.ln1 = nn.LayerNorm(n_embed)

        # Shared correction FFN (iterated K times)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)

        # Classification head: input is y (size C)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        y = processed_x + self.attn(self.ln1(processed_x))
        correction = self.corr_ffn(self.ln2(y))
        return y, correction

    def _get_head_input(self, y):
        return y


# ---------------------------------------------------------------------------
# Variant 2: Attention + Head FFN
# ---------------------------------------------------------------------------

class AttnHeadFFNModel(SplitBlockLookAhead):
    """Variant 2: Attention-only block, FFN at head.

    y[t] = processed_x[t] + attn(ln1(processed_x))[t]
    correction[t] = y[t] - processed_x[t]
    processed_x[t] = tok_emb[t] + shift(correction)[t]
    head_input[t] = head_ffn(ln2(y[t]))
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax, convergence_weight)

        # Shared attention (iterated K times)
        self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax)
        self.ln1 = nn.LayerNorm(n_embed)

        # Head FFN (runs once at the end)
        self.head_ffn = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)

        # Classification head: input is ffn(y) (size C)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        y = processed_x + self.attn(self.ln1(processed_x))
        correction = y - processed_x  # raw attention delta
        return y, correction

    def _get_head_input(self, y):
        return y + self.head_ffn(self.ln2(y))


# ---------------------------------------------------------------------------
# Variant 3: Standard Block + Head FFN
# ---------------------------------------------------------------------------

class BlockHeadFFNModel(SplitBlockLookAhead):
    """Variant 3: Standard block (attn+FFN) + extra FFN at head.

    y[t] = processed_x[t] + attn(ln1(processed_x))[t]
    z[t] = y[t] + ffn(ln2(y))[t]
    correction[t] = z[t] - processed_x[t]
    processed_x[t] = tok_emb[t] + shift(correction)[t]
    head_input[t] = head_ffn(ln3(z[t]))
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax, convergence_weight)

        # Shared block (attn + FFN, iterated K times)
        self.block = RoFormerBlock(n_embed, block_size, dropout, use_softmax)

        # Extra head FFN (runs once at the end)
        self.head_ffn = FeedForward(n_embed, dropout)
        self.ln3 = nn.LayerNorm(n_embed)

        # Classification head: input is head_ffn(z) (size C)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        z = self.block(processed_x)
        correction = z - processed_x
        return z, correction

    def _get_head_input(self, z):
        return z + self.head_ffn(self.ln3(z))


# ---------------------------------------------------------------------------
# Stacked Split-Block Models
# ---------------------------------------------------------------------------

class StackedSplitBlock(nn.Module):
    """Base class for stacked split-block look-ahead models.

    N units with separate weights, each iterated K times internally.
    Non-cumulative within each unit (reset to unit input).
    Cumulative between units (standard residual stacking).
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_units = n_units
        self.k_iters = n_layers // n_units
        self.block_size = block_size
        self.convergence_weight = convergence_weight

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

    def _unit_step(self, unit_idx, processed_x):
        """Run one iteration of unit. Returns (y, correction).
        Subclasses implement this.
        """
        raise NotImplementedError

    def _get_head_input(self, y):
        """Transform last unit's output for the head. Subclasses implement this."""
        raise NotImplementedError

    def _run_unit(self, unit_idx, anchor, k_iters):
        """Run K iterations of one unit with non-cumulative corrections."""
        B, T, C = anchor.shape
        processed = anchor
        prev_processed = None

        for k in range(k_iters):
            prev_processed = processed
            y, correction = self._unit_step(unit_idx, processed)

            zero = torch.zeros(B, 1, C, device=anchor.device)
            shifted = torch.cat([zero, correction[:, :-1, :]], dim=1)
            processed = anchor + shifted

        conv_loss = 0.0
        if self.convergence_weight > 0 and self.training and k_iters > 1:
            conv_loss = F.mse_loss(processed, prev_processed.detach())

        return processed, y, conv_loss

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        h = tok_emb
        total_conv_loss = 0.0
        y = None

        for i in range(self.n_units):
            h, y, conv_loss = self._run_unit(i, h, self.k_iters)
            total_conv_loss = total_conv_loss + conv_loss

        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            if total_conv_loss > 0:
                loss = loss + self.convergence_weight * total_conv_loss
        return logits, loss

    def forward_at_depth(self, idx, K, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        h = tok_emb
        y = None

        for i in range(self.n_units):
            h, y, _ = self._run_unit(i, h, K)

        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None):
        if self.k_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        h = tok_emb

        last_y = None
        for i in range(self.n_units):
            anchor = h.clone()
            processed = h.clone()

            for t in range(T):
                y_full, corr_full = self._unit_step(i, processed)
                if t < T - 1:
                    processed[:, t+1, :] = anchor[:, t+1, :] + corr_full[:, t, :]

            last_y = y_full
            h = processed

        head_input = self._get_head_input(last_y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        h = tok_emb
        all_norms = []
        all_ratios = []
        y = None

        for i in range(self.n_units):
            B, T, C = h.shape
            anchor = h
            processed = anchor
            prev_correction = None

            for k in range(self.k_iters):
                y, correction = self._unit_step(i, processed)

                zero = torch.zeros(B, 1, C, device=h.device)
                shifted = torch.cat([zero, correction[:, :-1, :]], dim=1)
                processed = anchor + shifted

                corr_norm = correction.norm(dim=-1).mean().item()
                all_norms.append(corr_norm)

                if prev_correction is not None:
                    diff = (correction - prev_correction).norm(dim=-1).mean()
                    prev_diff = prev_correction.norm(dim=-1).mean()
                    if prev_diff > 1e-8:
                        all_ratios.append((diff / prev_diff).item())
                prev_correction = correction

            h = processed

        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        diag = {
            'empirical_L': all_ratios[-1] if all_ratios else None,
            'correction_norms': all_norms,
            'contraction_ratios': all_ratios,
        }
        return logits, loss, diag

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    @torch.no_grad()
    def generate2(self, idx, max_new_tokens, prime_tokens=None):
        return self.generate(idx, max_new_tokens)


# ---------------------------------------------------------------------------
# Stacked Variant 1: Attention + Correction FFN
# ---------------------------------------------------------------------------

class StackedAttnCorrFFN(StackedSplitBlock):
    """Stacked version of AttnCorrFFNModel.

    N units, each with its own attention + correction FFN.
    Head sees y (attention output) from the last unit.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight)

        self.attns = nn.ModuleList([
            RoFormerAttention(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])
        self.corr_ffns = nn.ModuleList([
            FeedForward(n_embed, dropout) for _ in range(n_units)
        ])
        self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])

        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        y = processed_x + self.attns[unit_idx](self.ln1s[unit_idx](processed_x))
        correction = self.corr_ffns[unit_idx](self.ln2s[unit_idx](y))
        return y, correction

    def _get_head_input(self, y):
        return y


# ---------------------------------------------------------------------------
# Stacked Variant 2: Attention + Head FFN
# ---------------------------------------------------------------------------

class StackedAttnHeadFFN(StackedSplitBlock):
    """Stacked version of AttnHeadFFNModel.

    N units, each with its own attention. Correction = raw attn delta.
    Single head FFN applied once at the end.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight)

        self.attns = nn.ModuleList([
            RoFormerAttention(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])

        self.head_ffn = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)

        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        y = processed_x + self.attns[unit_idx](self.ln1s[unit_idx](processed_x))
        correction = y - processed_x
        return y, correction

    def _get_head_input(self, y):
        return y + self.head_ffn(self.ln2(y))


# ---------------------------------------------------------------------------
# Stacked Variant 3: Standard Block + Head FFN
# ---------------------------------------------------------------------------

class StackedBlockHeadFFN(StackedSplitBlock):
    """Stacked version of BlockHeadFFNModel.

    N units, each with its own standard block (attn + FFN).
    Single extra head FFN applied once at the end.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight)

        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])

        self.head_ffn = FeedForward(n_embed, dropout)
        self.ln3 = nn.LayerNorm(n_embed)

        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        z = self.blocks[unit_idx](processed_x)
        correction = z - processed_x
        return z, correction

    def _get_head_input(self, z):
        return z + self.head_ffn(self.ln3(z))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_attn_corr_ffn(vocab_size, n_embed, n_layers, block_size, dropout,
                       use_softmax=False, **kwargs):
    return AttnCorrFFNModel(vocab_size, n_embed, n_layers, block_size, dropout,
                            use_softmax=use_softmax, **kwargs)

def make_attn_head_ffn(vocab_size, n_embed, n_layers, block_size, dropout,
                       use_softmax=False, **kwargs):
    return AttnHeadFFNModel(vocab_size, n_embed, n_layers, block_size, dropout,
                            use_softmax=use_softmax, **kwargs)

def make_block_head_ffn(vocab_size, n_embed, n_layers, block_size, dropout,
                        use_softmax=False, **kwargs):
    return BlockHeadFFNModel(vocab_size, n_embed, n_layers, block_size, dropout,
                             use_softmax=use_softmax, **kwargs)

# Stacked factory functions

def make_stacked_attn_corr_ffn(vocab_size, n_embed, n_layers, block_size, dropout,
                                use_softmax=False, n_units=None, **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked model")
    return StackedAttnCorrFFN(vocab_size, n_embed, n_layers, block_size, dropout,
                              n_units=n_units, use_softmax=use_softmax, **kwargs)

def make_stacked_attn_head_ffn(vocab_size, n_embed, n_layers, block_size, dropout,
                                use_softmax=False, n_units=None, **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked model")
    return StackedAttnHeadFFN(vocab_size, n_embed, n_layers, block_size, dropout,
                              n_units=n_units, use_softmax=use_softmax, **kwargs)

def make_stacked_block_head_ffn(vocab_size, n_embed, n_layers, block_size, dropout,
                                 use_softmax=False, n_units=None, **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked model")
    return StackedBlockHeadFFN(vocab_size, n_embed, n_layers, block_size, dropout,
                               n_units=n_units, use_softmax=use_softmax, **kwargs)


# ---------------------------------------------------------------------------
# Model registry — includes all look_ahead3 models + new split-block variants
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    **LOOK_AHEAD3_MODELS,

    # Split-block variants (D=1, shared weights)
    'attn_corr_ffn': make_attn_corr_ffn,
    'attn_head_ffn': make_attn_head_ffn,
    'block_head_ffn': make_block_head_ffn,

    # Stacked split-block variants (N units x K iterations)
    'stacked_attn_corr_ffn': make_stacked_attn_corr_ffn,
    'stacked_attn_head_ffn': make_stacked_attn_head_ffn,
    'stacked_block_head_ffn': make_stacked_block_head_ffn,
}

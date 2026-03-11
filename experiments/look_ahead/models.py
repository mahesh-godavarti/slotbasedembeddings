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
Look-ahead models with split-block architecture.

The core mechanism: a shared-weight transformer block is iterated K times.
Each iteration produces a correction that contextualizes the next position's
embedding. At sequential K=1 inference, predecessors are already processed,
so the block receives contextualized inputs — matching the training regime.

Model variants:
  - block_head:          z = block(x), correction = z - x, head sees z
  - block_head_ffn:      z = block(x), correction = z - x, head sees z + head_ffn(z)
  - block_head_corr_ffn: z = block(x), correction = corr_ffn(ln(z)), head sees z  [BEST]
  - block_head_corr_ffn_add: z = block(x), correction = corr_ffn(ln(shift(z)+tok_emb)), head sees z
  - attn_corr_ffn:       y = x + attn(x), correction = corr_ffn(y), head sees y
  - attn_head_ffn:       y = x + attn(x), correction = y - x, head sees y + head_ffn(y)

All variants support deep block_head (D>1) via d_block parameter:
  D distinct blocks per iteration step, weights shared across iterations.

Stacked variants (N units x K iterations):
  N separate units with own weights, each doing K iterations internally.

Baselines:
  - roformer:          Standard transformer (separate weights per layer)
  - roformer_head_ffn: Standard transformer + extra FFN before head
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import (
    RoFormerBlock, RoFormerAttention, FeedForward, RoFormer,
    JoFormerFixedBlock, JoFormerLearnedBlock, JoFormerProjectedBlock,
    JoFormerFixed, JoFormerLearned, JoFormerProjected,
)


# ---------------------------------------------------------------------------
# Base class for split-block look-ahead models
# ---------------------------------------------------------------------------

class SplitBlockLookAhead(nn.Module):
    """Base class for split-block look-ahead variants.

    Subclasses implement _iteration_step() to define what happens each iteration
    and _get_head_input() to define what the head sees.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, k_min=0, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_iters = n_layers
        self.block_size = block_size
        self.convergence_weight = convergence_weight
        self.k_min = k_min  # 0 = disabled (always use n_iters), >0 = sample K ~ Uniform(k_min, n_iters)

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

    def _iteration_step(self, processed_x):
        """Run one iteration. Returns (y, correction)."""
        raise NotImplementedError

    def _get_head_input(self, y):
        """Transform block output into head input."""
        raise NotImplementedError

    def _run_iterations(self, tok_emb, n_iters):
        """Run n_iters shared-weight iterations.

        Returns: (processed_x, y, aux_loss)
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

        # Random K: sample K ~ Uniform(k_min, n_iters) during training
        if self.training and self.k_min > 0:
            n_iters = random.randint(self.k_min, self.n_iters)
        else:
            n_iters = self.n_iters

        processed_x, y, aux_loss = self._run_iterations(tok_emb, n_iters)
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
    """Attention-only block, FFN generates corrections."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax)
        self.ln1 = nn.LayerNorm(n_embed)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
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
    """Attention-only block, FFN at head."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax)
        self.ln1 = nn.LayerNorm(n_embed)
        self.head_ffn = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        y = processed_x + self.attn(self.ln1(processed_x))
        correction = y - processed_x
        return y, correction

    def _get_head_input(self, y):
        return y + self.head_ffn(self.ln2(y))


# ---------------------------------------------------------------------------
# Variant 3: Standard Block + Head FFN
# ---------------------------------------------------------------------------

class BlockHeadFFNModel(SplitBlockLookAhead):
    """Standard block (attn+FFN) + extra FFN at head.

    Supports deep block_head (D>1) via d_block parameter.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        if block_class is None:
            block_class = RoFormerBlock
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.blocks = nn.ModuleList([
                block_class(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax)
        self.head_ffn = FeedForward(n_embed, dropout)
        self.ln3 = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            z = h
        else:
            z = self.block(processed_x)
        correction = z - processed_x
        return z, correction

    def _get_head_input(self, z):
        return z + self.head_ffn(self.ln3(z))


# ---------------------------------------------------------------------------
# Variant 4: Standard Block, head sees z directly
# ---------------------------------------------------------------------------

class BlockHeadModel(SplitBlockLookAhead):
    """Standard block (attn+FFN), head sees block output directly.

    Supports deep block_head (D>1) via d_block parameter.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        if block_class is None:
            block_class = RoFormerBlock
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.blocks = nn.ModuleList([
                block_class(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            z = h
        else:
            z = self.block(processed_x)
        correction = z - processed_x
        return z, correction

    def _get_head_input(self, z):
        return z


# ---------------------------------------------------------------------------
# Variant 5: block_head + corr_ffn(z)  [BEST VARIANT]
# ---------------------------------------------------------------------------

class BlockHeadCorrFFNModel(SplitBlockLookAhead):
    """block_head with FFN-generated correction from z.

    z = block(processed_x)
    correction = corr_ffn(ln_corr(z))
    processed_x = tok_emb + shift(correction)
    head sees z

    Supports deep block_head (D>1) via d_block parameter.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        if block_class is None:
            block_class = RoFormerBlock
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.blocks = nn.ModuleList([
                block_class(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            z = h
        else:
            z = self.block(processed_x)
        correction = self.corr_ffn(self.ln_corr(z))
        return z, correction

    def _get_head_input(self, z):
        return z


# ---------------------------------------------------------------------------
# Variant 5c: block_head + corr_ffn(concat(shift(z), processed_x))
# ---------------------------------------------------------------------------

class BlockHeadCorrFFNConcatModel(SplitBlockLookAhead):
    """block_head with FFN correction from concat of shifted z and tok_emb.

    z = block(processed_x)
    correction = corr_ffn(concat(ln_corr(shift(z)), tok_emb))
    processed_x = tok_emb + correction
    head sees z

    The shift happens before the FFN, not after. The corr_ffn sees both
    past context (z[t-1]) and current token identity (tok_emb[t]).
    tok_emb is used instead of processed_x to avoid circular dependency
    that breaks sequential K=1 inference.
    corr_ffn input is 2C: Linear(2C→4C)→GELU→Linear(4C→C). Total: 24C².

    Supports deep block_head (D>1) via d_block parameter.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        if block_class is None:
            block_class = RoFormerBlock
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.blocks = nn.ModuleList([
                block_class(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax)
        # corr_ffn takes 2C input (concat of shifted z and tok_emb)
        self.corr_ffn = nn.Sequential(
            nn.Linear(2 * n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _run_iterations(self, tok_emb, n_iters):
        B, T, C = tok_emb.shape
        processed_x = tok_emb
        prev_processed_x = None
        z = None
        total_conv_loss = 0.0

        for k in range(n_iters):
            prev_processed_x = processed_x

            # Run block(s)
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            # Shift z: position t gets z from t-1
            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)

            # corr_ffn sees concat(ln(shifted_z), tok_emb) — tok_emb is constant, no circular dep
            ffn_input = torch.cat([self.ln_corr(shifted_z), tok_emb], dim=-1)
            correction = self.corr_ffn(ffn_input)

            # Non-cumulative: reset to tok_emb
            processed_x = tok_emb + correction

            # Convergence loss on last iteration
            if self.convergence_weight > 0 and self.training and k == n_iters - 1:
                total_conv_loss = F.mse_loss(
                    processed_x, prev_processed_x.detach()
                )

        return processed_x, z, total_conv_loss

    def _iteration_step(self, processed_x):
        # Not used by _run_iterations, but needed for sequential eval
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            z = h
        else:
            z = self.block(processed_x)
        return z, None  # correction computed in _run_iterations

    def _get_head_input(self, z):
        return z

    def forward_sequential(self, idx, targets=None):
        """Sequential evaluation: process positions one at a time."""
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            # Run block on full sequence (causal attention handles masking)
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            z_all[:, t, :] = z[:, t, :]

            # Correction for t+1: corr_ffn sees concat(ln(z[t]), tok_emb[t+1])
            if t < T - 1:
                shifted_z_t1 = self.ln_corr(z[:, t, :])  # z from position t
                te_t1 = tok_emb[:, t+1, :]  # constant tok_emb at t+1
                ffn_input = torch.cat([shifted_z_t1, te_t1], dim=-1)
                correction_t1 = self.corr_ffn(ffn_input)
                processed_x[:, t+1, :] = tok_emb[:, t+1, :] + correction_t1

        logits = self.head(self.ln_f(z_all))

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
        z = None
        prev_correction = None
        prev_prev_correction = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)
            ffn_input = torch.cat([self.ln_corr(shifted_z), tok_emb], dim=-1)
            correction = self.corr_ffn(ffn_input)
            processed_x = tok_emb + correction

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

        logits = self.head(self.ln_f(z))

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


# ---------------------------------------------------------------------------
# Variant 5d: block_head + corr_ffn(ln(shift(z) + tok_emb))
# ---------------------------------------------------------------------------

class BlockHeadCorrFFNAddModel(SplitBlockLookAhead):
    """block_head with FFN correction from sum of shifted z and tok_emb.

    z = block(processed_x)
    correction = corr_ffn(ln_corr(shift(z) + tok_emb))
    processed_x = tok_emb + correction
    head sees z

    Like concat variant but uses addition instead of concatenation.
    corr_ffn input is C (standard FeedForward). Total: 20C² per iteration.
    Same params/FLOPs as corr_ffn variant, but token-aware.
    No circular dependency (tok_emb is constant).

    Supports deep block_head (D>1) via d_block parameter.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        if block_class is None:
            block_class = RoFormerBlock
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.blocks = nn.ModuleList([
                block_class(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _run_iterations(self, tok_emb, n_iters):
        B, T, C = tok_emb.shape
        processed_x = tok_emb
        prev_processed_x = None
        z = None
        total_conv_loss = 0.0

        for k in range(n_iters):
            prev_processed_x = processed_x

            # Run block(s)
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            # Shift z: position t gets z from t-1
            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)

            # corr_ffn sees ln(shifted_z + tok_emb) — tok_emb is constant, no circular dep
            correction = self.corr_ffn(self.ln_corr(shifted_z + tok_emb))

            # Non-cumulative: reset to tok_emb
            processed_x = tok_emb + correction

            # Convergence loss on last iteration
            if self.convergence_weight > 0 and self.training and k == n_iters - 1:
                total_conv_loss = F.mse_loss(
                    processed_x, prev_processed_x.detach()
                )

        return processed_x, z, total_conv_loss

    def _iteration_step(self, processed_x):
        # Not used by _run_iterations, but needed for base class interface
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            z = h
        else:
            z = self.block(processed_x)
        return z, None  # correction computed in _run_iterations

    def _get_head_input(self, z):
        return z

    def forward_sequential(self, idx, targets=None):
        """Sequential evaluation: process positions one at a time."""
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            # Run block on full sequence (causal attention handles masking)
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            z_all[:, t, :] = z[:, t, :]

            # Correction for t+1: corr_ffn sees ln(z[t] + tok_emb[t+1])
            if t < T - 1:
                add_input = self.ln_corr(z[:, t, :] + tok_emb[:, t+1, :])
                correction_t1 = self.corr_ffn(add_input)
                processed_x[:, t+1, :] = tok_emb[:, t+1, :] + correction_t1

        logits = self.head(self.ln_f(z_all))

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
        z = None
        prev_correction = None
        prev_prev_correction = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)
            correction = self.corr_ffn(self.ln_corr(shifted_z + tok_emb))
            processed_x = tok_emb + correction

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

        logits = self.head(self.ln_f(z))

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


# ---------------------------------------------------------------------------
# Variant 5b: block_head + corr_ffn(z - processed_x)  [RETIRED]
# ---------------------------------------------------------------------------

class BlockHeadDeltaFFNModel(SplitBlockLookAhead):
    """block_head with FFN-generated correction from delta.

    z = block(processed_x)
    correction = corr_ffn(ln_corr(z - processed_x))
    processed_x = tok_emb + shift(correction)
    head sees z

    Supports deep block_head (D>1) via d_block parameter.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        if block_class is None:
            block_class = RoFormerBlock
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.blocks = nn.ModuleList([
                block_class(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _iteration_step(self, processed_x):
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            z = h
        else:
            z = self.block(processed_x)
        delta = z - processed_x
        correction = self.corr_ffn(self.ln_corr(delta))
        return z, correction

    def _get_head_input(self, z):
        return z


# ---------------------------------------------------------------------------
# Stacked Split-Block Models
# ---------------------------------------------------------------------------

class StackedSplitBlock(nn.Module):
    """Base class for stacked split-block look-ahead models.

    N units with separate weights, each iterated K times internally.
    Non-cumulative within each unit (reset to unit input).
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_units = n_units
        if n_layers % n_units != 0:
            raise ValueError(
                f"n_layers ({n_layers}) must be divisible by n_units ({n_units}). "
                f"Use n_layers = n_units * K (e.g. {n_units * (n_layers // n_units + 1)})."
            )
        self.k_iters = n_layers // n_units
        self.n_iters = self.k_iters
        self.block_size = block_size
        self.convergence_weight = convergence_weight

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

    def _unit_step(self, unit_idx, processed_x):
        raise NotImplementedError

    def _get_head_input(self, y):
        raise NotImplementedError

    def _run_unit(self, unit_idx, anchor, k_iters):
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
        """Evaluate at per-unit depth K (all units run K iterations each)."""
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
# Stacked Variants
# ---------------------------------------------------------------------------

class StackedAttnCorrFFN(StackedSplitBlock):
    """Stacked: N units, each with attention + correction FFN."""

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


class StackedAttnHeadFFN(StackedSplitBlock):
    """Stacked: N units, each with attention. Single head FFN at end."""

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


class StackedBlockHeadFFN(StackedSplitBlock):
    """Stacked: N units, each with standard block. Single head FFN at end."""

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


class StackedBlockHead(StackedSplitBlock):
    """Stacked: N units, each with standard block. Head sees z directly."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight)
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        z = self.blocks[unit_idx](processed_x)
        correction = z - processed_x
        return z, correction

    def _get_head_input(self, z):
        return z


class StackedBlockHeadCorrFFN(StackedSplitBlock):
    """Stacked: N units, each with block + corr_ffn. Head sees z."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight)
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.corr_ffns = nn.ModuleList([
            FeedForward(n_embed, dropout) for _ in range(n_units)
        ])
        self.ln_corrs = nn.ModuleList([
            nn.LayerNorm(n_embed) for _ in range(n_units)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        z = self.blocks[unit_idx](processed_x)
        correction = self.corr_ffns[unit_idx](self.ln_corrs[unit_idx](z))
        return z, correction

    def _get_head_input(self, z):
        return z


class StackedBlockHeadDeltaFFN(StackedSplitBlock):
    """Stacked version of BlockHeadDeltaFFNModel.

    N units, each with its own standard block + corr_ffn.
    z = block(processed_x)
    correction = corr_ffn(ln_corr(z - processed_x))
    Head sees z from the last unit.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight)
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.corr_ffns = nn.ModuleList([
            FeedForward(n_embed, dropout) for _ in range(n_units)
        ])
        self.ln_corrs = nn.ModuleList([
            nn.LayerNorm(n_embed) for _ in range(n_units)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        z = self.blocks[unit_idx](processed_x)
        delta = z - processed_x
        correction = self.corr_ffns[unit_idx](self.ln_corrs[unit_idx](delta))
        return z, correction

    def _get_head_input(self, z):
        return z


class StackedBlockHeadCorrFFNConcat(StackedSplitBlock):
    """Stacked concat variant: N units, each with block + concat corr_ffn.

    z = block(processed_x)
    correction = corr_ffn(concat(ln_corr(shift(z)), anchor))
    processed_x = anchor + correction
    Head sees z from the last unit.

    Uses anchor (constant within unit) instead of processed_x in concat
    to avoid circular dependency that breaks sequential K=1.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight)
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.corr_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_embed, 4 * n_embed),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * n_embed, n_embed),
                nn.Dropout(dropout),
            )
            for _ in range(n_units)
        ])
        self.ln_corrs = nn.ModuleList([
            nn.LayerNorm(n_embed) for _ in range(n_units)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        # Not used directly — _run_unit is overridden
        z = self.blocks[unit_idx](processed_x)
        return z, None

    def _get_head_input(self, z):
        return z

    def _run_unit(self, unit_idx, anchor, k_iters):
        B, T, C = anchor.shape
        processed = anchor
        prev_processed = None
        z = None

        for k in range(k_iters):
            prev_processed = processed
            z = self.blocks[unit_idx](processed)

            zero = torch.zeros(B, 1, C, device=anchor.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)
            ffn_input = torch.cat([self.ln_corrs[unit_idx](shifted_z), anchor], dim=-1)
            correction = self.corr_ffns[unit_idx](ffn_input)
            processed = anchor + correction

        conv_loss = 0.0
        if self.convergence_weight > 0 and self.training and k_iters > 1:
            conv_loss = F.mse_loss(processed, prev_processed.detach())

        return processed, z, conv_loss

    def forward_sequential(self, idx, targets=None):
        if self.k_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        h = tok_emb

        last_z = None
        for i in range(self.n_units):
            anchor = h.clone()
            processed = h.clone()

            for t in range(T):
                z = self.blocks[i](processed)
                if last_z is None and i == self.n_units - 1:
                    last_z = torch.zeros_like(tok_emb)
                if i == self.n_units - 1:
                    last_z[:, t, :] = z[:, t, :]

                if t < T - 1:
                    shifted_z_t1 = self.ln_corrs[i](z[:, t, :])
                    anc_t1 = anchor[:, t+1, :]  # constant anchor, no circular dep
                    ffn_input = torch.cat([shifted_z_t1, anc_t1], dim=-1)
                    correction_t1 = self.corr_ffns[i](ffn_input)
                    processed[:, t+1, :] = anchor[:, t+1, :] + correction_t1

            h = processed
            if i == self.n_units - 1 and last_z is None:
                last_z = z

        logits = self.head(self.ln_f(last_z))

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
        z = None

        for i in range(self.n_units):
            B, T, C = h.shape
            anchor = h
            processed = anchor
            prev_correction = None

            for k in range(self.k_iters):
                z = self.blocks[i](processed)

                zero = torch.zeros(B, 1, C, device=h.device)
                shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)
                ffn_input = torch.cat([self.ln_corrs[i](shifted_z), anchor], dim=-1)
                correction = self.corr_ffns[i](ffn_input)
                processed = anchor + correction

                corr_norm = correction.norm(dim=-1).mean().item()
                all_norms.append(corr_norm)

                if prev_correction is not None:
                    diff = (correction - prev_correction).norm(dim=-1).mean()
                    prev_diff = prev_correction.norm(dim=-1).mean()
                    if prev_diff > 1e-8:
                        all_ratios.append((diff / prev_diff).item())
                prev_correction = correction

            h = processed

        logits = self.head(self.ln_f(z))

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


# ---------------------------------------------------------------------------
# RoFormer + Head FFN (baseline with extra FFN before head)
# ---------------------------------------------------------------------------

class RoFormerHeadFFN(nn.Module):
    """Standard roformer with an extra FFN before the classification head."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [RoFormerBlock(n_embed, block_size, dropout, use_softmax)
             for _ in range(n_layers)]
        )
        self.head_ffn = FeedForward(n_embed, dropout)
        self.ln_h = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x)
        x = x + self.head_ffn(self.ln_h(x))
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


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _make_factory(cls):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, **kwargs):
        return cls(vocab_size, n_embed, n_layers, block_size, dropout,
                   use_softmax=use_softmax, **kwargs)
    return factory

def _make_stacked_factory(cls):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, n_units=None, **kwargs):
        if n_units is None:
            raise ValueError("n_units must be specified for stacked model")
        return cls(vocab_size, n_embed, n_layers, block_size, dropout,
                   n_units=n_units, use_softmax=use_softmax, **kwargs)
    return factory


# JoFormer variant factories (pass block_class to reuse RoFormer-based model classes)

def _make_joformer_block_head_ffn(block_class):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, **kwargs):
        return BlockHeadFFNModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 use_softmax=use_softmax, block_class=block_class, **kwargs)
    return factory

def _make_joformer_block_head(block_class):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, **kwargs):
        return BlockHeadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                              use_softmax=use_softmax, block_class=block_class, **kwargs)
    return factory

def _make_joformer_block_head_corr_ffn(block_class):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, **kwargs):
        return BlockHeadCorrFFNModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                     use_softmax=use_softmax, block_class=block_class, **kwargs)
    return factory

def _make_joformer_block_head_delta_ffn(block_class):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, **kwargs):
        return BlockHeadDeltaFFNModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                      use_softmax=use_softmax, block_class=block_class, **kwargs)
    return factory


# ---------------------------------------------------------------------------
# Legacy models (from look_ahead3 — kept for backwards compatibility)
# ---------------------------------------------------------------------------
from legacy_models import MODEL_CLASSES as LEGACY_MODEL_CLASSES


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    # Include all legacy models (marked NOT ACTIVELY USED — kept for reference)
    **LEGACY_MODEL_CLASSES,

    # Baselines (separate weights per layer)
    'roformer': RoFormer,
    'roformer_head_ffn': _make_factory(RoFormerHeadFFN),

    # JoFormer baselines
    'joformer_fixed': JoFormerFixed,
    'joformer_learned': JoFormerLearned,
    'joformer_projected': JoFormerProjected,

    # Split-block look-ahead variants (D=1 by default, supports deep block_head D>1 via --d_block)
    'attn_corr_ffn': _make_factory(AttnCorrFFNModel),
    'attn_head_ffn': _make_factory(AttnHeadFFNModel),
    'block_head_ffn': _make_factory(BlockHeadFFNModel),
    'block_head': _make_factory(BlockHeadModel),
    'block_head_corr_ffn': _make_factory(BlockHeadCorrFFNModel),
    'block_head_corr_ffn_concat': _make_factory(BlockHeadCorrFFNConcatModel),
    'block_head_corr_ffn_add': _make_factory(BlockHeadCorrFFNAddModel),
    'block_head_delta_ffn': _make_factory(BlockHeadDeltaFFNModel),

    # JoFormer block_head variants (NOT ACTIVELY USED)
    'joformer_fixed_block_head_ffn': _make_joformer_block_head_ffn(JoFormerFixedBlock),
    'joformer_learned_block_head_ffn': _make_joformer_block_head_ffn(JoFormerLearnedBlock),
    'joformer_projected_block_head_ffn': _make_joformer_block_head_ffn(JoFormerProjectedBlock),
    'joformer_fixed_block_head': _make_joformer_block_head(JoFormerFixedBlock),
    'joformer_learned_block_head': _make_joformer_block_head(JoFormerLearnedBlock),
    'joformer_projected_block_head': _make_joformer_block_head(JoFormerProjectedBlock),
    'joformer_fixed_block_head_corr_ffn': _make_joformer_block_head_corr_ffn(JoFormerFixedBlock),
    'joformer_learned_block_head_corr_ffn': _make_joformer_block_head_corr_ffn(JoFormerLearnedBlock),
    'joformer_projected_block_head_corr_ffn': _make_joformer_block_head_corr_ffn(JoFormerProjectedBlock),
    'joformer_fixed_block_head_delta_ffn': _make_joformer_block_head_delta_ffn(JoFormerFixedBlock),
    'joformer_learned_block_head_delta_ffn': _make_joformer_block_head_delta_ffn(JoFormerLearnedBlock),
    'joformer_projected_block_head_delta_ffn': _make_joformer_block_head_delta_ffn(JoFormerProjectedBlock),

    # Stacked split-block variants (N units x K iterations)
    'stacked_block_head': _make_stacked_factory(StackedBlockHead),
    'stacked_block_head_ffn': _make_stacked_factory(StackedBlockHeadFFN),
    'stacked_block_head_corr_ffn': _make_stacked_factory(StackedBlockHeadCorrFFN),
    'stacked_block_head_delta_ffn': _make_stacked_factory(StackedBlockHeadDeltaFFN),
    'stacked_block_head_corr_ffn_concat': _make_stacked_factory(StackedBlockHeadCorrFFNConcat),
    'stacked_attn_corr_ffn': _make_stacked_factory(StackedAttnCorrFFN),
    'stacked_attn_head_ffn': _make_stacked_factory(StackedAttnHeadFFN),
}

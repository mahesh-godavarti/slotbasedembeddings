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
  - attn_corr_ffn_sync:  h = x + attn(x), correction = corr_ffn(h), head sees x + correction
  - joformer_*_sync:     same as attn_corr_ffn_sync but with JoFormer attention variants
  - block_aligned:       correction = attn(x), classifier = f(x, corr), next = f(tok_emb, shift(corr))
                          where f(x,c) = x + c + ffn(ln2(x+c)). Classifier = roformer block output.
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
    JoFormerFixedAttention, JoFormerLearnedAttention,
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
                 use_softmax=False, convergence_weight=0.0, k_min=0,
                 head_sees_px=False, n_head=1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_iters = n_layers
        self.block_size = block_size
        self.convergence_weight = convergence_weight
        self.k_min = k_min  # 0 = disabled (always use n_iters), >0 = sample K ~ Uniform(k_min, n_iters)
        self.head_sees_px = head_sees_px
        self.n_head = n_head

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
        head_input = processed_x if self.head_sees_px else self._get_head_input(y)
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
        head_input = processed_x if self.head_sees_px else self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
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

        head_input = processed_x if self.head_sees_px else self._get_head_input(y)
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

        head_input = processed_x if self.head_sees_px else self._get_head_input(y)
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
                head_input = processed_x if self.head_sees_px else self._get_head_input(y)
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
                head_input = processed_x if self.head_sees_px else self._get_head_input(y)

            logits = self.head(self.ln_f(head_input))
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ---------------------------------------------------------------------------
# Variant 1: Attention + Correction FFN
# ---------------------------------------------------------------------------

class AttnCorrFFNModel(SplitBlockLookAhead):
    """Attention-only block, FFN generates corrections. Head sees h (attn output)."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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


class AttnCorrFFNSyncModel(SplitBlockLookAhead):
    """Attention + FFN, head sees processed_x + correction (in sync).

    h = x + attn(ln1(x))
    correction = corr_ffn(ln2(h))
    head sees: x + correction  (self-inclusive, unshifted)
    processed_x = tok_emb + shift(correction)  (past-only)

    Same params as a roformer block (12C² per D). Head and correction are aligned:
    the head sees what processed_x would be if the correction weren't shifted.

    Supports deep (D>1) via d_block parameter: D separate-weight (attn, corr_ffn) pairs.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.attns = nn.ModuleList([
                RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
            self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
            self.corr_ffns = nn.ModuleList([FeedForward(n_embed, dropout) for _ in range(d_block)])
            self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
        else:
            self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
            self.ln1 = nn.LayerNorm(n_embed)
            self.corr_ffn = FeedForward(n_embed, dropout)
            self.ln2 = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _apply_block(self, processed_x, d):
        """Apply the d-th (attn, corr_ffn) pair. Returns (y, correction)."""
        if self.d_block > 1:
            h = processed_x + self.attns[d](self.ln1s[d](processed_x))
            correction = self.corr_ffns[d](self.ln2s[d](h))
        else:
            h = processed_x + self.attn(self.ln1(processed_x))
            correction = self.corr_ffn(self.ln2(h))
        y = processed_x + correction
        return y, correction

    def _iteration_step(self, processed_x):
        if self.d_block > 1:
            correction = None
            for d in range(self.d_block):
                processed_x, correction = self._apply_block(processed_x, d)
            return processed_x, correction
        else:
            return self._apply_block(processed_x, 0)

    def _get_head_input(self, y):
        return y


# ---------------------------------------------------------------------------
# JoFormer Sync Variants
# ---------------------------------------------------------------------------

class JoFormerFixedSyncModel(SplitBlockLookAhead):
    """JoFormer Fixed attention + corr_ffn, synced head.

    Same as AttnCorrFFNSyncModel but uses JoFormerFixedAttention (rotates K,Q,V;
    inverse on output) instead of RoFormerAttention.
    Same param count: 12C² per D.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.attns = nn.ModuleList([
                JoFormerFixedAttention(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
            self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
            self.corr_ffns = nn.ModuleList([FeedForward(n_embed, dropout) for _ in range(d_block)])
            self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
        else:
            self.attn = JoFormerFixedAttention(n_embed, block_size, dropout, use_softmax)
            self.ln1 = nn.LayerNorm(n_embed)
            self.corr_ffn = FeedForward(n_embed, dropout)
            self.ln2 = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _apply_block(self, processed_x, d):
        if self.d_block > 1:
            h = processed_x + self.attns[d](self.ln1s[d](processed_x))
            correction = self.corr_ffns[d](self.ln2s[d](h))
        else:
            h = processed_x + self.attn(self.ln1(processed_x))
            correction = self.corr_ffn(self.ln2(h))
        y = processed_x + correction
        return y, correction

    def _iteration_step(self, processed_x):
        if self.d_block > 1:
            correction = None
            for d in range(self.d_block):
                processed_x, correction = self._apply_block(processed_x, d)
            return processed_x, correction
        else:
            return self._apply_block(processed_x, 0)

    def _get_head_input(self, y):
        return y


class JoFormerLearnedSyncModel(SplitBlockLookAhead):
    """JoFormer Learned attention + corr_ffn, synced head.

    Per-token learned angles (from angle_embedding_table), flip-cumsum-flip.
    Same angles for all iterations and D blocks.
    Token embedding is half-size + expander (matches original JoFormerLearned).
    Attention param count same (4C²), but adds angle_embedding (vocab × C/2)
    and expander (C/2 × C). Per-D cost: 12C².
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        # Override token embedding: half-size + expander (matches JoFormerLearned)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)

        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.attns = nn.ModuleList([
                JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
            self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
            self.corr_ffns = nn.ModuleList([FeedForward(n_embed, dropout) for _ in range(d_block)])
            self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
        else:
            self.attn = JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
            self.ln1 = nn.LayerNorm(n_embed)
            self.corr_ffn = FeedForward(n_embed, dropout)
            self.ln2 = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _apply_block(self, processed_x, d, angles):
        if self.d_block > 1:
            h = processed_x + self.attns[d](self.ln1s[d](processed_x), angles)
            correction = self.corr_ffns[d](self.ln2s[d](h))
        else:
            h = processed_x + self.attn(self.ln1(processed_x), angles)
            correction = self.corr_ffn(self.ln2(h))
        y = processed_x + correction
        return y, correction

    def _iteration_step(self, processed_x):
        """Uses self._current_angles set by forward/forward_at_depth."""
        angles = self._current_angles
        if self.d_block > 1:
            correction = None
            for d in range(self.d_block):
                processed_x, correction = self._apply_block(processed_x, d, angles)
            return processed_x, correction
        else:
            return self._apply_block(processed_x, 0, angles)

    def _get_head_input(self, y):
        return y

    def _compute_angles(self, idx):
        raw_angles = self.angle_embedding_table(idx)  # (B, T, C//2)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)

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
        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)
        processed_x, y, _ = self._run_iterations(tok_emb, K)
        head_input = self._get_head_input(y)
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        y = None

        for t in range(T):
            y_full, correction_full = self._iteration_step(processed_x)

            if y is None:
                y = torch.zeros_like(tok_emb)
            y[:, t, :] = y_full[:, t, :]

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
        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)
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


class JoFormerProjectedSyncModel(SplitBlockLookAhead):
    """JoFormer Projected attention + corr_ffn, synced head.

    Per-block angle_proj computes angles from block input (changes each iteration).
    No vector_proj — angles are projected but attention operates in original space.
    Uses JoFormerLearnedAttention internally.
    Per-D cost: 12C² (attn + corr_ffn) + 3C² (angle_proj) = 15C².
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        self.d_block = d_block
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.attns = nn.ModuleList([
                JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
                for _ in range(d_block)
            ])
            self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
            self.corr_ffns = nn.ModuleList([FeedForward(n_embed, dropout) for _ in range(d_block)])
            self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(d_block)])
            self.angle_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_embed, 2 * n_embed),
                    nn.GELU(),
                    nn.Linear(2 * n_embed, n_embed // 2),
                )
                for _ in range(d_block)
            ])
        else:
            self.attn = JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
            self.ln1 = nn.LayerNorm(n_embed)
            self.corr_ffn = FeedForward(n_embed, dropout)
            self.ln2 = nn.LayerNorm(n_embed)
            self.angle_proj = nn.Sequential(
                nn.Linear(n_embed, 2 * n_embed),
                nn.GELU(),
                nn.Linear(2 * n_embed, n_embed // 2),
            )
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _compute_angles(self, x, d):
        if self.d_block > 1:
            raw_angles = self.angle_projs[d](x)
        else:
            raw_angles = self.angle_proj(x)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def _apply_block(self, processed_x, d):
        angles = self._compute_angles(processed_x, d)
        if self.d_block > 1:
            h = processed_x + self.attns[d](self.ln1s[d](processed_x), angles)
            correction = self.corr_ffns[d](self.ln2s[d](h))
        else:
            h = processed_x + self.attn(self.ln1(processed_x), angles)
            correction = self.corr_ffn(self.ln2(h))
        y = processed_x + correction
        return y, correction

    def _iteration_step(self, processed_x):
        if self.d_block > 1:
            correction = None
            for d in range(self.d_block):
                processed_x, correction = self._apply_block(processed_x, d)
            return processed_x, correction
        else:
            return self._apply_block(processed_x, 0)

    def _get_head_input(self, y):
        return y


# ---------------------------------------------------------------------------
# Variant 7: Block-Aligned Look-Ahead
# ---------------------------------------------------------------------------

class BlockAlignedModel(nn.Module):
    """Block-aligned look-ahead: classifier and iterative process use the same formula.

    f(x, c) = x + c + ffn(ln2(x + c))

    - Classifier at t: f(processed_x[t], attn_corr[t])    — self-inclusive
    - Next position:   f(tok_emb[t+1], attn_corr[t])       — past-only

    Same formula, same shared weights. The classifier sees exactly a standard
    roformer block output. No architectural compromise at the head.

    D=1: single (attn, ffn) pair, shared across K iterations.
    D>1: blocks 1..D-1 are full standard blocks. Block D is split:
         attn_D saved for shifting, ffn_D applied in both classifier and next processed_x.
         12C² per D, same as roformer.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1, k_min=0, n_head=1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.convergence_weight = convergence_weight
        self.k_min = k_min
        self.d_block = d_block
        self.n_head = n_head

        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            # Blocks 1..D-1: full standard blocks
            self.inner_blocks = nn.ModuleList([
                RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block - 1)
            ])
            # Block D: split into attn + ffn
            self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
            self.ln1 = nn.LayerNorm(n_embed)
            self.ffn = FeedForward(n_embed, dropout)
            self.ln2 = nn.LayerNorm(n_embed)
        else:
            self.n_iters = n_layers
            self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
            self.ln1 = nn.LayerNorm(n_embed)
            self.ffn = FeedForward(n_embed, dropout)
            self.ln2 = nn.LayerNorm(n_embed)

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _apply_f(self, x, attn_corr):
        """f(x, c) = x + c + ffn(ln2(x + c))"""
        h = x + attn_corr
        return h + self.ffn(self.ln2(h))

    def _run_inner_blocks(self, x):
        """Run blocks 1..D-1 (full standard blocks). Identity if D=1."""
        if self.d_block > 1:
            for block in self.inner_blocks:
                x = block(x)
        return x

    def _get_attn_corr(self, x):
        """Run block D's attention only."""
        return self.attn(self.ln1(x))

    def _build_processed_x(self, tok_emb, attn_corr):
        """Build processed_x from tok_emb and shifted attn_corr.

        processed_x = f(tok_emb + shift(attn_corr)) through inner blocks,
        then ready for block D's attention.
        """
        B, T, C = tok_emb.shape
        zero = torch.zeros(B, 1, C, device=tok_emb.device)
        shifted = torch.cat([zero, attn_corr[:, :-1, :]], dim=1)

        # Apply f with block D's FFN: tok_emb + shifted + ffn(ln2(tok_emb + shifted))
        processed_x = self._apply_f(tok_emb, shifted)

        # Run inner blocks (D>1 only)
        processed_x = self._run_inner_blocks(processed_x)

        return processed_x

    def _run_iterations(self, tok_emb, n_iters):
        """Run n_iters iterations. Returns (processed_x, attn_corr, aux_loss)."""
        B, T, C = tok_emb.shape

        # Initial: raw tok_emb (bootstrapping step, like concat v2)
        processed_x = tok_emb

        prev_attn_corr = None
        attn_corr = None
        total_conv_loss = 0.0

        for k in range(n_iters):
            prev_attn_corr = attn_corr

            attn_corr = self._get_attn_corr(processed_x)

            if k < n_iters - 1:
                processed_x = self._build_processed_x(tok_emb, attn_corr)

            # Convergence loss
            if self.convergence_weight > 0 and self.training and k == n_iters - 1 and prev_attn_corr is not None:
                total_conv_loss = F.mse_loss(attn_corr, prev_attn_corr.detach())

        return processed_x, attn_corr, total_conv_loss

    def _classifier_input(self, processed_x, attn_corr):
        """Classifier sees f(processed_x, attn_corr) = standard block output."""
        return self._apply_f(processed_x, attn_corr)

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))

        if self.training and self.k_min > 0:
            n_iters = random.randint(self.k_min, self.n_iters)
        else:
            n_iters = self.n_iters

        processed_x, attn_corr, aux_loss = self._run_iterations(tok_emb, n_iters)
        logits = self.head(self.ln_f(self._classifier_input(processed_x, attn_corr)))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            if aux_loss > 0:
                loss = loss + self.convergence_weight * aux_loss
        return logits, loss

    def forward_at_depth(self, idx, K, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        processed_x, attn_corr, _ = self._run_iterations(tok_emb, K)
        logits = self.head(self.ln_f(self._classifier_input(processed_x, attn_corr)))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        # Initial: raw tok_emb (bootstrapping, matches iteration 0 of training)
        processed_x = tok_emb.clone()

        classifier_inputs = torch.zeros(B, T, C, device=tok_emb.device)

        for t in range(T):
            # Attention on full sequence (causal masking handles it)
            attn_corr = self._get_attn_corr(processed_x)

            # Classifier at position t
            ci = self._classifier_input(processed_x, attn_corr)
            classifier_inputs[:, t, :] = ci[:, t, :]

            # Set up position t+1
            if t < T - 1:
                # f(tok_emb[t+1], attn_corr[t])
                h = tok_emb[:, t+1, :] + attn_corr[:, t, :]
                new_px = h + self.ffn(self.ln2(h))

                # Run inner blocks on full sequence with updated position
                processed_x[:, t+1, :] = new_px
                # Re-run inner blocks to propagate the update
                processed_x = self._run_inner_blocks(processed_x)

        logits = self.head(self.ln_f(classifier_inputs))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        # Initial: raw tok_emb (bootstrapping)
        processed_x = tok_emb

        prev_attn_corr = None
        prev_prev_attn_corr = None
        attn_corr = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            prev_prev_attn_corr = prev_attn_corr
            prev_attn_corr = attn_corr

            attn_corr = self._get_attn_corr(processed_x)

            if k < self.n_iters - 1:
                processed_x = self._build_processed_x(tok_emb, attn_corr)

            if prev_attn_corr is not None:
                diff = (attn_corr - prev_attn_corr).norm(
                    p=float('inf'), dim=-1
                ).max().item()
                correction_norms.append(diff)

                if prev_prev_attn_corr is not None:
                    prev_diff = (prev_attn_corr - prev_prev_attn_corr).norm(
                        p=float('inf'), dim=-1
                    ).max().item()
                    if prev_diff > 1e-10:
                        contraction_ratios.append(diff / prev_diff)

        logits = self.head(self.ln_f(self._classifier_input(processed_x, attn_corr)))

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
        return self.generate(idx, max_new_tokens)


class BlockAlignedLightModel(BlockAlignedModel):
    """Block-aligned light: classifier sees processed_x + attn_corr (no extra FFN).

    processed_x already has FFN baked in from previous iteration, so the classifier
    input is already rich. Saves one FFN eval per forward pass.

    - Classifier at t: processed_x[t] + attn_corr[t]        — no FFN (8C² saved)
    - Next position:   f(tok_emb[t+1], attn_corr[t])          — has FFN
    """

    def _classifier_input(self, processed_x, attn_corr):
        """Classifier sees h = processed_x + attn_corr (no extra FFN)."""
        return processed_x + attn_corr


class BlockAlignedPureModel(BlockAlignedModel):
    """Block-aligned pure: classifier uses f(tok_emb, attn_corr) instead of f(processed_x, attn_corr).

    Maximally consistent:
        z[t]           = f(tok_emb[t], attn_corr[t])      — self-inclusive
        processed_x[t] = f(tok_emb[t], attn_corr[t-1])    — past-only

    Same f, same tok_emb, only the attention index differs.
    The original block_aligned uses f(processed_x, attn_corr) which nests two f applications.
    """

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))

        if self.training and self.k_min > 0:
            n_iters = random.randint(self.k_min, self.n_iters)
        else:
            n_iters = self.n_iters

        processed_x, attn_corr, aux_loss = self._run_iterations(tok_emb, n_iters)
        # Pure: classifier uses f(tok_emb, attn_corr) not f(processed_x, attn_corr)
        logits = self.head(self.ln_f(self._apply_f(tok_emb, attn_corr)))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            if aux_loss > 0:
                loss = loss + self.convergence_weight * aux_loss
        return logits, loss

    def forward_at_depth(self, idx, K, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        processed_x, attn_corr, _ = self._run_iterations(tok_emb, K)
        logits = self.head(self.ln_f(self._apply_f(tok_emb, attn_corr)))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        classifier_inputs = torch.zeros(B, T, C, device=tok_emb.device)

        for t in range(T):
            attn_corr = self._get_attn_corr(processed_x)

            # Pure: classifier uses f(tok_emb, attn_corr)
            ci = self._apply_f(tok_emb, attn_corr)
            classifier_inputs[:, t, :] = ci[:, t, :]

            if t < T - 1:
                h = tok_emb[:, t+1, :] + attn_corr[:, t, :]
                new_px = h + self.ffn(self.ln2(h))
                processed_x[:, t+1, :] = new_px
                processed_x = self._run_inner_blocks(processed_x)

        logits = self.head(self.ln_f(classifier_inputs))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb
        prev_attn_corr = None
        prev_prev_attn_corr = None
        attn_corr = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            prev_prev_attn_corr = prev_attn_corr
            prev_attn_corr = attn_corr

            attn_corr = self._get_attn_corr(processed_x)

            if k < self.n_iters - 1:
                processed_x = self._build_processed_x(tok_emb, attn_corr)

            if prev_attn_corr is not None:
                diff = (attn_corr - prev_attn_corr).norm(
                    p=float('inf'), dim=-1
                ).max().item()
                correction_norms.append(diff)

                if prev_prev_attn_corr is not None:
                    prev_diff = (prev_attn_corr - prev_prev_attn_corr).norm(
                        p=float('inf'), dim=-1
                    ).max().item()
                    if prev_diff > 1e-10:
                        contraction_ratios.append(diff / prev_diff)

        # Pure: classifier uses f(tok_emb, attn_corr)
        logits = self.head(self.ln_f(self._apply_f(tok_emb, attn_corr)))

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


# StackedBlockAlignedLight defined after StackedBlockAligned below


# ---------------------------------------------------------------------------
# Variant 2: Attention + Head FFN
# ---------------------------------------------------------------------------

class AttnHeadFFNModel(SplitBlockLookAhead):
    """Attention-only block, FFN at head."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        self.attn = RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
                 block_class=None, subtract_input=True, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight, **kwargs)
        if block_class is None:
            block_class = RoFormerBlock
        self.d_block = d_block
        self.subtract_input = subtract_input
        if d_block > 1:
            assert n_layers % d_block == 0
            self.n_iters = n_layers // d_block
            self.blocks = nn.ModuleList([
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
        correction = z - processed_x if self.subtract_input else z
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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
    correction = corr_ffn(ln_corr(concat(shift(z), tok_emb)))
    processed_x = tok_emb + correction
    head sees z

    The shift happens before the FFN, not after. The corr_ffn sees both
    past context (z[t-1]) and current token identity (tok_emb[t]).
    tok_emb is used instead of processed_x to avoid circular dependency
    that breaks sequential K=1 inference.
    Joint LN over full 2C concatenation (not separate LN on shifted_z only).
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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
        # corr_ffn takes 2C input (concat of shifted z and tok_emb)
        self.corr_ffn = nn.Sequential(
            nn.Linear(2 * n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        self.ln_corr = nn.LayerNorm(2 * n_embed)  # joint LN over full 2C concat
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

            # corr_ffn sees ln(concat(shifted_z, tok_emb)) — joint LN over full 2C
            ffn_input = self.ln_corr(torch.cat([shifted_z, tok_emb], dim=-1))
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

    def forward_sequential(self, idx, targets=None, seq_k=1):
        """Sequential evaluation: process positions one at a time."""
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        # Initialize position 0 to match parallel mode: corr_ffn(ln(concat(zeros, tok_emb[0])))
        zero_0 = torch.zeros(B, 1, C, device=tok_emb.device)
        ffn_input_0 = self.ln_corr(torch.cat([zero_0, tok_emb[:, :1, :]], dim=-1))
        init_corr_0 = self.corr_ffn(ffn_input_0)
        processed_x[:, 0, :] = tok_emb[:, 0, :] + init_corr_0.squeeze(1)
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            for _k in range(seq_k):
                if self.d_block > 1:
                    h = processed_x
                    for block in self.blocks:
                        h = block(h)
                    z = h
                else:
                    z = self.block(processed_x)

                if t > 0:
                    ffn_input = self.ln_corr(torch.cat([z[:, t-1, :], tok_emb[:, t, :]], dim=-1))
                    correction_t = self.corr_ffn(ffn_input)
                    processed_x[:, t, :] = tok_emb[:, t, :] + correction_t

            # Final block pass after last correction update
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            z_all[:, t, :] = z[:, t, :]

            # Correction for t+1: corr_ffn sees ln(concat(z[t], tok_emb[t+1]))
            if t < T - 1:
                shifted_z_t1 = z[:, t, :]
                te_t1 = tok_emb[:, t+1, :]
                ffn_input = self.ln_corr(torch.cat([shifted_z_t1, te_t1], dim=-1))
                correction_t1 = self.corr_ffn(ffn_input)
                processed_x[:, t+1, :] = tok_emb[:, t+1, :] + correction_t1

        head_input = processed_x if self.head_sees_px else z_all
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
            ffn_input = self.ln_corr(torch.cat([shifted_z, tok_emb], dim=-1))
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

        head_input = processed_x if self.head_sees_px else z
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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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

    def forward_sequential(self, idx, targets=None, seq_k=1):
        """Sequential evaluation: process positions one at a time.

        seq_k: number of times each position is processed through all D blocks.
        seq_k=1 is standard sequential inference.
        seq_k=2 means each position gets a second pass after its correction is refined.
        """
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        # Initialize position 0 to match parallel mode: corr_ffn(ln(zeros + tok_emb[0]))
        init_corr_0 = self.corr_ffn(self.ln_corr(torch.zeros(B, 1, C, device=tok_emb.device) + tok_emb[:, :1, :]))
        processed_x[:, 0, :] = tok_emb[:, 0, :] + init_corr_0.squeeze(1)
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            for _k in range(seq_k):
                # Run block on full sequence (causal attention handles masking)
                if self.d_block > 1:
                    h = processed_x
                    for block in self.blocks:
                        h = block(h)
                    z = h
                else:
                    z = self.block(processed_x)

                # Update correction for current position from its predecessor
                if t > 0:
                    add_input = self.ln_corr(z[:, t-1, :] + tok_emb[:, t, :])
                    correction_t = self.corr_ffn(add_input)
                    processed_x[:, t, :] = tok_emb[:, t, :] + correction_t

            # Final block pass after last correction update
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

        head_input = processed_x if self.head_sees_px else z_all
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

        head_input = processed_x if self.head_sees_px else z
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


# ---------------------------------------------------------------------------
# Variant 5e: corr_ffn tied to block FFN (12C² instead of 20C²)
# ---------------------------------------------------------------------------

class BlockHeadCorrFFNTiedModel(BlockHeadCorrFFNModel):
    """corr_ffn shares weights with the block's FFN. 12C² params (same as block_head).

    z = block(processed_x)
    correction = block.ffn(ln_corr(z))      ← same FFN as inside the block
    processed_x = tok_emb + shift(correction)
    head sees z
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight,
                         d_block=d_block, block_class=block_class, **kwargs)
        # Tie corr_ffn to the block's FFN (delete the separate one)
        del self.corr_ffn
        if d_block > 1:
            # Tie to last block's FFN
            self.corr_ffn = self.blocks[-1].ffn
        else:
            self.corr_ffn = self.block.ffn


class BlockHeadCorrFFNAddTiedModel(BlockHeadCorrFFNAddModel):
    """corr_ffn_add with FFN tied to block's FFN. 12C² params (same as block_head).

    z = block(processed_x)
    correction = block.ffn(ln_corr(shift(z) + tok_emb))   ← same FFN as inside the block
    processed_x = tok_emb + correction
    head sees z
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight,
                         d_block=d_block, block_class=block_class, **kwargs)
        # Tie corr_ffn to the block's FFN (delete the separate one)
        del self.corr_ffn
        if d_block > 1:
            self.corr_ffn = self.blocks[-1].ffn
        else:
            self.corr_ffn = self.block.ffn


# ---------------------------------------------------------------------------
# Variant 5f: pure residual pattern f(tok_emb, shift(z))
# ---------------------------------------------------------------------------

class BlockHeadCorrFFNAddPureModel(BlockHeadCorrFFNAddModel):
    """corr_ffn_add with f() residual pattern. 20C² params.

    z = block(processed_x)
    processed_x = tok_emb + shift(z) + corr_ffn(ln_corr(tok_emb + shift(z)))
    head sees z

    Unlike original corr_ffn_add which does processed_x = tok_emb + corr_ffn(ln_corr(shift(z) + tok_emb)),
    the pure version adds shift(z) as an explicit residual, matching f(x,c) = x + c + ffn(ln(x+c)).
    """

    def _run_iterations(self, tok_emb, n_iters):
        B, T, C = tok_emb.shape
        processed_x = tok_emb
        prev_processed_x = None
        z = None
        total_conv_loss = 0.0

        for k in range(n_iters):
            prev_processed_x = processed_x

            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)

            # f(tok_emb, shift(z)): residual pattern with separate corr_ffn/ln_corr
            processed_x = tok_emb + shifted_z + self.corr_ffn(self.ln_corr(tok_emb + shifted_z))

            if self.convergence_weight > 0 and self.training and k == n_iters - 1:
                total_conv_loss = F.mse_loss(
                    processed_x, prev_processed_x.detach()
                )

        return processed_x, z, total_conv_loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        # Initialize position 0 to match parallel mode: tok_emb[0] + zeros + corr_ffn(ln(tok_emb[0] + zeros))
        h0 = tok_emb[:, :1, :] + torch.zeros(B, 1, C, device=tok_emb.device)
        processed_x[:, 0, :] = (h0 + self.corr_ffn(self.ln_corr(h0))).squeeze(1)
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            z_all[:, t, :] = z[:, t, :]

            if t < T - 1:
                z_t = z[:, t, :]
                # f(tok_emb[t+1], z[t])
                h = tok_emb[:, t+1, :] + z_t
                processed_x[:, t+1, :] = h + self.corr_ffn(self.ln_corr(h))

        head_input = processed_x if self.head_sees_px else z_all
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
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

            # The "correction" for convergence tracking is the full delta
            correction = shifted_z + self.corr_ffn(self.ln_corr(tok_emb + shifted_z))
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

        head_input = processed_x if self.head_sees_px else z
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


class BlockHeadCorrFFNAddTiedPureModel(BlockHeadCorrFFNAddPureModel):
    """corr_ffn_add pure with FFN tied to block's FFN. 12C² params.

    z = block(processed_x)
    processed_x = tok_emb + shift(z) + block.ffn(block.ln2(tok_emb + shift(z)))
    head sees z

    Maximally consistent: the same ffn and ln2 are used both inside the block
    and for building the next processed_x.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, convergence_weight=0.0, d_block=1,
                 block_class=None, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         use_softmax=use_softmax, convergence_weight=convergence_weight,
                         d_block=d_block, block_class=block_class, **kwargs)
        # Tie corr_ffn and ln_corr to block's FFN and ln2
        del self.corr_ffn
        del self.ln_corr
        if d_block > 1:
            self.corr_ffn = self.blocks[-1].ffn
            self.ln_corr = self.blocks[-1].ln2
        else:
            self.corr_ffn = self.block.ffn
            self.ln_corr = self.block.ln2


# ---------------------------------------------------------------------------
# Variant 5g: block_head_recompute — shift delta, reapply block FFN with tok_emb
# ---------------------------------------------------------------------------

class BlockHeadRecomputeModel(SplitBlockLookAhead):
    """block_head variant that recomputes FFN at destination with tok_emb.

    z = block(processed_x)
    delta = z - processed_x                              # bounded block delta
    shifted_delta[t] = delta[t-1]                        # past context
    processed_x[t] = tok_emb[t] + shifted_delta[t] + block.ffn(block.ln2(tok_emb[t] + shifted_delta[t]))
    head sees z

    This is f(tok_emb, shifted_delta) where f(x,c) = x + c + ffn(ln2(x+c)).
    Token-aware (FFN sees tok_emb), contractive (shifts delta not z),
    12C² params (reuses block's FFN), 20C² FLOPs (FFN runs twice per iteration).

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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _get_block_and_last_ffn(self):
        """Returns the block(s) runner and last block's ffn/ln2 for recompute."""
        if self.d_block > 1:
            return self.blocks, self.blocks[-1].ffn, self.blocks[-1].ln2
        else:
            return self.block, self.block.ffn, self.block.ln2

    def _run_block(self, processed_x):
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            return h
        else:
            return self.block(processed_x)

    def _iteration_step(self, processed_x):
        z = self._run_block(processed_x)
        correction = z - processed_x
        return z, correction

    def _get_head_input(self, z):
        return z

    def _run_iterations(self, tok_emb, n_iters):
        B, T, C = tok_emb.shape
        _, ffn, ln2 = self._get_block_and_last_ffn()
        processed_x = tok_emb
        prev_processed_x = None
        z = None
        total_conv_loss = 0.0

        for k in range(n_iters):
            prev_processed_x = processed_x

            z = self._run_block(processed_x)
            delta = z - processed_x

            # Shift delta: position t gets delta from t-1
            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_delta = torch.cat([zero, delta[:, :-1, :]], dim=1)

            # f(tok_emb, shifted_delta) = tok_emb + shifted_delta + ffn(ln2(tok_emb + shifted_delta))
            h = tok_emb + shifted_delta
            processed_x = h + ffn(ln2(h))

            if self.convergence_weight > 0 and self.training and k == n_iters - 1:
                total_conv_loss = F.mse_loss(
                    processed_x, prev_processed_x.detach()
                )

        return processed_x, z, total_conv_loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        _, ffn, ln2 = self._get_block_and_last_ffn()

        processed_x = tok_emb.clone()
        # Initialize position 0 to match parallel mode: h = tok_emb[0] + zeros; processed_x = h + ffn(ln2(h))
        h0 = tok_emb[:, :1, :]
        processed_x[:, 0, :] = (h0 + ffn(ln2(h0))).squeeze(1)
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            z = self._run_block(processed_x)
            z_all[:, t, :] = z[:, t, :]

            if t < T - 1:
                delta_t = z[:, t, :] - processed_x[:, t, :]
                # f(tok_emb[t+1], delta[t])
                h = tok_emb[:, t+1, :] + delta_t
                processed_x[:, t+1, :] = h + ffn(ln2(h))

        head_input = processed_x if self.head_sees_px else z_all
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        _, ffn, ln2 = self._get_block_and_last_ffn()

        processed_x = tok_emb
        z = None
        prev_correction = None
        prev_prev_correction = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            z = self._run_block(processed_x)
            delta = z - processed_x

            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_delta = torch.cat([zero, delta[:, :-1, :]], dim=1)

            # The "correction" for tracking is shifted_delta + ffn contribution
            h = tok_emb + shifted_delta
            ffn_out = ffn(ln2(h))
            correction = shifted_delta + ffn_out
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

        head_input = processed_x if self.head_sees_px else z
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


# ---------------------------------------------------------------------------
# Variant 5h: block_head_recompute with separate FFN (20C² params, 20C² FLOPs)
# ---------------------------------------------------------------------------

class BlockHeadRecomputeSepModel(SplitBlockLookAhead):
    """block_head_recompute with a separate corr_ffn (not tied to block).

    z = block(processed_x)
    delta = z - processed_x
    shifted_delta[t] = delta[t-1]
    processed_x[t] = tok_emb[t] + shifted_delta[t] + corr_ffn(ln_corr(tok_emb[t] + shifted_delta[t]))
    head sees z

    Same f(tok_emb, shifted_delta) pattern as block_head_recompute, but with
    independent corr_ffn weights. 20C² params, 20C² FLOPs — same cost as corr_ffn_add.
    Tests whether the recompute pattern has merit when FFN can specialize.

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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
        self.corr_ffn = FeedForward(n_embed, dropout)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _run_block(self, processed_x):
        if self.d_block > 1:
            h = processed_x
            for block in self.blocks:
                h = block(h)
            return h
        else:
            return self.block(processed_x)

    def _iteration_step(self, processed_x):
        z = self._run_block(processed_x)
        correction = z - processed_x
        return z, correction

    def _get_head_input(self, z):
        return z

    def _run_iterations(self, tok_emb, n_iters):
        B, T, C = tok_emb.shape
        processed_x = tok_emb
        prev_processed_x = None
        z = None
        total_conv_loss = 0.0

        for k in range(n_iters):
            prev_processed_x = processed_x

            z = self._run_block(processed_x)
            delta = z - processed_x

            # Shift delta: position t gets delta from t-1
            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_delta = torch.cat([zero, delta[:, :-1, :]], dim=1)

            # f(tok_emb, shifted_delta) with separate corr_ffn
            h = tok_emb + shifted_delta
            processed_x = h + self.corr_ffn(self.ln_corr(h))

            if self.convergence_weight > 0 and self.training and k == n_iters - 1:
                total_conv_loss = F.mse_loss(
                    processed_x, prev_processed_x.detach()
                )

        return processed_x, z, total_conv_loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        # Initialize position 0 to match parallel mode: h = tok_emb[0] + zeros; processed_x = h + corr_ffn(ln_corr(h))
        h0 = tok_emb[:, :1, :]
        processed_x[:, 0, :] = (h0 + self.corr_ffn(self.ln_corr(h0))).squeeze(1)
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            z = self._run_block(processed_x)
            z_all[:, t, :] = z[:, t, :]

            if t < T - 1:
                delta_t = z[:, t, :] - processed_x[:, t, :]
                # f(tok_emb[t+1], delta[t]) with separate corr_ffn
                h = tok_emb[:, t+1, :] + delta_t
                processed_x[:, t+1, :] = h + self.corr_ffn(self.ln_corr(h))

        head_input = processed_x if self.head_sees_px else z_all
        logits = self.head(self.ln_f(head_input))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb
        z = None
        prev_correction = None
        prev_prev_correction = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            z = self._run_block(processed_x)
            delta = z - processed_x

            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_delta = torch.cat([zero, delta[:, :-1, :]], dim=1)

            h = tok_emb + shifted_delta
            ffn_out = self.corr_ffn(self.ln_corr(h))
            correction = shifted_delta + ffn_out
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

        head_input = processed_x if self.head_sees_px else z
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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
# Variant 5j: delta_ffn_add — shift delta, add tok_emb, FFN, no direct skip
# ---------------------------------------------------------------------------

class BlockHeadDeltaFFNAddModel(SplitBlockLookAhead):
    """block_head with token-aware FFN correction from shifted delta.

    z = block(processed_x)
    delta = z - processed_x
    shifted_delta[t] = delta[t-1]
    correction = corr_ffn(ln_corr(shifted_delta + tok_emb))
    processed_x = tok_emb + correction
    head sees z

    Like corr_ffn_add but with shift(delta) instead of shift(z).
    Delta extraction + FFN bottleneck + token-aware, no direct skip.
    20C² params, 20C² FLOPs.

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
                block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(d_block)
            ])
        else:
            self.block = block_class(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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

            # Delta extraction then shift
            delta = z - processed_x
            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_delta = torch.cat([zero, delta[:, :-1, :]], dim=1)

            # corr_ffn sees ln(shifted_delta + tok_emb) — no direct skip
            correction = self.corr_ffn(self.ln_corr(shifted_delta + tok_emb))

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

    def forward_sequential(self, idx, targets=None, seq_k=1):
        """Sequential evaluation: process positions one at a time."""
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape

        processed_x = tok_emb.clone()
        # Initialize position 0 to match parallel mode: corr_ffn(ln(zeros + tok_emb[0]))
        init_corr_0 = self.corr_ffn(self.ln_corr(torch.zeros(B, 1, C, device=tok_emb.device) + tok_emb[:, :1, :]))
        processed_x[:, 0, :] = tok_emb[:, 0, :] + init_corr_0.squeeze(1)
        z_all = torch.zeros_like(tok_emb)

        for t in range(T):
            for _k in range(seq_k):
                if self.d_block > 1:
                    h = processed_x
                    for block in self.blocks:
                        h = block(h)
                    z = h
                else:
                    z = self.block(processed_x)

                # Update correction for current position from its predecessor
                if t > 0:
                    delta_t_prev = z[:, t-1, :] - processed_x[:, t-1, :]
                    add_input = self.ln_corr(delta_t_prev + tok_emb[:, t, :])
                    correction_t = self.corr_ffn(add_input)
                    processed_x[:, t, :] = tok_emb[:, t, :] + correction_t

            # Final block pass after last correction update
            if self.d_block > 1:
                h = processed_x
                for block in self.blocks:
                    h = block(h)
                z = h
            else:
                z = self.block(processed_x)

            z_all[:, t, :] = z[:, t, :]

            # Correction for t+1: delta[t] = z[t] - processed_x[t], add tok_emb[t+1]
            if t < T - 1:
                delta_t = z[:, t, :] - processed_x[:, t, :]
                add_input = self.ln_corr(delta_t + tok_emb[:, t+1, :])
                correction_t1 = self.corr_ffn(add_input)
                processed_x[:, t+1, :] = tok_emb[:, t+1, :] + correction_t1

        head_input = processed_x if self.head_sees_px else z_all
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

            delta = z - processed_x
            zero = torch.zeros(B, 1, C, device=tok_emb.device)
            shifted_delta = torch.cat([zero, delta[:, :-1, :]], dim=1)
            correction = self.corr_ffn(self.ln_corr(shifted_delta + tok_emb))
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

        head_input = processed_x if self.head_sees_px else z
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


# ---------------------------------------------------------------------------
# Stacked Split-Block Models
# ---------------------------------------------------------------------------

class StackedSplitBlock(nn.Module):
    """Base class for stacked split-block look-ahead models.

    N units with separate weights, each iterated K times internally.
    Non-cumulative within each unit (reset to unit input).
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, d_block=1, k_min=0, n_head=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_units = n_units
        self.d_block = d_block
        self.k_min = k_min
        self.n_head = n_head
        total_divisor = n_units * d_block
        if n_layers % total_divisor != 0:
            raise ValueError(
                f"n_layers ({n_layers}) must be divisible by n_units * d_block ({n_units} * {d_block} = {total_divisor}). "
                f"Use n_layers = n_units * d_block * K."
            )
        self.k_iters = n_layers // total_divisor
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

        if self.training and self.k_min > 0:
            k_iters = random.randint(self.k_min, self.k_iters)
        else:
            k_iters = self.k_iters

        for i in range(self.n_units):
            h, y, conv_loss = self._run_unit(i, h, k_iters)
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

    def forward_sequential(self, idx, targets=None, seq_k=1):
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
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.attns = nn.ModuleList([
            RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.attns = nn.ModuleList([
            RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
    """Stacked: N units, each with standard block(s). Head sees z directly.

    Supports d_block > 1: each unit has D separate-weight blocks applied sequentially.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, d_block=1,
                 subtract_input=True, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         d_block=d_block, k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.subtract_input = subtract_input
        if d_block > 1:
            # Each unit gets D separate-weight blocks
            self.unit_blocks = nn.ModuleList([
                nn.ModuleList([
                    RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                    for _ in range(d_block)
                ])
                for _ in range(n_units)
            ])
        else:
            self.blocks = nn.ModuleList([
                RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                for _ in range(n_units)
            ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        if self.d_block > 1:
            h = processed_x
            for block in self.unit_blocks[unit_idx]:
                h = block(h)
            z = h
        else:
            z = self.blocks[unit_idx](processed_x)
        correction = z - processed_x if self.subtract_input else z
        return z, correction

    def _get_head_input(self, z):
        return z


class StackedBlockHeadCorrFFN(StackedSplitBlock):
    """Stacked: N units, each with block(s) + corr_ffn. Head sees z.

    Supports d_block > 1: each unit has D separate-weight blocks applied sequentially.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, d_block=1, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         d_block=d_block, k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        if d_block > 1:
            self.unit_blocks = nn.ModuleList([
                nn.ModuleList([
                    RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                    for _ in range(d_block)
                ])
                for _ in range(n_units)
            ])
        else:
            self.blocks = nn.ModuleList([
                RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
        if self.d_block > 1:
            h = processed_x
            for block in self.unit_blocks[unit_idx]:
                h = block(h)
            z = h
        else:
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
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.blocks = nn.ModuleList([
            RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
                 n_units, use_softmax=False, convergence_weight=0.0, d_block=1, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         d_block=d_block, k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        if d_block > 1:
            self.unit_blocks = nn.ModuleList([
                nn.ModuleList([
                    RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                    for _ in range(d_block)
                ])
                for _ in range(n_units)
            ])
        else:
            self.blocks = nn.ModuleList([
                RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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

    def _run_block(self, unit_idx, x):
        if self.d_block > 1:
            h = x
            for block in self.unit_blocks[unit_idx]:
                h = block(h)
            return h
        else:
            return self.blocks[unit_idx](x)

    def _unit_step(self, unit_idx, processed_x):
        # Not used directly — _run_unit is overridden
        z = self._run_block(unit_idx, processed_x)
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
            z = self._run_block(unit_idx, processed)

            zero = torch.zeros(B, 1, C, device=anchor.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)
            ffn_input = torch.cat([self.ln_corrs[unit_idx](shifted_z), anchor], dim=-1)
            correction = self.corr_ffns[unit_idx](ffn_input)
            processed = anchor + correction

        conv_loss = 0.0
        if self.convergence_weight > 0 and self.training and k_iters > 1:
            conv_loss = F.mse_loss(processed, prev_processed.detach())

        return processed, z, conv_loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.k_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        h = tok_emb

        last_z = None
        for i in range(self.n_units):
            anchor = h.clone()
            processed = h.clone()
            # Initialize position 0 to match parallel mode: corr_ffn(concat(ln_corr(zeros), anchor[0]))
            zero_0 = torch.zeros(B, 1, C, device=tok_emb.device)
            ffn_input_0 = torch.cat([self.ln_corrs[i](zero_0), anchor[:, :1, :]], dim=-1)
            init_corr_0 = self.corr_ffns[i](ffn_input_0)
            processed[:, 0, :] = anchor[:, 0, :] + init_corr_0.squeeze(1)

            for t in range(T):
                z = self._run_block(i, processed)
                if last_z is None and i == self.n_units - 1:
                    last_z = torch.zeros_like(tok_emb)
                if i == self.n_units - 1:
                    last_z[:, t, :] = z[:, t, :]

                if t < T - 1:
                    shifted_z_t1 = self.ln_corrs[i](z[:, t, :])
                    anc_t1 = anchor[:, t+1, :]
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
                z = self._run_block(i, processed)

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
# Stacked corr_ffn_add: N units, each with block + add corr_ffn
# ---------------------------------------------------------------------------

class StackedBlockHeadCorrFFNAdd(StackedSplitBlock):
    """Stacked add variant: N units, each with block + add corr_ffn.

    z = block(processed_x)
    correction = corr_ffn(ln_corr(shift(z) + anchor))
    processed_x = anchor + correction
    Head sees z from the last unit.

    Like concat variant but corr_ffn input is C (sum) instead of 2C (concat).
    20C² per unit vs 24C² for concat.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, d_block=1, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         d_block=d_block, k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        if d_block > 1:
            self.unit_blocks = nn.ModuleList([
                nn.ModuleList([
                    RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                    for _ in range(d_block)
                ])
                for _ in range(n_units)
            ])
        else:
            self.blocks = nn.ModuleList([
                RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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

    def _run_block(self, unit_idx, x):
        if self.d_block > 1:
            h = x
            for block in self.unit_blocks[unit_idx]:
                h = block(h)
            return h
        else:
            return self.blocks[unit_idx](x)

    def _unit_step(self, unit_idx, processed_x):
        z = self._run_block(unit_idx, processed_x)
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
            z = self._run_block(unit_idx, processed)

            zero = torch.zeros(B, 1, C, device=anchor.device)
            shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)
            correction = self.corr_ffns[unit_idx](self.ln_corrs[unit_idx](shifted_z + anchor))
            processed = anchor + correction

        conv_loss = 0.0
        if self.convergence_weight > 0 and self.training and k_iters > 1:
            conv_loss = F.mse_loss(processed, prev_processed.detach())

        return processed, z, conv_loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.k_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        h = tok_emb

        last_z = None
        for i in range(self.n_units):
            anchor = h.clone()
            processed = h.clone()
            # Initialize position 0 to match parallel mode: corr_ffn(ln_corr(zeros + anchor[0]))
            init_corr_0 = self.corr_ffns[i](self.ln_corrs[i](
                torch.zeros(B, 1, C, device=tok_emb.device) + anchor[:, :1, :]))
            processed[:, 0, :] = anchor[:, 0, :] + init_corr_0.squeeze(1)

            for t in range(T):
                z = self._run_block(i, processed)
                if last_z is None and i == self.n_units - 1:
                    last_z = torch.zeros_like(tok_emb)
                if i == self.n_units - 1:
                    last_z[:, t, :] = z[:, t, :]

                if t < T - 1:
                    add_input = self.ln_corrs[i](z[:, t, :] + anchor[:, t+1, :])
                    correction_t1 = self.corr_ffns[i](add_input)
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
                z = self._run_block(i, processed)

                zero = torch.zeros(B, 1, C, device=h.device)
                shifted_z = torch.cat([zero, z[:, :-1, :]], dim=1)
                correction = self.corr_ffns[i](self.ln_corrs[i](shifted_z + anchor))
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
# Stacked Sync Variants
# ---------------------------------------------------------------------------

class StackedAttnCorrFFNSync(StackedSplitBlock):
    """Stacked sync: N units, each with attn + corr_ffn.
    Head sees x + correction (synced), not h (attn output).
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.attns = nn.ModuleList([
            RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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
        h = processed_x + self.attns[unit_idx](self.ln1s[unit_idx](processed_x))
        correction = self.corr_ffns[unit_idx](self.ln2s[unit_idx](h))
        y = processed_x + correction  # sync: head sees x + correction
        return y, correction

    def _get_head_input(self, y):
        return y


class StackedJoFormerFixedSync(StackedSplitBlock):
    """Stacked sync with JoFormer Fixed attention."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.attns = nn.ModuleList([
            JoFormerFixedAttention(n_embed, block_size, dropout, use_softmax)
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
        h = processed_x + self.attns[unit_idx](self.ln1s[unit_idx](processed_x))
        correction = self.corr_ffns[unit_idx](self.ln2s[unit_idx](h))
        y = processed_x + correction
        return y, correction

    def _get_head_input(self, y):
        return y


class StackedJoFormerLearnedSync(StackedSplitBlock):
    """Stacked sync with JoFormer Learned attention.

    Per-token learned angles from angle_embedding_table, computed once per forward.
    Half-size token embedding + expander.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        # Override token embedding: half-size + expander
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)

        self.attns = nn.ModuleList([
            JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])
        self.corr_ffns = nn.ModuleList([
            FeedForward(n_embed, dropout) for _ in range(n_units)
        ])
        self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _compute_angles(self, idx):
        raw_angles = self.angle_embedding_table(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def _unit_step(self, unit_idx, processed_x):
        h = processed_x + self.attns[unit_idx](
            self.ln1s[unit_idx](processed_x), self._current_angles
        )
        correction = self.corr_ffns[unit_idx](self.ln2s[unit_idx](h))
        y = processed_x + correction
        return y, correction

    def _get_head_input(self, y):
        return y

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)
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
        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)
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

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.k_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)
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
        tok_emb = self.drop(self.expander(self.token_embedding_table(idx)))
        self._current_angles = self._compute_angles(idx)
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


class StackedJoFormerProjectedSync(StackedSplitBlock):
    """Stacked sync with JoFormer Projected attention.

    Per-unit angle_proj computes angles from unit input. No vector_proj.
    Uses JoFormerLearnedAttention internally.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, **kwargs):
        super().__init__(vocab_size, n_embed, n_layers, block_size, dropout,
                         n_units, use_softmax, convergence_weight,
                         k_min=kwargs.get('k_min', 0), n_head=kwargs.get('n_head', 1))
        self.attns = nn.ModuleList([
            JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
            for _ in range(n_units)
        ])
        self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])
        self.corr_ffns = nn.ModuleList([
            FeedForward(n_embed, dropout) for _ in range(n_units)
        ])
        self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])
        self.angle_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embed, 2 * n_embed),
                nn.GELU(),
                nn.Linear(2 * n_embed, n_embed // 2),
            )
            for _ in range(n_units)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _unit_step(self, unit_idx, processed_x):
        raw_angles = self.angle_projs[unit_idx](processed_x)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))

        h = processed_x + self.attns[unit_idx](
            self.ln1s[unit_idx](processed_x), angles
        )
        correction = self.corr_ffns[unit_idx](self.ln2s[unit_idx](h))
        y = processed_x + correction
        return y, correction

    def _get_head_input(self, y):
        return y


# ---------------------------------------------------------------------------
# Stacked Block-Aligned
# ---------------------------------------------------------------------------

class StackedBlockAligned(nn.Module):
    """Stacked block-aligned look-ahead: N units with separate weights.

    Each unit uses the block-aligned formula:
      f(x, c) = x + c + ffn(ln2(x + c))

    Within each unit (K iterations):
      - attn_corr = attn(ln1(processed_x))
      - processed_x = f(anchor, shift(attn_corr))   (past-only)
      - After K iters, unit output = processed_x (passed as anchor to next unit)

    Classifier sees f(final_processed_x, final_attn_corr) — standard block output.

    Each unit has D blocks (d_block parameter):
      - Blocks 1..D-1: full standard blocks
      - Block D: split (attn for shifting, ffn in f())
      - 12C² per D per unit
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 n_units, use_softmax=False, convergence_weight=0.0, d_block=1, k_min=0, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_units = n_units
        self.d_block = d_block
        self.k_min = k_min
        self.block_size = block_size
        self.convergence_weight = convergence_weight

        total_divisor = n_units * d_block
        if n_layers % total_divisor != 0:
            raise ValueError(
                f"n_layers ({n_layers}) must be divisible by n_units * d_block "
                f"({n_units} * {d_block} = {total_divisor}). "
                f"Use n_layers = n_units * d_block * K."
            )
        self.k_iters = n_layers // total_divisor
        self.n_iters = self.k_iters

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

        # Per-unit parameters
        if d_block > 1:
            # Blocks 1..D-1 per unit: full standard blocks
            self.unit_inner_blocks = nn.ModuleList([
                nn.ModuleList([
                    RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
                    for _ in range(d_block - 1)
                ])
                for _ in range(n_units)
            ])
        # Block D per unit: split attn + ffn
        self.attns = nn.ModuleList([
            RoFormerAttention(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
            for _ in range(n_units)
        ])
        self.ln1s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])
        self.ffns = nn.ModuleList([
            FeedForward(n_embed, dropout) for _ in range(n_units)
        ])
        self.ln2s = nn.ModuleList([nn.LayerNorm(n_embed) for _ in range(n_units)])

        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def _apply_f(self, unit_idx, x, attn_corr):
        """f(x, c) = x + c + ffn(ln2(x + c)) for unit unit_idx."""
        h = x + attn_corr
        return h + self.ffns[unit_idx](self.ln2s[unit_idx](h))

    def _run_inner_blocks(self, unit_idx, x):
        """Run blocks 1..D-1 for this unit. Identity if D=1."""
        if self.d_block > 1:
            for block in self.unit_inner_blocks[unit_idx]:
                x = block(x)
        return x

    def _get_attn_corr(self, unit_idx, x):
        """Run block D's attention for this unit."""
        return self.attns[unit_idx](self.ln1s[unit_idx](x))

    def _run_unit(self, unit_idx, anchor, k_iters):
        """Run one unit for k_iters iterations."""
        B, T, C = anchor.shape

        # Initial: raw anchor (bootstrapping step)
        processed_x = anchor

        prev_attn_corr = None
        attn_corr = None

        for k in range(k_iters):
            prev_attn_corr = attn_corr
            attn_corr = self._get_attn_corr(unit_idx, processed_x)

            if k < k_iters - 1:
                zero = torch.zeros(B, 1, C, device=anchor.device)
                shifted = torch.cat([zero, attn_corr[:, :-1, :]], dim=1)
                processed_x = self._apply_f(unit_idx, anchor, shifted)
                processed_x = self._run_inner_blocks(unit_idx, processed_x)

        conv_loss = 0.0
        if self.convergence_weight > 0 and self.training and k_iters > 1 and prev_attn_corr is not None:
            conv_loss = F.mse_loss(attn_corr, prev_attn_corr.detach())

        return processed_x, attn_corr, conv_loss

    def _classifier_input(self, unit_idx, processed_x, attn_corr):
        """Classifier sees f(processed_x, attn_corr) for the last unit."""
        return self._apply_f(unit_idx, processed_x, attn_corr)

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        h = tok_emb
        total_conv_loss = 0.0
        attn_corr = None
        last_processed_x = None

        if self.training and self.k_min > 0:
            k_iters = random.randint(self.k_min, self.k_iters)
        else:
            k_iters = self.k_iters

        for i in range(self.n_units):
            last_processed_x, attn_corr, conv_loss = self._run_unit(i, h, k_iters)
            total_conv_loss = total_conv_loss + conv_loss
            # Next unit's anchor = processed_x from this unit (with last shift applied)
            if i < self.n_units - 1:
                B, T, C = h.shape
                zero = torch.zeros(B, 1, C, device=h.device)
                shifted = torch.cat([zero, attn_corr[:, :-1, :]], dim=1)
                h = self._apply_f(i, h, shifted)
                h = self._run_inner_blocks(i, h)

        ci = self._classifier_input(self.n_units - 1, last_processed_x, attn_corr)
        logits = self.head(self.ln_f(ci))

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
        attn_corr = None
        last_processed_x = None

        for i in range(self.n_units):
            last_processed_x, attn_corr, _ = self._run_unit(i, h, K)
            if i < self.n_units - 1:
                B, T, C = h.shape
                zero = torch.zeros(B, 1, C, device=h.device)
                shifted = torch.cat([zero, attn_corr[:, :-1, :]], dim=1)
                h = self._apply_f(i, h, shifted)
                h = self._run_inner_blocks(i, h)

        ci = self._classifier_input(self.n_units - 1, last_processed_x, attn_corr)
        logits = self.head(self.ln_f(ci))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None, seq_k=1):
        if self.k_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        h = tok_emb

        classifier_inputs = torch.zeros(B, T, C, device=tok_emb.device)

        for i in range(self.n_units):
            anchor = h.clone()

            # Initial: raw anchor (bootstrapping step)
            processed_x = anchor.clone()

            for t in range(T):
                attn_corr = self._get_attn_corr(i, processed_x)

                # Classifier at position t (last unit only)
                if i == self.n_units - 1:
                    ci = self._classifier_input(i, processed_x, attn_corr)
                    classifier_inputs[:, t, :] = ci[:, t, :]

                # Set up t+1
                if t < T - 1:
                    h_next = anchor[:, t+1, :] + attn_corr[:, t, :]
                    new_px = h_next + self.ffns[i](self.ln2s[i](h_next))
                    processed_x[:, t+1, :] = new_px
                    processed_x = self._run_inner_blocks(i, processed_x)

            # Next unit's anchor: apply final shift
            if i < self.n_units - 1:
                attn_corr_final = self._get_attn_corr(i, processed_x)
                zero = torch.zeros(B, 1, C, device=h.device)
                shifted = torch.cat([zero, attn_corr_final[:, :-1, :]], dim=1)
                h = self._apply_f(i, anchor, shifted)
                h = self._run_inner_blocks(i, h)

        logits = self.head(self.ln_f(classifier_inputs))

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
        attn_corr = None
        last_processed_x = None

        for i in range(self.n_units):
            B, T, C = h.shape

            # Initial: raw anchor (bootstrapping step)
            processed_x = h

            prev_attn_corr = None
            attn_corr = None

            for k in range(self.k_iters):
                prev_attn_corr = attn_corr
                attn_corr = self._get_attn_corr(i, processed_x)

                if k < self.k_iters - 1:
                    zero = torch.zeros(B, 1, C, device=h.device)
                    shifted = torch.cat([zero, attn_corr[:, :-1, :]], dim=1)
                    processed_x = self._apply_f(i, h, shifted)
                    processed_x = self._run_inner_blocks(i, processed_x)

                corr_norm = attn_corr.norm(dim=-1).mean().item()
                all_norms.append(corr_norm)

                if prev_attn_corr is not None:
                    diff = (attn_corr - prev_attn_corr).norm(dim=-1).mean()
                    prev_diff = prev_attn_corr.norm(dim=-1).mean()
                    if prev_diff > 1e-8:
                        all_ratios.append((diff / prev_diff).item())

            last_processed_x = processed_x

            # Next unit's anchor
            if i < self.n_units - 1:
                zero = torch.zeros(B, 1, C, device=h.device)
                shifted = torch.cat([zero, attn_corr[:, :-1, :]], dim=1)
                h = self._apply_f(i, h, shifted)
                h = self._run_inner_blocks(i, h)

        ci = self._classifier_input(self.n_units - 1, last_processed_x, attn_corr)
        logits = self.head(self.ln_f(ci))

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


class StackedBlockAlignedLight(StackedBlockAligned):
    """Stacked block-aligned light: classifier sees processed_x + attn_corr (no extra FFN)."""

    def _classifier_input(self, unit_idx, processed_x, attn_corr):
        return processed_x + attn_corr


# ---------------------------------------------------------------------------
# RoFormer + Head FFN (baseline with extra FFN before head)
# ---------------------------------------------------------------------------

class RoFormerHeadFFN(nn.Module):
    """Standard roformer with an extra FFN before the classification head."""

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, n_head=1, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.n_head = n_head
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [RoFormerBlock(n_embed, block_size, dropout, use_softmax, n_head=self.n_head)
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

def _make_factory(cls, **fixed_kwargs):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, **kwargs):
        return cls(vocab_size, n_embed, n_layers, block_size, dropout,
                   use_softmax=use_softmax, **fixed_kwargs, **kwargs)
    return factory

def _make_stacked_factory(cls, **fixed_kwargs):
    def factory(vocab_size, n_embed, n_layers, block_size, dropout,
                use_softmax=False, n_units=None, **kwargs):
        if n_units is None:
            raise ValueError("n_units must be specified for stacked model")
        return cls(vocab_size, n_embed, n_layers, block_size, dropout,
                   n_units=n_units, use_softmax=use_softmax, **fixed_kwargs, **kwargs)
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
    'attn_corr_ffn_sync': _make_factory(AttnCorrFFNSyncModel),
    'block_aligned': _make_factory(BlockAlignedModel),
    'block_aligned_light': _make_factory(BlockAlignedLightModel),
    'block_aligned_pure': _make_factory(BlockAlignedPureModel),
    'joformer_fixed_sync': _make_factory(JoFormerFixedSyncModel),
    'joformer_learned_sync': _make_factory(JoFormerLearnedSyncModel),
    'joformer_projected_sync': _make_factory(JoFormerProjectedSyncModel),
    'attn_head_ffn': _make_factory(AttnHeadFFNModel),
    'block_head_ffn': _make_factory(BlockHeadFFNModel),
    'block_head': _make_factory(BlockHeadModel),
    'block_head_nosub': _make_factory(BlockHeadModel, subtract_input=False),
    'block_head_corr_ffn': _make_factory(BlockHeadCorrFFNModel),
    'block_head_corr_ffn_concat': _make_factory(BlockHeadCorrFFNConcatModel),
    'block_head_corr_ffn_add': _make_factory(BlockHeadCorrFFNAddModel),
    # NOTE: _px variants are duds. head sees processed_x instead of z — loses 4.2 PPL vs baseline.
    # Tested corr_ffn_add_px D=1 C=50 K=5: 86.82 vs corr_ffn_add 82.59. Do not use.
    'block_head_corr_ffn_px': _make_factory(BlockHeadCorrFFNModel, head_sees_px=True),
    'block_head_corr_ffn_concat_px': _make_factory(BlockHeadCorrFFNConcatModel, head_sees_px=True),
    'block_head_corr_ffn_add_px': _make_factory(BlockHeadCorrFFNAddModel, head_sees_px=True),
    'block_head_corr_ffn_tied': _make_factory(BlockHeadCorrFFNTiedModel),
    'block_head_corr_ffn_add_tied': _make_factory(BlockHeadCorrFFNAddTiedModel),
    'block_head_corr_ffn_add_pure': _make_factory(BlockHeadCorrFFNAddPureModel),
    'block_head_corr_ffn_add_tied_pure': _make_factory(BlockHeadCorrFFNAddTiedPureModel),
    'block_head_recompute': _make_factory(BlockHeadRecomputeModel),
    'block_head_recompute_sep': _make_factory(BlockHeadRecomputeSepModel),
    'block_head_delta_ffn': _make_factory(BlockHeadDeltaFFNModel),
    'block_head_delta_ffn_add': _make_factory(BlockHeadDeltaFFNAddModel),

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
    'stacked_block_head_nosub': _make_stacked_factory(StackedBlockHead, subtract_input=False),
    'stacked_block_head_ffn': _make_stacked_factory(StackedBlockHeadFFN),
    'stacked_block_head_corr_ffn': _make_stacked_factory(StackedBlockHeadCorrFFN),
    'stacked_block_head_delta_ffn': _make_stacked_factory(StackedBlockHeadDeltaFFN),
    'stacked_block_head_corr_ffn_concat': _make_stacked_factory(StackedBlockHeadCorrFFNConcat),
    'stacked_block_head_corr_ffn_add': _make_stacked_factory(StackedBlockHeadCorrFFNAdd),
    'stacked_attn_corr_ffn': _make_stacked_factory(StackedAttnCorrFFN),
    'stacked_attn_head_ffn': _make_stacked_factory(StackedAttnHeadFFN),
    'stacked_attn_corr_ffn_sync': _make_stacked_factory(StackedAttnCorrFFNSync),
    'stacked_joformer_fixed_sync': _make_stacked_factory(StackedJoFormerFixedSync),
    'stacked_joformer_learned_sync': _make_stacked_factory(StackedJoFormerLearnedSync),
    'stacked_joformer_projected_sync': _make_stacked_factory(StackedJoFormerProjectedSync),
    'stacked_block_aligned': _make_stacked_factory(StackedBlockAligned),
    'stacked_block_aligned_light': _make_stacked_factory(StackedBlockAlignedLight),
}

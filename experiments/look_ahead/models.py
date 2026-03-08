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
Look-ahead architecture models built on top of joformer block classes.

Wraps joformer blocks (RoFormerBlock, JoFormerFixedBlock, JoFormerLearnedBlock,
JoFormerProjectedBlock) with the look-ahead architecture:
  - Shared weights (one block repeated N times)
  - Non-cumulative corrections: x_k = x_0 + f(x_{k-1})
  - Past-only contextualization: position shift so position t depends on 0..t-1
  - Look-ahead concatenation: classification uses both past-only embedding
    and self-inclusive correction

Four variants via (non_cumulative, past_only) flags:
  look_ahead   - Non-cumulative + past-only (Model A in paper)
  baseline     - Cumulative + self-inclusive (Model B — shared-weight baseline)
  noncum_only  - Non-cumulative + self-inclusive (ablation)
  pastonly_only - Cumulative + past-only (ablation)
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
    RoFormerBlock, JoFormerFixedBlock, JoFormerLearnedBlock,
    JoFormerProjectedBlock,
    RoFormer, JoFormerFixed, JoFormerLearned, JoFormerProjected,
)


# ---------------------------------------------------------------------------
# Look-ahead model — wraps any joformer block with shared-weight iterations
# ---------------------------------------------------------------------------

class LookAheadModel(nn.Module):
    """Look-ahead model wrapping a joformer block class with shared weights.

    Uses a single block applied N times (shared weights). Supports all four
    (non_cumulative, past_only) variants. The block's built-in residual
    connection is factored out via correction = block(x) - x, and the
    residual strategy is controlled externally.

    For blocks that take only x: RoFormerBlock, JoFormerFixedBlock,
    JoFormerProjectedBlock. See LookAheadLearnedModel for JoFormerLearnedBlock.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 block_cls, non_cumulative=True, past_only=True,
                 use_softmax=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_iters = n_layers       # number of shared-weight iterations
        self.block_size = block_size
        self.non_cumulative = non_cumulative
        self.past_only = past_only

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

        # Single shared-weight block (same block reused N times)
        self.block = block_cls(n_embed, block_size, dropout, use_softmax)

        # Classification head
        head_in = 2 * n_embed if past_only else n_embed
        self.ln_f = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, vocab_size),
        )

    # ------------------------------------------------------------------
    # Internal helpers (override in subclasses for different block types)
    # ------------------------------------------------------------------

    def _get_embeddings(self, idx):
        """Compute token embeddings from indices."""
        return self.drop(self.token_embedding_table(idx))

    def _apply_block(self, x):
        """Apply the shared block. Returns block output with built-in residual."""
        return self.block(x)

    def _run_iterations(self, tok_emb, n_iters):
        """Run n_iters shared-weight iterations.

        Returns:
            processed_x: (B, T, C) — contextualized embeddings
            correction:  (B, T, C) — un-shifted correction from last iteration
        """
        B, T, C = tok_emb.shape
        processed_x = tok_emb
        correction = None

        for _ in range(n_iters):
            block_out = self._apply_block(processed_x)
            # Extract correction: block has built-in residual, so correction = out - in
            correction = block_out - processed_x

            if self.past_only:
                # Position shift: position t gets correction from position t-1
                zero = torch.zeros(B, 1, C, device=tok_emb.device)
                effective_correction = torch.cat(
                    [zero, correction[:, :-1, :]], dim=1
                )
            else:
                effective_correction = correction

            if self.non_cumulative:
                processed_x = tok_emb + effective_correction
            else:
                processed_x = processed_x + effective_correction

        return processed_x, correction

    def _build_output(self, processed_x, correction):
        """Build classification input from processed embeddings and correction."""
        if self.past_only and correction is not None:
            # Concatenate past-only embedding with self-inclusive look-ahead
            return torch.cat([processed_x, correction], dim=2)  # (B, T, 2C)
        return processed_x

    def _classify(self, output):
        """Classification head: LN -> FC -> GELU -> FC -> logits."""
        return self.head(self.ln_f(output))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, idx, targets=None):
        """Training forward pass. Returns (logits, loss)."""
        tok_emb = self._get_embeddings(idx)
        processed_x, correction = self._run_iterations(tok_emb, self.n_iters)
        output = self._build_output(processed_x, correction)
        logits = self._classify(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Full-depth autoregressive generation (N iterations per token)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    @torch.no_grad()
    def generate2(self, idx, max_new_tokens):
        """Single-step warm-started generation (look-ahead mode only).

        Uses the correction from the previous step as a warm start,
        requiring only one block evaluation per token.
        """
        if not (self.non_cumulative and self.past_only):
            return self.generate(idx, max_new_tokens)

        # Track effective correction (shifted) = processed_x - tok_emb
        eff_corr = None

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            tok_emb = self._get_embeddings(idx_cond)
            B, T, C = tok_emb.shape
            zero = torch.zeros(B, 1, C, device=tok_emb.device)

            if eff_corr is None:
                # Bootstrap: full-depth
                processed_x, correction = self._run_iterations(
                    tok_emb, self.n_iters
                )
                eff_corr = processed_x - tok_emb  # shifted correction
                output = self._build_output(processed_x, correction)
            else:
                # Warm start: reuse previous effective correction
                ec = eff_corr
                if ec.shape[1] >= T:
                    ec = ec[:, -T:, :]
                else:
                    pad = torch.zeros(
                        B, T - ec.shape[1], C, device=tok_emb.device
                    )
                    ec = torch.cat([ec, pad], dim=1)

                processed_x = tok_emb + ec  # warm-started

                # One block evaluation
                block_out = self._apply_block(processed_x)
                correction = block_out - processed_x

                # Save new effective correction for next step
                eff_corr = torch.cat(
                    [zero, correction[:, :-1, :]], dim=1
                )

                output = self._build_output(processed_x, correction)

            logits = self._classify(output)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    # ------------------------------------------------------------------
    # Experiment support methods
    # ------------------------------------------------------------------

    def forward_at_depth(self, idx, K, targets=None):
        """Evaluate at inference depth K (Section 4.5).

        Runs only K iterations instead of self.n_iters.
        """
        tok_emb = self._get_embeddings(idx)
        processed_x, correction = self._run_iterations(tok_emb, K)
        output = self._build_output(processed_x, correction)
        logits = self._classify(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        """Forward pass returning convergence diagnostics (Section 4.6).

        Returns:
            logits, loss, diagnostics_dict
        where diagnostics_dict contains:
            correction_norms: ||correction_k - correction_{k-1}|| at each iter
            contraction_ratios: successive ratio of correction_norms
        """
        tok_emb = self._get_embeddings(idx)
        B, T, C = tok_emb.shape

        processed_x = tok_emb
        prev_correction = None
        prev_prev_correction = None

        correction_norms = []
        contraction_ratios = []

        for k in range(self.n_iters):
            block_out = self._apply_block(processed_x)
            correction = block_out - processed_x

            if self.past_only:
                zero = torch.zeros(B, 1, C, device=tok_emb.device)
                effective_correction = torch.cat(
                    [zero, correction[:, :-1, :]], dim=1
                )
            else:
                effective_correction = correction

            if self.non_cumulative:
                processed_x = tok_emb + effective_correction
            else:
                processed_x = processed_x + effective_correction

            # Track convergence
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

        output = self._build_output(processed_x, correction)
        logits = self._classify(output)

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
    def generate_speculative(self, idx, max_new_tokens, draft_length=4):
        """Self-speculative multi-token generation (Section 4.4).

        Draft phase: generate draft_length tokens with single-step inference.
        Verify phase: run full-depth inference on drafted tokens.
        Accept/reject: accept longest matching prefix.
        """
        if not (self.non_cumulative and self.past_only):
            raise ValueError(
                "Self-speculative generation requires look-ahead model"
            )

        total_accepted = 0
        total_drafted = 0
        total_cycles = 0
        generated = 0

        while generated < max_new_tokens:
            k = min(draft_length, max_new_tokens - generated)

            # --- Draft phase: single-step generation ---
            draft_tokens = []
            draft_idx = idx.clone()
            eff_corr = None

            for i in range(k):
                idx_cond = draft_idx[:, -self.block_size:]
                tok_emb = self.token_embedding_table(idx_cond)
                B, T, C = tok_emb.shape
                zero = torch.zeros(B, 1, C, device=tok_emb.device)

                if eff_corr is None:
                    processed_x, correction = self._run_iterations(
                        tok_emb, self.n_iters
                    )
                    eff_corr = processed_x - tok_emb
                    output = self._build_output(processed_x, correction)
                else:
                    ec = eff_corr
                    if ec.shape[1] >= T:
                        ec = ec[:, -T:, :]
                    else:
                        pad = torch.zeros(
                            B, T - ec.shape[1], C, device=tok_emb.device
                        )
                        ec = torch.cat([ec, pad], dim=1)
                    processed_x = tok_emb + ec
                    block_out = self._apply_block(processed_x)
                    correction = block_out - processed_x
                    eff_corr = torch.cat(
                        [zero, correction[:, :-1, :]], dim=1
                    )
                    output = self._build_output(processed_x, correction)

                logits = self._classify(output)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(tok)
                draft_idx = torch.cat([draft_idx, tok], dim=1)

            # --- Verify phase: full-depth on drafted tokens ---
            verify_idx = torch.cat([idx] + draft_tokens, dim=1)
            verify_cond = verify_idx[:, -self.block_size:]
            verify_logits, _ = self(verify_cond)

            # Verifier predictions at draft positions
            n_draft = len(draft_tokens)
            verify_start = verify_logits.shape[1] - n_draft
            verified_tokens = []
            for j in range(n_draft):
                vprobs = F.softmax(
                    verify_logits[:, verify_start + j, :], dim=-1
                )
                vtok = torch.multinomial(vprobs, num_samples=1)
                verified_tokens.append(vtok)

            # --- Accept/reject: longest matching prefix ---
            n_accept = 0
            for j in range(n_draft):
                if (draft_tokens[j] == verified_tokens[j]).all():
                    n_accept += 1
                else:
                    break

            # Accept tokens
            for j in range(n_accept):
                idx = torch.cat([idx, draft_tokens[j]], dim=1)
                generated += 1
                if generated >= max_new_tokens:
                    break

            if generated < max_new_tokens:
                if n_accept < n_draft:
                    idx = torch.cat([idx, verified_tokens[n_accept]], dim=1)
                else:
                    last_probs = F.softmax(
                        verify_logits[:, -1, :], dim=-1
                    )
                    next_tok = torch.multinomial(last_probs, num_samples=1)
                    idx = torch.cat([idx, next_tok], dim=1)
                generated += 1

            total_accepted += n_accept
            total_drafted += n_draft
            total_cycles += 1

        stats = {
            'total_cycles': total_cycles,
            'total_drafted': total_drafted,
            'total_accepted': total_accepted,
            'acceptance_rate': total_accepted / max(total_drafted, 1),
            'tokens_per_cycle': generated / max(total_cycles, 1),
        }
        return idx, stats


# ---------------------------------------------------------------------------
# LookAheadLearnedModel — for JoFormerLearnedBlock (takes x and angles)
# ---------------------------------------------------------------------------

class LookAheadLearnedModel(LookAheadModel):
    """Look-ahead model for JoFormerLearnedBlock, which needs angle embeddings.

    JoFormerLearned uses separate token and angle embeddings. Angles are
    computed once from the input and reused across all shared-weight iterations.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 non_cumulative=True, past_only=True, use_softmax=False):
        # Initialize with JoFormerLearnedBlock
        super().__init__(
            vocab_size, n_embed, n_layers, block_size, dropout,
            JoFormerLearnedBlock, non_cumulative, past_only, use_softmax
        )
        # Override embeddings to match JoFormerLearned
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self._angles = None  # computed in _get_embeddings, used in _apply_block

    def _get_embeddings(self, idx):
        """Compute token embeddings and cache angles for block calls."""
        x = self.expander(self.token_embedding_table(idx))
        raw_angles = self.angle_embedding_table(idx)  # (B, T, C//2)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        self._angles = angles
        return self.drop(x)

    def _apply_block(self, x):
        """Apply block with cached angles."""
        return self.block(x, self._angles)


# ---------------------------------------------------------------------------
# Factory functions (matching joformer constructor interface)
# ---------------------------------------------------------------------------

# --- Look-ahead (Model A): non-cumulative + past-only ---

def make_roformer_look_ahead(vocab_size, n_embed, n_layers, block_size,
                              dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax)

def make_joformer_fixed_look_ahead(vocab_size, n_embed, n_layers, block_size,
                                    dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax)

def make_joformer_learned_look_ahead(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False):
    return LookAheadLearnedModel(vocab_size, n_embed, n_layers, block_size,
                                  dropout, non_cumulative=True, past_only=True,
                                  use_softmax=use_softmax)

def make_joformer_projected_look_ahead(vocab_size, n_embed, n_layers, block_size,
                                        dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlock, non_cumulative=True,
                          past_only=True, use_softmax=use_softmax)

# --- Baseline (Model B): cumulative + self-inclusive (shared weights) ---

def make_roformer_baseline(vocab_size, n_embed, n_layers, block_size,
                            dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=False, past_only=False,
                          use_softmax=use_softmax)

def make_joformer_fixed_baseline(vocab_size, n_embed, n_layers, block_size,
                                  dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=False, past_only=False,
                          use_softmax=use_softmax)

def make_joformer_learned_baseline(vocab_size, n_embed, n_layers, block_size,
                                    dropout, use_softmax=False):
    return LookAheadLearnedModel(vocab_size, n_embed, n_layers, block_size,
                                  dropout, non_cumulative=False, past_only=False,
                                  use_softmax=use_softmax)

def make_joformer_projected_baseline(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlock, non_cumulative=False,
                          past_only=False, use_softmax=use_softmax)

# --- Ablation: non-cumulative + self-inclusive ---

def make_joformer_fixed_noncum_only(vocab_size, n_embed, n_layers, block_size,
                                     dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=False,
                          use_softmax=use_softmax)

# --- Ablation: cumulative + past-only ---

def make_joformer_fixed_pastonly_only(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=False, past_only=True,
                          use_softmax=use_softmax)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    # Look-ahead variants (Model A: shared weights, non-cumulative, past-only)
    'roformer_look_ahead':           make_roformer_look_ahead,
    'joformer_fixed_look_ahead':     make_joformer_fixed_look_ahead,
    'joformer_learned_look_ahead':   make_joformer_learned_look_ahead,
    'joformer_projected_look_ahead': make_joformer_projected_look_ahead,

    # Baseline variants (Model B: shared weights, cumulative, self-inclusive)
    'roformer_baseline':             make_roformer_baseline,
    'joformer_fixed_baseline':       make_joformer_fixed_baseline,
    'joformer_learned_baseline':     make_joformer_learned_baseline,
    'joformer_projected_baseline':   make_joformer_projected_baseline,

    # Ablations (on joformer_fixed)
    'joformer_fixed_noncum_only':    make_joformer_fixed_noncum_only,
    'joformer_fixed_pastonly_only':   make_joformer_fixed_pastonly_only,

    # Original joformer models (separate blocks, for reference)
    'roformer':            RoFormer,
    'joformer_fixed':      JoFormerFixed,
    'joformer_learned':    JoFormerLearned,
    'joformer_projected':  JoFormerProjected,
}

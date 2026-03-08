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
    JoFormerLearnedAttention,
    RoFormer, JoFormerFixed, JoFormerLearned, JoFormerProjected,
    FeedForward,
    build_rotation_matrix, apply_rotation, apply_inverse_rotation,
)

# Windowed attention variants
from windowed_attention import (
    WindowedRoFormerBlock,
    WindowedJoFormerProjectedBlockCausal,
    WindowedRoFormer,
)


# ---------------------------------------------------------------------------
# JoFormerProjectedBlock with causal angle shift for look-ahead architecture
# ---------------------------------------------------------------------------

class JoFormerProjectedBlockCausal(nn.Module):
    """JoFormerProjectedBlock with causal angle shift.

    In standard JoFormerProjectedBlock, angles at position t are derived from
    x_t. This means Q_t is rotated by angle(x_t), so the attention score
    between position t and t-1 involves angle(x_t) — leaking current-token
    information into the "past-only" attention.

    Fix: shift the angles so position t uses angle(t-1). The query rotation
    at position t only involves information from past positions, making the
    block strictly causal/past-only.
    """
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

    def forward(self, x, return_raw_angles=False):
        x_proj = self.vector_proj(x)
        raw_angles = self.angle_proj(x)  # (B, T, C//2)

        # Shift angles: position t gets angle from position t-1
        # Position 0 gets zero angles (no past information)
        zero = torch.zeros_like(raw_angles[:, :1, :])
        shifted_angles = torch.cat([zero, raw_angles[:, :-1, :]], dim=1)

        # Apply flip-cumsum-flip to the shifted angles
        angles = torch.flip(shifted_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))

        x_proj = x_proj + self.sa_head(self.ln1(x_proj), angles)
        x_proj = x_proj + self.ffn(self.ln2(x_proj))
        if return_raw_angles:
            return x_proj, raw_angles
        return x_proj


# ---------------------------------------------------------------------------
# Vector Quantizer for discrete convergence
# ---------------------------------------------------------------------------


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
                 use_softmax=False, concat_head=True, mlp_head=False,
                 use_combiner=False,
                 convergence_weight=0.0, d_block=1, correct_rotation=False,
                 full_correction=False, no_self_embed=False, window_size=None,
                 correction_head=False, additive_head=False, proj_head=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.d_block = d_block         # number of layers per shared unit
        self.n_iters = n_layers // d_block  # number of shared-weight iterations
        self.block_size = block_size
        self.non_cumulative = non_cumulative
        self.past_only = past_only
        self.concat_head = concat_head
        self.correction_head = correction_head  # head uses correction[t] (self-inclusive, size C)
        self.additive_head = additive_head  # head uses processed_x[t] + correction[t]
        self.proj_head = proj_head  # head uses Linear(2C→C)([processed_x; correction])
        self.correct_rotation = correct_rotation
        self.full_correction = full_correction
        self.no_self_embed = no_self_embed  # don't add tok_emb back, just use shifted output

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

        # Shared-weight unit: D blocks with separate weights, iterated K times
        block_kwargs = dict(n_embed=n_embed, block_size=block_size,
                            dropout=dropout, use_softmax=use_softmax)
        if window_size is not None:
            block_kwargs['window_size'] = window_size
        self.blocks = nn.ModuleList([
            block_cls(**block_kwargs)
            for _ in range(d_block)
        ])

        # Optional learned combiner: f(correction, original) -> corrected
        # Replaces the additive processed_x = tok_emb + correction
        self.use_combiner = use_combiner
        if use_combiner:
            self.combiner = nn.Sequential(
                nn.Linear(2 * n_embed, 4 * n_embed),
                nn.GELU(),
                nn.Linear(4 * n_embed, n_embed),
            )

        self.convergence_weight = convergence_weight

        # Projection layer for proj_head: 2C → C
        if proj_head:
            self.head_proj = nn.Linear(2 * n_embed, n_embed)

        # Classification head
        head_in = 2 * n_embed if (past_only and concat_head) else n_embed
        self.ln_f = nn.LayerNorm(head_in)
        if mlp_head:
            self.head = nn.Sequential(
                nn.Linear(head_in, 4 * n_embed),
                nn.GELU(),
                nn.Linear(4 * n_embed, vocab_size),
            )
        else:
            self.head = nn.Linear(head_in, vocab_size)

    # ------------------------------------------------------------------
    # Internal helpers (override in subclasses for different block types)
    # ------------------------------------------------------------------

    def _get_embeddings(self, idx):
        """Compute token embeddings from indices."""
        return self.drop(self.token_embedding_table(idx))

    def _build_correction_rotation(self, T, C, device):
        """Build rotation matrix to apply to corrections before shifting.

        For fixed JoFormer: constant R(-freq) for all positions.
        V[t-1] appears in output at t rotated by R(Θ_{t-1} - Θ_t) = R(-freq_flipped),
        where Θ_t = t * freq_flipped is the cumsum angle. The correction inherits
        this rotation and must be matched when shifted to position t+1.
        Returns shape (1, 1, C//2, 2, 2) — broadcasts over B and T.
        """
        freq = torch.flip(torch.arange(C // 2, device=device), dims=(0,))
        angle = (-freq).float().unsqueeze(0).unsqueeze(0)  # (1, 1, C//2)
        return build_rotation_matrix(torch.cos(angle), torch.sin(angle))

    def _build_projected_rotation(self, raw_angles):
        """Build per-position rotation matrices for projected JoFormer correction.

        For projected JoFormer, angles are input-dependent. The angle difference
        between position t and t+1 is shifted_angles[t] = raw_angles[t-1] (t>=1)
        or 0 (t=0). We apply R(-shifted_angles[t]) to correction[t] before
        shifting it to position t+1.

        raw_angles: (B, T, C//2) from angle_proj.
        Returns: (B, T, C//2, 2, 2) rotation matrices.
        """
        B, T, half_C = raw_angles.shape
        zero = torch.zeros(B, 1, half_C, device=raw_angles.device)
        neg_shifted = torch.cat([zero, -raw_angles[:, :-1, :]], dim=1)  # (B, T, C//2)
        return build_rotation_matrix(torch.cos(neg_shifted), torch.sin(neg_shifted))

    def _has_projected_block(self):
        """Check if the last block has input-dependent angles (projected JoFormer)."""
        return hasattr(self.blocks[-1], 'angle_proj')

    def _apply_block(self, x, return_raw_angles=False):
        """Apply the shared unit and return the correction.

        full_correction=False (default):
            D=1: correction = block(x) - x
            D>1: blocks 1..D-1 run with standard residuals,
                 correction = last block's output - last block's input.
        full_correction=True:
            Run all D blocks with standard residuals.
            Return full output h (no subtraction).
            The network output IS the correction added to tok_emb.

        If return_raw_angles=True and the block supports it (JoFormerProjectedBlockCausal),
        also returns raw_angles from the last block for position-dependent rotation correction.
        """
        h = x
        raw_angles = None
        if self.full_correction:
            for block in self.blocks:
                if return_raw_angles and block is self.blocks[-1] and hasattr(block, 'angle_proj'):
                    h, raw_angles = block(h, return_raw_angles=True)
                else:
                    h = block(h)
            if return_raw_angles:
                return h, raw_angles
            return h
        else:
            for block in self.blocks[:-1]:
                h = block(h)  # standard residual blocks
            # Last block: extract delta only
            last_block = self.blocks[-1]
            if return_raw_angles and hasattr(last_block, 'angle_proj'):
                out, raw_angles = last_block(h, return_raw_angles=True)
                if return_raw_angles:
                    return out - h, raw_angles
            return self.blocks[-1](h) - h

    def _run_iterations(self, tok_emb, n_iters):
        """Run n_iters shared-weight iterations.

        Returns:
            processed_x: (B, T, C) — contextualized embeddings
            correction:  (B, T, C) — un-shifted correction from last iteration
            aux_loss:    scalar — convergence loss
        """
        B, T, C = tok_emb.shape
        processed_x = tok_emb
        prev_processed_x = None
        correction = None
        total_conv_loss = 0.0

        # Build correction rotation matrix (JoFormer blocks only)
        rot_matrix = None
        use_projected_rotation = self.correct_rotation and self._has_projected_block()
        if self.correct_rotation and not use_projected_rotation:
            rot_matrix = self._build_correction_rotation(T, C, tok_emb.device)

        for k in range(n_iters):
            prev_processed_x = processed_x

            if use_projected_rotation:
                correction, raw_angles = self._apply_block(processed_x, return_raw_angles=True)
                rot_matrix = self._build_projected_rotation(raw_angles)
            else:
                correction = self._apply_block(processed_x)

            if self.past_only:
                # Apply rotation to correction before shifting (JoFormer fix)
                corr_to_shift = correction
                if rot_matrix is not None:
                    corr_to_shift = apply_rotation(correction, rot_matrix)

                # Position shift: position t gets correction from position t-1
                zero = torch.zeros(B, 1, C, device=tok_emb.device)
                effective_correction = torch.cat(
                    [zero, corr_to_shift[:, :-1, :]], dim=1
                )
            else:
                effective_correction = correction

            if self.non_cumulative:
                if self.use_combiner:
                    # Learned combination: f(original, correction) -> corrected
                    processed_x = self.combiner(
                        torch.cat([effective_correction, tok_emb], dim=-1)
                    )
                elif self.no_self_embed:
                    processed_x = effective_correction
                else:
                    processed_x = tok_emb + effective_correction
            else:
                processed_x = processed_x + effective_correction

            # Convergence loss: push last iteration output toward previous
            if self.convergence_weight > 0 and self.training and k == n_iters - 1:
                total_conv_loss = F.mse_loss(
                    processed_x, prev_processed_x.detach()
                )

        aux_loss = total_conv_loss
        return processed_x, correction, aux_loss

    def _build_output(self, processed_x, correction):
        """Build classification input from processed embeddings and correction."""
        if self.past_only and correction is not None:
            if self.concat_head:
                return torch.cat([processed_x, correction], dim=2)  # (B, T, 2C)
            if self.correction_head:
                return correction  # (B, T, C) — self-inclusive
            if self.additive_head:
                return processed_x + correction  # (B, T, C) — zero extra params
            if self.proj_head:
                cat = torch.cat([processed_x, correction], dim=2)  # (B, T, 2C)
                return self.head_proj(cat)  # (B, T, C)
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
        processed_x, correction, aux_loss = self._run_iterations(tok_emb, self.n_iters)
        output = self._build_output(processed_x, correction)
        logits = self._classify(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            if aux_loss > 0:
                loss = loss + self.convergence_weight * aux_loss
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
    def generate2(self, idx, max_new_tokens, prime_tokens=None):
        """Single-step warm-started generation (look-ahead mode only).

        Primes the first `prime_tokens` tokens at full depth N so the
        correction cache is fully built up. After priming, each subsequent
        token requires only one block evaluation.

        Args:
            prime_tokens: number of tokens to generate at full depth before
                switching to K=1. Defaults to n_iters.
        """
        if not (self.non_cumulative and self.past_only):
            return self.generate(idx, max_new_tokens)

        if prime_tokens is None:
            prime_tokens = self.n_iters

        # Track effective correction (shifted) = processed_x - tok_emb
        eff_corr = None

        for i in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            tok_emb = self._get_embeddings(idx_cond)
            B, T, C = tok_emb.shape
            zero = torch.zeros(B, 1, C, device=tok_emb.device)

            if i < prime_tokens:
                # Priming phase: full-depth iterations
                processed_x, correction, _ = self._run_iterations(
                    tok_emb, self.n_iters
                )
                eff_corr = processed_x - tok_emb
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
                    ec = torch.cat([pad, ec], dim=1)

                if self.use_combiner:
                    processed_x = self.combiner(
                        torch.cat([ec, tok_emb], dim=-1)
                    )
                else:
                    processed_x = tok_emb + ec  # warm-started

                # One block evaluation
                correction = self._apply_block(processed_x)

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
        processed_x, correction, _ = self._run_iterations(tok_emb, K)
        output = self._build_output(processed_x, correction)
        logits = self._classify(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None):
        """Sequential evaluation: process positions one at a time.

        Each position sees already-contextualized previous positions.
        This matches autoregressive deployment when K>1 during training
        (the unit learned to handle contextualized inputs).

        For K=1 (e.g. D=N), the unit only saw raw tok_emb during training,
        so sequential buildup is invalid — falls back to parallel K=1.

        Cost: T block evaluations.
        """
        if self.n_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self._get_embeddings(idx)
        B, T, C = tok_emb.shape

        # Build correction rotation matrix (JoFormer blocks only)
        rot_matrix = None
        use_projected_rotation = self.correct_rotation and self._has_projected_block()
        if self.correct_rotation and not use_projected_rotation:
            rot_matrix = self._build_correction_rotation(T, C, tok_emb.device)

        processed_x = tok_emb.clone()
        correction = torch.zeros_like(tok_emb)

        for t in range(T):
            if use_projected_rotation:
                corr, raw_angles = self._apply_block(processed_x, return_raw_angles=True)
            else:
                corr = self._apply_block(processed_x)
            correction[:, t, :] = corr[:, t, :]

            # Past-only shift: correction at t updates position t+1
            if t < T - 1:
                corr_t = correction[:, t, :]
                if use_projected_rotation and t >= 1:
                    # Position-dependent rotation: R(-raw_angles[t-1])
                    angle_t = -raw_angles[:, t - 1, :]  # (B, C//2)
                    rot_t = build_rotation_matrix(
                        torch.cos(angle_t).unsqueeze(1),
                        torch.sin(angle_t).unsqueeze(1)
                    )  # (B, 1, C//2, 2, 2)
                    corr_t = apply_rotation(
                        corr_t.unsqueeze(1), rot_t
                    ).squeeze(1)
                elif rot_matrix is not None:
                    corr_t = apply_rotation(
                        corr_t.unsqueeze(1), rot_matrix
                    ).squeeze(1)

                if self.use_combiner:
                    combo_in = torch.cat([
                        corr_t.unsqueeze(1),
                        tok_emb[:, t+1:t+2, :]
                    ], dim=-1)
                    processed_x[:, t+1:t+2, :] = self.combiner(combo_in)
                elif self.no_self_embed:
                    processed_x[:, t+1, :] = corr_t
                else:
                    processed_x[:, t+1, :] = tok_emb[:, t+1, :] + corr_t

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

        # Build correction rotation matrix (JoFormer blocks only)
        rot_matrix = None
        use_projected_rotation = self.correct_rotation and self._has_projected_block()
        if self.correct_rotation and not use_projected_rotation:
            rot_matrix = self._build_correction_rotation(T, C, tok_emb.device)

        for k in range(self.n_iters):
            if use_projected_rotation:
                correction, raw_angles = self._apply_block(processed_x, return_raw_angles=True)
                rot_matrix = self._build_projected_rotation(raw_angles)
            else:
                correction = self._apply_block(processed_x)

            if self.past_only:
                corr_to_shift = correction
                if rot_matrix is not None:
                    corr_to_shift = apply_rotation(correction, rot_matrix)

                zero = torch.zeros(B, 1, C, device=tok_emb.device)
                effective_correction = torch.cat(
                    [zero, corr_to_shift[:, :-1, :]], dim=1
                )
            else:
                effective_correction = correction

            if self.non_cumulative:
                if self.use_combiner:
                    processed_x = self.combiner(
                        torch.cat([effective_correction, tok_emb], dim=-1)
                    )
                elif self.no_self_embed:
                    processed_x = effective_correction
                else:
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
                    processed_x, correction, _ = self._run_iterations(
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
                    correction = self._apply_block(processed_x)
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
                 non_cumulative=True, past_only=True, use_softmax=False,
                 concat_head=True, mlp_head=False, **kwargs):
        # Initialize with JoFormerLearnedBlock
        super().__init__(
            vocab_size, n_embed, n_layers, block_size, dropout,
            JoFormerLearnedBlock, non_cumulative, past_only, use_softmax,
            concat_head, mlp_head, **kwargs
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

        # Causal angle shift: position t uses angle from position t-1
        if self.past_only:
            zero = torch.zeros_like(raw_angles[:, :1, :])
            raw_angles = torch.cat([zero, raw_angles[:, :-1, :]], dim=1)

        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        self._angles = angles
        return self.drop(x)

    def _apply_block(self, x):
        """Apply block with cached angles, return correction (delta)."""
        h = x
        for block in self.blocks[:-1]:
            h = block(h, self._angles)
        return self.blocks[-1](h, self._angles) - h


# ---------------------------------------------------------------------------
# Stacked Look-Ahead Model
# ---------------------------------------------------------------------------

class StackedLookAheadModel(nn.Module):
    """Stacked look-ahead model: N units with separate weights, each iterated K times.

    Each unit has its own shared-weight block. Within each unit, K iterations
    run with non-cumulative corrections anchored to the unit's input (not tok_emb).
    Between units, representations flow cumulatively (standard residual stacking).

    At inference K=1: equivalent to a standard N-layer transformer.
    During training: N*K effective depth.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 block_cls, n_units, use_softmax=False,
                 convergence_weight=0.0, full_correction=False,
                 concat_head=False, correction_head=False,
                 additive_head=False, proj_head=False, window_size=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_units = n_units
        self.k_iters = n_layers // n_units  # iterations per unit
        self.block_size = block_size
        self.convergence_weight = convergence_weight
        self.full_correction = full_correction
        self.concat_head = concat_head
        self.correction_head = correction_head
        self.additive_head = additive_head
        self.proj_head = proj_head

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

        # N units, each with its own block (separate weights between units)
        block_kwargs = dict(n_embed=n_embed, block_size=block_size,
                            dropout=dropout, use_softmax=use_softmax)
        if window_size is not None:
            block_kwargs['window_size'] = window_size
        self.units = nn.ModuleList([
            block_cls(**block_kwargs)
            for _ in range(n_units)
        ])

        # Projection layer for proj_head: 2C → C
        if proj_head:
            self.head_proj = nn.Linear(2 * n_embed, n_embed)

        head_in = 2 * n_embed if concat_head else n_embed
        self.ln_f = nn.LayerNorm(head_in)
        self.head = nn.Linear(head_in, vocab_size)

    def _apply_unit(self, block, x):
        """Apply one block and return correction."""
        if self.full_correction:
            return block(x)
        else:
            return block(x) - x

    def _run_unit(self, block, anchor, k_iters):
        """Run K iterations of one unit with non-cumulative corrections."""
        B, T, C = anchor.shape
        processed = anchor
        prev_processed = None

        for k in range(k_iters):
            prev_processed = processed
            correction = self._apply_unit(block, processed)

            # Past-only shift: position t gets correction from t-1
            zero = torch.zeros(B, 1, C, device=anchor.device)
            shifted = torch.cat([zero, correction[:, :-1, :]], dim=1)

            # Non-cumulative: reset to unit input
            processed = anchor + shifted

        # Convergence loss for this unit
        conv_loss = 0.0
        if self.convergence_weight > 0 and self.training and k_iters > 1:
            conv_loss = F.mse_loss(processed, prev_processed.detach())

        return processed, correction, conv_loss

    def _build_output(self, processed, correction):
        if correction is not None:
            if self.concat_head:
                return torch.cat([processed, correction], dim=2)
            if self.correction_head:
                return correction
            if self.additive_head:
                return processed + correction
            if self.proj_head:
                cat = torch.cat([processed, correction], dim=2)
                return self.head_proj(cat)
        return processed

    def forward(self, idx, targets=None):
        tok_emb = self.drop(self.token_embedding_table(idx))
        h = tok_emb
        total_conv_loss = 0.0
        correction = None

        for block in self.units:
            h, correction, conv_loss = self._run_unit(block, h, self.k_iters)
            total_conv_loss = total_conv_loss + conv_loss

        output = self._build_output(h, correction)
        logits = self.head(self.ln_f(output))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            if total_conv_loss > 0:
                loss = loss + self.convergence_weight * total_conv_loss
        return logits, loss

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
        """At K=1, each unit is just one block — standard transformer generation."""
        return self.generate(idx, max_new_tokens)

    def forward_at_depth(self, idx, K, targets=None):
        """Evaluate with K iterations per unit."""
        tok_emb = self.drop(self.token_embedding_table(idx))
        h = tok_emb
        correction = None

        for block in self.units:
            h, correction, _ = self._run_unit(block, h, K)

        output = self._build_output(h, correction)
        logits = self.head(self.ln_f(output))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_sequential(self, idx, targets=None):
        """Sequential evaluation: process positions one at a time per unit.

        For each unit, processes positions left-to-right so each position
        sees contextualized previous positions from that unit.
        """
        if self.k_iters <= 1:
            return self.forward_at_depth(idx, 1, targets)

        tok_emb = self.drop(self.token_embedding_table(idx))
        B, T, C = tok_emb.shape
        h = tok_emb

        last_corr = None
        for block in self.units:
            anchor = h.clone()
            processed = h.clone()

            for t in range(T):
                corr = self._apply_unit(block, processed)
                if t < T - 1:
                    zero_pad = torch.zeros(B, 1, C, device=h.device)
                    shifted = torch.cat([zero_pad, corr[:, :-1, :]], dim=1)
                    processed[:, t+1, :] = anchor[:, t+1, :] + shifted[:, t+1, :]

            last_corr = corr
            h = processed

        output = self._build_output(h, last_corr)
        logits = self.head(self.ln_f(output))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def forward_with_diagnostics(self, idx, targets=None):
        """Forward pass with convergence diagnostics (per-unit)."""
        tok_emb = self.drop(self.token_embedding_table(idx))
        h = tok_emb
        all_ratios = []
        all_norms = []

        for block in self.units:
            B, T, C = h.shape
            anchor = h
            processed = anchor
            prev_correction = None
            unit_ratios = []

            for k in range(self.k_iters):
                correction = self._apply_unit(block, processed)

                zero = torch.zeros(B, 1, C, device=h.device)
                shifted = torch.cat([zero, correction[:, :-1, :]], dim=1)
                processed = anchor + shifted

                corr_norm = correction.norm(dim=-1).mean().item()
                all_norms.append(corr_norm)

                if prev_correction is not None:
                    diff = (correction - prev_correction).norm(dim=-1).mean()
                    prev_diff = prev_correction.norm(dim=-1).mean()
                    if prev_diff > 1e-8:
                        unit_ratios.append((diff / prev_diff).item())
                prev_correction = correction

            last_corr = correction
            h = processed
            all_ratios.extend(unit_ratios)

        output = self._build_output(h, last_corr)
        logits = self.head(self.ln_f(output))

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
# Factory functions (matching joformer constructor interface)
# ---------------------------------------------------------------------------
#
# Model factory guide:
#
# LOOK-AHEAD VARIANTS (shared weights, non-cumulative, past-only):
#   correction = block(x) - x (delta extraction)
#   processed_x[t] = tok_emb[t] + correction[t-1]
#
#   {block}_look_ahead           — concat head (2C -> vocab), correction cat'd with processed_x
#   {block}_look_ahead_nocat     — linear head (C -> vocab), uses processed_x only
#   {block}_look_ahead_mlp       — MLP head (C -> 4C -> vocab), uses processed_x only
#
#   block types: roformer, joformer_fixed, joformer_learned, joformer_projected
#
# FULL CORRECTION VARIANT:
#   No delta subtraction — network output h is used directly as correction.
#   processed_x[t] = tok_emb[t] + h[t-1]   (h includes input due to residuals)
#
#   roformer_look_ahead_nocat_full — linear head, full correction
#
# PAST-ONLY BASELINE:
#   Like full correction but without adding tok_emb back.
#   processed_x[t] = h[t-1]   (position t gets only past network output)
#
#   roformer_look_ahead_nocat_pastonly — linear head, no self-embed
#
# ROTATION-CORRECTED VARIANTS (JoFormer only):
#   Pre-rotates correction by R(-raw_angle) before shifting to fix
#   rotation frame mismatch between positions.
#
#   joformer_fixed_look_ahead[_nocat|_mlp]_corrected
#
# BASELINES (shared weights, cumulative, self-inclusive):
#   Standard shared-weight iteration: x_k = x_{k-1} + f(x_{k-1})
#   No past-only shift — position t sees its own correction.
#
#   {block}_baseline
#
# ABLATIONS:
#   joformer_fixed_noncum_only   — non-cumulative but NO past-only shift
#   joformer_fixed_pastonly_only  — past-only shift but cumulative
#
# STANDARD TRANSFORMERS (separate weights per layer, no look-ahead):
#   roformer, joformer_fixed, joformer_learned, joformer_projected
#
# ---------------------------------------------------------------------------

# --- Look-ahead (Model A): non-cumulative + past-only ---

def make_roformer_look_ahead(vocab_size, n_embed, n_layers, block_size,
                              dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, **kwargs)

def make_roformer_look_ahead_nocat(vocab_size, n_embed, n_layers, block_size,
                                    dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, **kwargs)

def make_roformer_look_ahead_corrhead(vocab_size, n_embed, n_layers, block_size,
                                       dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False,
                          correction_head=True, **kwargs)

def make_roformer_look_ahead_addhead(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False,
                          additive_head=True, **kwargs)

def make_roformer_look_ahead_projhead(vocab_size, n_embed, n_layers, block_size,
                                       dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False,
                          proj_head=True, **kwargs)

def make_roformer_look_ahead_mlp(vocab_size, n_embed, n_layers, block_size,
                                  dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, mlp_head=True, **kwargs)

def make_roformer_look_ahead_nocat_full(vocab_size, n_embed, n_layers, block_size,
                                        dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False,
                          full_correction=True, **kwargs)

def make_roformer_look_ahead_nocat_pastonly(vocab_size, n_embed, n_layers, block_size,
                                            dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False,
                          full_correction=True, no_self_embed=True, **kwargs)

def make_joformer_fixed_look_ahead(vocab_size, n_embed, n_layers, block_size,
                                    dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, **kwargs)

def make_joformer_fixed_look_ahead_nocat(vocab_size, n_embed, n_layers, block_size,
                                          dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, **kwargs)

def make_joformer_fixed_look_ahead_corrhead(vocab_size, n_embed, n_layers, block_size,
                                              dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False,
                          correction_head=True, **kwargs)

def make_joformer_fixed_look_ahead_mlp(vocab_size, n_embed, n_layers, block_size,
                                        dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, mlp_head=True, **kwargs)

def make_joformer_learned_look_ahead(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False, **kwargs):
    return LookAheadLearnedModel(vocab_size, n_embed, n_layers, block_size,
                                  dropout, non_cumulative=True, past_only=True,
                                  use_softmax=use_softmax, **kwargs)

def make_joformer_learned_look_ahead_nocat(vocab_size, n_embed, n_layers, block_size,
                                            dropout, use_softmax=False, **kwargs):
    return LookAheadLearnedModel(vocab_size, n_embed, n_layers, block_size,
                                  dropout, non_cumulative=True, past_only=True,
                                  use_softmax=use_softmax, concat_head=False, **kwargs)

def make_joformer_learned_look_ahead_corrhead(vocab_size, n_embed, n_layers, block_size,
                                                dropout, use_softmax=False, **kwargs):
    return LookAheadLearnedModel(vocab_size, n_embed, n_layers, block_size,
                                  dropout, non_cumulative=True, past_only=True,
                                  use_softmax=use_softmax, concat_head=False,
                                  correction_head=True, **kwargs)

def make_joformer_learned_look_ahead_mlp(vocab_size, n_embed, n_layers, block_size,
                                          dropout, use_softmax=False, **kwargs):
    return LookAheadLearnedModel(vocab_size, n_embed, n_layers, block_size,
                                  dropout, non_cumulative=True, past_only=True,
                                  use_softmax=use_softmax, concat_head=False, mlp_head=True, **kwargs)

def make_joformer_projected_look_ahead(vocab_size, n_embed, n_layers, block_size,
                                        dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlockCausal, non_cumulative=True,
                          past_only=True, use_softmax=use_softmax, **kwargs)

def make_joformer_projected_look_ahead_nocat(vocab_size, n_embed, n_layers, block_size,
                                              dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlockCausal, non_cumulative=True,
                          past_only=True, use_softmax=use_softmax, concat_head=False, **kwargs)

def make_joformer_projected_look_ahead_corrhead(vocab_size, n_embed, n_layers, block_size,
                                                  dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlockCausal, non_cumulative=True,
                          past_only=True, use_softmax=use_softmax, concat_head=False,
                          correction_head=True, **kwargs)

def make_joformer_projected_look_ahead_mlp(vocab_size, n_embed, n_layers, block_size,
                                            dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlockCausal, non_cumulative=True,
                          past_only=True, use_softmax=use_softmax, concat_head=False, mlp_head=True, **kwargs)

# --- Look-ahead with rotation-corrected shift (JoFormer blocks only) ---

def make_joformer_fixed_look_ahead_corrected(vocab_size, n_embed, n_layers, block_size,
                                              dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, correct_rotation=True, **kwargs)

def make_joformer_fixed_look_ahead_nocat_corrected(vocab_size, n_embed, n_layers, block_size,
                                                    dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, correct_rotation=True, **kwargs)

def make_joformer_fixed_look_ahead_mlp_corrected(vocab_size, n_embed, n_layers, block_size,
                                                  dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, mlp_head=True, correct_rotation=True, **kwargs)

def make_joformer_projected_look_ahead_corrected(vocab_size, n_embed, n_layers, block_size,
                                                  dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlockCausal, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, correct_rotation=True, **kwargs)

def make_joformer_projected_look_ahead_nocat_corrected(vocab_size, n_embed, n_layers, block_size,
                                                        dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlockCausal, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, correct_rotation=True, **kwargs)

def make_joformer_projected_look_ahead_mlp_corrected(vocab_size, n_embed, n_layers, block_size,
                                                      dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlockCausal, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, mlp_head=True, correct_rotation=True, **kwargs)

# --- Windowed attention variants ---

def make_roformer_look_ahead_nocat_windowed(vocab_size, n_embed, n_layers, block_size,
                                             dropout, use_softmax=False, window_size=64, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          WindowedRoFormerBlock, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False,
                          window_size=window_size, **kwargs)

def make_joformer_projected_look_ahead_nocat_corrected_windowed(vocab_size, n_embed, n_layers, block_size,
                                                                 dropout, use_softmax=False, window_size=64, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          WindowedJoFormerProjectedBlockCausal, non_cumulative=True, past_only=True,
                          use_softmax=use_softmax, concat_head=False, correct_rotation=True,
                          window_size=window_size, **kwargs)

# --- Baseline (Model B): cumulative + self-inclusive (shared weights) ---

def make_roformer_baseline(vocab_size, n_embed, n_layers, block_size,
                            dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          RoFormerBlock, non_cumulative=False, past_only=False,
                          use_softmax=use_softmax, **kwargs)

def make_joformer_fixed_baseline(vocab_size, n_embed, n_layers, block_size,
                                  dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=False, past_only=False,
                          use_softmax=use_softmax, **kwargs)

def make_joformer_learned_baseline(vocab_size, n_embed, n_layers, block_size,
                                    dropout, use_softmax=False, **kwargs):
    return LookAheadLearnedModel(vocab_size, n_embed, n_layers, block_size,
                                  dropout, non_cumulative=False, past_only=False,
                                  use_softmax=use_softmax, **kwargs)

def make_joformer_projected_baseline(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerProjectedBlock, non_cumulative=False,
                          past_only=False, use_softmax=use_softmax, **kwargs)

# --- Stacked look-ahead (N units x K iterations) ---

def make_roformer_stacked_look_ahead_nocat(vocab_size, n_embed, n_layers, block_size,
                                            dropout, use_softmax=False, n_units=None,
                                            **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 RoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, **kwargs)

def make_roformer_stacked_look_ahead(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False, n_units=None,
                                      **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 RoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, concat_head=True, **kwargs)

def make_roformer_stacked_look_ahead_corrhead(vocab_size, n_embed, n_layers, block_size,
                                                dropout, use_softmax=False, n_units=None,
                                                **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 RoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, correction_head=True, **kwargs)

def make_roformer_stacked_look_ahead_projhead(vocab_size, n_embed, n_layers, block_size,
                                                dropout, use_softmax=False, n_units=None,
                                                **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 RoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, proj_head=True, **kwargs)

def make_roformer_stacked_look_ahead_addhead(vocab_size, n_embed, n_layers, block_size,
                                               dropout, use_softmax=False, n_units=None,
                                               **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 RoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, additive_head=True, **kwargs)

def make_roformer_stacked_look_ahead_windowed(vocab_size, n_embed, n_layers, block_size,
                                               dropout, use_softmax=False, n_units=None,
                                               window_size=64, **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 WindowedRoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, concat_head=True,
                                 window_size=window_size, **kwargs)

def make_roformer_stacked_look_ahead_corrhead_windowed(vocab_size, n_embed, n_layers, block_size,
                                                        dropout, use_softmax=False, n_units=None,
                                                        window_size=64, **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 WindowedRoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, correction_head=True,
                                 window_size=window_size, **kwargs)

def make_roformer_stacked_look_ahead_addhead_windowed(vocab_size, n_embed, n_layers, block_size,
                                                       dropout, use_softmax=False, n_units=None,
                                                       window_size=64, **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 WindowedRoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, additive_head=True,
                                 window_size=window_size, **kwargs)

def make_roformer_stacked_look_ahead_projhead_windowed(vocab_size, n_embed, n_layers, block_size,
                                                        dropout, use_softmax=False, n_units=None,
                                                        window_size=64, **kwargs):
    if n_units is None:
        raise ValueError("n_units must be specified for stacked look-ahead model")
    return StackedLookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                                 WindowedRoFormerBlock, n_units=n_units,
                                 use_softmax=use_softmax, proj_head=True,
                                 window_size=window_size, **kwargs)

# --- Ablation: non-cumulative + self-inclusive ---

def make_joformer_fixed_noncum_only(vocab_size, n_embed, n_layers, block_size,
                                     dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=True, past_only=False,
                          use_softmax=use_softmax, **kwargs)

# --- Ablation: cumulative + past-only ---

def make_joformer_fixed_pastonly_only(vocab_size, n_embed, n_layers, block_size,
                                      dropout, use_softmax=False, **kwargs):
    return LookAheadModel(vocab_size, n_embed, n_layers, block_size, dropout,
                          JoFormerFixedBlock, non_cumulative=False, past_only=True,
                          use_softmax=use_softmax, **kwargs)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    # Look-ahead variants (Model A: shared weights, non-cumulative, past-only)
    'roformer_look_ahead':                make_roformer_look_ahead,
    'roformer_look_ahead_nocat':          make_roformer_look_ahead_nocat,
    'roformer_look_ahead_mlp':            make_roformer_look_ahead_mlp,
    'roformer_look_ahead_nocat_full':     make_roformer_look_ahead_nocat_full,
    'roformer_look_ahead_nocat_pastonly': make_roformer_look_ahead_nocat_pastonly,
    'joformer_fixed_look_ahead':          make_joformer_fixed_look_ahead,
    'joformer_fixed_look_ahead_nocat':    make_joformer_fixed_look_ahead_nocat,
    'joformer_fixed_look_ahead_mlp':      make_joformer_fixed_look_ahead_mlp,
    'joformer_learned_look_ahead':        make_joformer_learned_look_ahead,
    'joformer_learned_look_ahead_nocat':  make_joformer_learned_look_ahead_nocat,
    'joformer_learned_look_ahead_mlp':    make_joformer_learned_look_ahead_mlp,
    'joformer_projected_look_ahead':      make_joformer_projected_look_ahead,
    'joformer_projected_look_ahead_nocat': make_joformer_projected_look_ahead_nocat,
    'joformer_projected_look_ahead_mlp':  make_joformer_projected_look_ahead_mlp,

    # Look-ahead with rotation-corrected shift (JoFormer only)
    'joformer_fixed_look_ahead_corrected':          make_joformer_fixed_look_ahead_corrected,
    'joformer_fixed_look_ahead_nocat_corrected':    make_joformer_fixed_look_ahead_nocat_corrected,
    'joformer_fixed_look_ahead_mlp_corrected':      make_joformer_fixed_look_ahead_mlp_corrected,
    'joformer_projected_look_ahead_corrected':      make_joformer_projected_look_ahead_corrected,
    'joformer_projected_look_ahead_nocat_corrected': make_joformer_projected_look_ahead_nocat_corrected,
    'joformer_projected_look_ahead_mlp_corrected':  make_joformer_projected_look_ahead_mlp_corrected,

    # Windowed attention variants
    'roformer_look_ahead_nocat_windowed': make_roformer_look_ahead_nocat_windowed,
    'joformer_projected_look_ahead_nocat_corrected_windowed': make_joformer_projected_look_ahead_nocat_corrected_windowed,

    # Baseline variants (Model B: shared weights, cumulative, self-inclusive)
    'roformer_baseline':             make_roformer_baseline,
    'joformer_fixed_baseline':       make_joformer_fixed_baseline,
    'joformer_learned_baseline':     make_joformer_learned_baseline,
    'joformer_projected_baseline':   make_joformer_projected_baseline,

    # Stacked look-ahead (N units x K iterations per unit)
    'roformer_stacked_look_ahead_nocat': make_roformer_stacked_look_ahead_nocat,
    'roformer_stacked_look_ahead': make_roformer_stacked_look_ahead,
    'roformer_stacked_look_ahead_corrhead': make_roformer_stacked_look_ahead_corrhead,
    'roformer_stacked_look_ahead_projhead': make_roformer_stacked_look_ahead_projhead,
    'roformer_stacked_look_ahead_addhead': make_roformer_stacked_look_ahead_addhead,
    'roformer_stacked_look_ahead_windowed': make_roformer_stacked_look_ahead_windowed,
    'roformer_stacked_look_ahead_corrhead_windowed': make_roformer_stacked_look_ahead_corrhead_windowed,
    'roformer_stacked_look_ahead_addhead_windowed': make_roformer_stacked_look_ahead_addhead_windowed,
    'roformer_stacked_look_ahead_projhead_windowed': make_roformer_stacked_look_ahead_projhead_windowed,

    # Head variants (different ways to combine processed_x and correction)
    'roformer_look_ahead_corrhead': make_roformer_look_ahead_corrhead,
    'joformer_fixed_look_ahead_corrhead': make_joformer_fixed_look_ahead_corrhead,
    'joformer_learned_look_ahead_corrhead': make_joformer_learned_look_ahead_corrhead,
    'joformer_projected_look_ahead_corrhead': make_joformer_projected_look_ahead_corrhead,
    'roformer_look_ahead_addhead': make_roformer_look_ahead_addhead,
    'roformer_look_ahead_projhead': make_roformer_look_ahead_projhead,

    # Ablations (on joformer_fixed)
    'joformer_fixed_noncum_only':    make_joformer_fixed_noncum_only,
    'joformer_fixed_pastonly_only':   make_joformer_fixed_pastonly_only,

    # Original joformer models (separate blocks, for reference)
    'roformer':            RoFormer,
    'joformer_fixed':      JoFormerFixed,
    'joformer_learned':    JoFormerLearned,
    'joformer_projected':  JoFormerProjected,

    # Windowed standalone models
    'roformer_windowed':   lambda vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=False, window_size=64, **kwargs: WindowedRoFormer(vocab_size, n_embed, n_layers, block_size, dropout, use_softmax=use_softmax, window_size=window_size),
}

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Tokenizer


class CharCompositionalEmbedding(nn.Module):
    """Derives BPE token embeddings from character-level embeddings via
    algebraic composition: (a, A) . (b, B) = (a + Ab, AB).

    Character vectors are tied to the single-byte token rows of wte.
    Only rotation angles are separate learnable parameters.

    Uses block-diagonal 2x2 rotation matrices so composition reduces to
    angle accumulation: R_c1 * R_c2 * ... * R_ck -> R(theta_c1 + ... + theta_ck).
    """

    def __init__(self, vocab_size: int = 50257, embed_dim: int = 768,
                 n_chars: int = 256, tokenizer_name: str = "gpt2"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_chars = n_chars
        self.n_blocks = embed_dim // 2  # 384 blocks of 2x2

        # Only angles are learnable here; char vectors come from wte
        self.char_angles = nn.Embedding(n_chars, self.n_blocks)
        nn.init.normal_(self.char_angles.weight, std=0.02)

        # Build token-to-byte mapping from the GPT-2 tokenizer
        self._build_token_byte_mapping(tokenizer_name)

    def _build_token_byte_mapping(self, tokenizer_name: str):
        """Precompute mapping from each token ID to its byte sequence.

        Also builds byte_to_token_id (ties chars to wte rows) and
        multi_token_ids (multi-char tokens sorted by length for efficient batching).
        """
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

        max_token_len = 0
        byte_sequences = []

        for token_id in range(self.vocab_size):
            token_bytes = tokenizer.decode([token_id]).encode("utf-8")
            byte_sequences.append(list(token_bytes))
            max_token_len = max(max_token_len, len(token_bytes))

        padded = torch.zeros(self.vocab_size, max_token_len, dtype=torch.long)
        lengths = torch.zeros(self.vocab_size, dtype=torch.long)
        byte_to_token = torch.zeros(self.n_chars, dtype=torch.long)
        multi_ids = []

        for token_id, seq in enumerate(byte_sequences):
            lengths[token_id] = len(seq)
            for i, b in enumerate(seq):
                padded[token_id, i] = b
            if len(seq) == 1:
                byte_to_token[seq[0]] = token_id
            else:
                multi_ids.append(token_id)

        # Sort multi-char token IDs by length for efficient batching
        multi_ids_tensor = torch.tensor(multi_ids, dtype=torch.long)
        multi_lengths = lengths[multi_ids_tensor]
        sort_order = torch.argsort(multi_lengths)
        multi_ids_sorted = multi_ids_tensor[sort_order]

        self.register_buffer("token_bytes", padded)
        self.register_buffer("token_lengths", lengths)
        self.register_buffer("byte_to_token_id", byte_to_token)
        self.register_buffer("multi_token_ids", multi_ids_sorted)
        self.max_token_len = max_token_len

    def _apply_rotation(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Apply block-diagonal 2x2 rotation to x using angles.

        Args:
            x: (..., embed_dim) vector to rotate
            angles: (..., n_blocks) rotation angles

        Returns:
            Rotated vector of shape (..., embed_dim)
        """
        cos_a = torch.cos(angles)  # (..., n_blocks)
        sin_a = torch.sin(angles)  # (..., n_blocks)

        x_even = x[..., 0::2]  # (..., n_blocks)
        x_odd = x[..., 1::2]   # (..., n_blocks)

        out_even = cos_a * x_even - sin_a * x_odd
        out_odd = sin_a * x_even + cos_a * x_odd

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def _compose_by_ids(self, token_ids: torch.Tensor, wte: nn.Embedding) -> torch.Tensor:
        """Compose embeddings for given token IDs using wte for char vectors."""
        device = wte.weight.device
        n = token_ids.shape[0]

        result = torch.zeros(n, self.embed_dim, device=device)
        cum_angles = torch.zeros(n, self.n_blocks, device=device)

        chunk_bytes = self.token_bytes[token_ids]
        chunk_lengths = self.token_lengths[token_ids]
        max_len = chunk_lengths.max().item()

        for pos in range(max_len):
            mask = (pos < chunk_lengths).float()
            char_ids = chunk_bytes[:, pos]

            # Character vectors come from wte via byte_to_token_id
            v = wte.weight[self.byte_to_token_id[char_ids]]
            theta = self.char_angles(char_ids)

            rotated_v = self._apply_rotation(v, cum_angles)

            result = result + rotated_v * mask.unsqueeze(1)
            cum_angles = cum_angles + theta * mask.unsqueeze(1)

        return result

    def sample_compose_loss(self, wte: nn.Embedding, sample_size: int = 1024) -> torch.Tensor:
        """Sample multi-char tokens (contiguous block, sorted by length),
        compose them, return MSE vs wte entries."""
        n_multi = self.multi_token_ids.shape[0]
        if sample_size > n_multi:
            sample_size = n_multi

        # Sample a contiguous block from length-sorted list (minimal masking)
        start = torch.randint(0, n_multi - sample_size, (1,)).item()
        sampled_ids = self.multi_token_ids[start:start + sample_size]

        composed = self._compose_by_ids(sampled_ids, wte)
        table = wte.weight[sampled_ids]

        return nn.functional.mse_loss(table, composed)

    def sample_prediction_loss(self, wte: nn.Embedding, sample_size: int = 1024) -> torch.Tensor:
        """Control auxiliary loss: next-token prediction on mini-sequences
        built from tokens and their characters.

        For each sampled multi-char token, builds two mini-sequences:
          chars→token: [char_tok_1, char_tok_2, ..., token_id]
          token→chars: [token_id, char_tok_1, char_tok_2, ...]
        and computes cross-entropy next-token prediction loss.
        """
        n_multi = self.multi_token_ids.shape[0]
        if sample_size > n_multi:
            sample_size = n_multi

        start = torch.randint(0, n_multi - sample_size, (1,)).item()
        sampled_ids = self.multi_token_ids[start:start + sample_size]

        device = wte.weight.device
        chunk_bytes = self.token_bytes[sampled_ids]       # (N, max_len)
        chunk_lengths = self.token_lengths[sampled_ids]    # (N,)
        max_len = chunk_lengths.max().item()

        # Convert byte IDs to token IDs
        char_token_ids = self.byte_to_token_id[chunk_bytes[:, :max_len]]  # (N, max_len)

        # Build chars→token sequences: [c1, c2, ..., cn, token]
        # seq length = max_len + 1
        seq_len = max_len + 1
        c2t_seqs = torch.zeros(sample_size, seq_len, dtype=torch.long, device=device)
        c2t_mask = torch.zeros(sample_size, seq_len, dtype=torch.bool, device=device)

        for i in range(max_len):
            valid = (i < chunk_lengths)
            c2t_seqs[:, i] = torch.where(valid, char_token_ids[:, i], torch.zeros_like(sampled_ids))
            c2t_mask[:, i] = valid

        # Token goes at position = chunk_length (right after last char)
        for i in range(sample_size):
            pos = chunk_lengths[i].item()
            c2t_seqs[i, pos] = sampled_ids[i]
            c2t_mask[i, pos] = True

        # Build token→chars sequences: [token, c1, c2, ..., cn]
        t2c_seqs = torch.zeros(sample_size, seq_len, dtype=torch.long, device=device)
        t2c_mask = torch.zeros(sample_size, seq_len, dtype=torch.bool, device=device)
        t2c_seqs[:, 0] = sampled_ids
        t2c_mask[:, 0] = True

        for i in range(max_len):
            valid = (i < chunk_lengths)
            t2c_seqs[:, i + 1] = torch.where(valid, char_token_ids[:, i], torch.zeros_like(sampled_ids))
            t2c_mask[:, i + 1] = valid

        # Combine both directions
        all_seqs = torch.cat([c2t_seqs, t2c_seqs], dim=0)    # (2N, seq_len)
        all_mask = torch.cat([c2t_mask, t2c_mask], dim=0)     # (2N, seq_len)

        # Process in chunks to avoid OOM on the (2N, seq_len, vocab) logits
        chunk_size = 256
        total_loss = 0.0
        total_valid = 0

        for c_start in range(0, all_seqs.shape[0], chunk_size):
            c_end = min(c_start + chunk_size, all_seqs.shape[0])
            chunk_seqs = all_seqs[c_start:c_end]
            chunk_mask = all_mask[c_start:c_end]

            embeds = wte(chunk_seqs)
            logits = embeds @ wte.weight.T

            pred_logits = logits[:, :-1, :].contiguous()
            targets = chunk_seqs[:, 1:].contiguous()
            t_mask = chunk_mask[:, 1:]

            loss_per_pos = nn.functional.cross_entropy(
                pred_logits.view(-1, self.vocab_size),
                targets.view(-1),
                reduction='none'
            ).view(targets.shape)

            total_loss = total_loss + (loss_per_pos * t_mask.float()).sum()
            total_valid += t_mask.sum().item()

        return total_loss / max(total_valid, 1)

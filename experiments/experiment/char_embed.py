import torch
import torch.nn as nn
from transformers import GPT2Tokenizer


class CharCompositionalEmbedding(nn.Module):
    """Derives BPE token embeddings from character-level embeddings via
    algebraic composition: (a, A) . (b, B) = (a + Ab, AB).

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

        # Learnable character-level parameters
        self.char_vectors = nn.Embedding(n_chars, embed_dim)
        self.char_angles = nn.Embedding(n_chars, self.n_blocks)

        # Initialize: small vectors, small angles
        nn.init.normal_(self.char_vectors.weight, std=0.02)
        nn.init.normal_(self.char_angles.weight, std=0.02)

        # Build token-to-byte mapping from the GPT-2 tokenizer
        self._build_token_byte_mapping(tokenizer_name)

    def _build_token_byte_mapping(self, tokenizer_name: str):
        """Precompute mapping from each token ID to its byte sequence."""
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

        max_token_len = 0
        byte_sequences = []

        for token_id in range(self.vocab_size):
            # Decode token to bytes
            token_bytes = tokenizer.decode([token_id]).encode("utf-8")
            byte_sequences.append(list(token_bytes))
            max_token_len = max(max_token_len, len(token_bytes))

        # Pad all sequences to max_token_len
        # Use 0 as padding (will be masked out)
        padded = torch.zeros(self.vocab_size, max_token_len, dtype=torch.long)
        lengths = torch.zeros(self.vocab_size, dtype=torch.long)

        for token_id, seq in enumerate(byte_sequences):
            lengths[token_id] = len(seq)
            for i, b in enumerate(seq):
                padded[token_id, i] = b

        self.register_buffer("token_bytes", padded)
        self.register_buffer("token_lengths", lengths)
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

    def _compose_chunk(self, start: int, end: int) -> torch.Tensor:
        """Compose embeddings for a chunk of tokens [start, end)."""
        device = self.char_vectors.weight.device
        chunk_size = end - start

        result = torch.zeros(chunk_size, self.embed_dim, device=device)
        cum_angles = torch.zeros(chunk_size, self.n_blocks, device=device)

        chunk_bytes = self.token_bytes[start:end]       # (chunk, max_len)
        chunk_lengths = self.token_lengths[start:end]    # (chunk,)

        for pos in range(self.max_token_len):
            mask = (pos < chunk_lengths).float()         # (chunk,)
            char_ids = chunk_bytes[:, pos]               # (chunk,)

            v = self.char_vectors(char_ids)              # (chunk, embed_dim)
            theta = self.char_angles(char_ids)           # (chunk, n_blocks)

            rotated_v = self._apply_rotation(v, cum_angles)

            result = result + rotated_v * mask.unsqueeze(1)
            cum_angles = cum_angles + theta * mask.unsqueeze(1)

        return result

    def compose_all_tokens(self, chunk_size: int = 8192) -> torch.Tensor:
        """Compose embeddings for all tokens from character-level parameters.

        Processes in chunks to limit peak memory usage.

        Returns:
            (vocab_size, embed_dim) tensor of token embeddings.
        """
        chunks = []
        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            chunks.append(self._compose_chunk(start, end))
        return torch.cat(chunks, dim=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up composed embeddings for input token IDs.

        Args:
            input_ids: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, embed_dim) embeddings
        """
        # Compose all token embeddings (recomputed each forward pass for gradients)
        composed = self.compose_all_tokens()  # (vocab_size, embed_dim)
        return composed[input_ids]

    def get_output_embeddings(self) -> torch.Tensor:
        """Return the composed embedding matrix for use as output projection."""
        return self.compose_all_tokens()

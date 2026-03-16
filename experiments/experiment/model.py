import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

from .char_embed import CharCompositionalEmbedding
from .config import ExperimentConfig


def create_model(config: ExperimentConfig) -> nn.Module:
    """Create a GPT-2 model with either standard or compositional embeddings."""
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.block_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        resid_pdrop=config.dropout,
        embd_pdrop=config.dropout,
        attn_pdrop=config.dropout,
    )

    if config.model_type == "baseline":
        model = GPT2LMHeadModel(gpt2_config)
        return model
    elif config.model_type == "char_compose":
        model = CharComposeGPT2(gpt2_config, config)
        return model
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


class CharComposeGPT2(nn.Module):
    """GPT-2 with character-compositional embeddings replacing the standard
    token embedding table. Positional embeddings are kept unchanged."""

    def __init__(self, gpt2_config: GPT2Config, exp_config: ExperimentConfig):
        super().__init__()
        self.config = gpt2_config

        # Build the base GPT-2 model
        self.transformer = GPT2LMHeadModel(gpt2_config)

        # Replace the token embedding with our compositional module
        self.char_embed = CharCompositionalEmbedding(
            vocab_size=exp_config.vocab_size,
            embed_dim=exp_config.n_embd,
            n_chars=exp_config.n_chars,
        )

        # Remove the original embedding table and lm_head to save memory
        self.transformer.transformer.wte = None
        self.transformer.lm_head = None

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass with composed embeddings for both input and output.

        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target token IDs for loss computation

        Returns:
            CausalLMOutput-like object with loss and logits.
        """
        device = input_ids.device

        # Compose all token embeddings once per forward pass
        composed_matrix = self.char_embed.compose_all_tokens()  # (vocab, embed_dim)

        # Input embeddings: lookup + positional
        inputs_embeds = composed_matrix[input_ids]  # (batch, seq, embed_dim)

        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        position_embeds = self.transformer.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.transformer.transformer.drop(hidden_states)

        # Pass through transformer blocks
        for block in self.transformer.transformer.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

        hidden_states = self.transformer.transformer.ln_f(hidden_states)

        # Output projection: tie weights with composed embeddings
        logits = hidden_states @ composed_matrix.T  # (batch, seq, vocab)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

    def parameters_count(self):
        """Return a dict of parameter counts by component."""
        char_params = sum(p.numel() for p in self.char_embed.parameters())
        transformer_params = sum(
            p.numel() for name, p in self.transformer.named_parameters()
            if p.requires_grad and "wte" not in name and "lm_head" not in name
        )
        return {
            "char_embed_params": char_params,
            "transformer_params": transformer_params,
            "total_params": char_params + transformer_params,
        }


def count_parameters(model: nn.Module) -> dict:
    """Count parameters for any model."""
    if isinstance(model, CharComposeGPT2):
        return model.parameters_count()

    total = sum(p.numel() for p in model.parameters())
    embed_params = model.transformer.wte.weight.numel()
    return {
        "embedding_params": embed_params,
        "transformer_params": total - embed_params,
        "total_params": total,
    }

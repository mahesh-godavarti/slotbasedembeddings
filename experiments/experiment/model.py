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
    elif config.model_type == "predict_control":
        model = PredictControlGPT2(gpt2_config, config)
        return model
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


class CharComposeGPT2(nn.Module):
    """GPT-2 with standard embedding table + character-compositional regularization.

    The forward pass uses a normal embedding lookup (fast). A regularization loss
    forces the embedding table to match character-composed embeddings, computed on
    a random subset of tokens each step."""

    def __init__(self, gpt2_config: GPT2Config, exp_config: ExperimentConfig):
        super().__init__()
        self.config = gpt2_config
        self.compose_sample_size = 1024  # tokens to sample for composition loss
        self.compose_weight = 1.0  # weight of composition loss

        # Build the base GPT-2 model (keeps wte and lm_head intact)
        self.transformer = GPT2LMHeadModel(gpt2_config)

        # Character composition module (for regularization only)
        self.char_embed = CharCompositionalEmbedding(
            vocab_size=exp_config.vocab_size,
            embed_dim=exp_config.n_embd,
            n_chars=exp_config.n_chars,
        )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass with standard embeddings + composition regularization.

        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target token IDs for loss computation

        Returns:
            dict with loss and logits.
        """
        # Standard GPT-2 forward (fast lookup)
        outputs = self.transformer(input_ids, labels=labels)
        lm_loss = outputs.loss
        logits = outputs.logits

        # Composition regularization on multi-char tokens only
        # (single-char tokens are already tied via wte)
        if self.training:
            wte = self.transformer.transformer.wte
            compose_loss = self.char_embed.sample_compose_loss(
                wte, sample_size=self.compose_sample_size)
            loss = lm_loss + self.compose_weight * compose_loss
        else:
            compose_loss = None
            loss = lm_loss

        return {"loss": loss, "logits": logits, "lm_loss": lm_loss, "aux_loss": compose_loss}

    def parameters_count(self):
        """Return a dict of parameter counts by component."""
        char_params = sum(p.numel() for p in self.char_embed.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        return {
            "char_embed_params": char_params,
            "transformer_params": transformer_params,
            "total_params": char_params + transformer_params,
        }


class PredictControlGPT2(nn.Module):
    """Control model: GPT-2 + auxiliary next-token prediction on
    char/token mini-sequences (no algebraic composition)."""

    def __init__(self, gpt2_config: GPT2Config, exp_config: ExperimentConfig):
        super().__init__()
        self.config = gpt2_config
        self.predict_sample_size = 64
        self.predict_weight = 1.0

        self.transformer = GPT2LMHeadModel(gpt2_config)

        # Reuse CharCompositionalEmbedding only for its byte mappings
        self.char_embed = CharCompositionalEmbedding(
            vocab_size=exp_config.vocab_size,
            embed_dim=exp_config.n_embd,
            n_chars=exp_config.n_chars,
        )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        outputs = self.transformer(input_ids, labels=labels)
        lm_loss = outputs.loss
        logits = outputs.logits

        if self.training:
            wte = self.transformer.transformer.wte
            predict_loss = self.char_embed.sample_prediction_loss(
                wte, sample_size=self.predict_sample_size)
            loss = lm_loss + self.predict_weight * predict_loss
        else:
            predict_loss = None
            loss = lm_loss

        return {"loss": loss, "logits": logits, "lm_loss": lm_loss, "aux_loss": predict_loss}

    def parameters_count(self):
        char_params = sum(p.numel() for p in self.char_embed.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        return {
            "char_embed_params": char_params,
            "transformer_params": transformer_params,
            "total_params": char_params + transformer_params,
        }


def count_parameters(model: nn.Module) -> dict:
    """Count parameters for any model."""
    if isinstance(model, (CharComposeGPT2, PredictControlGPT2)):
        return model.parameters_count()

    total = sum(p.numel() for p in model.parameters())
    embed_params = model.transformer.wte.weight.numel()
    return {
        "embedding_params": embed_params,
        "transformer_params": total - embed_params,
        "total_params": total,
    }

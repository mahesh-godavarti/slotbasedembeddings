from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # Model
    model_type: str = "char_compose"  # "baseline" or "char_compose"
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.1

    # Character compositional embedding
    n_chars: int = 256
    n_rotation_blocks: int = 384  # n_embd // 2

    # Training
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 2000
    max_steps: int = 10_000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    fp16: bool = True

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    seq_length: int = 1024

    # Evaluation
    eval_interval: int = 1000
    eval_steps: int = 200

    # Logging
    use_wandb: bool = False
    wandb_project: str = "char-compose-embeddings"
    run_name: str = ""

    # Checkpointing
    output_dir: str = "checkpoints"
    save_interval: int = 10000


def baseline_config() -> ExperimentConfig:
    return ExperimentConfig(model_type="baseline", run_name="baseline")


def char_compose_config() -> ExperimentConfig:
    return ExperimentConfig(model_type="char_compose", run_name="char_compose")

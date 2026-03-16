import os
import math
import time
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from .config import ExperimentConfig, baseline_config, char_compose_config
from .model import create_model, count_parameters


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset chunked into fixed-length sequences."""

    def __init__(self, token_ids: torch.Tensor, seq_length: int):
        self.seq_length = seq_length
        # Truncate to a multiple of seq_length
        n_tokens = (len(token_ids) // seq_length) * seq_length
        self.data = token_ids[:n_tokens].view(-1, seq_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_wikitext(config: ExperimentConfig, split: str = "train"):
    """Load and tokenize WikiText-103."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=split)

    # Concatenate all text and tokenize
    all_text = "\n".join(dataset["text"])
    token_ids = tokenizer.encode(all_text, return_tensors="pt").squeeze(0)

    return TokenizedDataset(token_ids, config.seq_length)


def get_lr(step: int, config: ExperimentConfig) -> float:
    """Cosine decay learning rate with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def evaluate(model, val_loader, config: ExperimentConfig, device: torch.device, max_steps: int = None):
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    n_steps = 0
    max_steps = max_steps or config.eval_steps

    for batch in val_loader:
        if n_steps >= max_steps:
            break
        input_ids = batch.to(device)
        labels = input_ids.clone()

        with torch.cuda.amp.autocast(enabled=config.fp16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        total_loss += loss.item()
        n_steps += 1

    avg_loss = total_loss / max(n_steps, 1)
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def train(config: ExperimentConfig):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading training data...")
    train_dataset = load_wikitext(config, split="train")
    val_dataset = load_wikitext(config, split="validation")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # Model
    print(f"Creating {config.model_type} model...")
    model = create_model(config).to(device)
    params = count_parameters(model)
    print(f"Parameter counts: {params}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16 and device.type == "cuda")

    # Wandb
    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, name=config.run_name, config=vars(config))

    # Training loop
    os.makedirs(config.output_dir, exist_ok=True)
    model.train()
    step = 0
    optimizer.zero_grad()
    train_iter = iter(train_loader)

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        # Get batch (cycle through data)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch.to(device)
        labels = input_ids.clone()

        # Forward
        with torch.cuda.amp.autocast(enabled=config.fp16 and device.type == "cuda"):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss = loss / config.gradient_accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Step every gradient_accumulation_steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Update LR
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging
        train_loss = loss.item() * config.gradient_accumulation_steps
        pbar.set_postfix(loss=f"{train_loss:.4f}")
        pbar.update(1)

        if config.use_wandb and step % 100 == 0:
            import wandb
            wandb.log({"train/loss": train_loss, "train/lr": lr}, step=step)

        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, config, device)
            print(f"\nStep {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
            if config.use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)

        # Save checkpoint
        if step > 0 and step % config.save_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f"{config.model_type}_step{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(config),
            }, ckpt_path)
            print(f"\nSaved checkpoint to {ckpt_path}")

        step += 1

    pbar.close()

    # Final save
    ckpt_path = os.path.join(config.output_dir, f"{config.model_type}_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": vars(config),
    }, ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")

    # Final eval
    val_loss, val_ppl = evaluate(model, val_loader, config, device)
    print(f"Final: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

    if config.use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="char_compose",
                        choices=["baseline", "char_compose"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.model_type == "baseline":
        config = baseline_config()
    else:
        config = char_compose_config()

    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.use_wandb:
        config.use_wandb = True
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    train(config)


if __name__ == "__main__":
    main()

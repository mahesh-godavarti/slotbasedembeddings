import os
import math
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig, baseline_config, char_compose_config
from .model import create_model, count_parameters, CharComposeGPT2
from .train import load_wikitext


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, fp16=True):
    """Compute perplexity over a full dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch.to(device)
        labels = input_ids.clone()

        with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        # Count non-padding tokens (all tokens minus the first position per sequence)
        n_tokens = input_ids.numel() - input_ids.shape[0]
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def load_checkpoint(checkpoint_path: str, config: ExperimentConfig, device: torch.device):
    """Load a model from checkpoint."""
    model = create_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def compare_models(baseline_path: str, char_compose_path: str):
    """Compare baseline and char_compose models side by side."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    b_config = baseline_config()
    c_config = char_compose_config()

    print("Loading baseline model...")
    baseline = load_checkpoint(baseline_path, b_config, device)
    print("Loading char_compose model...")
    char_compose = load_checkpoint(char_compose_path, c_config, device)

    # Load test data
    print("Loading test data...")
    test_dataset = load_wikitext(b_config, split="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=2, pin_memory=True, drop_last=True)

    # Evaluate both
    print("\n--- Baseline ---")
    b_params = count_parameters(baseline)
    print(f"Parameters: {b_params}")
    b_loss, b_ppl = evaluate_perplexity(baseline, test_loader, device)
    print(f"Test loss: {b_loss:.4f}, Test perplexity: {b_ppl:.2f}")

    print("\n--- Char Compose ---")
    c_params = count_parameters(char_compose)
    print(f"Parameters: {c_params}")
    c_loss, c_ppl = evaluate_perplexity(char_compose, test_loader, device)
    print(f"Test loss: {c_loss:.4f}, Test perplexity: {c_ppl:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Baseline':>15} {'CharCompose':>15}")
    print("-" * 60)
    print(f"{'Total params':<25} {b_params['total_params']:>15,} {c_params['total_params']:>15,}")
    embed_key_b = "embedding_params"
    embed_key_c = "char_embed_params"
    print(f"{'Embedding params':<25} {b_params[embed_key_b]:>15,} {c_params[embed_key_c]:>15,}")
    print(f"{'Test perplexity':<25} {b_ppl:>15.2f} {c_ppl:>15.2f}")
    print(f"{'Test loss':<25} {b_loss:>15.4f} {c_loss:>15.4f}")
    print("=" * 60)


def evaluate_rare_tokens(model, config, device, min_count=0, max_count=5):
    """Evaluate perplexity specifically on positions where the target token is rare.

    Finds tokens that appear <= max_count times in training data,
    then measures loss only at positions predicting those tokens.
    This is where char-compose should shine vs baseline.
    """
    from collections import Counter
    from transformers import GPT2Tokenizer
    import numpy as np

    print(f"\n--- Rare Token Evaluation (train count <= {max_count}) ---")

    # Count token frequencies in training data
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    cache_path = os.path.join(cache_dir, f"{config.dataset_config}_train.bin")

    if os.path.exists(cache_path):
        train_ids = np.memmap(cache_path, dtype=np.int32, mode='r')
    else:
        print("  Training cache not found, skipping rare token eval.")
        return None

    # Count frequencies
    print("  Counting token frequencies in training data...")
    counts = Counter(train_ids.tolist())
    rare_tokens = set(tid for tid, c in counts.items() if min_count <= c <= max_count)
    all_tokens = set(range(config.vocab_size))
    unseen_tokens = all_tokens - set(counts.keys())
    rare_tokens |= unseen_tokens

    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Tokens seen in training: {len(counts)}")
    print(f"  Rare tokens (count <= {max_count}): {len(rare_tokens)}")

    # Load test data
    test_dataset = load_wikitext(config, split="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=True)

    # Evaluate: compute loss only at positions where the target is a rare token
    model.eval()
    rare_total_loss = 0.0
    rare_total_count = 0
    common_total_loss = 0.0
    common_total_count = 0

    rare_tokens_tensor = torch.zeros(config.vocab_size, dtype=torch.bool, device=device)
    for tid in rare_tokens:
        if tid < config.vocab_size:
            rare_tokens_tensor[tid] = True

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Rare token eval"):
            input_ids = batch.to(device)
            labels = input_ids[:, 1:]  # targets
            inputs = input_ids[:, :-1]

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(input_ids, labels=input_ids.clone())
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

            # Per-token cross entropy
            shift_logits = logits[:, :-1, :].contiguous()
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.contiguous().view(-1),
                reduction='none'
            ).view(labels.shape)

            # Mask for rare vs common
            rare_mask = rare_tokens_tensor[labels]
            common_mask = ~rare_mask

            if rare_mask.any():
                rare_total_loss += per_token_loss[rare_mask].sum().item()
                rare_total_count += rare_mask.sum().item()
            if common_mask.any():
                common_total_loss += per_token_loss[common_mask].sum().item()
                common_total_count += common_mask.sum().item()

    results = {}
    if rare_total_count > 0:
        rare_avg = rare_total_loss / rare_total_count
        rare_ppl = math.exp(min(rare_avg, 20))
        results['rare_loss'] = rare_avg
        results['rare_ppl'] = rare_ppl
        results['rare_count'] = rare_total_count
        print(f"  Rare tokens:   loss={rare_avg:.4f}, PPL={rare_ppl:.2f} ({rare_total_count} positions)")
    else:
        print(f"  No rare tokens found in test set.")

    if common_total_count > 0:
        common_avg = common_total_loss / common_total_count
        common_ppl = math.exp(min(common_avg, 20))
        results['common_loss'] = common_avg
        results['common_ppl'] = common_ppl
        results['common_count'] = common_total_count
        print(f"  Common tokens: loss={common_avg:.4f}, PPL={common_ppl:.2f} ({common_total_count} positions)")

    return results


def cosine_similarity_check(model, tokenizer_name="gpt2"):
    """Check that tokens sharing character prefixes have higher similarity."""
    from transformers import GPT2Tokenizer
    import torch.nn.functional as F

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    if isinstance(model, CharComposeGPT2):
        embeddings = model.char_embed.compose_all_tokens().detach()
    else:
        embeddings = model.transformer.wte.weight.detach()

    # Test pairs: tokens sharing prefixes vs random
    test_words = [("run", "running"), ("play", "playing"), ("cat", "cats")]
    print("\nCosine similarity for related tokens:")
    for w1, w2 in test_words:
        id1 = tokenizer.encode(w1)
        id2 = tokenizer.encode(w2)
        if len(id1) == 1 and len(id2) == 1:
            sim = F.cosine_similarity(
                embeddings[id1[0]].unsqueeze(0),
                embeddings[id2[0]].unsqueeze(0),
            ).item()
            print(f"  '{w1}' vs '{w2}': {sim:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="char_compose",
                        choices=["baseline", "char_compose"])
    parser.add_argument("--compare_baseline", type=str, default=None,
                        help="Path to baseline checkpoint for comparison")
    args = parser.parse_args()

    if args.compare_baseline:
        compare_models(args.compare_baseline, args.checkpoint)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = baseline_config() if args.model_type == "baseline" else char_compose_config()

        print(f"Loading {args.model_type} model from {args.checkpoint}...")
        model = load_checkpoint(args.checkpoint, config, device)
        params = count_parameters(model)
        print(f"Parameters: {params}")

        print("Loading test data...")
        test_dataset = load_wikitext(config, split="test")
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                 num_workers=2, pin_memory=True, drop_last=True)

        loss, ppl = evaluate_perplexity(model, test_loader, device)
        print(f"Test loss: {loss:.4f}, Test perplexity: {ppl:.2f}")

        cosine_similarity_check(model)


if __name__ == "__main__":
    main()

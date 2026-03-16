#!/usr/bin/env python3
"""Evaluate rare-token perplexity for baseline vs char-compose models.

Usage:
    python -m experiment.eval_rare \
        --char_compose_ckpt checkpoints/char_compose_final.pt \
        --baseline_ckpt checkpoints/baseline_final.pt \
        --max_count 5
"""

import argparse
import torch

from .config import baseline_config, char_compose_config
from .model import create_model
from .evaluate import evaluate_rare_tokens


def main():
    parser = argparse.ArgumentParser(description="Rare token PPL comparison")
    parser.add_argument("--char_compose_ckpt", type=str, required=True)
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--max_count", type=int, default=5,
                        help="Tokens with <= this many training occurrences are 'rare'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Char-compose ---
    print("=" * 60)
    print("CHAR-COMPOSE")
    print("=" * 60)
    c_config = char_compose_config()
    c_model = create_model(c_config).to(device)
    ckpt = torch.load(args.char_compose_ckpt, map_location=device)
    c_model.load_state_dict(ckpt["model_state_dict"])
    c_results = evaluate_rare_tokens(c_model, c_config, device,
                                     max_count=args.max_count)

    # Free GPU before loading next model
    del c_model
    torch.cuda.empty_cache()

    # --- Baseline ---
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    b_config = baseline_config()
    b_model = create_model(b_config).to(device)
    ckpt = torch.load(args.baseline_ckpt, map_location=device)
    b_model.load_state_dict(ckpt["model_state_dict"])
    b_results = evaluate_rare_tokens(b_model, b_config, device,
                                     max_count=args.max_count)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    if c_results and b_results:
        print(f"{'Metric':<25} {'Baseline':>15} {'CharCompose':>15}")
        print("-" * 60)
        if 'rare_ppl' in b_results and 'rare_ppl' in c_results:
            print(f"{'Rare token PPL':<25} {b_results['rare_ppl']:>15.2f} {c_results['rare_ppl']:>15.2f}")
            print(f"{'Rare token count':<25} {b_results['rare_count']:>15,} {c_results['rare_count']:>15,}")
        if 'common_ppl' in b_results and 'common_ppl' in c_results:
            print(f"{'Common token PPL':<25} {b_results['common_ppl']:>15.2f} {c_results['common_ppl']:>15.2f}")
            print(f"{'Common token count':<25} {b_results['common_count']:>15,} {c_results['common_count']:>15,}")
        print("=" * 60)


if __name__ == "__main__":
    main()

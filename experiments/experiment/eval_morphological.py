#!/usr/bin/env python3
"""Evaluate morphological pattern generalization.

Measures perplexity on morphological validation sentences
(held out from training) to test whether character-compositional
embeddings help the model generalize morphological patterns.

Usage:
    python -m experiment.eval_morphological \
        --char_compose_ckpt checkpoints/char_compose_final.pt \
        --baseline_ckpt checkpoints/baseline_final.pt \
        --control_ckpt checkpoints/predict_control_final.pt
"""

import argparse
import math
import torch
from transformers import GPT2Tokenizer

from .config import ExperimentConfig
from .model import create_model
from .morphological_data import (get_train_with_pairs, get_val_with_pairs,
                                 get_novel_with_pairs)


@torch.no_grad()
def eval_sentences(model, sentences, tokenizer, device, word_pairs=None):
    """Compute per-sentence perplexity.

    If word_pairs is provided (list of (derived, base) tuples parallel to sentences),
    also computes loss restricted to tokens of the morphological words.
    """
    model.eval()
    results = []
    morph_total_loss = 0.0
    morph_total_count = 0

    for i, sent in enumerate(sentences):
        token_ids = tokenizer.encode(sent)
        input_ids = torch.tensor([token_ids], device=device)
        labels = input_ids.clone()

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        ppl = math.exp(min(loss.item(), 20))
        result = {"sentence": sent, "loss": loss.item(), "ppl": ppl,
                  "n_tokens": len(token_ids)}

        # Compute loss on morphological words only
        if word_pairs is not None:
            derived, base = word_pairs[i]
            derived_ids = tokenizer.encode(" " + derived)
            base_ids = tokenizer.encode(" " + base)
            morph_token_set = set(derived_ids + base_ids)

            # Per-token loss
            shift_logits = logits[0, :-1, :].contiguous()
            shift_labels = labels[0, 1:].contiguous()
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction='none')

            # Find positions where the target token is part of a morph word
            morph_mask = torch.tensor(
                [t.item() in morph_token_set for t in shift_labels],
                device=device, dtype=torch.bool)

            if morph_mask.any():
                morph_loss = per_token_loss[morph_mask].sum().item()
                morph_count = morph_mask.sum().item()
                morph_total_loss += morph_loss
                morph_total_count += morph_count
                result["morph_loss"] = morph_loss / morph_count
                result["morph_ppl"] = math.exp(min(result["morph_loss"], 20))

        results.append(result)

    avg_loss = sum(r["loss"] for r in results) / len(results)
    avg_ppl = math.exp(min(avg_loss, 20))

    morph_avg_loss = morph_total_loss / max(morph_total_count, 1)
    morph_avg_ppl = math.exp(min(morph_avg_loss, 20))

    return results, avg_loss, avg_ppl, morph_avg_loss, morph_avg_ppl


def load_model(model_type, ckpt_path, device):
    """Load a model from checkpoint."""
    config = ExperimentConfig(model_type=model_type)
    model = create_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def run_eval(model_name, model, tokenizer, device):
    """Run morphological eval on one model."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    train_sents, train_pairs = get_train_with_pairs()
    val_sents, val_pairs = get_val_with_pairs()
    novel_sents, novel_pairs = get_novel_with_pairs()

    _, train_loss, train_ppl, train_m_loss, train_m_ppl = eval_sentences(
        model, train_sents, tokenizer, device, word_pairs=train_pairs)
    print(f"  Train         ({len(train_sents):>4}): PPL={train_ppl:>10.2f}  morph_PPL={train_m_ppl:>10.2f}")

    val_results, val_loss, val_ppl, val_m_loss, val_m_ppl = eval_sentences(
        model, val_sents, tokenizer, device, word_pairs=val_pairs)
    print(f"  Val same tpl  ({len(val_sents):>4}): PPL={val_ppl:>10.2f}  morph_PPL={val_m_ppl:>10.2f}")

    novel_results, novel_loss, novel_ppl, novel_m_loss, novel_m_ppl = eval_sentences(
        model, novel_sents, tokenizer, device, word_pairs=novel_pairs)
    print(f"  Val novel tpl ({len(novel_sents):>4}): PPL={novel_ppl:>10.2f}  morph_PPL={novel_m_ppl:>10.2f}")

    # Show a few examples
    print(f"\n  Sample novel template predictions:")
    for r in novel_results[:5]:
        m_ppl = r.get('morph_ppl', 0)
        print(f"    PPL={r['ppl']:>10.2f}  morph_PPL={m_ppl:>10.2f}  {r['sentence']}")

    return {
        "train_ppl": train_ppl, "train_morph_ppl": train_m_ppl,
        "val_ppl": val_ppl, "val_morph_ppl": val_m_ppl,
        "novel_ppl": novel_ppl, "novel_morph_ppl": novel_m_ppl,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--char_compose_ckpt", type=str, default=None)
    parser.add_argument("--control_ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    results = {}

    models = []
    if args.char_compose_ckpt:
        models.append(("Char-Compose", "char_compose", args.char_compose_ckpt))
    if args.control_ckpt:
        models.append(("Predict-Control", "predict_control", args.control_ckpt))
    if args.baseline_ckpt:
        models.append(("Baseline", "baseline", args.baseline_ckpt))

    for name, model_type, ckpt_path in models:
        model = load_model(model_type, ckpt_path, device)
        r = run_eval(name, model, tokenizer, device)
        results[name] = r
        del model
        torch.cuda.empty_cache()

    # Summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY — Full sentence PPL")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Train':>10} {'Val':>10} {'Novel':>10}")
        print("-" * 54)
        for name, r in results.items():
            print(f"{name:<20} {r['train_ppl']:>10.2f} {r['val_ppl']:>10.2f} {r['novel_ppl']:>10.2f}")

        print(f"\nSUMMARY — Morphological words only PPL")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Train':>10} {'Val':>10} {'Novel':>10}")
        print("-" * 54)
        for name, r in results.items():
            print(f"{name:<20} {r['train_morph_ppl']:>10.2f} {r['val_morph_ppl']:>10.2f} {r['novel_morph_ppl']:>10.2f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

"""Test script to verify batched eval produces identical results to non-batched eval.

Loads checkpoints, runs both versions, compares results.
"""

import sys
import os
import torch
import numpy as np
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F
import kg_text_experiment as kge


# ============================================================================
# Original (non-batched) eval functions — copied from pre-edit version
# ============================================================================

def evaluate_model_kg_ORIG(model, kg_eval_prompts, vocab, config, model_type="A"):
    model.eval()
    model.to(config.device)
    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for p in kg_eval_prompts:
            tier, relation = p["tier"], p["relation"]
            head_tokens = vocab.encode_entity(p["head"])
            tail_tokens = vocab.encode_entity(p["tail"])
            head_len, tail_len = len(head_tokens), len(tail_tokens)
            rel_name = p["rel"]

            total_log_prob = 0.0
            log_prob_first = log_prob_last = None
            all_in_top5 = True

            for t_idx in range(tail_len):
                tokens, targets, head_lens, rel_names = kge._build_kg_eval_batch(
                    head_tokens, rel_name, tail_tokens, [head_len + t_idx], vocab, model_type)
                tokens, targets = tokens.to(config.device), targets.to(config.device)
                logits = kge._forward_kg_eval(model, tokens, targets, head_lens, rel_names, model_type)

                mask_pos = head_len + t_idx if model_type == "E" else head_len + 1 + t_idx
                step_logits = logits[0, mask_pos, :]
                log_probs = F.log_softmax(step_logits, dim=0)
                true_token = tail_tokens[t_idx]
                total_log_prob += log_probs[true_token].item()
                if t_idx == 0: log_prob_first = log_probs[true_token].item()
                log_prob_last = log_probs[true_token].item()
                top5 = torch.topk(step_logits, k=min(5, step_logits.shape[0])).indices.tolist()
                if true_token not in top5: all_in_top5 = False

            ppl = np.exp(-total_log_prob / max(tail_len, 1))
            first_char_ppl = np.exp(-log_prob_first) if log_prob_first is not None else ppl
            last_char_ppl = np.exp(-log_prob_last) if log_prob_last is not None else ppl

            all_mask_positions = list(range(head_len, head_len + tail_len))
            tokens, targets, head_lens, rel_names = kge._build_kg_eval_batch(
                head_tokens, rel_name, tail_tokens, all_mask_positions, vocab, model_type)
            tokens, targets = tokens.to(config.device), targets.to(config.device)
            logits = kge._forward_kg_eval(model, tokens, targets, head_lens, rel_names, model_type)

            hit1 = 1
            for t_idx in range(tail_len):
                mask_pos = head_len + t_idx if model_type == "E" else head_len + 1 + t_idx
                if torch.argmax(logits[0, mask_pos, :]).item() != tail_tokens[t_idx]:
                    hit1 = 0; break

            results[tier][relation].append({
                "hit1": hit1, "hit5": 1 if all_in_top5 else 0,
                "ppl": ppl, "first_char_ppl": first_char_ppl, "last_char_ppl": last_char_ppl,
                "head": p["head"], "rel": rel_name, "tail": p["tail"],
            })
    model.train()
    return results


def evaluate_model_ORIG(model, eval_prompts, vocab, config):
    model.eval()
    model.to(config.device)
    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for p in eval_prompts:
            tier, relation = p["tier"], p["relation"]
            prompt_tokens = p["prompt_tokens"]
            target_tokens = p["target_tokens"]
            if len(prompt_tokens) > config.block_size:
                prompt_tokens = prompt_tokens[-config.block_size:]

            full_correct = True
            total_log_prob = 0.0
            generated_name = []
            current_tokens = prompt_tokens.copy()
            log_probs_first = log_probs_last = None

            for t_idx, t_tok in enumerate(target_tokens):
                x = torch.tensor([current_tokens[-config.block_size:]], dtype=torch.long, device=config.device)
                logits = model.predict_text(x)
                step_logits = logits[0, -1, :]
                log_probs = F.log_softmax(step_logits, dim=0)
                total_log_prob += log_probs[t_tok].item()
                if t_idx == 0: log_probs_first = log_probs[t_tok]
                log_probs_last = log_probs[t_tok]
                pred = torch.argmax(step_logits).item()
                generated_name.append(pred)
                if pred != t_tok: full_correct = False
                current_tokens.append(t_tok)

            ppl = np.exp(-total_log_prob / len(target_tokens))
            first_char_ppl = np.exp(-log_probs_first.item()) if target_tokens else ppl
            last_char_ppl = np.exp(-log_probs_last.item()) if target_tokens else ppl
            hit1 = 1 if generated_name == target_tokens else 0

            all_in_top5 = True
            current_tokens_t5 = prompt_tokens.copy()
            for t_idx, t_tok in enumerate(target_tokens):
                x = torch.tensor([current_tokens_t5[-config.block_size:]], dtype=torch.long, device=config.device)
                logits = model.predict_text(x)
                step_logits = logits[0, -1, :]
                top5 = torch.topk(step_logits, k=min(5, step_logits.shape[0])).indices.tolist()
                if t_tok not in top5:
                    all_in_top5 = False; break
                current_tokens_t5.append(t_tok)

            results[tier][relation].append({
                "hit1": hit1, "hit5": 1 if all_in_top5 else 0,
                "ppl": ppl, "first_char_ppl": first_char_ppl, "last_char_ppl": last_char_ppl,
                "full_correct": 1 if full_correct else 0,
                "prompt": p["prompt"], "target": p["target"],
            })
    model.train()
    return results


def evaluate_model_kg_causal_ORIG(model, kg_eval_prompts, vocab, config):
    model.eval()
    model.to(config.device)
    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for p in kg_eval_prompts:
            tier, relation = p["tier"], p["relation"]
            head_tokens = vocab.encode_entity(p["head"])
            tail_tokens = vocab.encode_entity(p["tail"])
            head_len, tail_len = len(head_tokens), len(tail_tokens)

            for direction in ("forward", "backward"):
                if direction == "forward":
                    seq = list(head_tokens) + list(tail_tokens)
                    ctx_len, pred_tokens, negate = head_len, tail_tokens, False
                else:
                    seq = list(tail_tokens) + list(head_tokens)
                    ctx_len, pred_tokens, negate = tail_len, head_tokens, True

                pred_len = len(pred_tokens)
                seq_t = torch.tensor([seq], dtype=torch.long, device=config.device)
                pad_mask = (seq_t != 0)
                x = model.expander(model.token_embedding(seq_t))
                angles = model._cumsum_angles_kg(seq_t, [ctx_len], [p["rel"]], config.device, negate_angles=[negate])
                for block in model.blocks:
                    x = block(x, angles, causal=True, pad_mask=pad_mask)
                logits = model.lm_head(x)

                total_log_prob = 0.0
                all_in_top5 = True
                all_top1 = True
                for j in range(pred_len):
                    pred_pos = ctx_len - 1 + j
                    true_token = pred_tokens[j]
                    step_logits = logits[0, pred_pos, :]
                    log_probs = F.log_softmax(step_logits, dim=0)
                    total_log_prob += log_probs[true_token].item()
                    top5 = torch.topk(step_logits, k=min(5, step_logits.shape[0])).indices.tolist()
                    if true_token not in top5: all_in_top5 = False
                    if torch.argmax(step_logits).item() != true_token: all_top1 = False

                ppl = np.exp(-total_log_prob / max(pred_len, 1))
                results[tier][relation].append({
                    "hit1": 1 if all_top1 else 0, "hit5": 1 if all_in_top5 else 0,
                    "ppl": ppl, "first_char_ppl": ppl, "last_char_ppl": ppl,
                    "head": p["head"], "rel": p["rel"], "tail": p["tail"], "direction": direction,
                })
    model.train()
    return results


# ============================================================================
# Comparison
# ============================================================================

def flatten_results(results):
    flat = []
    for tier in sorted(results.keys()):
        for rel in sorted(results[tier].keys()):
            for r in results[tier][rel]:
                flat.append((tier, rel, r))
    return flat


def result_key(tier, rel, r):
    """Unique key for a result entry for matching between orig and batched."""
    # For text eval: use prompt+target
    if "prompt" in r:
        return (tier, rel, r["prompt"], r["target"])
    # For KG causal: use head+rel+tail+direction
    if "direction" in r:
        return (tier, rel, r["head"], r["rel"], r["tail"], r["direction"])
    # For KG MLM: use head+rel+tail
    return (tier, rel, r["head"], r["rel"], r["tail"])


def compare_results(orig_results, batched_results, label):
    orig_flat = flatten_results(orig_results)
    batched_flat = flatten_results(batched_results)

    if len(orig_flat) != len(batched_flat):
        print(f"  {label}: MISMATCH count: orig={len(orig_flat)} batched={len(batched_flat)}")
        return False

    # Build lookup by key
    orig_by_key = {}
    for tier, rel, r in orig_flat:
        k = result_key(tier, rel, r)
        orig_by_key[k] = r

    batched_by_key = {}
    for tier, rel, r in batched_flat:
        k = result_key(tier, rel, r)
        batched_by_key[k] = r

    if set(orig_by_key.keys()) != set(batched_by_key.keys()):
        missing = set(orig_by_key.keys()) - set(batched_by_key.keys())
        extra = set(batched_by_key.keys()) - set(orig_by_key.keys())
        if missing:
            print(f"  {label}: {len(missing)} keys in orig but not batched")
        if extra:
            print(f"  {label}: {len(extra)} keys in batched but not orig")
        return False

    mismatches = 0
    for k in sorted(orig_by_key.keys()):
        o, b = orig_by_key[k], batched_by_key[k]
        for field in ("hit1", "hit5", "ppl", "first_char_ppl", "last_char_ppl", "full_correct"):
            if field not in o or field not in b:
                continue
            ov, bv = o[field], b[field]
            if isinstance(ov, float):
                if abs(ov - bv) / max(abs(ov), 1e-8) > 1e-5:
                    if mismatches < 5:
                        print(f"  {label} {k[:3]}...: {field} mismatch: orig={ov:.6f} batched={bv:.6f}")
                    mismatches += 1
            elif ov != bv:
                if mismatches < 5:
                    print(f"  {label} {k[:3]}...: {field} mismatch: orig={ov} batched={bv}")
                mismatches += 1

    if mismatches == 0:
        print(f"  {label}: ALL {len(orig_by_key)} results MATCH perfectly")
    else:
        print(f"  {label}: {mismatches} mismatches out of {len(orig_by_key)} results")
    return mismatches == 0


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  Batched Eval Verification Test")
    print("=" * 60)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    # Read config from first checkpoint
    ckpt = torch.load(os.path.join(ckpt_dir, "7a_A_seed0.pt"), map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {})
    kge.cfg.n_embed = saved_cfg.get("n_embed", 50)
    kge.cfg.n_layers = saved_cfg.get("n_layers", 20)
    kge.cfg.batch_size = 8  # small batch for testing variety

    print(f"\nConfig: n_embed={kge.cfg.n_embed}, n_layers={kge.cfg.n_layers}, batch_size={kge.cfg.batch_size}")
    print("Preparing data...")

    vocab, text_ds_base, text_ds_lin, kg_ds, eval_prompts, kg_eval_prompts = kge.prepare_data(generous_linearization=True)

    # Use subsets for speed
    eval_sub = eval_prompts[:100]
    kg_sub = kg_eval_prompts[:100]

    # Only test models with same config (130 chains, n_embed=50, n_layers=20)
    test_configs = [
        ("A",  "A",  "7a_A_seed0.pt"),
        ("A'", "A",  "7a_Ap_seed0.pt"),
        ("E'", "E",  "7a_Ep_seed0.pt"),
    ]

    all_pass = True
    for model_name, model_type, ckpt_file in test_configs:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {model_name}: checkpoint not found")
            continue

        print(f"\n--- Testing {model_name} ---")
        ckpt = torch.load(ckpt_path, map_location=kge.cfg.device, weights_only=False)
        model = kge.create_model(model_name, vocab.size, kge.cfg)
        model.load_state_dict(ckpt["model_state_dict"])

        # Text eval
        t0 = time.time()
        orig_text = evaluate_model_ORIG(model, eval_sub, vocab, kge.cfg)
        t_orig = time.time() - t0

        t0 = time.time()
        _, _, batched_text = kge.evaluate_model(model, eval_sub, vocab, kge.cfg, model_name)
        t_batch = time.time() - t0

        ok = compare_results(orig_text, batched_text, f"{model_name} TEXT")
        all_pass = all_pass and ok
        print(f"  TEXT timing: orig={t_orig:.1f}s  batched={t_batch:.1f}s  speedup={t_orig/max(t_batch,0.001):.1f}x")

        # KG MLM eval
        if model_type is not None:
            t0 = time.time()
            orig_kg = evaluate_model_kg_ORIG(model, kg_sub, vocab, kge.cfg, model_type)
            t_orig = time.time() - t0

            t0 = time.time()
            _, _, batched_kg = kge.evaluate_model_kg(model, kg_sub, vocab, kge.cfg, model_name, model_type)
            t_batch = time.time() - t0

            ok = compare_results(orig_kg, batched_kg, f"{model_name} KG_MLM")
            all_pass = all_pass and ok
            print(f"  KG_MLM timing: orig={t_orig:.1f}s  batched={t_batch:.1f}s  speedup={t_orig/max(t_batch,0.001):.1f}x")

        # KG causal eval (E/H only)
        base_name = model_name.replace("'", "")
        if base_name in ("E", "H"):
            t0 = time.time()
            orig_kgc = evaluate_model_kg_causal_ORIG(model, kg_sub, vocab, kge.cfg)
            t_orig = time.time() - t0

            t0 = time.time()
            _, _, batched_kgc = kge.evaluate_model_kg_causal(model, kg_sub, vocab, kge.cfg, model_name)
            t_batch = time.time() - t0

            ok = compare_results(orig_kgc, batched_kgc, f"{model_name} KG_CAUSAL")
            all_pass = all_pass and ok
            print(f"  KG_CAUSAL timing: orig={t_orig:.1f}s  batched={t_batch:.1f}s  speedup={t_orig/max(t_batch,0.001):.1f}x")

        del model

    print(f"\n{'='*60}")
    print(f"  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

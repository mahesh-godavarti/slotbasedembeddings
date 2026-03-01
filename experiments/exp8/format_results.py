#!/usr/bin/env python3
"""Format experiment results into 4 markdown tables (Text h@5, Text PPL, KG h@5, KG PPL).

Usage:
    python format_results.py <logfile> [output_file]
    python format_results.py expanded_names_random_n500_2layers_2000iters.log
    python format_results.py expanded_names_random_n500_2layers_2000iters.log results.md
"""
import sys
from parse_results import parse_log, MODEL_ORDER, ALL_TIERS

TIER_SHORT = {
    "memorization": "mem",
    "transfer": "trans",
    "generalization": "gen",
    "kg_exclusive_memorization": "kgExcl_m",
    "kg_exclusive_generalization": "kgExcl_g",
    "text_exclusive_memorization": "txtExcl_m",
    "text_exclusive_generalization": "txtExcl_g",
}

# Models that don't get KG eval
TEXT_ONLY_MODELS = {"B", "B'", "C", "C'"}


def format_tables(vocab_size, text_results, kg_results):
    lines = []
    if vocab_size:
        lines.append(f"Vocabulary size: {vocab_size}\n")

    # --- Text Eval h@5 ---
    lines.append("**Text Eval h@5:**\n")
    hdr = "| Model | " + " | ".join(TIER_SHORT[t] for t in ALL_TIERS) + " |"
    sep = "|-------|" + "------|" * len(ALL_TIERS)
    lines.append(hdr)
    lines.append(sep)
    # Sort by memorization h@5 descending
    models_with_data = [(m, text_results.get(m, {})) for m in MODEL_ORDER if m in text_results]
    models_with_data.sort(key=lambda x: -x[1].get("memorization", {}).get("hit5", 0))
    for m, tiers in models_with_data:
        vals = []
        for t in ALL_TIERS:
            d = tiers.get(t)
            vals.append(f"{d['hit5']:.3f}" if d else "...")
        lines.append(f"| {m} | " + " | ".join(vals) + " |")

    # --- Text Eval PPL ---
    lines.append("\n**Text Eval PPL:**\n")
    lines.append(hdr)
    lines.append(sep)
    # Sort by memorization PPL ascending
    models_with_data.sort(key=lambda x: x[1].get("memorization", {}).get("ppl", 1e18))
    for m, tiers in models_with_data:
        vals = []
        for t in ALL_TIERS:
            d = tiers.get(t)
            vals.append(f"{d['ppl']:.2f}" if d else "...")
        lines.append(f"| {m} | " + " | ".join(vals) + " |")

    # --- KG Eval h@5 ---
    kg_models = [(m, kg_results.get(m, {})) for m in MODEL_ORDER
                 if m in kg_results and m not in TEXT_ONLY_MODELS]
    if kg_models:
        lines.append("\n**KG Eval h@5:**\n")
        lines.append(hdr)
        lines.append(sep)
        kg_models.sort(key=lambda x: -x[1].get("memorization", {}).get("hit5", 0))
        for m, tiers in kg_models:
            vals = []
            for t in ALL_TIERS:
                d = tiers.get(t)
                vals.append(f"{d['hit5']:.3f}" if d else "...")
            lines.append(f"| {m} | " + " | ".join(vals) + " |")

        # --- KG Eval PPL ---
        lines.append("\n**KG Eval PPL:**\n")
        lines.append(hdr)
        lines.append(sep)
        kg_models.sort(key=lambda x: x[1].get("memorization", {}).get("ppl", 1e18))
        for m, tiers in kg_models:
            vals = []
            for t in ALL_TIERS:
                d = tiers.get(t)
                vals.append(f"{d['ppl']:.2f}" if d else "...")
            lines.append(f"| {m} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile> [output_file]")
        sys.exit(1)

    vocab_size, text_results, kg_results = parse_log(sys.argv[1])
    output = format_tables(vocab_size, text_results, kg_results)
    print(output)

    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w") as f:
            f.write(output + "\n")
        print(f"\nWritten to {sys.argv[2]}")

#!/usr/bin/env python3
"""Parse experiment output logs and print summary tables.

Usage:
    python parse_results.py <logfile>
    python parse_results.py expanded_names_n90_1000iters.log
"""
import re
import sys

MODEL_ORDER = ["A", "A'", "B", "B'", "C", "C'", "D", "D'", "E", "E'", "F", "F'", "G", "G'", "H", "H'"]
ALL_TIERS = [
    "memorization", "transfer", "generalization",
    "kg_exclusive_memorization", "kg_exclusive_generalization",
    "text_exclusive_memorization", "text_exclusive_generalization",
]

def parse_log(path):
    with open(path) as f:
        lines = f.readlines()

    # Extract vocab size
    vocab_size = None
    for line in lines:
        m = re.search(r'Vocabulary size: (\d+)', line)
        if m:
            vocab_size = int(m.group(1))
            break

    text_results = {}  # model -> tier -> {hit1, hit5, ppl, full_acc}
    kg_results = {}    # model -> tier -> {hit1, hit5, ppl}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match evaluation headers
        m = re.match(r'(KG )?Evaluation: ([A-H]\'?)$', line)
        if m:
            is_kg = m.group(1) is not None
            model = m.group(2)
            store = kg_results if is_kg else text_results
            if model not in store:
                store[model] = {}

            # Scan ahead for tier lines
            for j in range(i + 1, min(i + 80, len(lines))):
                tl = lines[j].strip()
                if re.match(r'(KG )?Evaluation:', tl) or tl.startswith('Per-relation'):
                    break
                # Format: tier: hit@1=X hit@5=X ppl=X [fc_ppl=X] [lc_ppl=X] [full_acc=X] (n=N)
                tm = re.match(
                    r'(\w[\w_]*?):\s+hit@1=([0-9.]+)\s+hit@5=([0-9.]+)\s+ppl=([0-9.]+)'
                    r'(?:\s+fc_ppl=([0-9.]+))?'
                    r'(?:\s+lc_ppl=([0-9.]+))?'
                    r'(?:\s+full_acc=([0-9.]+))?\s+\(n=(\d+)\)',
                    tl
                )
                if tm:
                    tier = tm.group(1)
                    entry = {
                        'hit1': float(tm.group(2)),
                        'hit5': float(tm.group(3)),
                        'ppl': float(tm.group(4)),
                        'n': int(tm.group(8)),
                    }
                    if tm.group(5) is not None:
                        entry['fc_ppl'] = float(tm.group(5))
                    if tm.group(6) is not None:
                        entry['lc_ppl'] = float(tm.group(6))
                    if tm.group(7) is not None:
                        entry['full_acc'] = float(tm.group(7))
                    store[model][tier] = entry
        i += 1

    return vocab_size, text_results, kg_results


def print_summary(vocab_size, text_results, kg_results):
    if vocab_size:
        print(f"Vocabulary size: {vocab_size}")
    print()

    # Check what extra columns are available
    has_fc_ppl = any(
        'fc_ppl' in t
        for store in [text_results, kg_results]
        for m in store
        for t in store[m].values()
    )
    has_lc_ppl = any(
        'lc_ppl' in t
        for store in [text_results, kg_results]
        for m in store
        for t in store[m].values()
    )

    # Compact table: memorization tier PPL + hit@5 for text and KG
    print("=== Memorization Tier Summary ===")
    print(f"{'Model':<6} {'Text PPL':>9} {'Text h@5':>9} {'KG PPL':>9} {'KG h@5':>9}")
    print("-" * 46)
    for m in MODEL_ORDER:
        t = text_results.get(m, {}).get('memorization')
        k = kg_results.get(m, {}).get('memorization')
        tp = f"{t['ppl']:.2f}" if t else "..."
        th = f"{t['hit5']:.3f}" if t else "..."
        kp = f"{k['ppl']:.2f}" if k else "n/a"
        kh = f"{k['hit5']:.3f}" if k else "n/a"
        print(f"{m:<6} {tp:>9} {th:>9} {kp:>9} {kh:>9}")

    # Build header and format string dynamically
    def make_header_fmt(has_fc, has_lc):
        cols = [f"{'Model':<6}", f"{'Tier':<35}", f"{'PPL':>7}"]
        if has_fc:
            cols.append(f"{'fcPPL':>7}")
        if has_lc:
            cols.append(f"{'lcPPL':>7}")
        cols.extend([f"{'h@1':>7}", f"{'h@5':>7}", f"{'n':>6}"])
        return "  ".join(cols[2:])  # skip model/tier for width calc

    def print_row(m, tier, t, has_fc, has_lc):
        parts = [f"{m:<6}", f"{tier:<35}", f"{t['ppl']:>7.2f}"]
        if has_fc:
            parts.append(f"{t.get('fc_ppl', 0):>7.2f}")
        if has_lc:
            parts.append(f"{t.get('lc_ppl', 0):>7.2f}")
        parts.extend([f"{t['hit1']:>7.3f}", f"{t['hit5']:>7.3f}", f"{t['n']:>6}"])
        print(" ".join(parts))

    # Full tier table for text eval
    print()
    print("=== All Tiers (Text Eval) ===")
    hdr_parts = [f"{'Model':<6}", f"{'Tier':<35}", f"{'PPL':>7}"]
    if has_fc_ppl:
        hdr_parts.append(f"{'fcPPL':>7}")
    if has_lc_ppl:
        hdr_parts.append(f"{'lcPPL':>7}")
    hdr_parts.extend([f"{'h@1':>7}", f"{'h@5':>7}", f"{'n':>6}"])
    hdr = " ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))
    for m in MODEL_ORDER:
        tiers = text_results.get(m, {})
        if not tiers:
            continue
        for tier in ALL_TIERS:
            t = tiers.get(tier)
            if t:
                print_row(m, tier, t, has_fc_ppl, has_lc_ppl)

    kg_models = [m for m in MODEL_ORDER if m in kg_results]
    if kg_models:
        print()
        print("=== All Tiers (KG Eval) ===")
        print(hdr)
        print("-" * len(hdr))
        for m in kg_models:
            tiers = kg_results.get(m, {})
            for tier in ALL_TIERS:
                t = tiers.get(tier)
                if t:
                    print_row(m, tier, t, has_fc_ppl, has_lc_ppl)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile>")
        sys.exit(1)
    vocab_size, text_results, kg_results = parse_log(sys.argv[1])
    print_summary(vocab_size, text_results, kg_results)

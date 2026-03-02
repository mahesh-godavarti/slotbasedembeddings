#!/usr/bin/env python3
"""Parse experiment log files, handling tqdm \\r carriage returns.

Works with: kg_text_experiment.py, kg_text_experiment_dual.py,
            exp8/word_experiment.py, exp8/word_experiment_dual.py,
            joformer/train_wiki.py

Usage:
    # Parse a specific log file
    python parse_log.py <logfile>

    # Find and list recent log files
    python parse_log.py --find

    # Parse the most recent log file found
    python parse_log.py --latest

    # Show only training progress (no eval results)
    python parse_log.py <logfile> --progress

    # Show only the last N training progress lines per model
    python parse_log.py <logfile> --progress --tail 5
"""
import argparse
import glob
import os
import re
import sys


def clean_lines(path):
    """Read a log file and handle tqdm \\r carriage returns.

    tqdm uses \\r to overwrite the same line, creating one enormous line
    with no \\n. We convert \\r to \\n and take the last overwrite of each
    logical line.
    """
    with open(path, "rb") as f:
        raw = f.read()
    # Convert \r\n to \n first (Windows-style), then remaining \r to \n
    text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", errors="replace")
    return text.splitlines()


def find_log_files():
    """Search common locations for experiment log files."""
    candidates = []

    # Claude Code temp files
    for pattern in ["/tmp/claude-*/tasks/*.output",
                    "/tmp/claude-*/**/*.output",
                    "/tmp/claude-*/**/*.out",
                    "/private/tmp/claude-*/tasks/*.output",
                    "/private/tmp/claude-*/**/*.output"]:
        candidates.extend(glob.glob(pattern, recursive=True))

    # Working directory and common locations
    cwd = os.getcwd()
    for pattern in ["*.log", "*.out", "**/*.log", "**/*.out"]:
        candidates.extend(glob.glob(os.path.join(cwd, pattern), recursive=True))

    # Deduplicate and sort by modification time (newest first)
    candidates = list(set(candidates))
    candidates = [(f, os.path.getmtime(f), os.path.getsize(f))
                  for f in candidates if os.path.isfile(f) and os.path.getsize(f) > 0]
    candidates.sort(key=lambda x: -x[1])

    return candidates


def detect_experiment(lines):
    """Detect which experiment script produced this log."""
    for line in lines[:200]:
        if "joformer" in line.lower() or "roformer" in line.lower():
            return "joformer"
        if "word_experiment" in line.lower() or "Exp 8" in line:
            return "exp8"
        if "kg_text_experiment" in line.lower() or "Exp 7" in line:
            return "exp7"
        if "BPE" in line and "vocab" in line.lower():
            return "exp8"
        if "Vocabulary size:" in line and "chain" in "".join(lines[:200]).lower():
            return "exp7"
    return "unknown"


def extract_config(lines):
    """Extract configuration from log lines."""
    config = {}
    for line in lines[:100]:
        # Common config patterns
        m = re.search(r'n_embed=(\d+)', line)
        if m:
            config['n_embed'] = int(m.group(1))
        m = re.search(r'n_layers?=(\d+)', line)
        if m:
            config['n_layers'] = int(m.group(1))
        m = re.search(r'max_iters?=(\d+)', line)
        if m:
            config['max_iters'] = int(m.group(1))
        m = re.search(r'batch_size=(\d+)', line)
        if m:
            config['batch_size'] = int(m.group(1))
        m = re.search(r'lr=([0-9.e-]+)', line)
        if m:
            config['lr'] = m.group(1)
        m = re.search(r'[Vv]ocab\w*[\s_]?[Ss]?ize[=:\s]+(\d+)', line)
        if m:
            config['vocab_size'] = int(m.group(1))
        m = re.search(r'[Dd]evice:\s*(\S+)', line)
        if m:
            config['device'] = m.group(1)
        if 'dual_objective=True' in line or 'dual_objective' in line.lower() and 'ENABLED' in line:
            config['dual_objective'] = True
        m = re.search(r'[Mm]odels?:\s*\[(.+?)\]', line)
        if m:
            config['models'] = m.group(1)
        # Joformer config
        m = re.search(r'block_size=(\d+)', line)
        if m:
            config['block_size'] = int(m.group(1))
    return config


def extract_training_progress(lines):
    """Extract training progress lines (iter, loss, PPL)."""
    progress = {}  # model_name -> list of progress strings

    for line in lines:
        # Exp 7/8 format: [ModelName] iter N, ...
        m = re.match(r'\s*\[([^\]]+)\]\s+iter\s+(\d+)[,\s]+(.*)', line)
        if m:
            model = m.group(1)
            progress.setdefault(model, []).append(line.strip())
            continue

        # Joformer format: [model_name] iter N: train loss ...
        m = re.match(r'\s*\[([^\]]+)\]\s+iter\s+(\d+):\s+(.*)', line)
        if m:
            model = m.group(1)
            progress.setdefault(model, []).append(line.strip())
            continue

    return progress


def extract_eval_results(lines):
    """Extract evaluation results for exp7/exp8 logs."""
    text_results = {}
    kg_results = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match evaluation headers: "Evaluation: X" or "KG Evaluation: X"
        m = re.match(r'(KG )?Evaluation:\s+([A-IJ]\'?)$', line)
        if m:
            is_kg = m.group(1) is not None
            model = m.group(2)
            store = kg_results if is_kg else text_results
            if model not in store:
                store[model] = {}

            for j in range(i + 1, min(i + 80, len(lines))):
                tl = lines[j].strip()
                if re.match(r'(KG )?Evaluation:', tl) or tl.startswith('Per-relation'):
                    break
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
                    if tm.group(7) is not None:
                        entry['full_acc'] = float(tm.group(7))
                    store[model][tier] = entry
        i += 1

    return text_results, kg_results


def extract_joformer_results(lines):
    """Extract final results from joformer logs."""
    results = {}
    for line in lines:
        m = re.match(r'\s*\[(\w+)\]\s+final val loss:\s+([0-9.]+)\s+\(PPL\s+([0-9.]+)\)', line)
        if m:
            model = m.group(1)
            results[model] = {
                'val_loss': float(m.group(2)),
                'val_ppl': float(m.group(3)),
            }
    return results


def print_config(config):
    if not config:
        return
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()


def print_progress(progress, tail=None):
    if not progress:
        print("No training progress found.\n")
        return
    print("=" * 60)
    print("TRAINING PROGRESS")
    print("=" * 60)
    for model in sorted(progress.keys()):
        entries = progress[model]
        if tail:
            entries = entries[-tail:]
        print(f"\n  [{model}]")
        for e in entries:
            print(f"    {e}")
    print()


ALL_TIERS = [
    "memorization", "transfer", "generalization",
    "kg_exclusive_memorization", "kg_exclusive_generalization",
    "text_exclusive_memorization", "text_exclusive_generalization",
]

TIER_SHORT = {
    "memorization": "mem",
    "transfer": "trans",
    "generalization": "gen",
    "kg_exclusive_memorization": "kgExcl_m",
    "kg_exclusive_generalization": "kgExcl_g",
    "text_exclusive_memorization": "txtExcl_m",
    "text_exclusive_generalization": "txtExcl_g",
}


def print_eval_results(text_results, kg_results):
    if not text_results and not kg_results:
        print("No evaluation results found.\n")
        return

    models = sorted(set(list(text_results.keys()) + list(kg_results.keys())))

    if text_results:
        print("=" * 60)
        print("TEXT EVAL RESULTS")
        print("=" * 60)
        tier_labels = [TIER_SHORT.get(t, t) for t in ALL_TIERS]
        hdr = f"{'Model':<6} " + " ".join(f"{t:>9}" for t in tier_labels)
        print(f"\n  h@5:")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for m in models:
            if m not in text_results:
                continue
            vals = []
            for t in ALL_TIERS:
                d = text_results[m].get(t)
                vals.append(f"{d['hit5']:.3f}" if d else "    ...")
            print(f"  {m:<6} " + " ".join(f"{v:>9}" for v in vals))

        print(f"\n  PPL:")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for m in models:
            if m not in text_results:
                continue
            vals = []
            for t in ALL_TIERS:
                d = text_results[m].get(t)
                vals.append(f"{d['ppl']:.2f}" if d else "    ...")
            print(f"  {m:<6} " + " ".join(f"{v:>9}" for v in vals))

    if kg_results:
        print()
        print("=" * 60)
        print("KG EVAL RESULTS")
        print("=" * 60)
        tier_labels = [TIER_SHORT.get(t, t) for t in ALL_TIERS]
        hdr = f"{'Model':<6} " + " ".join(f"{t:>9}" for t in tier_labels)
        print(f"\n  h@5:")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for m in models:
            if m not in kg_results:
                continue
            vals = []
            for t in ALL_TIERS:
                d = kg_results[m].get(t)
                vals.append(f"{d['hit5']:.3f}" if d else "    ...")
            print(f"  {m:<6} " + " ".join(f"{v:>9}" for v in vals))

        print(f"\n  PPL:")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for m in models:
            if m not in kg_results:
                continue
            vals = []
            for t in ALL_TIERS:
                d = kg_results[m].get(t)
                vals.append(f"{d['ppl']:.2f}" if d else "    ...")
            print(f"  {m:<6} " + " ".join(f"{v:>9}" for v in vals))
    print()


def print_joformer_results(results):
    if not results:
        return
    print("=" * 60)
    print("JOFORMER FINAL RESULTS")
    print("=" * 60)
    print(f"  {'Model':<25} {'Val Loss':>10} {'Val PPL':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for m in sorted(results.keys()):
        r = results[m]
        print(f"  {m:<25} {r['val_loss']:>10.4f} {r['val_ppl']:>10.2f}")
    if results:
        best = min(results, key=lambda k: results[k]['val_ppl'])
        print(f"\n  Best: {best} (PPL {results[best]['val_ppl']:.2f})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Parse experiment log files (handles tqdm \\r)")
    parser.add_argument('logfile', nargs='?', help='Log file to parse')
    parser.add_argument('--find', action='store_true',
                        help='Find and list recent log files')
    parser.add_argument('--latest', action='store_true',
                        help='Parse the most recent log file found')
    parser.add_argument('--progress', action='store_true',
                        help='Show only training progress')
    parser.add_argument('--tail', type=int, default=None,
                        help='Show only last N progress lines per model')
    args = parser.parse_args()

    if args.find or (args.latest and not args.logfile):
        candidates = find_log_files()
        if not candidates:
            print("No log files found.")
            sys.exit(1)

        if args.find:
            print(f"Found {len(candidates)} log files (newest first):\n")
            for path, mtime, size in candidates[:20]:
                import datetime
                ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                size_mb = size / (1024 * 1024)
                if size_mb >= 1:
                    size_str = f"{size_mb:.1f} MB"
                else:
                    size_str = f"{size / 1024:.0f} KB"
                print(f"  {ts}  {size_str:>8}  {path}")
            if not args.latest:
                sys.exit(0)

        if args.latest:
            args.logfile = candidates[0][0]
            print(f"Using: {args.logfile}\n")

    if not args.logfile:
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.logfile):
        print(f"Error: {args.logfile} not found")
        sys.exit(1)

    lines = clean_lines(args.logfile)
    experiment = detect_experiment(lines)
    config = extract_config(lines)
    progress = extract_training_progress(lines)

    print(f"Log file: {args.logfile}")
    print(f"Lines (after \\r cleanup): {len(lines)}")
    print(f"Detected experiment: {experiment}")
    print()

    print_config(config)

    if args.progress:
        print_progress(progress, tail=args.tail)
        return

    print_progress(progress, tail=args.tail or 3)

    if experiment == "joformer":
        results = extract_joformer_results(lines)
        print_joformer_results(results)
    else:
        text_results, kg_results = extract_eval_results(lines)
        print_eval_results(text_results, kg_results)


if __name__ == "__main__":
    main()

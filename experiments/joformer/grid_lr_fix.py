#!/usr/bin/env python3
"""LR grid search for diverged joformer_projected configs."""

import subprocess
import sys
import json
import time
from pathlib import Path

CONFIGS = [
    (200, 6),
    (500, 4),
]

LR_VALUES = [1e-4, 2e-4, 3e-4]

FIXED_ARGS = {
    "max_iters": 50000,
    "wiki_lines": 2000000,
    "vocab_size": 16000,
    "block_size": 64,
    "batch_size": 32,
    "checkpoint_dir": "",
}

SCRIPT = Path(__file__).parent / "train_wiki.py"
LOG_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent


def result_file(n_embed, n_layers, lr):
    return RESULTS_DIR / f"joformer_lrfix_n{n_embed}_L{n_layers}_lr{lr}.json"


def log_file(n_embed, n_layers, lr):
    return LOG_DIR / f"joformer_lrfix_n{n_embed}_L{n_layers}_lr{lr}.log"


def run_config(n_embed, n_layers, lr):
    cmd = [
        sys.executable, str(SCRIPT),
        "--softmax",
        "--models", "joformer_projected",
        "--n_embed", str(n_embed),
        "--n_layers", str(n_layers),
        "--lr", str(lr),
    ]
    for k, v in FIXED_ARGS.items():
        cmd.extend([f"--{k}", str(v)])

    lf = log_file(n_embed, n_layers, lr)
    print(f"  Logging to {lf}", flush=True)

    with open(lf, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    return proc.returncode


def collect_results():
    latest = RESULTS_DIR / "joformer_results_latest.json"
    if latest.exists():
        with open(latest) as f:
            return json.load(f)
    return None


def main():
    all_results = {}
    runs = [(n, l, lr) for n, l in CONFIGS for lr in LR_VALUES]
    total = len(runs)
    completed = 0
    start_time = time.time()

    print(f"LR fix grid: {total} runs (joformer_projected only)", flush=True)
    print(flush=True)

    for i, (n_embed, n_layers, lr) in enumerate(runs):
        key = f"n{n_embed}_L{n_layers}_lr{lr}"
        rf = result_file(n_embed, n_layers, lr)

        if rf.exists():
            print(f"[{i+1}/{total}] n{n_embed} L{n_layers} lr={lr} — SKIPPING (result exists)", flush=True)
            with open(rf) as f:
                all_results[key] = json.load(f)
            completed += 1
            continue

        elapsed = time.time() - start_time
        if completed > 0:
            avg = elapsed / completed
            remaining = avg * (total - completed)
            eta = f"~{remaining/60:.0f} min remaining"
        else:
            eta = "estimating..."

        print(f"[{i+1}/{total}] n{n_embed} L{n_layers} lr={lr} — RUNNING ({eta})", flush=True)

        t0 = time.time()
        rc = run_config(n_embed, n_layers, lr)
        dt = time.time() - t0

        if rc != 0:
            print(f"  FAILED (exit code {rc}) after {dt/60:.1f} min", flush=True)
            continue

        data = collect_results()
        if data:
            with open(rf, "w") as f:
                json.dump(data, f, indent=2)
            all_results[key] = data
            ppl = data["results"]["joformer_projected"]["val_ppl"]
            print(f"  joformer_projected: val PPL = {ppl:.2f}", flush=True)
        else:
            print(f"  WARNING: no results file found", flush=True)

        completed += 1
        print(f"  Done in {dt/60:.1f} min", flush=True)
        print(flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("LR FIX SUMMARY (joformer_projected val PPL)", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Config':<20} {'lr=1e-4':>10} {'lr=2e-4':>10} {'lr=3e-4':>10}", flush=True)
    print("-" * 60, flush=True)
    for n, l in CONFIGS:
        vals = []
        for lr in LR_VALUES:
            key = f"n{n}_L{l}_lr{lr}"
            if key in all_results:
                ppl = all_results[key]["results"]["joformer_projected"]["val_ppl"]
                vals.append(f"{ppl:.2f}")
            else:
                vals.append("--")
        print(f"n{n}_L{l:<14} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

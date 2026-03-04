#!/usr/bin/env python3
"""Grid search over n_embed x n_layers for all JoFormer models."""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

# Grid: (n_embed, n_layers)
GRID = [
    (100, 2), (100, 4), (100, 6), (100, 8),
    (200, 2), (200, 4), (200, 6),
    (500, 2), (500, 4),
]

FIXED_ARGS = {
    "max_iters": 50000,
    "wiki_lines": 2000000,
    "vocab_size": 16000,
    "block_size": 64,
    "batch_size": 32,
    "lr": 5e-4,
    "checkpoint_dir": "",
}

SCRIPT = Path(__file__).parent / "train_wiki.py"
LOG_DIR = Path(__file__).parent.parent  # ~/
RESULTS_DIR = Path(__file__).parent


def result_file(n_embed, n_layers):
    return RESULTS_DIR / f"joformer_grid_n{n_embed}_L{n_layers}.json"


def log_file(n_embed, n_layers):
    return LOG_DIR / f"joformer_grid_n{n_embed}_L{n_layers}.log"


def run_config(n_embed, n_layers):
    """Run train_wiki.py for one config, wait for completion."""
    cmd = [
        sys.executable, str(SCRIPT),
        "--softmax",
        "--n_embed", str(n_embed),
        "--n_layers", str(n_layers),
    ]
    for k, v in FIXED_ARGS.items():
        cmd.extend([f"--{k}", str(v)])

    lf = log_file(n_embed, n_layers)
    print(f"  Logging to {lf}", flush=True)

    with open(lf, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    return proc.returncode


def collect_results():
    """Read joformer_results_latest.json after each run and rename it."""
    latest = RESULTS_DIR / "joformer_results_latest.json"
    if latest.exists():
        with open(latest) as f:
            return json.load(f)
    return None


def print_summary(all_results):
    """Print a summary table of all completed configs."""
    print("\n" + "=" * 90, flush=True)
    print("GRID SEARCH SUMMARY", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Config':<15} {'roformer':>12} {'jo_fixed':>12} {'jo_learned':>12} {'jo_projected':>12}", flush=True)
    print("-" * 90, flush=True)

    for key in sorted(all_results.keys()):
        res = all_results[key]["results"]
        vals = []
        for m in ["roformer", "joformer_fixed", "joformer_learned", "joformer_projected"]:
            if m in res and res[m]["val_ppl"] is not None:
                v = res[m]["val_ppl"]
                if v != v:  # NaN check
                    vals.append("NaN")
                else:
                    vals.append(f"{v:.2f}")
            else:
                vals.append("--")
        print(f"{key:<15} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}", flush=True)
    print("=" * 90, flush=True)


def main():
    all_results = {}
    total = len(GRID)
    completed = 0
    start_time = time.time()

    print(f"Grid search: {total} configs, 4 models each", flush=True)
    print(f"Grid: {GRID}", flush=True)
    print(flush=True)

    for i, (n_embed, n_layers) in enumerate(GRID):
        key = f"n{n_embed}_L{n_layers}"
        rf = result_file(n_embed, n_layers)

        # Skip if already done
        if rf.exists():
            print(f"[{i+1}/{total}] n_embed={n_embed}, n_layers={n_layers} — SKIPPING (result exists)", flush=True)
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

        print(f"[{i+1}/{total}] n_embed={n_embed}, n_layers={n_layers} — RUNNING ({eta})", flush=True)

        t0 = time.time()
        rc = run_config(n_embed, n_layers)
        dt = time.time() - t0

        if rc != 0:
            print(f"  FAILED (exit code {rc}) after {dt/60:.1f} min", flush=True)
            continue

        # Collect and rename results
        data = collect_results()
        if data:
            with open(rf, "w") as f:
                json.dump(data, f, indent=2)
            all_results[key] = data
            # Print per-model val PPL
            for m in data["results"]:
                ppl = data["results"][m]["val_ppl"]
                print(f"  {m}: val PPL = {ppl:.2f}", flush=True)
        else:
            print(f"  WARNING: no results file found", flush=True)

        completed += 1
        print(f"  Done in {dt/60:.1f} min", flush=True)
        print(flush=True)

    # Save combined results
    combined_path = RESULTS_DIR / "joformer_grid_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}", flush=True)

    print_summary(all_results)


if __name__ == "__main__":
    main()

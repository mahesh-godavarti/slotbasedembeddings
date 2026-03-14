#!/usr/bin/env python3
"""Grid search v4: Full wiki (streaming/memmap), lr=2e-4, 200k iters, softmax,
vocab=8000, 3 models (no joformer_learned).
Tracks best val PPL and the iteration it was achieved.

Two phases:
  1. Preprocess full wiki.en.txt to memmap binary (one-time)
  2. Train grid configs using train_wiki_streaming.py train
"""

import subprocess
import sys
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
    "max_iters": 200000,
    "block_size": 64,
    "batch_size": 32,
    "lr": 2e-4,
    "checkpoint_dir": "",
    "eval_interval": 500,
}

SCRIPT = Path(__file__).parent / "train_wiki_streaming.py"
WIKI_PATH = Path(__file__).parent.parent / "exp8" / "data" / "wiki.en.txt"
DATA_DIR = Path(__file__).parent / "data_full_v8k"
LOG_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent
MODELS = ["roformer", "joformer_fixed", "joformer_projected"]
VOCAB_SIZE = 8000


def preprocess():
    """Preprocess full wiki to memmap binary if not already done."""
    bin_path = DATA_DIR / "wiki_tokens.bin"
    if bin_path.exists():
        meta_path = DATA_DIR / "wiki_tokens.meta"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"Preprocessed data exists: {meta['total_tokens']:,} tokens from {meta.get('line_count', '?')} lines")
            return True
        print(f"Binary exists but no metadata — re-preprocessing")

    print(f"Preprocessing full wiki: {WIKI_PATH}")
    print(f"Output dir: {DATA_DIR}")
    cmd = [
        sys.executable, str(SCRIPT), "preprocess",
        "--wiki_path", str(WIKI_PATH),
        "--vocab_size", str(VOCAB_SIZE),
        "--data_dir", str(DATA_DIR),
    ]
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"Preprocessing FAILED (exit code {proc.returncode})")
        return False
    return True


def result_file(n_embed, n_layers):
    return RESULTS_DIR / f"joformer_gridv4_n{n_embed}_L{n_layers}.json"


def log_file(n_embed, n_layers):
    return LOG_DIR / f"joformer_gridv4_n{n_embed}_L{n_layers}.log"


def run_config(n_embed, n_layers):
    cmd = [
        sys.executable, str(SCRIPT), "train",
        "--softmax",
        "--data_dir", str(DATA_DIR),
        "--n_embed", str(n_embed),
        "--n_layers", str(n_layers),
        "--models",
    ] + MODELS
    for k, v in FIXED_ARGS.items():
        cmd.extend([f"--{k}", str(v)])

    lf = log_file(n_embed, n_layers)
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


def extract_best_ppl(data):
    """Extract best val PPL and its iteration for each model."""
    summary = {}
    for m in data.get("results", {}):
        r = data["results"][m]
        curve = r.get("ppl_curve", {})
        iters = curve.get("iter", [])
        val_ppls = curve.get("val_ppl", [])
        if val_ppls:
            best_val = min(val_ppls)
            best_iter = iters[val_ppls.index(best_val)]
        else:
            best_val = r.get("val_ppl", None)
            best_iter = None
        summary[m] = {
            "final_val_ppl": r.get("val_ppl"),
            "best_val_ppl": best_val,
            "best_iter": best_iter,
        }
    return summary


def print_summary(all_results):
    print("\n" + "=" * 90, flush=True)
    print("GRID SEARCH V4 SUMMARY (Full Wiki, vocab=8000) — Best Val PPL (iteration) / Final Val PPL", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Config':<12} {'roformer':>20} {'jo_fixed':>20} {'jo_projected':>20}", flush=True)
    print("-" * 90, flush=True)

    for key in sorted(all_results.keys()):
        best = all_results[key]["best"]
        vals = []
        for m in MODELS:
            if m in best and best[m]["best_val_ppl"] is not None:
                bv = best[m]["best_val_ppl"]
                bi = best[m]["best_iter"]
                fv = best[m]["final_val_ppl"]
                if bv != bv:  # NaN
                    vals.append("NaN")
                else:
                    vals.append(f"{bv:.1f}({bi//1000}k)/{fv:.1f}")
            else:
                vals.append("--")
        print(f"{key:<12} {vals[0]:>20} {vals[1]:>20} {vals[2]:>20}", flush=True)
    print("=" * 90, flush=True)


def main():
    # Phase 1: Preprocess
    print("=" * 60, flush=True)
    print("PHASE 1: Preprocessing full wiki (vocab=8000)", flush=True)
    print("=" * 60, flush=True)
    if not preprocess():
        print("Aborting grid search — preprocessing failed")
        sys.exit(1)

    # Phase 2: Grid search
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2: Grid search", flush=True)
    print("=" * 60, flush=True)

    all_results = {}
    total = len(GRID)
    completed = 0
    start_time = time.time()

    print(f"Grid search v4 (full wiki, vocab=8000): {total} configs, {len(MODELS)} models each", flush=True)
    print(f"Models: {MODELS}", flush=True)
    print(f"lr=2e-4, 200k iters, softmax, no cosine decay, vocab={VOCAB_SIZE}", flush=True)
    print(f"Grid: {GRID}", flush=True)
    print(flush=True)

    for i, (n_embed, n_layers) in enumerate(GRID):
        key = f"n{n_embed}_L{n_layers}"
        rf = result_file(n_embed, n_layers)

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
            eta = f"~{remaining/3600:.1f}h remaining"
        else:
            eta = "estimating..."

        print(f"[{i+1}/{total}] n_embed={n_embed}, n_layers={n_layers} — RUNNING ({eta})", flush=True)

        t0 = time.time()
        rc = run_config(n_embed, n_layers)
        dt = time.time() - t0

        if rc != 0:
            print(f"  FAILED (exit code {rc}) after {dt/60:.1f} min", flush=True)
            continue

        data = collect_results()
        if data:
            best = extract_best_ppl(data)
            entry = {"data": data, "best": best}
            with open(rf, "w") as f:
                json.dump(entry, f, indent=2)
            all_results[key] = entry
            for m in MODELS:
                if m in best:
                    b = best[m]
                    print(f"  {m}: best val PPL = {b['best_val_ppl']:.2f} (iter {b['best_iter']}), final = {b['final_val_ppl']:.2f}", flush=True)
        else:
            print(f"  WARNING: no results file found", flush=True)

        completed += 1
        print(f"  Done in {dt/60:.1f} min ({dt/3600:.1f}h)", flush=True)
        print(flush=True)

    # Save combined results
    combined_path = RESULTS_DIR / "joformer_gridv4_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}", flush=True)

    print_summary(all_results)


if __name__ == "__main__":
    main()

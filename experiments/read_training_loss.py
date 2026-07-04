#!/usr/bin/env python3
"""Extract training loss lines from experiment output files.

Handles tqdm \r carriage returns that interleave with print() output.
Works with both raw task output files and tee'd log files.

Usage:
    python read_training_loss.py <output_file>
    python read_training_loss.py <output_file> --tail 5    # last 5 loss lines
    python read_training_loss.py <output_file> --watch      # poll every 30s
    python read_training_loss.py <output_file> --watch 10   # poll every 10s
"""
import argparse
import os
import sys
import time


def extract_loss_lines(path):
    """Read a file and extract all loss-reporting lines, handling tqdm \r."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", errors="replace")
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Match loss lines from train_model_mixed or train_model_text_only
        if "loss" in s and "iter" in s and ("[" in s):
            # Strip tqdm prefix if present (e.g. "Model I:  5%|... [I] iter 500, ...")
            # Find the [ModelName] iter N, ... part
            import re
            m = re.search(r'\[([A-Za-z][^\]]*)\]\s+iter\s+\d+.*loss.*', s)
            if m:
                lines.append(m.group(0).strip())
    return lines


def main():
    parser = argparse.ArgumentParser(description="Extract training loss lines from experiment output")
    parser.add_argument("output_file", help="Path to output/log file")
    parser.add_argument("--tail", type=int, default=None, help="Show only last N loss lines")
    parser.add_argument("--watch", nargs="?", const=30, type=int, default=None,
                        help="Poll every N seconds (default 30)")
    args = parser.parse_args()

    if not os.path.isfile(args.output_file):
        print(f"Error: {args.output_file} not found")
        sys.exit(1)

    if args.watch is not None:
        prev_count = 0
        try:
            while True:
                lines = extract_loss_lines(args.output_file)
                if len(lines) > prev_count:
                    for line in lines[prev_count:]:
                        print(line)
                    prev_count = len(lines)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        lines = extract_loss_lines(args.output_file)
        if args.tail:
            lines = lines[-args.tail:]
        if not lines:
            print("No loss lines found.")
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()

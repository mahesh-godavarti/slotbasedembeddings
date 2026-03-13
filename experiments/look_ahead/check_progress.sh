#!/bin/bash
# Check progress of a running experiment
# Usage: bash check_progress.sh [output_file]
#
# tqdm writes progress bars using \r (carriage return) rather than \n (newline).
# Log files captured via `tee` preserve these \r characters, so the file may
# appear as one giant line to tools that split on \n. We convert \r to \n
# before parsing to handle both tee'd log files and task output files.
OUTPUT="${1:-/tmp/claude-1000/-home-ubuntu/tasks/b3jnvwyw9.output}"

echo "=== Current progress ==="
tr '\r' '\n' < "$OUTPUT" | grep -oP '\d+/100000.*?val_ppl=[\d.]+' | tail -1

echo ""
echo "=== Val PPL at each eval point ==="
tr '\r' '\n' < "$OUTPUT" | grep -oP '\d+/100000.*?val_ppl=[\d.]+' | python3 -c "
import sys
prev_ppl = None
for line in sys.stdin:
    parts = line.strip().split('val_ppl=')
    if len(parts) == 2:
        ppl = parts[1]
        it = int(parts[0].split('/')[0].strip())
        if ppl != prev_ppl and it > 0:
            print(f'{it+1:>6d}  {ppl}')
            prev_ppl = ppl
"

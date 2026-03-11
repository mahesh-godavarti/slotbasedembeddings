#!/bin/bash
# Check progress of a running experiment
# Usage: bash check_progress.sh [output_file]
OUTPUT="${1:-/tmp/claude-1000/-home-ubuntu/tasks/b3jnvwyw9.output}"

echo "=== Current progress ==="
grep -oP '\d+/100000.*?val_ppl=[\d.]+' "$OUTPUT" | tail -1

echo ""
echo "=== Val PPL at each eval point ==="
python3 -c "
import re, sys
with open('$OUTPUT', 'r') as f:
    text = f.read()
matches = re.findall(r'(\d+)/100000.*?val_ppl=([\d.]+)', text)
prev_ppl = None
for iter_str, ppl in matches:
    if ppl != prev_ppl:
        it = int(iter_str)
        if it > 0:
            print(f'{it+1:>6d}  {ppl}')
        prev_ppl = ppl
"

#!/bin/bash
# Check progress of the running experiment
OUTPUT=/tmp/claude-1000/-home-ubuntu/tasks/b6zx8xsav.output

echo "=== Current progress ==="
tail -c 500 "$OUTPUT" | tr '\r' '\n' | grep -o '[0-9]*%.*' | tail -1

echo ""
echo "=== Val PPL at each eval point ==="
grep -oP 'val_ppl=[\d.]+' "$OUTPUT" | tr '\r' '\n' | sort -u -t= -k2 -g | grep -v '19216\|19381'

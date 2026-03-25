#!/bin/bash
# Parse .output files from ViT training runs
# Usage: bash parse_results.sh <output_file>
#
# Example:
#   bash parse_results.sh /tmp/claude-1000/-home-ubuntu/.../tasks/TASKID.output

OUTPUT="${1:?Usage: bash parse_results.sh <output_file>}"

echo "=== Models in this run ==="
grep "ViT with" "$OUTPUT"

echo ""
echo "=== Final results ==="
grep "Final" "$OUTPUT"

echo ""
echo "=== Convergence (every 10 epochs) ==="
grep "Epoch\|Final" "$OUTPUT"

echo ""
echo "=== Parameter counts ==="
grep "params\|PE params" "$OUTPUT"

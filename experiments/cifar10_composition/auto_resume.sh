#!/bin/bash
# auto_resume.sh — Automatically resume interrupted training runs.
# Scans checkpoints/ for latest_*.pt files and restarts each on its original GPU.
#
# Uses a lock file to prevent multiple invocations from clashing.
# Safe to call from cron @reboot, .profile, or manually — only the first
# caller does anything; subsequent calls within the same boot are no-ops.

set -euo pipefail

SCRIPT_DIR="/home/ubuntu/cifar10_composition"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
LOG_DIR="$SCRIPT_DIR/logs"
VENV="/home/ubuntu/exp8/venv/bin/python"

# Lock: one run per boot. /tmp is cleared on reboot, so the lock resets automatically.
LOCK_FILE="/tmp/.auto_resume_training.lock"

if [ -f "$LOCK_FILE" ]; then
    echo "auto_resume: already ran this boot (lock exists at $LOCK_FILE). Exiting."
    exit 0
fi
touch "$LOCK_FILE"

# Wait for GPUs to be ready (driver may take a moment after boot)
sleep 10

mkdir -p "$LOG_DIR"

found=0
for ckpt in "$CHECKPOINT_DIR"/latest_*.pt; do
    [ -f "$ckpt" ] || continue

    # Extract pe_type from filename: latest_rope2d.pt -> rope2d
    base=$(basename "$ckpt" .pt)
    pe_type="${base#latest_}"

    # Read checkpoint: check if training is complete, get GPU and epoch
    info=$($VENV -c "
import torch, sys, json
ckpt = torch.load('$ckpt', map_location='cpu', weights_only=False)
args = ckpt['args']
epoch = ckpt['epoch']
total = args['epochs']
if epoch >= total:
    sys.exit(1)
print(json.dumps({'gpu': args['gpu'], 'epoch': epoch, 'total': total}))
" 2>/dev/null) || { echo "Skipping $pe_type — training complete or checkpoint unreadable"; continue; }

    gpu=$(echo "$info" | $VENV -c "import sys,json; print(json.load(sys.stdin)['gpu'])")
    epoch=$(echo "$info" | $VENV -c "import sys,json; print(json.load(sys.stdin)['epoch'])")
    total=$(echo "$info" | $VENV -c "import sys,json; print(json.load(sys.stdin)['total'])")

    # Reconstruct the full command from saved args
    cmd=$($VENV -c "
import torch
ckpt = torch.load('$ckpt', map_location='cpu', weights_only=False)
a = ckpt['args']
parts = []
for k, v in a.items():
    if k == 'resume':
        continue
    if isinstance(v, bool):
        if v:
            parts.append(f'--{k}')
    else:
        parts.append(f'--{k} {v}')
parts.append('--resume')
print(' '.join(parts))
")

    log_file="$LOG_DIR/imagenet_${pe_type}_resumed.log"

    echo "$(date): Resuming $pe_type on GPU $gpu from epoch $((epoch + 1))/$total" | tee -a "$log_file"
    nohup $VENV "$SCRIPT_DIR/vit_imagenet.py" $cmd >> "$log_file" 2>&1 &
    echo "  PID: $!, log: $log_file"
    found=$((found + 1))
done

if [ "$found" -eq 0 ]; then
    echo "auto_resume: no incomplete training runs found."
fi

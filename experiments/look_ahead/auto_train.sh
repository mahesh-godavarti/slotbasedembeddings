#!/bin/bash
# Auto-start training on container boot.
# Uses a lock file to ensure it only runs once per boot.
# Called from /etc/profile.d/auto_train.sh on first SSH login.

LOCK="/tmp/train_launched.lock"

# Already launched this boot
[ -f "$LOCK" ] && exit 0

# Claim the lock atomically
(set -C; echo $$ > "$LOCK") 2>/dev/null || exit 0

echo "=== Auto-launching training $(date) ==="
/home/ubuntu/look_ahead6/train_both.sh

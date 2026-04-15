#!/bin/bash
# Wait for continuation 1 to finish, then run continuation 2
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for run_scaling_continuation.sh to finish..."
while pgrep -f 'run_scaling_continuation.sh' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): Continuation 1 done. Starting continuation 2."

bash run_scaling_continuation2.sh

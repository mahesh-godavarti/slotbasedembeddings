#!/bin/bash
# Wait for scaling experiment to finish, then run continuation
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for run_scaling_experiment_resume.sh (PID 136677) to finish..."
while kill -0 136677 2>/dev/null; do
    sleep 30
done
echo "$(date): Scaling experiment done. Starting continuation."

bash run_scaling_continuation.sh

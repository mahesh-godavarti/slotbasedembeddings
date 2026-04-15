#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for D=2 C=512 to finish..."
while pgrep -f 'run_width_d2_scratch' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): D=2 C=512 done. Starting D=1 C=1944."

bash run_d1_c1943.sh

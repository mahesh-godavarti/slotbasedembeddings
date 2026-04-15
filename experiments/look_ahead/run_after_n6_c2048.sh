#!/bin/bash
set -e
cd /home/ubuntu/look_ahead6

echo "$(date): Waiting for N=6 C=2048 to finish..."
while pgrep -f 'n_embed 2048.*n_layers 6.*gpu 0' > /dev/null 2>&1; do
    sleep 30
done
echo "$(date): N=6 C=2048 done. Starting N=24 C=1088."

bash run_n24_c1088.sh

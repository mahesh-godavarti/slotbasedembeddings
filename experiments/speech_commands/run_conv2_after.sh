#!/bin/bash
source ~/experiments/venv/bin/activate
cd ~/speech_commands
while pgrep -f "speech_commands.py" > /dev/null; do
    sleep 10
done
echo "Conv done. Starting Conv2..."
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_conv2 --epochs 40 2>&1 | tee /tmp/learned_spec_conv2.log
echo "Done."

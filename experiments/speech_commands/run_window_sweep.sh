#!/bin/bash
source ~/experiments/venv/bin/activate
cd ~/speech_commands
# Wait for Conv2 to finish (it's queued after Conv)
while pgrep -f "run_conv2_after" > /dev/null; do
    sleep 10
done
# Also wait for any speech_commands.py
while pgrep -f "speech_commands.py" > /dev/null; do
    sleep 10
done
echo "Starting LearnedSpecCNN window sweep..."
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec --window 200 --epochs 40 2>&1 | tee /tmp/learned_spec_w200.log
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec --window 80 --epochs 40 2>&1 | tee /tmp/learned_spec_w80.log
echo "Done."

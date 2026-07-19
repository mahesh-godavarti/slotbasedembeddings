#!/bin/bash
source ~/experiments/venv/bin/activate
cd ~/speech_commands

# Wait for current training to finish
while pgrep -f "speech_commands.py" > /dev/null; do
    sleep 10
done

echo "LearnedSpecCNNMod done. Starting Mod2..."
PYTHONUNBUFFERED=1 python speech_commands.py --model learned_spec_mod2 --epochs 40 2>&1 | tee /tmp/learned_spec_mod2.log
echo "Done."

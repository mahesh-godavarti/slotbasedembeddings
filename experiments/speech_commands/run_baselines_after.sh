#!/bin/bash
# Wait for BlockDecayS5V2 to finish, then run baselines
source ~/experiments/venv/bin/activate
cd ~/speech_commands

# Wait for any running speech_commands process to finish
while pgrep -f "speech_commands.py" > /dev/null; do
    sleep 10
done

echo "BlockDecayS5V2 done. Starting baselines..."
PYTHONUNBUFFERED=1 python speech_commands.py --model mel --epochs 40 2>&1 | tee /tmp/mel_full.log
PYTHONUNBUFFERED=1 python speech_commands.py --model raw --epochs 40 2>&1 | tee /tmp/raw_full.log
echo "All done."

#!/bin/bash
source ~/experiments/venv/bin/activate
cd ~/speech_commands

# Wait for current learned_spec_multi4 training to finish
while pgrep -f "learned_spec_multi4" > /dev/null; do
    sleep 10
done

echo "Starting FilterbankSinCosMulti (2/bin, k=400)..."
PYTHONUNBUFFERED=1 python speech_commands.py --model filterbank_sincos_multi --window 400 --epochs 40 2>&1 | tee /tmp/filterbank_sincos_multi.log

echo "All done."

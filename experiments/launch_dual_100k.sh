#!/bin/bash
# Wait for all current experiments to finish, then launch dual 100K sweep
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/experiments

echo "$(date): Waiting for current experiments to finish..."

# Poll until all 4 current background tasks are done
while true; do
    running=0
    for pid_file in /tmp/claude-1000/-home-ubuntu-experiments/tasks/*.pid; do
        [ -f "$pid_file" ] || continue
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            running=$((running + 1))
        fi
    done
    if [ "$running" -eq 0 ]; then
        break
    fi
    echo "$(date): $running experiments still running, checking again in 60s..."
    sleep 60
done

echo "$(date): All experiments done! Launching dual 100K sweep..."

# Group 1: A/A'/F/F'/G/G'/J/J'
echo "$(date): Launching Group 1: A/A'/F/F'/G/G'/J/J'"
python kg_text_experiment_dual.py --models A "A'" F "F'" G "G'" J "J'" --n_layers 2 --n_embed 500 --dual_objective --iters 100000 --seeds 1 --exp 7a 2>&1 | tee logs/dual_100K_group1.log &
PID1=$!

# Group 2: E/E'/H/H'/I/I'
echo "$(date): Launching Group 2: E/E'/H/H'/I/I'"
python kg_text_experiment_dual.py --models E "E'" H "H'" I "I'" --n_layers 2 --n_embed 500 --dual_objective --iters 100000 --seeds 1 --exp 7a 2>&1 | tee logs/dual_100K_group2.log &
PID2=$!

# Group 3: B/B'/C/C' (kg_as_text)
echo "$(date): Launching Group 3: B/B'/C/C' (kg_as_text)"
python kg_text_experiment_dual.py --models B "B'" C "C'" --kg_as_text --n_layers 2 --n_embed 500 --dual_objective --iters 100000 --seeds 1 --exp 7a 2>&1 | tee logs/dual_100K_group3.log &
PID3=$!

echo "$(date): All 3 groups launched (PIDs: $PID1, $PID2, $PID3)"
wait $PID1 $PID2 $PID3
echo "$(date): All dual 100K experiments complete!"

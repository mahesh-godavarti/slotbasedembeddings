#!/bin/bash
# Grid sweep: K/K' backfill for both dual and non-dual
# Each (n_embed, n_layers) combo runs 2 groups sequentially:
#   1. Dual K/K' (kg_text_experiment_dual.py --dual_objective)
#   2. Non-dual K/K' (kg_text_experiment.py)
#
# Usage:
#   ./grid_sweep_K.sh              # run the full grid
#   ./grid_sweep_K.sh --dry-run    # print commands without executing
#   ./grid_sweep_K.sh --resume     # skip combos where log contains "Done."

set -e
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/experiments

# ============================================================================
# GRID CONFIGURATION — edit these to change the sweep
# ============================================================================
ITERS=100000
SEEDS=1
EXP="7a"

# Grid: (n_embed, n_layers) pairs — same grid as main sweeps
GRID=(
    # n_embed  n_layers
    "50  2"
    "50  4"
    "50  8"
    "50  16"
    "50  20"
    "100 2"
    "100 4"
    "100 8"
    "100 16"
    "100 20"
    "250 2"
    "250 4"
    "250 8"
    "500 2"
    "500 4"
)

# ============================================================================
# Parse arguments
# ============================================================================
DRY_RUN=false
RESUME=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --resume)  RESUME=true ;;
    esac
done

mkdir -p logs

# ============================================================================
# Run the grid
# ============================================================================
TOTAL=${#GRID[@]}
DONE=0

for entry in "${GRID[@]}"; do
    read -r N_EMBED N_LAYERS <<< "$entry"
    DONE=$((DONE + 1))

    echo ""
    echo "========================================================================"
    echo "$(date): Grid point $DONE/$TOTAL: n_embed=$N_EMBED, n_layers=$N_LAYERS"
    echo "========================================================================"

    # --- Group 1: Dual K/K' ---
    LOG1="logs/grid_n${N_EMBED}_l${N_LAYERS}_K.log"
    CMD1="python kg_text_experiment_dual.py --models K \"K'\" --softmax --n_embed $N_EMBED --n_layers $N_LAYERS --dual_objective --iters $ITERS --seeds $SEEDS --exp $EXP"

    if $RESUME && [ -f "$LOG1" ] && grep -q "^Done\." "$LOG1" 2>/dev/null; then
        echo "$(date): SKIP Group 1 (dual K/K') — already done: $LOG1"
    else
        echo "$(date): Group 1 (dual K/K'): $CMD1"
        if ! $DRY_RUN; then
            eval "$CMD1" 2>&1 | tee "$LOG1"
            echo "$(date): Group 1 (dual K/K') finished."
        fi
    fi

    # --- Group 2: Non-dual K/K' ---
    LOG2="logs/nd_grid_n${N_EMBED}_l${N_LAYERS}_K.log"
    CMD2="python kg_text_experiment.py --models K \"K'\" --softmax --n_embed $N_EMBED --n_layers $N_LAYERS --iters $ITERS --seeds $SEEDS --exp $EXP"

    if $RESUME && [ -f "$LOG2" ] && grep -q "^Done\." "$LOG2" 2>/dev/null; then
        echo "$(date): SKIP Group 2 (non-dual K/K') — already done: $LOG2"
    else
        echo "$(date): Group 2 (non-dual K/K'): $CMD2"
        if ! $DRY_RUN; then
            eval "$CMD2" 2>&1 | tee "$LOG2"
            echo "$(date): Group 2 (non-dual K/K') finished."
        fi
    fi

    echo "$(date): Grid point $DONE/$TOTAL complete: n${N_EMBED}_l${N_LAYERS}"
done

echo ""
echo "========================================================================"
echo "$(date): K/K' grid sweep complete! $TOTAL points evaluated."
echo "========================================================================"

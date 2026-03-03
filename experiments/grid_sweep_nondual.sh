#!/bin/bash
# Grid sweep: n_embed x n_layers using kg_text_experiment.py (non-dual)
# Each (n_embed, n_layers) combo runs 3 groups sequentially:
#   1. Mixed KG models (A-J, standard MLM KG training) + softmax
#   2. Causal KG models (E/H/I with --causal_kg) + softmax
#   3. kg_as_text models (B/C with --kg_as_text) + softmax
#
# Usage:
#   ./grid_sweep_nondual.sh              # run the full grid
#   ./grid_sweep_nondual.sh --dry-run    # print commands without executing
#   ./grid_sweep_nondual.sh --resume     # skip combos where log contains "Done."

set -e
source /home/ubuntu/experiments/venv/bin/activate
cd /home/ubuntu/experiments

# ============================================================================
# GRID CONFIGURATION — edit these to change the sweep
# ============================================================================
ITERS=100000
SEEDS=1
EXP="7a"

# Grid: (n_embed, n_layers) pairs
# At lower n_embed we can afford more layers; at higher n_embed fewer layers
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

    # --- Group 1: Mixed KG models (standard MLM KG) ---
    LOG1="logs/nd_grid_n${N_EMBED}_l${N_LAYERS}_mixed.log"
    CMD1="python kg_text_experiment.py --models A \"A'\" D \"D'\" E \"E'\" F \"F'\" G \"G'\" H \"H'\" I \"I'\" J \"J'\" --softmax --n_embed $N_EMBED --n_layers $N_LAYERS --iters $ITERS --seeds $SEEDS --exp $EXP"

    if $RESUME && [ -f "$LOG1" ] && grep -q "^Done\." "$LOG1" 2>/dev/null; then
        echo "$(date): SKIP Group 1 (mixed) — already done: $LOG1"
    else
        echo "$(date): Group 1 (mixed): $CMD1"
        if ! $DRY_RUN; then
            eval "$CMD1" 2>&1 | tee "$LOG1"
            echo "$(date): Group 1 (mixed) finished."
        fi
    fi

    # --- Group 2: Causal KG models (E/H/I with --causal_kg) ---
    LOG2="logs/nd_grid_n${N_EMBED}_l${N_LAYERS}_causal.log"
    CMD2="python kg_text_experiment.py --models E \"E'\" H \"H'\" I \"I'\" --causal_kg --softmax --n_embed $N_EMBED --n_layers $N_LAYERS --iters $ITERS --seeds $SEEDS --exp $EXP"

    if $RESUME && [ -f "$LOG2" ] && grep -q "^Done\." "$LOG2" 2>/dev/null; then
        echo "$(date): SKIP Group 2 (causal) — already done: $LOG2"
    else
        echo "$(date): Group 2 (causal): $CMD2"
        if ! $DRY_RUN; then
            eval "$CMD2" 2>&1 | tee "$LOG2"
            echo "$(date): Group 2 (causal) finished."
        fi
    fi

    # --- Group 3: kg_as_text models (B/C) ---
    LOG3="logs/nd_grid_n${N_EMBED}_l${N_LAYERS}_kat.log"
    CMD3="python kg_text_experiment.py --models B \"B'\" C \"C'\" --kg_as_text --softmax --n_embed $N_EMBED --n_layers $N_LAYERS --iters $ITERS --seeds $SEEDS --exp $EXP"

    if $RESUME && [ -f "$LOG3" ] && grep -q "^Done\." "$LOG3" 2>/dev/null; then
        echo "$(date): SKIP Group 3 (kat) — already done: $LOG3"
    else
        echo "$(date): Group 3 (kat): $CMD3"
        if ! $DRY_RUN; then
            eval "$CMD3" 2>&1 | tee "$LOG3"
            echo "$(date): Group 3 (kat) finished."
        fi
    fi

    echo "$(date): Grid point $DONE/$TOTAL complete: n${N_EMBED}_l${N_LAYERS}"
done

echo ""
echo "========================================================================"
echo "$(date): Grid sweep complete! $TOTAL points evaluated."
echo "========================================================================"

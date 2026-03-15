#!/bin/bash
# =============================================================================
# Large-scale C=768 experiments: corr_ffn variants vs roformer baselines
# =============================================================================
#
# MACHINE: 8× A100 80GB (AWS p4de.24xlarge, ~$40/hr)
#   - All 12 experiments finish in ~2.5 hours ($100 total)
#   - Alternatively: 8× H100 (p5.48xlarge, ~$98/hr) finishes in ~1.5 hours
#   - A100 40GB (p4d.24xlarge, ~$33/hr) works too — may need batch=32 for D=5
#
# SETUP on new machine:
#   1. Copy look_ahead6/ directory (models.py + train_wiki_streaming.py + this script)
#   2. Copy data: scp -r look_ahead/look_ahead/data_full/ to DATA_DIR below (3.7GB)
#   3. Install deps: pip install torch numpy tqdm tokenizers
#   4. Adjust paths below, then: bash run_c768_experiments.sh
#
# FLOP comparison (sequential K=1 inference):
#   corr_ffn_add:    (12D+8)C²    corr_ffn_concat: (12D+12)C²
#   roformer N:      12NC²
#
#   D=2 add=32C²(44%) D=3 add=44C²(61%) D=4 add=56C²(78%) D=5 add=68C²(94%)
#   D=2 cat=36C²(50%) D=3 cat=48C²(67%) D=4 cat=60C²(83%) D=5 cat=72C²(100%)
#   roformer: N=3=36C² N=4=48C² N=5=60C² N=6=72C² (baseline)
# =============================================================================

set -euo pipefail

# ---- Configurable paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_wiki_streaming.py"
PYTHON="${PYTHON:-/home/ubuntu/exp8/venv/bin/python}"
DATA_DIR="${DATA_DIR:-/home/ubuntu/look_ahead/look_ahead/data_full}"
LOG_DIR="${SCRIPT_DIR}/logs/c768"

# ---- Training settings ----
N_EMBED=768
BLOCK_SIZE=256
BATCH_SIZE=64
LR="2e-4"
MAX_ITERS=100000
EVAL_INTERVAL=5000
COMMON="--n_embed ${N_EMBED} --block_size ${BLOCK_SIZE} --batch_size ${BATCH_SIZE} \
--lr ${LR} --softmax \
--max_iters ${MAX_ITERS} --eval_interval ${EVAL_INTERVAL} \
--data_dir ${DATA_DIR} --amp"
# Extra args only for corr_ffn models (ignored by roformer)
CORR_FFN_ARGS="--convergence_weight 0.1 --k_min 2"

# ---- Setup ----
mkdir -p "${LOG_DIR}"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "============================================"
echo "  C=768 Experiment Suite"
echo "  GPUs detected: ${NUM_GPUS}"
echo "  Logs: ${LOG_DIR}/"
echo "============================================"
echo ""

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi

# ---- Define experiments ----
# Priority 1-4 first (the core story), then the rest slowest-first
# Format: "name|model|extra_args"
EXPERIMENTS=(
    # --- Priority: these 4 tell the full story ---
    # 1. The baseline to beat
    "roformer_n6|roformer|--n_layers 6"
    # 2. FLOP-matched to N=6 — validates scaling advantage
    "corr_ffn_concat_d5|block_head_corr_ffn_concat|--n_layers 25 --d_block 5 ${CORR_FFN_ARGS}"
    # 3. 39% fewer FLOPs — the big savings story
    "corr_ffn_add_d3|block_head_corr_ffn_add|--n_layers 15 --d_block 3 ${CORR_FFN_ARGS}"
    # 4. Intermediate reference for D=3 comparison
    "roformer_n4|roformer|--n_layers 4"

    # --- Rest: fill in the full picture, slowest first ---
    "corr_ffn_add_d5|block_head_corr_ffn_add|--n_layers 25 --d_block 5 ${CORR_FFN_ARGS}"
    "corr_ffn_concat_d4|block_head_corr_ffn_concat|--n_layers 20 --d_block 4 ${CORR_FFN_ARGS}"
    "corr_ffn_add_d4|block_head_corr_ffn_add|--n_layers 20 --d_block 4 ${CORR_FFN_ARGS}"
    "corr_ffn_concat_d3|block_head_corr_ffn_concat|--n_layers 15 --d_block 3 ${CORR_FFN_ARGS}"
    "corr_ffn_concat_d2|block_head_corr_ffn_concat|--n_layers 10 --d_block 2 ${CORR_FFN_ARGS}"
    "corr_ffn_add_d2|block_head_corr_ffn_add|--n_layers 10 --d_block 2 ${CORR_FFN_ARGS}"
    "roformer_n5|roformer|--n_layers 5"
    "roformer_n3|roformer|--n_layers 3"
)

# ---- GPU job scheduler ----
declare -A GPU_PIDS   # GPU -> PID
declare -A GPU_NAMES  # GPU -> experiment name

launch_on_gpu() {
    local gpu=$1 name=$2 model=$3 extra=$4
    echo "[$(date '+%H:%M:%S')] GPU ${gpu}: ${name}"
    CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PYTHON $TRAIN_SCRIPT train \
        --models $model \
        $COMMON $extra \
        > "${LOG_DIR}/${name}_c768.log" 2>&1 &
    GPU_PIDS[$gpu]=$!
    GPU_NAMES[$gpu]=$name
}

wait_for_free_gpu() {
    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            local pid=${GPU_PIDS[$gpu]:-0}
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                if [ "$pid" -ne 0 ]; then
                    wait "$pid" 2>/dev/null || true
                    echo "[$(date '+%H:%M:%S')] GPU ${gpu}: ${GPU_NAMES[$gpu]} DONE"
                fi
                echo "$gpu"
                return
            fi
        done
        sleep 5
    done
}

# ---- Launch all experiments ----
echo "Launching ${#EXPERIMENTS[@]} experiments..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name model extra <<< "$exp"
    gpu=$(wait_for_free_gpu)
    launch_on_gpu "$gpu" "$name" "$model" "$extra"
done

# Wait for all remaining
echo ""
echo "All launched. Waiting for stragglers..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    pid=${GPU_PIDS[$gpu]:-0}
    if [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        echo "[$(date '+%H:%M:%S')] GPU ${gpu}: ${GPU_NAMES[$gpu]} DONE"
    fi
done

# ---- Extract results ----
echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
echo ""
printf "%-25s  %8s  %8s\n" "Experiment" "FLOPs" "Final PPL"
printf "%-25s  %8s  %8s\n" "-------------------------" "--------" "---------"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name model extra <<< "$exp"
    logfile="${LOG_DIR}/${name}_c768.log"
    if [ -f "$logfile" ]; then
        # Extract last val_ppl from log
        ppl=$(tr '\r' '\n' < "$logfile" | grep -oP 'val_ppl=[\d.]+' | tail -1 | cut -d= -f2)
        # Determine FLOP budget
        case "$name" in
            corr_ffn_add_d2)    flops="32C²" ;;
            corr_ffn_concat_d2) flops="36C²" ;;
            corr_ffn_add_d3)    flops="44C²" ;;
            corr_ffn_concat_d3) flops="48C²" ;;
            corr_ffn_add_d4)    flops="56C²" ;;
            corr_ffn_concat_d4) flops="60C²" ;;
            corr_ffn_add_d5)    flops="68C²" ;;
            corr_ffn_concat_d5) flops="72C²" ;;
            roformer_n3)        flops="36C²" ;;
            roformer_n4)        flops="48C²" ;;
            roformer_n5)        flops="60C²" ;;
            roformer_n6)        flops="72C²" ;;
        esac
        printf "%-25s  %8s  %8s\n" "$name" "$flops" "${ppl:-FAILED}"
    else
        printf "%-25s  %8s  %8s\n" "$name" "—" "NO LOG"
    fi
done

echo ""
echo "Detailed logs: ${LOG_DIR}/"
echo "Check progress while running: bash check_progress.sh ${LOG_DIR}/<name>_c768.log"

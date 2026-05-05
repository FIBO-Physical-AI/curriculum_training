#!/usr/bin/env bash
# Parameter sweep for gait emergence (Run 4).
#
# Usage:
#   bash src/scripts/run_sweep_v4.sh SIGMA_LOW SIGMA_MID SIGMA_HIGH
#
# Example (values chosen after reviewing Phase 0 diagnostic):
#   bash src/scripts/run_sweep_v4.sh 250 500 1000
#
# Add RESUME=1 to skip runs whose eval CSV already exists:
#   RESUME=1 bash src/scripts/run_sweep_v4.sh 250 500 1000
#
# Tunable env vars (with defaults shown):
#   MAX_ITERATIONS=3000
#   NUM_ENVS=4096
#   ROLLOUTS_PER_BIN=100
set -uo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SIGMA_LOW="${1:?Usage: bash src/scripts/run_sweep_v4.sh SIGMA_LOW SIGMA_MID SIGMA_HIGH}"
SIGMA_MID="${2:?Usage: bash src/scripts/run_sweep_v4.sh SIGMA_LOW SIGMA_MID SIGMA_HIGH}"
SIGMA_HIGH="${3:?Usage: bash src/scripts/run_sweep_v4.sh SIGMA_LOW SIGMA_MID SIGMA_HIGH}"

MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROLLOUTS_PER_BIN="${ROLLOUTS_PER_BIN:-100}"
RESUME="${RESUME:-0}"

RESULTS_DIR="$PROJECT_ROOT/src/results"
SWEEP_LOG="$RESULTS_DIR/sweep_v4_output.log"
TIMING_LOG="$RESULTS_DIR/sweep_v4_timings.txt"
EXP_DIR="$PROJECT_ROOT/unitree_rl_lab/logs/rsl_rl/curriculum_go2_velocity_uniform"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "=== run_sweep_v4 starting at $(date) — output captured to $SWEEP_LOG ==="
echo "=== σ_low=$SIGMA_LOW  σ_mid=$SIGMA_MID  σ_high=$SIGMA_HIGH ==="
echo "=== MAX_ITERATIONS=$MAX_ITERATIONS  NUM_ENVS=$NUM_ENVS ==="

: > "$TIMING_LOG"
echo "===== SWEEP_V4 $(date '+%Y-%m-%d %H:%M:%S') σ=($SIGMA_LOW,$SIGMA_MID,$SIGMA_HIGH) =====" >> "$TIMING_LOG"

# 9-run grid: (sigma std alpha_en)
SIGMAS=("$SIGMA_LOW"  "$SIGMA_MID"  "$SIGMA_HIGH"
        "$SIGMA_LOW"  "$SIGMA_MID"  "$SIGMA_HIGH"
        "$SIGMA_LOW"  "$SIGMA_MID"  "$SIGMA_HIGH")
STDS=(  "0.5"         "0.5"         "0.5"
        "1.0"         "1.0"         "1.0"
        "1.0"         "1.0"         "1.0")
ALPHAS=("1.0"         "1.0"         "1.0"
        "1.0"         "1.0"         "1.0"
        "0.5"         "0.5"         "0.5")

for run_id in 1 2 3 4 5 6 7 8 9; do
    idx=$((run_id - 1))
    sigma="${SIGMAS[$idx]}"
    std="${STDS[$idx]}"
    alpha_en="${ALPHAS[$idx]}"
    sigma_z=$(python -c "print($sigma * 0.5)")

    eval_csv="$RESULTS_DIR/sweep_v4_run${run_id}_eval.csv"
    traces_dir="$RESULTS_DIR/sweep_v4_run${run_id}_traces"
    trace_npz="$traces_dir/uniform_seed0.npz"

    echo ""
    echo "=========================================="
    echo "RUN $run_id/9  σ=$sigma  σ_z=$sigma_z  std=$std  α_en=$alpha_en"
    echo "=========================================="

    if [ "$RESUME" = "1" ] && [ -s "$eval_csv" ]; then
        echo "  -> already done (eval CSV exists), skipping"
        continue
    fi

    # ── TRAIN ──────────────────────────────────────────────────────────────
    train_start=$(date +%s)
    train_ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[TRAIN_START] run=$run_id  at $train_ts" | tee -a "$TIMING_LOG"

    export SWEEP_SIGMA_EN_X="$sigma"
    export SWEEP_SIGMA_EN_Z="$sigma_z"
    export SWEEP_TRACK_STD="$std"
    export SWEEP_ENERGY_WEIGHT="$alpha_en"

    python src/scripts/train.py \
        --condition uniform \
        --seed 0 \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITERATIONS"
    train_rc=$?

    train_stop=$(date +%s)
    train_elapsed=$((train_stop - train_start))
    train_min=$(echo "scale=1; $train_elapsed / 60" | bc)
    printf "[TRAIN_STOP]  run=%d  elapsed=%ds (%.1fmin)  rc=%d\n" \
        "$run_id" "$train_elapsed" "$train_min" "$train_rc" | tee -a "$TIMING_LOG"

    if [ "$train_rc" -ne 0 ]; then
        echo "  WARN: train failed for run $run_id (rc=$train_rc) — logging NaN row"
        python src/scripts/sweep_param_grid.py \
            --append-run \
            --run-id   "$run_id" \
            --sigma    "$sigma" \
            --std      "$std" \
            --alpha-en "$alpha_en" \
            --wall-sec "$train_elapsed" || true
        continue
    fi

    # find checkpoint (newest dir, newest model_*.pt inside)
    latest=$(ls -td "$EXP_DIR"/*/ 2>/dev/null | head -1)
    latest="${latest%/}"
    if [ -z "$latest" ]; then
        echo "FATAL: no log dir under $EXP_DIR after run $run_id" >&2
        exit 1
    fi
    ckpt=$(ls -t "$latest"/model_*.pt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "FATAL: no model_*.pt in $latest (run $run_id)" >&2
        exit 1
    fi
    echo "  -> checkpoint: $ckpt"

    # ── EVAL ───────────────────────────────────────────────────────────────
    eval_start=$(date +%s)
    echo "[EVAL_START]  run=$run_id  at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$TIMING_LOG"

    mkdir -p "$traces_dir"
    python src/scripts/eval_epte.py \
        --condition uniform \
        --seed 0 \
        --checkpoint "$ckpt" \
        --rollouts-per-bin "$ROLLOUTS_PER_BIN" \
        --out-csv "$eval_csv" \
        --traces-dir "$traces_dir"
    eval_rc=$?

    eval_elapsed=$(($(date +%s) - eval_start))
    printf "[EVAL_STOP]   run=%d  elapsed=%ds  rc=%d\n" \
        "$run_id" "$eval_elapsed" "$eval_rc" | tee -a "$TIMING_LOG"

    if [ "$eval_rc" -ne 0 ] || [ ! -s "$eval_csv" ]; then
        echo "  WARN: eval failed for run $run_id — logging NaN row"
        python src/scripts/sweep_param_grid.py \
            --append-run \
            --run-id   "$run_id" \
            --sigma    "$sigma" \
            --std      "$std" \
            --alpha-en "$alpha_en" \
            --wall-sec "$train_elapsed" \
            --ckpt     "$ckpt"   || true
        continue
    fi

    # ── APPEND ROW TO MASTER CSV ────────────────────────────────────────────
    python src/scripts/sweep_param_grid.py \
        --append-run \
        --run-id    "$run_id" \
        --sigma     "$sigma" \
        --std       "$std" \
        --alpha-en  "$alpha_en" \
        --eval-csv  "$eval_csv" \
        --trace-npz "$trace_npz" \
        --wall-sec  "$train_elapsed" \
        --ckpt      "$ckpt"

    echo ""
done

# ── GENERATE COMPARISON MARKDOWN ────────────────────────────────────────────
echo ""
echo "=========================================="
echo "COMPARISON TABLE"
echo "=========================================="
python src/scripts/sweep_param_grid.py --compare

# ── QUICK SUMMARY ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "SWEEP V4 SUMMARY"
echo "=========================================="
if [ -s "$RESULTS_DIR/sweep_v4.csv" ]; then
    echo ""
    echo "--- tracking error per run (b0=walk, b4=mid, b7=sprint) ---"
    awk -F, 'NR>1 {
        printf "  run=%-2s  σ=%-6s  std=%-3s  α=%-3s  err_b0=%-6s  err_b4=%-6s  err_b7=%-6s  wall=%smin\n",
            $1, $2, $3, $4, $7, $11, $14, $55
    }' "$RESULTS_DIR/sweep_v4.csv"
    echo ""
    echo "--- duty factor per run (b0=walk, b4=mid, b7=sprint) ---"
    awk -F, 'NR>1 {
        printf "  run=%-2s  duty_b0=%-6s  duty_b4=%-6s  duty_b7=%-6s\n",
            $1, $23, $27, $30
    }' "$RESULTS_DIR/sweep_v4.csv"
else
    echo "  (sweep_v4.csv not found)"
fi

echo ""
echo "=========================================="
echo ">>> run_sweep_v4 done at $(date)"
echo ">>> sweep_v4.csv         -> $RESULTS_DIR/sweep_v4.csv"
echo ">>> sweep_v4_comparison  -> $RESULTS_DIR/sweep_v4_comparison.md"
echo ">>> gait diagrams        -> $RESULTS_DIR/figures/sweep_v4_run_*_gait.png"
echo ">>> full log             -> $SWEEP_LOG"

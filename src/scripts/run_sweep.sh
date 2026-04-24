#!/usr/bin/env bash
set -uo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/src/results"
SWEEP_LOG="$PROJECT_ROOT/src/results/sweep_output.log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "=== sweep script starting at $(date) — output captured to $SWEEP_LOG ==="

CONDITIONS=(${CONDITIONS:-uniform task_specific teacher})
SEEDS=(${SEEDS:-0})
MAX_ITERATIONS=${MAX_ITERATIONS:-6000}
NUM_ENVS=${NUM_ENVS:-2048}
VIDEO_ENVS=${VIDEO_ENVS:-1}
VIDEO_LENGTH=${VIDEO_LENGTH:-400}
NUM_BINS=${NUM_BINS:-8}
V_MAX=${V_MAX:-3.0}
RECORD_VIDEOS=${RECORD_VIDEOS:-1}
STEPS_PER_ITER=${STEPS_PER_ITER:-48}
export CURRICULUM_STEPS_PER_ITER="$STEPS_PER_ITER"

TIMING_LOG="$PROJECT_ROOT/src/results/run_timings.txt"
mkdir -p "$(dirname "$TIMING_LOG")"
mkdir -p "$PROJECT_ROOT/.sweep_runs"
: > "$TIMING_LOG"

declare -A EXP_NAME=(
    [uniform]=curriculum_go2_velocity_uniform
    [task_specific]=curriculum_go2_velocity_taskspec
    [teacher]=curriculum_go2_velocity_teacher
)

echo ">>> sweep config: conditions=(${CONDITIONS[*]}) seeds=(${SEEDS[*]}) I_max=${MAX_ITERATIONS}"
echo "===== SWEEP $(date '+%Y-%m-%d %H:%M:%S') conditions=(${CONDITIONS[*]}) seeds=(${SEEDS[*]}) I_max=${MAX_ITERATIONS} =====" | tee -a "$TIMING_LOG"

for condition in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "=========================================="
        echo "TRAIN  condition=${condition}  seed=${seed}"
        echo "=========================================="
        start_epoch=$(date +%s)
        start_ts=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[START] ${condition} seed=${seed}  at ${start_ts}" | tee -a "$TIMING_LOG"

        python src/scripts/train.py \
            --condition "$condition" \
            --seed "$seed" \
            --headless \
            --num_envs "$NUM_ENVS" \
            --max_iterations "$MAX_ITERATIONS"
        train_rc=$?

        stop_epoch=$(date +%s)
        stop_ts=$(date '+%Y-%m-%d %H:%M:%S')
        elapsed=$((stop_epoch - start_epoch))
        printf "[STOP]  %s seed=%d  at %s  elapsed=%ds (%dh%02dm%02ds)  rc=%d\n" \
            "$condition" "$seed" "$stop_ts" "$elapsed" \
            $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) "$train_rc" \
            | tee -a "$TIMING_LOG"

        exp_name="${EXP_NAME[$condition]}"
        exp_dir="$PROJECT_ROOT/unitree_rl_lab/logs/rsl_rl/$exp_name"
        marker_file="$PROJECT_ROOT/.sweep_runs/${condition}_seed${seed}.path"
        latest=""
        if [ -d "$exp_dir" ]; then
            latest=$(ls -td "$exp_dir"/*/ 2>/dev/null | head -1)
            latest="${latest%/}"
        fi
        if [ -z "$latest" ]; then
            echo "FATAL: no log directory under $exp_dir after training ${condition} seed=${seed}" >&2
            echo "Check $TIMING_LOG — training likely failed." >&2
            exit 1
        fi
        echo "$latest" > "$marker_file"
        echo "  -> marker ${condition}_seed${seed}.path = $latest" | tee -a "$TIMING_LOG"

        ckpt=$(ls -t "${latest}"/model_*.pt 2>/dev/null | head -1)
        if [ -z "$ckpt" ]; then
            echo "FATAL: no model_*.pt checkpoint in $latest (${condition} seed=${seed})" >&2
            exit 1
        fi

        echo ""
        echo "------ EVAL  ${condition} seed=${seed}  <- ${ckpt}"
        eval_start=$(date +%s)
        echo "[EVAL_START] ${condition} seed=${seed}  at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$TIMING_LOG"
        python src/scripts/eval_epte.py \
            --condition "$condition" --seed "$seed" --checkpoint "$ckpt"
        eval_elapsed=$(( $(date +%s) - eval_start ))
        printf "[EVAL_STOP]  %s seed=%d  elapsed=%ds\n" "$condition" "$seed" "$eval_elapsed" \
            | tee -a "$TIMING_LOG"

        if [ ! -s "$PROJECT_ROOT/src/results/epte_sp.csv" ]; then
            echo "FATAL: eval_epte.py ran but epte_sp.csv is empty/missing for ${condition} seed=${seed}" >&2
            exit 1
        fi

        if [ "$RECORD_VIDEOS" != "1" ]; then
            echo ""
            echo "------ PLAY  ${condition} seed=${seed}  SKIPPED (RECORD_VIDEOS=${RECORD_VIDEOS})"
            echo "[PLAY_SKIP]  ${condition} seed=${seed}" | tee -a "$TIMING_LOG"
            continue
        fi

        echo ""
        echo "------ PLAY  ${condition} seed=${seed}  (per-bin videos, ${NUM_BINS} bins, v_max=${V_MAX})"
        play_start=$(date +%s)
        echo "[PLAY_START] ${condition} seed=${seed}  at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$TIMING_LOG"

        bin_width=$(python -c "print($V_MAX / $NUM_BINS)")
        video_dir="$latest/videos/play"

        for B in $(seq 0 $((NUM_BINS-1))); do
            v_center=$(python -c "print(round(($B + 0.5) * $bin_width, 3))")
            echo ""
            echo "  --- bin $B  v=${v_center} m/s"
            python src/scripts/play.py \
                --condition "$condition" \
                --bin "$B" \
                --num_envs "$VIDEO_ENVS" \
                --video \
                --video_length "$VIDEO_LENGTH" \
                --headless
            play_rc=$?
            if [ "$play_rc" -ne 0 ]; then
                echo "  WARN: play.py returned rc=$play_rc for bin $B (continuing)" >&2
                continue
            fi
            if [ -d "$video_dir" ]; then
                latest_mp4=$(ls -t "$video_dir"/*.mp4 2>/dev/null | grep -v "/bin[0-9]*_v" | head -1)
                if [ -n "$latest_mp4" ]; then
                    target="$video_dir/bin${B}_v${v_center}.mp4"
                    mv -f "$latest_mp4" "$target"
                    echo "  -> saved $target"
                fi
            fi
        done

        play_elapsed=$(( $(date +%s) - play_start ))
        printf "[PLAY_STOP]  %s seed=%d  elapsed=%ds\n" "$condition" "$seed" "$play_elapsed" \
            | tee -a "$TIMING_LOG"
    done
done

echo ""
echo "=========================================="
echo "PLOT   regenerating all figures"
echo "=========================================="
python src/scripts/plot_all.py

echo ""
echo ">>> sweep done. checkpoints + videos in unitree_rl_lab/logs/rsl_rl/*/"
echo ">>> figures in src/results/figures/"

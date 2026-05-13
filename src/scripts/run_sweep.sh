#!/usr/bin/env bash
set -uo pipefail
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/src/results}"
mkdir -p "$RESULTS_DIR"
SWEEP_LOG="$RESULTS_DIR/sweep_output.log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "=== sweep script starting at $(date) — output captured to $SWEEP_LOG ==="
echo "=== RESULTS_DIR = $RESULTS_DIR ==="

CONDITIONS=(${CONDITIONS:-uniform task_specific teacher})
SEEDS=(${SEEDS:-0})
MAX_ITERATIONS=${MAX_ITERATIONS:-3000}
NUM_ENVS=${NUM_ENVS:-4096}
VIDEO_ENVS=${VIDEO_ENVS:-1}
VIDEO_LENGTH=${VIDEO_LENGTH:-200}
NUM_BINS=${NUM_BINS:-8}
V_MAX=${V_MAX:-4.0}
RECORD_VIDEOS=${RECORD_VIDEOS:-1}
STEPS_PER_ITER=${STEPS_PER_ITER:-48}
export CURRICULUM_STEPS_PER_ITER="$STEPS_PER_ITER"

TIMING_LOG="$RESULTS_DIR/run_timings.txt"
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
        marker_file="$RESULTS_DIR/.sweep_runs/${condition}_seed${seed}.path"
        mkdir -p "$RESULTS_DIR/.sweep_runs"
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

        if [ -f "$latest/curriculum.csv" ]; then
            mirror_dir="$RESULTS_DIR/curriculum_runs/${exp_name}/seed${seed}"
            mkdir -p "$mirror_dir"
            cp -f "$latest/curriculum.csv" "$mirror_dir/curriculum.csv"
            echo "  -> mirrored curriculum.csv to $mirror_dir" | tee -a "$TIMING_LOG"
        else
            echo "  WARN: $latest/curriculum.csv not found — learning_curves/heatmap/ITM/convergence plots will skip" | tee -a "$TIMING_LOG"
        fi

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
            --condition "$condition" --seed "$seed" --checkpoint "$ckpt" \
            --out-csv "$RESULTS_DIR/epte_sp.csv" \
            --traces-dir "$RESULTS_DIR/eval_traces"
        eval_elapsed=$(( $(date +%s) - eval_start ))
        printf "[EVAL_STOP]  %s seed=%d  elapsed=%ds\n" "$condition" "$seed" "$eval_elapsed" \
            | tee -a "$TIMING_LOG"

        if [ ! -s "$RESULTS_DIR/epte_sp.csv" ]; then
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
        push_dir="$RESULTS_DIR/videos/${condition}_seed${seed}"
        mkdir -p "$push_dir"

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
                    cp -f "$target" "$push_dir/bin${B}_v${v_center}.mp4"
                    echo "  -> saved $target"
                    echo "  -> copied to $push_dir/bin${B}_v${v_center}.mp4"
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
echo "RAMP   best seed per condition (by bin-$((NUM_BINS-1)) tracking error)"
echo "=========================================="
TARGET_BIN=$((NUM_BINS-1))
for condition in "${CONDITIONS[@]}"; do
    if [ ! -s "$RESULTS_DIR/epte_sp.csv" ]; then
        echo "SKIP RAMP ${condition}: epte_sp.csv missing" >&2
        continue
    fi

    best_seed=$(awk -F, -v cond="$condition" -v tb="$TARGET_BIN" '
        NR>1 && $1==cond && ($3+0)==tb {sum[$2]+=$6; n[$2]++}
        END {
            best_seed=""; best_err=1e18
            for (s in sum) {
                avg = sum[s]/n[s]
                if (avg < best_err) { best_err = avg; best_seed = s }
            }
            print best_seed
        }
    ' "$RESULTS_DIR/epte_sp.csv")

    if [ -z "$best_seed" ]; then
        echo "SKIP RAMP ${condition}: no rows for bin ${TARGET_BIN} in epte_sp.csv" >&2
        continue
    fi

    marker_file="$RESULTS_DIR/.sweep_runs/${condition}_seed${best_seed}.path"
    if [ ! -f "$marker_file" ]; then
        echo "SKIP RAMP ${condition}: no marker $marker_file" >&2
        continue
    fi
    run_dir=$(cat "$marker_file")
    ckpt=$(ls -t "${run_dir}"/model_*.pt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "SKIP RAMP ${condition}: no checkpoint in $run_dir" >&2
        continue
    fi

    echo ""
    echo "------ RAMP  ${condition}  best_seed=${best_seed}  <- ${ckpt}"
    ramp_start=$(date +%s)
    echo "[RAMP_START] ${condition} best_seed=${best_seed}  at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$TIMING_LOG"

    ramp_push_dir="$RESULTS_DIR/videos/${condition}_seed${best_seed}"
    mkdir -p "$ramp_push_dir"
    ramp_video_args=()
    if [ "$RECORD_VIDEOS" = "1" ]; then
        ramp_video_args=(--video --video-dir "$ramp_push_dir")
    fi

    rm -f "$RESULTS_DIR/ramp_${condition}_seed"*.npz

    python src/scripts/eval_ramp.py \
        --condition "$condition" \
        --seed "$best_seed" \
        --checkpoint "$ckpt" \
        --out "$RESULTS_DIR/ramp_${condition}_seed${best_seed}.npz" \
        "${ramp_video_args[@]}"
    ramp_rc=$?

    ramp_elapsed=$(( $(date +%s) - ramp_start ))
    printf "[RAMP_STOP]  %s best_seed=%s  elapsed=%ds  rc=%d\n" \
        "$condition" "$best_seed" "$ramp_elapsed" "$ramp_rc" | tee -a "$TIMING_LOG"
    if [ "$ramp_rc" -ne 0 ]; then
        echo "WARN: eval_ramp.py returned rc=$ramp_rc for ${condition} best_seed=${best_seed} — gait transition plot may be incomplete" >&2
    fi
done

echo ""
echo "=========================================="
echo "PLOT   regenerating all figures"
echo "=========================================="
if [ -d "$RESULTS_DIR/curriculum_runs" ]; then
    PLOT_LOGS_ROOT="$RESULTS_DIR/curriculum_runs"
else
    PLOT_LOGS_ROOT="$PROJECT_ROOT/unitree_rl_lab/logs/rsl_rl"
fi
python src/scripts/plot_all.py \
    --logs-root "$PLOT_LOGS_ROOT" \
    --out-dir "$RESULTS_DIR/figures" \
    --epte-csv "$RESULTS_DIR/epte_sp.csv" \
    --traces-dir "$RESULTS_DIR/eval_traces" \
    --ramps-dir "$RESULTS_DIR" \
    --sigma-phase0-csv "$RESULTS_DIR/phase0_table.csv" \
    --sigma-v4-csv "$RESULTS_DIR/sweep_v4.csv"

echo ""
echo "=========================================="
echo "EPTE SUMMARY"
echo "=========================================="
if [ -s "$RESULTS_DIR/epte_sp.csv" ]; then
    echo ""
    echo "--- mean EPTE per condition ---"
    awk -F, 'NR>1 {sum[$1]+=$7; n[$1]++} END {for (k in sum) printf "  %-15s %.3f  (n=%d)\n", k, sum[k]/n[k], n[k]}' \
        "$RESULTS_DIR/epte_sp.csv" | sort
    echo ""
    echo "--- per-bin diagnostic (cond bin  fall  err  epte  v_act_signed  early/100) ---"
    awk -F, 'NR>1 {key=$1"|"$3; fall[key]+=$5; err[key]+=$6; epte[key]+=$7; vsig[key]+=$9; n[key]++; if($5<999)e[key]++} END {for(k in n){split(k,a,"|"); printf "  %-15s b%s  fall=%6.1f  err=%.3f  epte=%.3f  v_act=%+5.2f  early=%d/%d\n", a[1], a[2], fall[k]/n[k], err[k]/n[k], epte[k]/n[k], vsig[k]/n[k], e[k]+0, n[k]}}' \
        "$RESULTS_DIR/epte_sp.csv" | sort
else
    echo "  (no epte_sp.csv found)"
fi

echo ""
echo "=========================================="
echo "CURRICULUM STATE (final weights per bin)"
echo "=========================================="
for condition in "${CONDITIONS[@]}"; do
    exp_name="${EXP_NAME[$condition]}"
    marker_file="$RESULTS_DIR/.sweep_runs/${condition}_seed${SEEDS[0]}.path"
    if [ -f "$marker_file" ]; then
        run_dir=$(cat "$marker_file")
        curr_csv="$run_dir/curriculum.csv"
        if [ -s "$curr_csv" ]; then
            echo ""
            echo "--- $condition (last iter) ---"
            python -c "
import csv, sys
rows = list(csv.reader(open('$curr_csv')))[1:]
if not rows: sys.exit()
last_step = max(int(r[0]) for r in rows)
last = [r for r in rows if int(r[0]) == last_step]
last.sort(key=lambda r: int(r[1]))
print(f'  step {last_step}: ' + '  '.join(f'b{r[1]}={float(r[2]):.2f}' for r in last))
print(f'  rewards    : ' + '  '.join(f'b{r[1]}={float(r[3]):.2f}' for r in last))
print(f'  n_samples  : ' + '  '.join(f'b{r[1]}={r[4]}' for r in last))
"
        fi
    fi
done

echo ""
echo ">>> sweep done. checkpoints + videos in unitree_rl_lab/logs/rsl_rl/*/"
echo ">>> figures in $RESULTS_DIR/figures/"

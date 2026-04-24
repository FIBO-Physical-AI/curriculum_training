#!/usr/bin/env bash
# Record one short video per task bin for a trained policy.
#
# Usage:
#   bash src/scripts/play_per_bin.sh <condition> [num_bins] [video_length] [num_envs]
# Example:
#   bash src/scripts/play_per_bin.sh uniform 8 400 1
#
# Videos land in unitree_rl_lab/logs/rsl_rl/<exp>/<latest_run>/videos/play/
# and are renamed to bin<i>_v<center>.mp4 between runs so they don't collide.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDITION="${1:?condition required: uniform|task_specific|teacher}"
NUM_BINS="${2:-8}"
VIDEO_LENGTH="${3:-400}"
NUM_ENVS="${4:-1}"
V_MAX="${V_MAX:-3.0}"

declare -A EXP_NAME=(
    [uniform]=curriculum_go2_velocity_uniform
    [task_specific]=curriculum_go2_velocity_taskspec
    [teacher]=curriculum_go2_velocity_teacher
)
exp="${EXP_NAME[$CONDITION]}"
latest=$(ls -td "$PROJECT_ROOT/unitree_rl_lab/logs/rsl_rl/$exp"/*/ | head -1)
latest="${latest%/}"
video_dir="$latest/videos/play"

bin_width=$(python -c "print($V_MAX / $NUM_BINS)")

for B in $(seq 0 $((NUM_BINS-1))); do
    v_center=$(python -c "print(($B + 0.5) * $bin_width)")
    echo ""
    echo "=========================================="
    echo "PLAY  condition=$CONDITION  bin=$B  v=${v_center} m/s"
    echo "=========================================="

    python src/scripts/play.py \
        --condition "$CONDITION" \
        --bin "$B" \
        --num_envs "$NUM_ENVS" \
        --video \
        --video_length "$VIDEO_LENGTH" \
        --headless

    if [ -d "$video_dir" ]; then
        # rename the freshly-written mp4 so the next iteration doesn't overwrite it
        latest_mp4=$(ls -t "$video_dir"/*.mp4 2>/dev/null | head -1)
        if [ -n "$latest_mp4" ]; then
            target="$video_dir/bin${B}_v${v_center}.mp4"
            mv -f "$latest_mp4" "$target"
            echo "  -> saved $target"
        fi
    fi
done

echo ""
echo ">>> per-bin videos in $video_dir"

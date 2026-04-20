#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONDITIONS=(uniform task_specific teacher)
SEEDS=(0 1 2)

for condition in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=== condition=${condition} seed=${seed} ==="
        python scripts/train.py --condition "$condition" --seed "$seed"
    done
done

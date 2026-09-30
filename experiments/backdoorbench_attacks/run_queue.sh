#!/usr/bin/env bash
# Runs the GPU stages of this experiment resumably, 1 model per GPU lock slot.
#
#   bash experiments/backdoorbench_attacks/run_queue.sh evaluate
#   bash experiments/backdoorbench_attacks/run_queue.sh sweep
#
# evaluate writes results/_experiments/backdoorbench_attacks/evaluation/<folder>.json
# and sweep writes results/bb_<folder>/, and a model whose output is complete is
# skipped, so a rerun picks up where the last one stopped. Each model holds 1 of
# the 2 shared lock slots. No model starts after 06:30, or when its estimated
# duration would carry it past 07:00, or before 17:00.

set -uo pipefail
WORKTREE=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WORKTREE"
source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS=4

MODE=$1
LOCKS=(scratch/gpu.lock scratch/gpu2.lock)
LOG_DIR=scratch/backdoorbench_attacks
mkdir -p "$LOG_DIR"
# Minutes a model holds the GPU, measured on the smoke model and rounded up.
declare -A ESTIMATE_MINUTES=([evaluate]=3 [sweep]=${SWEEP_MINUTES:-40})

fits_in_window() {
    local now finish
    now=$(date +%H%M)
    finish=$(date -d "+${ESTIMATE_MINUTES[$MODE]} minutes" +%H%M)
    # 10# forces base 10, so 0630 is not read as octal.
    if ((10#$now >= 630 && 10#$now < 1700)); then
        return 1
    fi
    if ((10#$now < 700 && 10#$finish >= 700)); then
        return 1
    fi
    return 0
}

# -E 75 makes a held lock exit with 75, which tells a busy slot apart from a
# stage that ran and failed.
on_free_slot() {
    local lock
    for lock in "${LOCKS[@]}"; do
        if flock -n -E 75 "$lock" "$@"; then
            return 0
        else
            local status=$?
            if ((status != 75)); then
                return "$status"
            fi
        fi
    done
    flock "${LOCKS[0]}" "$@"
}

is_done() {
    local folder=$1
    if [[ $MODE == evaluate ]]; then
        [[ -f results/_experiments/backdoorbench_attacks/evaluation/$folder.json ]]
    else
        python - "$folder" <<'PY'
import os, sys
from experiments.backdoorbench_attacks.sweep_model import results_dir_of
folder = sys.argv[1]
sys.exit(0 if os.path.exists(os.path.join(results_dir_of(folder), f"bb_{folder}", "psbd_metrics.json")) else 1)
PY
    fi
}

run_model() {
    local folder=$1
    if is_done "$folder"; then
        return 0
    fi
    if ! fits_in_window; then
        echo "$(date +%T) outside the GPU window, stopping before $folder"
        return 1
    fi
    echo "$(date +%T) $MODE $folder"
    local module=experiments.backdoorbench_attacks.sweep_model
    [[ $MODE == evaluate ]] && module=experiments.backdoorbench_attacks.evaluate
    if on_free_slot python -m "$module" --folder "$folder" >>"$LOG_DIR/$MODE.$folder.log" 2>&1; then
        echo "$(date +%T) $(tail -n 1 "$LOG_DIR/$MODE.$folder.log")"
    else
        echo "$(date +%T) failed $folder, see $LOG_DIR/$MODE.$folder.log"
    fi
}

mapfile -t FOLDERS < <(python -m experiments.backdoorbench_attacks.queue)
for folder in "${FOLDERS[@]}"; do
    run_model "$folder" || break
done
echo "$(date +%T) $MODE queue finished"

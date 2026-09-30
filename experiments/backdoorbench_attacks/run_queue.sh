#!/usr/bin/env bash
# Runs the GPU stages of this experiment resumably, 1 model per GPU lock slot.
#
#   bash experiments/backdoorbench_attacks/run_queue.sh model
#
# Each model runs model_job.py (evaluation, reproduction gate, sweep) inside 1 of
# the 2 shared lock slots and leaves jobs/<folder>.json under
# results/_experiments/backdoorbench_attacks/. A model with that record is
# skipped, so a rerun picks up where the last one stopped. Exit 3 from a model
# (not reproduced) stops the queue. No model starts after 06:30, or when its
# estimated duration would carry it past 07:00, or before 17:00.

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
ESTIMATE_MINUTES=${MODEL_MINUTES:-40}

fits_in_window() {
    local now finish
    now=$(date +%H%M)
    finish=$(date -d "+${ESTIMATE_MINUTES} minutes" +%H%M)
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
    [[ -f results/_experiments/backdoorbench_attacks/jobs/$1.json ]]
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
    echo "$(date +%T) start $folder"
    on_free_slot python -m experiments.backdoorbench_attacks.model_job --folder "$folder" >>"$LOG_DIR/model.$folder.log" 2>&1
    local status=$?
    echo "$(date +%T) exit $status $folder: $(tail -n 1 "$LOG_DIR/model.$folder.log")"
    if ((status == 3)); then
        echo "$(date +%T) $folder not reproduced, stopping the queue"
        return 1
    fi
    return 0
}

mapfile -t FOLDERS < <(python -m experiments.backdoorbench_attacks.queue)
for folder in "${FOLDERS[@]}"; do
    run_model "$folder" || break
done
echo "$(date +%T) $MODE queue finished"

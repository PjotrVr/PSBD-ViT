#!/usr/bin/env bash
# Runs run_passes.py over the plan in order (tuning, dev, holdout, rest), resumably.
#
# A model whose run_record.json exists is skipped, so a restart on a later night
# continues where this one stopped. Each model holds 1 of the 2 shared GPU lock
# slots for its whole run, and no model starts between 06:15 and 17:00, which leaves
# the slowest model (about 15 minutes) room to finish before the 07:00 curfew.
#
#   bash experiments/tact_calibration/queue.sh

set -uo pipefail
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS=4

MODELS=results/_experiments/tact_calibration/models
PLAN=results/_experiments/tact_calibration/plan.json
LOCKS=(scratch/gpu.lock scratch/gpu2.lock)
LOG_DIR=results/_experiments/tact_calibration/logs
mkdir -p "$LOG_DIR"

inside_gpu_window() {
    local now
    now=$(date +%H%M)
    # 10# forces base 10, so 0615 is not read as octal.
    if ((10#$now >= 615 && 10#$now < 1700)); then
        return 1
    fi
    return 0
}

# -E 75 makes a held lock exit with 75, which tells a busy slot apart from a
# run that started and failed.
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

folders=$(python -c "import json; print(' '.join(m['folder'] for m in json.load(open('$PLAN'))['models']))")
for folder in $folders; do
    if [[ -f "$MODELS/$folder/run_record.json" ]]; then
        continue
    fi
    if ! inside_gpu_window; then
        echo "$(date +%T) outside the GPU window, stopping before $folder"
        exit 0
    fi
    echo "$(date +%T) start $folder"
    if on_free_slot python -m experiments.tact_calibration.run_passes run --folder "$folder" \
        >"$LOG_DIR/$folder.log" 2>&1; then
        echo "$(date +%T) done $folder"
    else
        echo "$(date +%T) FAILED $folder, see $LOG_DIR/$folder.log"
    fi
done
echo "$(date +%T) queue finished"

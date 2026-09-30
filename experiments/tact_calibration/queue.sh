#!/usr/bin/env bash
# Runs run_passes.py over the plan in order (tuning, dev, holdout, rest), resumably.
#
# A model with a run_record.json or a failed.json is skipped, so a restart on a
# later night continues where this one stopped. Each lock hold runs pending models
# for about 40 minutes (run_passes.py's budget), because up to 6 agents wait on the
# 2 shared slots and a slot released after every model is rarely won back. No
# model starts between 06:15 and 17:00, which leaves the slowest model (about 15
# minutes) room to finish before the 07:00 curfew.
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
LOG=results/_experiments/tact_calibration/logs/run_passes.log

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

pending_folders() {
    python - "$PLAN" "$MODELS" <<'PY'
import json, os, sys
plan, models = sys.argv[1], sys.argv[2]
folders = [m["folder"] for m in json.load(open(plan))["models"]]
done = ("run_record.json", "failed.json")
print(" ".join(f for f in folders if not any(os.path.exists(os.path.join(models, f, d)) for d in done)))
PY
}

previous=""
while true; do
    pending=$(pending_folders)
    if [[ -z "$pending" ]]; then
        echo "$(date +%T) queue finished"
        exit 0
    fi
    if [[ "$pending" == "$previous" ]]; then
        echo "$(date +%T) a lock hold made no progress, stopping"
        exit 1
    fi
    if ! inside_gpu_window; then
        echo "$(date +%T) outside the GPU window, stopping with $(wc -w <<<"$pending") models left"
        exit 0
    fi
    previous=$pending
    echo "$(date +%T) waiting for a slot, $(wc -w <<<"$pending") models left"
    # shellcheck disable=SC2086
    on_free_slot python -m experiments.tact_calibration.run_passes run --folders $pending >>"$LOG" 2>&1
done

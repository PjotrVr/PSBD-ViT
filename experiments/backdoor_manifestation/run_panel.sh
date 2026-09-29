#!/usr/bin/env bash
# Runs the whole panel on the login node GPU, resumably.
#
# A run whose JSON already exists is skipped, so the script can be restarted any
# night. Pairs are prepared on the CPU outside the lock. Each probe holds the
# shared GPU lock for 1 model only and releases it, since other agents queue on
# the same lock. No probe starts between 06:30 and 17:00, the hours the GPU
# belongs to interactive users. The marker file tells the next agent in the GPU
# queue that this experiment is finished for the night.
#
#   bash experiments/backdoor_manifestation/run_panel.sh

set -uo pipefail
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export PYTHONPATH=.

SCRIPT=experiments/backdoor_manifestation/measure.py
RUNS_DIR=results/_experiments/backdoor_manifestation/runs
PAIRS_DIR=scratch/backdoor_manifestation/pairs
LOCKS=(scratch/gpu.lock scratch/gpu2.lock)
MARKER=scratch/gpu_done_manifestation
LOG_DIR=scratch/backdoor_manifestation/logs
mkdir -p "$LOG_DIR"

inside_gpu_window() {
    local now
    now=$(date +%H%M)
    # 10# forces base 10, so 0630 is not read as octal.
    if ((10#$now >= 630 && 10#$now < 1700)); then
        return 1
    fi
    return 0
}

# 2 lock slots share the GPU. Take whichever is free, and queue on the first
# when both are busy. -E 75 makes a held lock exit with 75, which tells a busy
# slot apart from a probe that ran and failed.
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

for run in $(python "$SCRIPT" list 2>/dev/null); do
    if [[ -f "$RUNS_DIR/$run.json" ]]; then
        continue
    fi
    if ! inside_gpu_window; then
        echo "$(date +%T) outside the GPU window, stopping before $run"
        break
    fi
    if [[ ! -f "$PAIRS_DIR/$run.pt" ]]; then
        python "$SCRIPT" prepare --run "$run" >"$LOG_DIR/$run.prepare.log" 2>&1 \
            || { echo "$(date +%T) prepare failed for $run"; continue; }
    fi
    echo "$(date +%T) probing $run"
    if on_free_slot python "$SCRIPT" probe --run "$run" >"$LOG_DIR/$run.probe.log" 2>&1; then
        tail -n 1 "$LOG_DIR/$run.probe.log"
    else
        echo "$(date +%T) probe failed for $run"
    fi
done

touch "$MARKER"
echo "$(date +%T) done, marker written"

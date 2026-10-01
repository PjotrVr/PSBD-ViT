#!/usr/bin/env bash
# Runs the GPU stages of this experiment resumably, 1 model per GPU lock slot.
#
#   bash experiments/backdoorbench_attacks/run_queue.sh model
#
# Each model runs model_job.py (evaluation, reproduction gate, sweep) inside 1 of
# the 2 shared lock slots and leaves jobs/<folder>.json under
# results/_experiments/backdoorbench_attacks/. A model with that record is
# skipped, so a rerun picks up where the last one stopped. Exit 3 from a model
# (not reproduced) is recorded in its jobs record and the queue moves on, since
# 1 attack's reproduction says nothing about another's (changed 2026-10-01 after
# CIFAR-10 Input-Aware stopped the queue). No model starts after 06:30, or when its
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

# Takes whichever slot is free, and otherwise waits on both slots at once as a
# blocking waiter, the pattern of experiments/resnet_tact/run_queue.sh. A waiter
# blocked on 1 slot never sees the other come free, which left a slot idle for
# half an hour on 2026-10-01. The first waiter to get its slot claims the stage
# with an atomic mkdir, and the other waiter, still blocked with no child, is
# stopped. -E 75 tells a busy slot apart from a stage that ran and failed.
on_free_slot() {
    local lock status
    for lock in "${LOCKS[@]}"; do
        flock -n -E 75 "$lock" "$@"
        status=$?
        ((status != 75)) && return "$status"
    done

    local claim="$LOG_DIR/claim.$$.$RANDOM"
    local claimed='mkdir "$0" 2>/dev/null || exit 0; exec "${@}"'
    flock "${LOCKS[0]}" bash -c "$claimed" "$claim" "$@" &
    local first=$!
    flock "${LOCKS[1]}" bash -c "$claimed" "$claim" "$@" &
    local second=$!
    until [[ -d "$claim" ]]; do
        sleep 5
    done
    sleep 2

    local winner=$first loser=$second
    if [[ -z "$(ps -o pid= --ppid "$first")" ]]; then
        winner=$second
        loser=$first
    fi
    kill "$loser" 2>/dev/null
    wait "$winner"
    status=$?
    wait "$loser" 2>/dev/null
    rmdir "$claim"
    return "$status"
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
        echo "$(date +%T) $folder not reproduced, recorded and skipped"
    fi
    return 0
}

mapfile -t FOLDERS < <(python -m experiments.backdoorbench_attacks.queue)
for folder in "${FOLDERS[@]}"; do
    run_model "$folder" || break
done
echo "$(date +%T) $MODE queue finished"

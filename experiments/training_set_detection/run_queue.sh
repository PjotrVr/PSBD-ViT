#!/usr/bin/env bash
# Scores the training-set detection queue on the shared login GPU, resumably.
#
# The paper-mirror set runs first, then the development models it does not hold.
# Pass 1 scores every model with PSBD only and pass 2 adds STRIP and CD-L, so the
# lock time the shared card grants goes to the main question first.
# Each model holds 1 of the 2 shared GPU lock slots for its whole scoring and
# releases it before the next, no model starts between 06:30 and 17:00, and a
# model whose every part is on disk is skipped by score.py itself, so rerunning
# this script after a curfew resumes the queue.
#
#   bash experiments/training_set_detection/run_queue.sh            # full queue
#   bash experiments/training_set_detection/run_queue.sh smoke MODEL  # 128-image smoke

set -uo pipefail
# Overridable so a frozen copy of this script can run while the original is
# edited, since bash reads a running script lazily.
WORKTREE=${WORKTREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MAIN=/lustre/home/pstika/projects/PSBD-ViT
cd "$MAIN"
source .venv/bin/activate
export PYTHONPATH="$WORKTREE"

SCORE="$WORKTREE/experiments/training_set_detection/score.py"
LOCKS=(scratch/gpu.lock scratch/gpu2.lock)
LOG_DIR=scratch/training_set_detection
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

# -E 75 makes a held lock exit with 75, which tells a busy slot apart from a
# stage that ran and failed. Both slots are polled, since a process blocked on
# the first slot never notices the second one coming free.
on_free_slot() {
    local lock status
    while true; do
        for lock in "${LOCKS[@]}"; do
            flock -n -E 75 "$lock" "$@"
            status=$?
            if ((status != 75)); then
                return "$status"
            fi
        done
        sleep 20
    done
}

if [[ "${1:-}" == "smoke" ]]; then
    on_free_slot python "$SCORE" --models "$2" --smoke 128 2>&1 | tee "$LOG_DIR/smoke.$2.log"
    exit $?
fi

# A stdin script puts the working directory first on sys.path, where the main
# checkout's own experiments/ package would shadow the worktree's.
mapfile -t QUEUE < <(python -c "
import sys
sys.path.insert(0, '$WORKTREE')
from experiments.training_set_detection.common import model_queue
print('\n'.join(model_queue()))
")
if ((${#QUEUE[@]} == 0)); then
    echo "empty queue" >&2
    exit 1
fi

run_pass() {
    local pass=$1
    shift
    local folder marker
    for folder in "${QUEUE[@]}"; do
        marker="$LOG_DIR/done_$pass.$folder"
        if [[ -f "$marker" ]]; then
            continue
        fi
        if ! inside_gpu_window; then
            echo "$(date +%T) outside the GPU window, stopping before $pass $folder" | tee -a "$LOG_DIR/queue.log"
            return 1
        fi
        echo "$(date +%T) start $pass $folder" | tee -a "$LOG_DIR/queue.log"
        if on_free_slot python "$SCORE" --models "$folder" "$@" >>"$LOG_DIR/score.$folder.log" 2>&1; then
            touch "$marker"
            echo "$(date +%T) $pass $(tail -n 1 "$LOG_DIR/score.$folder.log")" | tee -a "$LOG_DIR/queue.log"
        else
            echo "$(date +%T) failed $pass $folder" | tee -a "$LOG_DIR/queue.log"
        fi
    done
}

run_pass psbd --skip-detectors && run_pass detectors
echo "$(date +%T) training-set detection queue finished" | tee -a "$LOG_DIR/queue.log"

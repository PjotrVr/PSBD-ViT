#!/usr/bin/env bash
# Scores the training-set detection queue on the shared login GPU, resumably.
#
# The paper-mirror set runs first, then the development models it does not hold.
# Each model holds 1 of the 2 shared GPU lock slots for its whole scoring and
# releases it before the next, no model starts between 06:30 and 17:00, and a
# model whose every part is on disk is skipped by score.py itself, so rerunning
# this script after a curfew resumes the queue.
#
#   bash experiments/training_set_detection/run_queue.sh            # full queue
#   bash experiments/training_set_detection/run_queue.sh smoke MODEL  # 128-image smoke

set -uo pipefail
WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
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

if [[ "${1:-}" == "smoke" ]]; then
    on_free_slot python "$SCORE" --models "$2" --smoke 128 2>&1 | tee "$LOG_DIR/smoke.$2.log"
    exit $?
fi

mapfile -t QUEUE < <(python - <<'PY'
from experiments.training_set_detection.common import model_queue
print("\n".join(model_queue()))
PY
)

for folder in "${QUEUE[@]}"; do
    if [[ -f "$LOG_DIR/done.$folder" ]]; then
        continue
    fi
    if ! inside_gpu_window; then
        echo "$(date +%T) outside the GPU window, stopping before $folder" | tee -a "$LOG_DIR/queue.log"
        break
    fi
    echo "$(date +%T) start $folder" | tee -a "$LOG_DIR/queue.log"
    if on_free_slot python "$SCORE" --models "$folder" >>"$LOG_DIR/score.$folder.log" 2>&1; then
        touch "$LOG_DIR/done.$folder"
        echo "$(date +%T) $(tail -n 1 "$LOG_DIR/score.$folder.log")" | tee -a "$LOG_DIR/queue.log"
    else
        echo "$(date +%T) failed $folder" | tee -a "$LOG_DIR/queue.log"
    fi
done
echo "$(date +%T) training-set detection queue finished" | tee -a "$LOG_DIR/queue.log"

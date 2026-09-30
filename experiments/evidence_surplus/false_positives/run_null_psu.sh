#!/usr/bin/env bash
# Runs the GPU stages of the null-referenced PSU, resumably: the bfloat16 nulls and the
# fresh 3-pass dumps, the 12 A3 models first and then the rest of the panel.
#
# A model whose dump exists is skipped. Each model holds 1 of the 2 shared GPU lock
# slots, and no model starts between 06:30 and 17:00.
#
#   bash experiments/evidence_surplus/false_positives/run_null_psu.sh

set -uo pipefail
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export PYTHONPATH=.

MODULE=experiments.evidence_surplus.false_positives.null_psu
DUMPS=results/_experiments/evidence_surplus/false_positives/null_psu/dumps
LOCKS=(scratch/gpu.lock scratch/gpu2.lock)
LOG_DIR=scratch/evidence_surplus/logs
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

run_model() {
    local folder=$1
    if [[ -f "$DUMPS/$folder.pt" ]]; then
        return 0
    fi
    if ! inside_gpu_window; then
        echo "$(date +%T) outside the GPU window, stopping before $folder"
        return 1
    fi
    echo "$(date +%T) $folder"
    if on_free_slot python -m "$MODULE" null --device cuda --models "$folder" >"$LOG_DIR/null_psu.$folder.log" 2>&1 \
        && on_free_slot python -m "$MODULE" dump --device cuda --models "$folder" >>"$LOG_DIR/null_psu.$folder.log" 2>&1; then
        tail -n 1 "$LOG_DIR/null_psu.$folder.log"
    else
        echo "$(date +%T) failed $folder"
    fi
}

mapfile -t A3 < <(python - <<'PY'
from experiments.evidence_surplus.surplus_factor.removal import MODELS
print("\n".join(MODELS))
PY
)
mapfile -t PANEL < <(python - <<'PY'
from scripts.paper._common import clearing_cells, load_coverage
print("\n".join(c["folder_name"] for c in clearing_cells(load_coverage("results"))))
PY
)

for folder in "${A3[@]}" "${PANEL[@]}"; do
    run_model "$folder" || break
done
echo "$(date +%T) null-referenced PSU queue finished"

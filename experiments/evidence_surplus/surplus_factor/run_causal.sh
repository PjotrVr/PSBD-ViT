#!/usr/bin/env bash
# Runs A2 (manufactured surplus) and A3 (surplus removal) after experiment A, resumably.
#
# A2 reads A's last-block surplus factors, so it waits for A's marker. A model
# whose JSON exists is skipped. Each probe holds 1 of the 2 shared GPU lock slots
# for 1 model, and no probe starts between 06:30 and 17:00.
#
#   bash experiments/evidence_surplus/surplus_factor/run_causal.sh

set -uo pipefail
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export PYTHONPATH=.

HERE=experiments/evidence_surplus/surplus_factor
RESULTS=results/_experiments/evidence_surplus
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
# probe that ran and failed.
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
    local script=$1 folder=$2 output=$3
    if [[ -f "$output" ]]; then
        return 0
    fi
    if ! inside_gpu_window; then
        echo "$(date +%T) outside the GPU window, skipping $script $folder"
        return 1
    fi
    echo "$(date +%T) $script $folder"
    if on_free_slot python "$HERE/$script" probe --model "$folder" >"$LOG_DIR/$script.$folder.log" 2>&1; then
        tail -n 3 "$LOG_DIR/$script.$folder.log"
    else
        echo "$(date +%T) failed $script $folder"
    fi
}

while [[ ! -f scratch/evidence_surplus/surplus_factor_done ]]; do
    sleep 60
done
python "$HERE/measure.py" summarize >"$LOG_DIR/a_summary.log" 2>&1

for folder in vit_cifar10_badnet_a2o_0_01 vit_cifar10_blend_0_1 vit_cifar10_benign vit_gtsrb_benign; do
    run_model manufactured.py "$folder" "$RESULTS/manufactured/$folder.json" || break
done
for folder in vit_cifar10_badnet_a2o_0_01 vit_tiny_badnet_a2o_0_05 vit_cifar100_blend_0_1 \
    vit_cifar10_blend_0_1 vit_cifar10_bpp_0_05 vit_gtsrb_lf_0_01 vit_cifar10_wanet_0_1 \
    vit_tiny_wanet_0_05 vit_gtsrb_tact_0_05 vit_cifar10_tact_0_01 vit_cifar10_benign vit_gtsrb_benign; do
    run_model removal.py "$folder" "$RESULTS/removal/$folder.json" || break
done
echo "$(date +%T) causal queue finished"

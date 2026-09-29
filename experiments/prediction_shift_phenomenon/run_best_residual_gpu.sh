#!/usr/bin/env bash
# Claim 3 for the best residual placement, added to the ViT feature records.
#
# Same lock protocol as run_gpu.sh: 1 of the 2 slots per model, 14 GB cap, no
# new model after 06:30. Only the br_adaptive condition runs and is merged into
# each existing record, so the other conditions are not recomputed.
#
#     bash experiments/prediction_shift_phenomenon/run_best_residual_gpu.sh
set -u
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS=8
LOG_DIR=logs/prediction_shift_phenomenon
MEASURE=experiments/prediction_shift_phenomenon/measure.py

run_locked() {
    local code
    flock -n -E 75 scratch/gpu.lock "$@"
    code=$?
    [[ $code -ne 75 ]] && return $code
    flock -n -E 75 scratch/gpu2.lock "$@"
    code=$?
    [[ $code -ne 75 ]] && return $code
    flock scratch/gpu.lock "$@"
}

FOLDERS=(
    vit_gtsrb_badnet_a2o_0_1 vit_gtsrb_blend_0_1 vit_gtsrb_tact_0_05 vit_gtsrb_benign
    vit_cifar10_badnet_a2o_0_1 vit_cifar10_tact_0_01 vit_cifar10_tact_0_05
    vit_cifar10_blend_0_1 vit_cifar10_lf_0_1 vit_cifar10_wanet_0_1
    vit_cifar10_bpp_0_1 vit_cifar10_benign
)
for folder in "${FOLDERS[@]}"; do
    now=$(date +%H%M)
    if [[ "$now" > "0629" && "$now" < "1700" ]]; then
        echo "cutoff reached before $folder" >> "$LOG_DIR/run.log"
        exit 0
    fi
    echo "$(date -Is) start best residual $folder" >> "$LOG_DIR/run.log"
    run_locked python "$MEASURE" --stage features --folders "$folder" \
        --add-conditions br_adaptive --gpu-memory-gb 14 --batch-size 128 \
        >> "$LOG_DIR/$folder.log" 2>&1
    echo "$(date -Is) end best residual $folder exit $?" >> "$LOG_DIR/run.log"
done

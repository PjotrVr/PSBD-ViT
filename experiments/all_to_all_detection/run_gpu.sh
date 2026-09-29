#!/bin/bash
# The placements the caches lack for this experiment, swept on the login node GPU
# between 17:00 and 06:30, 1 model per hold of scratch/gpu.lock. Each sweep writes
# under results/_experiments/all_to_all_detection/caches/, never the canonical
# tree, and reuses the canonical baselines and manifest through symlinks. A sweep
# whose rates are all on disk is skipped, so the script resumes.
#
#     bash experiments/all_to_all_detection/run_gpu.sh
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export OMP_NUM_THREADS=4
ROOT=results/_experiments/all_to_all_detection/caches

sweep() {
    local folder=$1
    shift
    local now
    now=$(date +%H%M)
    if [ "$now" -ge 0630 ] && [ "$now" -lt 1700 ]; then
        echo "window closed, $folder not started"
        exit 0
    fi
    mkdir -p "$ROOT/$folder/psbd"
    for file in results/"$folder"/psbd/baseline_*.pt results/"$folder"/psbd/split_manifest.json; do
        [ -e "$ROOT/$folder/psbd/$(basename "$file")" ] || ln -s "$(realpath "$file")" "$ROOT/$folder/psbd/"
    done
    echo "[start] $folder $* $(date +%H:%M)"
    flock scratch/gpu.lock python experiments/all_to_all_detection/sweep_capped.py \
        --checkpoint-folder "$folder" --results-dir "$ROOT" --skip-existing "$@" 2>&1 \
        | grep -E "FAILED|Error|rate|skip" | tail -3
    echo "[done] $folder $(date +%H:%M)"
}

LATE_VIT=(--position pre_residual --operator dropout --block-range 9 12)
LATE_SWIN=(--position pre_residual --operator dropout --block-range 17 24)
TOKEN_MASK=(--position before_attention_norm --operator token_mask)
BENIGN_PROBE=(--probe-attack badnet_a2o --probe-target-label 0)

sweep vit_cifar100_badnet_a2a_0_1 "${LATE_VIT[@]}"
sweep vit_tiny_badnet_a2a_0_05 "${LATE_VIT[@]}"
sweep vit_tiny_badnet_a2a_0_1 "${LATE_VIT[@]}"
sweep vit_gtsrb_badnet_a2a_0_005 "${LATE_VIT[@]}"
sweep swin_cifar10_badnet_a2a_0_01 "${TOKEN_MASK[@]}"
sweep swin_cifar10_badnet_a2a_0_05 "${TOKEN_MASK[@]}"
sweep swin_cifar10_badnet_a2a_0_1 "${TOKEN_MASK[@]}"
sweep swin_gtsrb_badnet_a2a_0_01 "${LATE_SWIN[@]}"
sweep swin_gtsrb_badnet_a2a_0_05 "${LATE_SWIN[@]}"
sweep swin_gtsrb_badnet_a2a_0_1 "${LATE_SWIN[@]}"
sweep swin_gtsrb_badnet_a2a_0_005 "${LATE_SWIN[@]}"
sweep swin_cifar10_benign "${LATE_SWIN[@]}" "${BENIGN_PROBE[@]}"
sweep swin_gtsrb_benign "${LATE_SWIN[@]}" "${BENIGN_PROBE[@]}"
sweep swin_cifar100_benign "${LATE_SWIN[@]}" "${BENIGN_PROBE[@]}"
sweep swin_cifar100_badnet_a2a_0_1 "${TOKEN_MASK[@]}"

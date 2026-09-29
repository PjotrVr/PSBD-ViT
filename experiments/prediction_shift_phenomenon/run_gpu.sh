#!/usr/bin/env bash
# The GPU half of the prediction shift experiment, queued on the shared login-node GPU.
#
# Every model is 1 process under 1 of the 2 GPU lock slots (scratch/gpu.lock and
# scratch/gpu2.lock), so a slot is held per model and released between models for
# the other agents sharing the GPU. A free slot is taken without waiting, and when
# both are busy the job queues on the first.
# measure.py skips a model whose output already exists, so rerunning this script
# resumes where the last run stopped. No job starts after 06:30, because the GPU
# window closes at 07:00. The marker file tells the next agent the GPU is free.
#
#     bash experiments/prediction_shift_phenomenon/run_gpu.sh
set -u
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export PYTHONPATH=.
# The login node is shared by several CPU-heavy agents, and an unbounded thread
# pool per process drove the load past the core count.
export OMP_NUM_THREADS=8

MARKER=scratch/gpu_done_phenomenon
LOG_DIR=logs/prediction_shift_phenomenon
MEASURE=experiments/prediction_shift_phenomenon/measure.py
# 14 GB of the 40 GB A100, the per-process cap 2 concurrent jobs are shared on.
MEMORY_FRACTION=0.35
MEMORY_GB=14
# The GPU is compute bound, so a larger batch fills the cap better than a
# second process would.
BATCH_SIZE=128
mkdir -p "$LOG_DIR"

past_cutoff() {
    local now
    now=$(date +%H%M)
    [[ "$now" > "0629" && "$now" < "1700" ]]
}

# flock exits 75 only when the slot is busy, so the command's own exit code
# passes through untouched.
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

log_gpu() {
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader >> "$LOG_DIR/nvidia_smi.csv"
}

run_features() {
    local folder=$1
    if [[ -f results/_experiments/prediction_shift_phenomenon/features/$folder.json ]]; then
        return
    fi
    if past_cutoff; then
        echo "cutoff reached before $folder" | tee -a "$LOG_DIR/run.log"
        touch "$MARKER"
        exit 0
    fi
    log_gpu
    echo "$(date -Is) start features $folder" >> "$LOG_DIR/run.log"
    run_locked python "$MEASURE" --stage features --folders "$folder" \
        --gpu-memory-gb "$MEMORY_GB" --batch-size "$BATCH_SIZE" \
        >> "$LOG_DIR/$folder.log" 2>&1
    echo "$(date -Is) end features $folder exit $?" >> "$LOG_DIR/run.log"
    log_gpu
}

run_benign_sweep() {
    local folder=$1 position=$2 operator=$3
    shift 3
    local placement=$position
    [[ "$operator" != "dropout" ]] && placement=${position}_${operator}
    if [[ -f results/$folder/psbd/run_$placement.json ]]; then
        return
    fi
    if past_cutoff; then
        echo "cutoff reached before $folder $placement" | tee -a "$LOG_DIR/run.log"
        touch "$MARKER"
        exit 0
    fi
    log_gpu
    echo "$(date -Is) start sweep $folder $placement" >> "$LOG_DIR/run.log"
    # cli.sweep takes no memory cap, so the cap is set in the same process before
    # the module runs.
    run_locked python -c "
import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction($MEMORY_FRACTION)
sys.argv = ['cli.sweep'] + sys.argv[1:]
runpy.run_module('cli.sweep', run_name='__main__')
" --checkpoint-folder "$folder" --position "$position" --operator "$operator" \
        --probe-attack badnet_a2o --probe-target-label 0 --batch-size "$BATCH_SIZE" \
        --num-workers 4 --rates "$@" >> "$LOG_DIR/sweep_$folder.log" 2>&1
    echo "$(date -Is) end sweep $folder $placement exit $?" >> "$LOG_DIR/run.log"
    log_gpu
}

run_gate() {
    local folder=$1
    local record=results/_experiments/prediction_shift_phenomenon/sanity/$folder.json
    if [[ ! -f $record ]]; then
        log_gpu
        run_locked python experiments/prediction_shift_phenomenon/sanity.py \
            --folder "$folder" --gpu-memory-gb "$MEMORY_GB" \
            >> "$LOG_DIR/sanity_$folder.log" 2>&1
    fi
    # A failed gate stops the queue, since every later number would inherit it.
    python -c "
import json, sys
record = json.load(open('$record'))
gates = [value for key, value in record.items() if key.startswith('gate_')]
sys.exit(0 if all(gate['passes'] for gate in gates) else 1)
" || { echo "sanity gate failed on $folder" | tee -a "$LOG_DIR/run.log"; exit 1; }
}

# The first 3 gates ran on GTSRB BadNets only. These add a global trigger on
# ResNet-18, and CIFAR-100 Blend and Tiny BPP on the transformers.
for folder in resnet18_gtsrb_blend_0_1 vit_cifar100_blend_0_1 swin_tiny_bpp_0_1; do
    run_gate "$folder"
done

RESIDUAL_RATES=(0.005 0.01 0.02 0.03 0.05 0.07 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
TOKEN_RATES=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

FIRST_FEATURES=(
    resnet18_gtsrb_badnet_a2o_0_1 resnet18_gtsrb_blend_0_1
    resnet18_cifar10_badnet_a2o_0_1_smoke
    vit_gtsrb_badnet_a2o_0_1 vit_gtsrb_blend_0_1 vit_gtsrb_tact_0_05 vit_gtsrb_benign
    vit_cifar10_badnet_a2o_0_1 vit_cifar10_tact_0_01 vit_cifar10_tact_0_05
    vit_cifar10_blend_0_1 vit_cifar10_lf_0_1
    vit_cifar10_wanet_0_1 vit_cifar10_bpp_0_1 vit_cifar10_benign
    swin_gtsrb_badnet_a2o_0_1 swin_gtsrb_blend_0_1
    swin_cifar10_badnet_a2o_0_1 swin_cifar10_tact_0_01 swin_cifar10_tact_0_05
    swin_cifar10_blend_0_1 swin_cifar10_lf_0_1
    swin_cifar10_wanet_0_1 swin_cifar10_bpp_0_1
)
for folder in "${FIRST_FEATURES[@]}"; do
    run_features "$folder"
done

# The Swin benign models of CIFAR-10 and GTSRB were never swept, so they need
# stage-1 caches before they can serve as controls for claims 1, 2 and 4, and
# an adaptive rate before claim 3 can read them.
for folder in swin_gtsrb_benign swin_cifar10_benign; do
    run_benign_sweep "$folder" post_residual dropout "${RESIDUAL_RATES[@]}"
    run_benign_sweep "$folder" before_attention_norm token_mask "${TOKEN_RATES[@]}"
    python "$MEASURE" --stage cached --folders "$folder" --overwrite \
        >> "$LOG_DIR/$folder.log" 2>&1
    run_features "$folder"
done

touch "$MARKER"
echo "$(date -Is) all GPU work done" >> "$LOG_DIR/run.log"

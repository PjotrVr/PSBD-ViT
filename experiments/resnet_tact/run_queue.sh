#!/usr/bin/env bash
# The GPU stages of the ResNet-18 TaCT test, resumably: training, the post_residual
# dropout sweep with its analysis, the trigger sufficiency readings and 2 competitor
# detectors. A stage whose output exists is skipped, so the queue can be rerun
# after an interruption. Each GPU stage holds 1 of the 2 shared lock slots and no
# stage starts between 06:30 and 17:00.
#
#   bash experiments/resnet_tact/run_queue.sh

set -uo pipefail
WORKTREE=$(cd "$(dirname "$0")/../.." && pwd)
MAIN=/lustre/home/pstika/projects/PSBD-ViT
cd "$WORKTREE"
source .venv/bin/activate
export PYTHONPATH=.

CHECKPOINTS=$MAIN/checkpoints
RESULTS=$MAIN/results
RAW=$MAIN/raw_data
LOCKS=("$MAIN/scratch/gpu.lock" "$MAIN/scratch/gpu2.lock")
LOG_DIR=$MAIN/scratch/resnet_tact
mkdir -p "$LOG_DIR"
CAPPED="python -m experiments.resnet_tact.capped"

# folder, dataset, attack, poison rate, source classes ("-" keeps TaCT's default),
# in the order the night runs them: the GTSRB models and their benign reference
# first, then CIFAR-10.
MODELS=(
    "resnet18_gtsrb_tact_0_05_src6 gtsrb tact 0.05 1,2,3,4,5,6"
    "resnet18_gtsrb_tact_0_1_src12 gtsrb tact 0.1 1,2,3,4,5,6,7,8,9,10,11,12"
    "resnet18_gtsrb_benign gtsrb benign - -"
    "resnet18_cifar10_tact_0_01 cifar10 tact 0.01 -"
    "resnet18_cifar10_tact_0_05_src3 cifar10 tact 0.05 1,2,3"
    "resnet18_cifar10_badnet_a2o_0_1 cifar10 badnet_a2o 0.1 -"
    "resnet18_cifar10_benign cifar10 benign - -"
)
SUFFICIENCY=(
    resnet18_gtsrb_tact_0_05_src6
    resnet18_gtsrb_tact_0_1_src12
    resnet18_cifar10_tact_0_01
    resnet18_cifar10_tact_0_05_src3
    resnet18_cifar10_badnet_a2o_0_1
    resnet18_gtsrb_badnet_a2o_0_1
    resnet18_gtsrb_blend_0_1
)
TACT_FOLDERS=("${SUFFICIENCY[@]:0:4}")

inside_gpu_window() {
    local now
    now=$(date +%H%M)
    # 10# forces base 10, so 0630 is not read as octal.
    if ((10#$now >= 630 && 10#$now < 1700)); then
        return 1
    fi
    return 0
}

# Queues on the first slot for up to 30 seconds and then tries the second, in a
# loop, so a free second slot is taken while the first is held by a long job
# (the pattern of experiments/why_psbd_works/gpu_slot.sh). -E 75 tells a busy
# slot apart from a stage that ran and failed.
on_free_slot() {
    local status
    while true; do
        flock -w 30 -E 75 "${LOCKS[0]}" "$@"
        status=$?
        ((status != 75)) && return "$status"
        flock -n -E 75 "${LOCKS[1]}" "$@"
        status=$?
        ((status != 75)) && return "$status"
    done
}

stage() {
    local name=$1
    shift
    if ! inside_gpu_window; then
        echo "$(date +%T) outside the GPU window, stopping before $name"
        exit 1
    fi
    echo "$(date +%T) start $name"
    local started=$SECONDS
    if on_free_slot "$@" >"$LOG_DIR/$name.log" 2>&1; then
        echo "$(date +%T) done $name in $(((SECONDS - started) / 60)) min"
    else
        echo "$(date +%T) failed $name, see $LOG_DIR/$name.log"
    fi
}

model_done() {
    local folder=$1 attack=$2
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] || return 1
    [[ "$attack" == "benign" || -f "$RESULTS/$folder/psbd_metrics.json" ]]
}

for run in "${MODELS[@]}"; do
    read -r folder dataset attack rate sources <<<"$run"
    model_done "$folder" "$attack" && continue
    stage "model.$folder" bash experiments/resnet_tact/one_model.sh \
        "$folder" "$dataset" "$attack" "$rate" "$sources"
done

missing_sufficiency=()
for folder in "${SUFFICIENCY[@]}"; do
    [[ -f "results/_experiments/resnet_tact/sufficiency/$folder.json" ]] && continue
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] && missing_sufficiency+=("$folder")
done
if ((${#missing_sufficiency[@]})); then
    stage sufficiency python experiments/why_psbd_works/sufficiency.py \
        --models "${missing_sufficiency[@]}" --output-slug resnet_tact --gpu-memory-gb 8 \
        --checkpoints-dir "$CHECKPOINTS" --raw-data-dir "$RAW" --results-dir results
fi

present_tact=()
for folder in "${TACT_FOLDERS[@]}"; do
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] && present_tact+=("$folder")
done
stage baselines $CAPPED cli.baselines --checkpoint-folder "${present_tact[@]}" \
    --detectors ted beatrix --skip-existing \
    --checkpoints-dir "$CHECKPOINTS" --results-dir "$RESULTS" --raw-data-dir "$RAW"
echo "$(date +%T) resnet tact queue finished"

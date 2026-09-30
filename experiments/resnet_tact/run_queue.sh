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

# The ladder and pass count of the existing ResNet-18 reproductions
# (results/resnet18_gtsrb_badnet_a2o_0_1/psbd/run_post_residual.json).
RATES="0.005 0.01 0.02 0.03 0.05 0.07 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# folder, dataset, attack, poison rate, source classes ("-" keeps TaCT's default).
TACT_RUNS=(
    "resnet18_gtsrb_tact_0_05_src6 gtsrb tact 0.05 1,2,3,4,5,6"
    "resnet18_gtsrb_tact_0_1_src12 gtsrb tact 0.1 1,2,3,4,5,6,7,8,9,10,11,12"
    "resnet18_cifar10_tact_0_01 cifar10 tact 0.01 -"
    "resnet18_cifar10_tact_0_05_src3 cifar10 tact 0.05 1,2,3"
    "resnet18_cifar10_badnet_a2o_0_1 cifar10 badnet_a2o 0.1 -"
)
BENIGN_DATASETS=(gtsrb cifar10)
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

train_backdoor() {
    local folder=$1 dataset=$2 attack=$3 rate=$4 sources=$5
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] && return 0
    local override=()
    [[ "$sources" != "-" ]] && override=(--attack-override "source_classes=$sources")
    stage "train.$folder" $CAPPED cli.train_backdoor \
        --dataset "$dataset" --attack "$attack" --poison-rate "$rate" \
        --target-label 0 --architecture resnet18 --epochs 100 --seed 0 \
        --num-workers 4 --raw-data-dir "$RAW" "${override[@]}" \
        --output "$CHECKPOINTS/$folder/attack_result.pt"
}

sweep_and_analyze() {
    local folder=$1
    [[ -f "$RESULTS/$folder/psbd_metrics.json" ]] && return 0
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] || return 0
    stage "sweep.$folder" $CAPPED cli.sweep --checkpoint-folder "$folder" \
        --position post_residual --operator dropout --rates $RATES \
        --forward-passes 3 --skip-existing \
        --checkpoints-dir "$CHECKPOINTS" --results-dir "$RESULTS" --raw-data-dir "$RAW"
    python -m cli.analyze --checkpoint-folder "$folder" \
        --checkpoints-dir "$CHECKPOINTS" --results-dir "$RESULTS" \
        >"$LOG_DIR/analyze.$folder.log" 2>&1
}

# Each model is swept right after it trains, so the GTSRB TaCT models read out
# before the longer CIFAR-10 runs start.
for run in "${TACT_RUNS[@]}"; do
    read -r folder dataset attack rate sources <<<"$run"
    train_backdoor "$folder" "$dataset" "$attack" "$rate" "$sources"
    sweep_and_analyze "$folder"
done

for dataset in "${BENIGN_DATASETS[@]}"; do
    folder=resnet18_${dataset}_benign
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] && continue
    stage "train.$folder" $CAPPED cli.train_benign \
        --datasets "$dataset" --architecture resnet18 --epochs 100 --seed 0 \
        --num-workers 4 --raw-data-dir "$RAW" --weights-dir "$CHECKPOINTS"
done

for folder in "${SUFFICIENCY[@]}"; do
    [[ -f "results/_experiments/resnet_tact/sufficiency/$folder.json" ]] && continue
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] || continue
    stage "sufficiency.$folder" python experiments/why_psbd_works/sufficiency.py \
        --models "$folder" --output-slug resnet_tact --gpu-memory-gb 8 \
        --checkpoints-dir "$CHECKPOINTS" --raw-data-dir "$RAW" --results-dir results
done

for folder in "${TACT_FOLDERS[@]}"; do
    [[ -f "$CHECKPOINTS/$folder/attack_result.pt" ]] || continue
    stage "baselines.$folder" $CAPPED cli.baselines --checkpoint-folder "$folder" \
        --detectors ted beatrix --skip-existing \
        --checkpoints-dir "$CHECKPOINTS" --results-dir "$RESULTS" --raw-data-dir "$RAW"
done
echo "$(date +%T) resnet tact queue finished"

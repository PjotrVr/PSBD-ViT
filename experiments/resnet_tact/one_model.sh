#!/usr/bin/env bash
# Trains 1 ResNet-18 model when its checkpoint is missing and sweeps it when its
# PSBD metrics are missing, inside 1 hold of a GPU lock slot. The lock queue on
# the login node is the bottleneck of the night, so a model does not queue twice.
#
#   bash experiments/resnet_tact/one_model.sh <folder> <dataset> <attack> <rate> <sources>
#   bash experiments/resnet_tact/one_model.sh resnet18_gtsrb_benign gtsrb benign - -

set -euo pipefail
folder=$1 dataset=$2 attack=$3 rate=$4 sources=$5
MAIN=/lustre/home/pstika/projects/PSBD-ViT
CHECKPOINTS=$MAIN/checkpoints
RESULTS=$MAIN/results
RAW=$MAIN/raw_data
CAPPED="python -m experiments.resnet_tact.capped"

# The ladder and pass count of the existing ResNet-18 reproductions
# (results/resnet18_gtsrb_badnet_a2o_0_1/psbd/run_post_residual.json).
RATES="0.005 0.01 0.02 0.03 0.05 0.07 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

if [[ ! -f "$CHECKPOINTS/$folder/attack_result.pt" ]]; then
    echo "$(date +%T) train $folder"
    if [[ "$attack" == "benign" ]]; then
        $CAPPED cli.train_benign \
            --datasets "$dataset" --architecture resnet18 --epochs 100 --seed 0 \
            --num-workers 4 --raw-data-dir "$RAW" --weights-dir "$CHECKPOINTS"
    else
        override=()
        [[ "$sources" != "-" ]] && override=(--attack-override "source_classes=$sources")
        $CAPPED cli.train_backdoor \
            --dataset "$dataset" --attack "$attack" --poison-rate "$rate" \
            --target-label 0 --architecture resnet18 --epochs 100 --seed 0 \
            --num-workers 4 --raw-data-dir "$RAW" "${override[@]}" \
            --output "$CHECKPOINTS/$folder/attack_result.pt"
    fi
fi

# A benign model is only the clean-accuracy reference, it is not swept.
if [[ "$attack" != "benign" && ! -f "$RESULTS/$folder/psbd_metrics.json" ]]; then
    echo "$(date +%T) sweep $folder"
    $CAPPED cli.sweep --checkpoint-folder "$folder" \
        --position post_residual --operator dropout --rates $RATES \
        --forward-passes 3 --skip-existing \
        --checkpoints-dir "$CHECKPOINTS" --results-dir "$RESULTS" --raw-data-dir "$RAW"
    python -m cli.analyze --checkpoint-folder "$folder" \
        --checkpoints-dir "$CHECKPOINTS" --results-dir "$RESULTS"
fi
echo "$(date +%T) finished $folder"

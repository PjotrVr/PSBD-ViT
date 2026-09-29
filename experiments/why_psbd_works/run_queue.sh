#!/bin/bash
# The night's GPU queue for why_psbd_works, in priority order: 1 ViT model per
# attack category, then the same on Swin, then the rest of each architecture,
# then the seed ensembles. Start 2 copies. Each copy claims the next entry with
# an atomic mkdir, runs it on a free lock slot (gpu_slot.sh) and releases the
# slot before the next entry. No entry starts after 06:30. Every finished entry
# leaves its JSON, so a rerun skips it and the queue resumes where it stopped.
#
#   experiments/why_psbd_works/run_queue.sh worker_a &
#   experiments/why_psbd_works/run_queue.sh worker_b &
REPO=/lustre/home/pstika/projects/PSBD-ViT
CLAIMS=$REPO/scratch/why_psbd_works_claims
LOGS=$REPO/scratch/why_psbd_works_logs
mkdir -p "$CLAIMS" "$LOGS"
cd "$REPO" || exit 1
source .venv/bin/activate
export PYTHONPATH=.

QUEUE=(
    vit_cifar100_badnet_a2o_0_05 vit_cifar10_tact_0_05 vit_cifar100_blend_0_05
    vit_cifar100_lf_0_05 vit_tiny_wanet_0_1 vit_cifar100_bpp_0_05
    vit_cifar100_benign:badnet_a2o vit_cifar100_benign:blend
    swin_cifar100_badnet_a2o_0_05 swin_cifar10_tact_0_05 swin_cifar100_blend_0_05
    swin_cifar100_lf_0_05 swin_cifar100_wanet_0_1 swin_cifar100_bpp_0_05
    swin_cifar100_benign:badnet_a2o swin_cifar100_benign:blend
    vit_tiny_badnet_a2o_0_05 vit_cifar10_tact_0_01 vit_gtsrb_tact_0_05
    vit_tiny_blend_0_05 vit_tiny_lf_0_05 vit_gtsrb_wanet_0_1 vit_cifar10_wanet_0_1
    vit_cifar10_wanet_0_05 vit_tiny_wanet_0_05 vit_tiny_bpp_0_05
    vit_tiny_benign:badnet_a2o vit_tiny_benign:blend
    swin_tiny_badnet_a2o_0_05 swin_cifar10_tact_0_01 swin_tiny_blend_0_05
    swin_tiny_lf_0_05 swin_tiny_wanet_0_1 swin_tiny_bpp_0_05
    swin_tiny_benign:badnet_a2o swin_tiny_benign:blend
    seed:vit_cifar100_badnet_a2o_0_05 seed:vit_cifar100_blend_0_05
    seed:vit_cifar100_lf_0_05 seed:vit_cifar100_bpp_0_05 seed:vit_tiny_wanet_0_1
    seed:vit_cifar10_tact_0_05 seed:vit_tiny_badnet_a2o_0_05 seed:vit_tiny_blend_0_05
    seed:vit_tiny_lf_0_05 seed:vit_tiny_bpp_0_05 seed:vit_gtsrb_wanet_0_1
    seed:vit_cifar10_wanet_0_1 seed:vit_gtsrb_tact_0_05 seed:vit_cifar10_tact_0_01
)

for entry in "${QUEUE[@]}"; do
    if [ "$(date +%H%M)" -ge 0630 ] && [ "$(date +%H%M)" -lt 1700 ]; then
        echo "$(date) $1 stops, no new entry after 06:30"
        break
    fi
    claim="$CLAIMS/${entry//:/__}"
    mkdir "$claim" 2>/dev/null || continue
    echo "$(date) $1 starts $entry"
    if [[ $entry == seed:* ]]; then
        command=(python experiments/why_psbd_works/seed_ensemble.py --cells "${entry#seed:}")
    else
        command=(python experiments/why_psbd_works/measure.py --models "$entry")
    fi
    experiments/why_psbd_works/gpu_slot.sh "${command[@]}" \
        > "$LOGS/${entry//:/__}.log" 2>&1
    echo "$(date) $1 ends $entry with status $?"
done
echo "$(date) $1 done"

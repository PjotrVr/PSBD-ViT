#!/bin/bash
# The GPU queue of the novel probes: the 10 development models, then the 4 benign
# controls. Each model runs under a blocking flock on scratch/gpu.lock, so the
# queue waits behind whoever holds it and releases it between models. The login
# GPU is ours only from 17:00 to 07:00 and no model starts after 06:30, so an
# entry reached outside that window sleeps until 17:00. Finished models leave
# their passes file and are skipped on a rerun.
#
#   nohup experiments/novel_probes/run_queue.sh > scratch/novel_probes/queue.log 2>&1 &
REPO=/lustre/home/pstika/projects/PSBD-ViT
LOCK=$REPO/scratch/gpu.lock
cd "$REPO" || exit 1
source .venv/bin/activate
export PYTHONPATH=.

QUEUE=(
    vit_cifar10_wanet_0_1 vit_cifar10_badnet_a2o_0_01 vit_cifar10_tact_0_01
    vit_cifar10_bpp_0_05 vit_cifar10_blend_0_1 vit_gtsrb_tact_0_05
    vit_gtsrb_bpp_0_01 vit_gtsrb_bpp_0_1 vit_gtsrb_lf_0_01 vit_gtsrb_wanet_0_1
    vit_cifar10_benign vit_gtsrb_benign vit_cifar100_benign vit_tiny_benign
)

in_window() {
    local now
    now=$(date +%H%M)
    [ "$now" -ge 1700 ] || [ "$now" -lt 0630 ]
}

wait_for_window() {
    while ! in_window; do
        echo "$(date) outside the GPU window, sleeping until 17:00"
        sleep "$(( $(date -d 'today 17:00' +%s) - $(date +%s) ))"
    done
}

# A GPU smoke on a small split first, so a bug that only bfloat16 or CUDA
# exposes stops the queue before it spends a night slot.
wait_for_window
flock "$LOCK" python -m experiments.novel_probes.measure --smoke --overwrite \
    --folders vit_cifar10_badnet_a2o_0_01 --pairs 64 --validation-count 256 \
    || { echo "$(date) GPU smoke failed, queue stopped"; exit 1; }
echo "$(date) GPU smoke passed"

for folder in "${QUEUE[@]}"; do
    if [ -f "results/_experiments/novel_probes/passes/$folder.pt" ]; then
        echo "$(date) skip $folder"
        continue
    fi
    wait_for_window
    echo "$(date) waiting for the lock for $folder"
    # The window is checked again once the lock is ours, since the wait for it
    # can run past 06:30.
    flock "$LOCK" bash -c "
        now=\$(date +%H%M)
        if [ \"\$now\" -ge 0630 ] && [ \"\$now\" -lt 1700 ]; then exit 75; fi
        python -m experiments.novel_probes.measure --folders $folder
    "
    status=$?
    if [ $status -eq 75 ]; then
        echo "$(date) lock came after 06:30, $folder waits for 17:00"
        wait_for_window
        flock "$LOCK" python -m experiments.novel_probes.measure --folders "$folder"
        status=$?
    fi
    echo "$(date) $folder exited $status"
done
touch "$REPO/scratch/gpu_done_novel_probes"

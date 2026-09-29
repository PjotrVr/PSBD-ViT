#!/bin/bash
# The night queue of the final method's GPU work on the login A100, inference only.
# It waits for 17:00 and for the analysis agents' done markers, so their leftovers
# run first, then runs item 1, 2 and 3 of README.md in order. Every sweep holds
# scratch/gpu.lock for 1 model only and caps itself at 0.15 of the card's memory,
# so it interleaves with the other agents' queues. No job starts from 06:30 to
# 17:00, and a stopped run resumes where it left off, since gpu_jobs.py lists only
# work not yet cached. It never touches another process.
#
#     nohup experiments/final_method/gpu_driver.sh > /dev/null 2>&1 &
cd /lustre/home/pstika/projects/PSBD-ViT || exit 1
source .venv/bin/activate
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4
LOGS=results/_experiments/final_method/gpu_logs
mkdir -p "$LOGS"
SUMMARY=$LOGS/summary.txt
MARKERS="general tm phenomenon manifestation"

in_window() {
    local now
    now=$(date +%H%M)
    [ "$now" -ge 1700 ] || [ "$now" -lt 0630 ]
}

wait_for_start() {
    while true; do
        local ready=1
        for marker in $MARKERS; do [ -e "scratch/gpu_done_$marker" ] || ready=0; done
        if [ $ready = 1 ] && in_window; then return 0; fi
        sleep 60
    done
}

sweep() {
    local folder=$1; shift
    local started log
    started=$(date +%s)
    log="$LOGS/${folder}_$(echo "$*" | tr ' ' '_' | cut -c1-80).log"
    flock scratch/gpu.lock python -c "
import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.15, 0)
sys.argv = ['cli.sweep', '--checkpoint-folder', '$folder', '--skip-existing'] + '$*'.split()
runpy.run_module('cli.sweep', run_name='__main__')
print('PEAK_MEM_MB', torch.cuda.max_memory_allocated() // 2**20, flush=True)
" > "$log" 2>&1 < /dev/null
    local status=$?
    echo "$(date -Is) $folder status=$status wall_s=$(( $(date +%s) - started )) $*" >> "$SUMMARY"
    # The CPU analysis runs outside the lock, beside the next sweep.
    ( python -m cli.analyze --checkpoint-folder "$folder" > "$LOGS/${folder}_analyze.log" 2>&1 \
        || echo "$(date -Is) $folder analyze FAILED" >> "$SUMMARY" ) &
}

run_item() {
    local item=$1 jobs="$LOGS/jobs_$1.txt"
    python -m experiments.final_method.gpu_jobs --item "$item" > "$jobs"
    echo "$(date -Is) $item start, $(wc -l < "$jobs") jobs" >> "$SUMMARY"
    while read -r _ folder arguments; do
        if ! in_window; then
            echo "$(date -Is) STOP 06:30 reached before $folder ($item)" >> "$SUMMARY"
            return 1
        fi
        sweep "$folder" $arguments
    done < "$jobs"
    wait
    local left
    left=$(python -m experiments.final_method.gpu_jobs --item "$item" | wc -l)
    echo "$(date -Is) $item done, $left jobs left" >> "$SUMMARY"
    [ "$left" = 0 ]
}

wait_for_start
echo "$(date -Is) driver start" >> "$SUMMARY"

# Item 1 found its band already cached on every scorable Swin model, and the
# pre-registered Swin readout was read once on 2026-09-30, so it is not rerun.
# The sweep stays in the queue only to catch a model added to the panel since.
run_item item_1_swin_band

# Item 2, then the compute control on both architectures.
if run_item item_2_k6; then
    python -m experiments.final_method.compute_control --architecture vit >> "$SUMMARY" 2>&1
    python -m experiments.final_method.compute_control --architecture swin >> "$SUMMARY" 2>&1
fi

# Item 3, then the final method on both kinds of adaptive attacker.
if run_item item_3_evaders; then
    python -m experiments.final_method.fusion_readout --set evaders >> "$SUMMARY" 2>&1
fi

wait
touch scratch/gpu_done_final_method
echo "$(date -Is) driver end" >> "$SUMMARY"

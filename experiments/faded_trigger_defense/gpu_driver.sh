#!/bin/bash
# The login GPU queue of the faded trigger defenses, inference only. It waits for
# 17:00 and for the done markers of the queues already waiting (FADED_AFTER, by
# default the final method's, empty to wait for 17:00 alone), then scores 1 entry and 1 part per hold of
# scratch/gpu.lock, so the card is released between jobs. score.py caps itself at
# 0.15 of the card's memory. No job starts from 06:30 to 17:00, and 1 job takes a
# few minutes, so every job ends before the 07:00 curfew. A stopped run resumes
# where it left off, since score.py skips every record already on disk. It never
# touches another process. The CPU readout and the README follow the last job.
#
#     setsid nohup experiments/faded_trigger_defense/gpu_driver.sh > /dev/null 2>&1 &
cd /lustre/home/pstika/projects/PSBD-ViT || exit 1
source .venv/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=4
LOGS=results/_experiments/faded_trigger_defense/gpu_logs
mkdir -p "$LOGS"
SUMMARY=$LOGS/summary.txt
MARKERS=${FADED_AFTER-final_method}
SCORE=experiments/faded_trigger_defense/score.py

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

wait_for_start
echo "$(date -Is) driver start" >> "$SUMMARY"

finished=1
for entry in $(python "$SCORE" --list); do
    for part in plain features amplified; do
        if ! in_window; then
            echo "$(date -Is) STOP 06:30 reached before $entry $part" >> "$SUMMARY"
            finished=0
            break 2
        fi
        started=$(date +%s)
        flock scratch/gpu.lock python "$SCORE" --models "$entry" --parts "$part" \
            > "$LOGS/${entry//:/__}_$part.log" 2>&1 < /dev/null
        status=$?
        echo "$(date -Is) $entry $part status=$status wall_s=$(( $(date +%s) - started ))" >> "$SUMMARY"
    done
done

python experiments/faded_trigger_defense/analyze.py \
    --readme experiments/faded_trigger_defense/README.md >> "$SUMMARY" 2>&1
echo "$(date -Is) readout status=$?" >> "$SUMMARY"
[ $finished = 1 ] && touch scratch/gpu_done_faded_trigger_defense
echo "$(date -Is) driver end" >> "$SUMMARY"

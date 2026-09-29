#!/bin/bash
# Runs 1 command on whichever of the 2 shared GPU lock slots is free. While both
# are busy it queues on the first slot for up to 30 seconds, as a blocking flock
# waiter the kernel wakes in turn with everyone else's, then tries the second.
# The lock is held for this command only, so a caller that loops over models
# releases the GPU between models. Exit code 200 means a slot was busy, which
# keeps it apart from the command's own failures.
LOCKS=/lustre/home/pstika/projects/PSBD-ViT/scratch
while true; do
    flock -w 30 -E 200 "$LOCKS/gpu.lock" "$@"
    status=$?
    [ $status -ne 200 ] && exit $status
    flock -n -E 200 "$LOCKS/gpu2.lock" "$@"
    status=$?
    [ $status -ne 200 ] && exit $status
done

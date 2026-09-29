#!/bin/bash
# tokens.py on the login node GPU, 1 model per hold of scratch/gpu.lock, no model
# started between 06:30 and 17:00.
#
#     bash experiments/all_to_all_detection/run_tokens.sh FOLDER [FOLDER ...]
cd /lustre/home/pstika/projects/PSBD-ViT
source .venv/bin/activate
export OMP_NUM_THREADS=4
for folder in "$@"; do
    now=$(date +%H%M)
    if [ "$now" -ge 0630 ] && [ "$now" -lt 1700 ]; then echo "window closed"; exit 0; fi
    flock scratch/gpu.lock python experiments/all_to_all_detection/tokens.py --folders "$folder" 2>&1 | grep -E "^\[|Error|error"
done

"""cli.sweep under this experiment's share of the login node GPU.

Sets the per-process memory fraction and the CPU thread count before cli.sweep
builds anything, then hands it the command line unchanged.

    python experiments/all_to_all_detection/sweep_capped.py --checkpoint-folder F ...
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from cli.sweep import main  # noqa: E402

GPU_MEMORY_FRACTION = 0.15
TORCH_THREADS = 4

if __name__ == "__main__":
    torch.set_num_threads(TORCH_THREADS)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    main()

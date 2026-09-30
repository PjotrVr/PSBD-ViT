"""Runs 1 module with the GPU memory and CPU thread caps of the shared login node.

python -m experiments.resnet_tact.capped cli.train_backdoor --dataset gtsrb ...
"""

import runpy
import sys

import torch

# Other queues share the A100 at night, so this process may take at most a fifth
# of it. The CPUs are oversubscribed too, and PyTorch's default thread count
# spins OpenMP threads that starve the GPU.
GPU_MEMORY_FRACTION = 0.2
CPU_THREADS = 4

torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
torch.set_num_threads(CPU_THREADS)

module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__", alter_sys=True)

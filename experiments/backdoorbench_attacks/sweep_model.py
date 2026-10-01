"""Stage 1 and 2 of PSBD for 1 BackdoorBench checkpoint, under a GPU memory cap.

Runs cli.sweep for the 2 placements the final method reads, PSBD-TM and its
partner pre_residual_blocks_5_8, on their ladders from configs/psbd_basis.json at
k = 3 and the standard 2000-image validation split, and then cli.analyze. Both
CLIs read results/bb_<folder> through data.backdoorbench. This wrapper adds the
per-process memory cap the shared login GPU needs, since cli.sweep has no flag
for it. It also checks completeness, since cli.sweep reports a failed checkpoint
on stdout and exits 0.

PSBD-RD (post_residual) is out of scope by the user's cut of 2026-09-30 and is
not swept.

    .venv/bin/python -m experiments.backdoorbench_attacks.sweep_model --folder cifar10_ssba_0_1
"""

import argparse
import json
import os
import sys
import time

import torch

import cli.analyze
import cli.sweep
from data.backdoorbench import BACKDOORBENCH_RESULTS_PREFIX
from experiments.backdoorbench_attacks.common import SWEEPS_DIR, write_json
from utils.provenance import current_git_commit

PLACEMENTS = ("before_attention_norm_token_mask", "pre_residual_blocks_5_8")
BASIS_PATH = "configs/psbd_basis.json"
GPU_MEMORY_FRACTION = 0.15
BATCH_SIZE = 256
NUM_WORKERS = 4
RESULTS_DIR = "results"


def main():
    args = parse_args()
    started = time.perf_counter()
    results_folder = f"{BACKDOORBENCH_RESULTS_PREFIX}{args.folder}"
    results_dir = results_dir_of(args.folder)
    basis = basis_entries()

    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    for placement in PLACEMENTS:
        run_cli(
            cli.sweep.main, sweep_argv(results_folder, results_dir, basis[placement])
        )
    run_cli(
        cli.analyze.main,
        ["--checkpoint-folder", results_folder, "--results-dir", results_dir],
    )

    missing = [
        placement
        for placement in PLACEMENTS
        if not cli.sweep.already_complete(
            os.path.join(results_dir, results_folder, "psbd"),
            placement,
            tuple(basis[placement]["rates"]),
        )
    ]
    metrics_path = os.path.join(results_dir, results_folder, "psbd_metrics.json")
    if missing or not os.path.exists(metrics_path):
        raise SystemExit(
            f"{args.folder}: incomplete {missing}, metrics {os.path.exists(metrics_path)}"
        )
    timing = {
        "folder": args.folder,
        "placements": list(PLACEMENTS),
        "wall_seconds": time.perf_counter() - started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "git_commit": current_git_commit(),
    }
    write_json(timing, os.path.join(SWEEPS_DIR, f"{args.folder}.json"))
    print(f"[done] {args.folder} in {timing['wall_seconds']:.0f} s", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    return args


def results_dir_of(folder):
    # 1 place to change should a model's cache ever need a separate root.
    results_dir = RESULTS_DIR
    return results_dir


def basis_entries():
    with open(BASIS_PATH) as handle:
        basis = json.load(handle)["basis"]
    entries = {entry["id"]: entry for entry in basis if entry["id"] in PLACEMENTS}
    assert set(entries) == set(PLACEMENTS), sorted(entries)
    return entries


def sweep_argv(results_folder, results_dir, entry):
    argv = [
        "--checkpoint-folder", results_folder,
        "--position", entry["position"],
        "--operator", entry["operator"],
        "--rates", *[str(rate) for rate in entry["rates"]],
        "--results-dir", results_dir,
        "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--skip-existing",
    ]  # fmt: skip
    if entry["block_range"] is not None:
        argv += ["--block-range", *[str(block) for block in entry["block_range"]]]
    return argv


def run_cli(entry_point, argv):
    # Both CLIs parse sys.argv, so each call gets its own.
    saved = sys.argv
    sys.argv = [entry_point.__module__, *argv]
    try:
        entry_point()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()

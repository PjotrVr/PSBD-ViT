"""Check 3 for the models that joined the panel after its first run: PSBD-TM at mask seeds 1 and 2.

Same protocol as measure.check3_model: cli.sweep at PSBD-TM's already-selected
rate with --mask-seed 1 and 2 into results/_experiments/reviewer_checks/mask_seeds/,
then cli.analyze over that tree, and the readings at seed 0 from the canonical
cache. Both CLIs run in this process under a GPU memory cap, since the login A100
is shared and cli.sweep has no flag for it. TPR is read at the 1%, 5% and 10%
quantiles of clean validation, check 3's table reads the 10% one.

Run from the main checkout, which holds checkpoints/ and results/:

    PYTHONPATH=<worktree> .venv/bin/python <worktree>/experiments/reviewer_checks/mask_seed_followup.py
"""

import json
import os
import sys

import torch

import cli.analyze
import cli.sweep
from defenses.decision import RECOMMENDED_PLACEMENT
from experiments.reviewer_checks.measure import (
    CHECK3_MASK_SEEDS,
    CHECK3_OPERATOR,
    CHECK3_POSITION,
    rate_row_detection,
)
from scripts.paper._common import load_psbd_metrics, std_or_none
from utils.provenance import current_git_commit, utc_timestamp

# The panel's lowest PSBD-TM model since 2026-09-30, absent from the first run.
FOLLOWUP_MODELS = ("vit_gtsrb_tact_0_01_cos",)
RESULTS_DIR = "results"
MASK_SEED_RESULTS_DIR = "results/_experiments/reviewer_checks/mask_seeds"
OUTPUT_PATH = "results/_experiments/reviewer_checks/check3_followup_2026-10-01.json"
QUANTILE_KEYS = ("q0.01", "q0.05", "q0.10")
GPU_MEMORY_FRACTION = 0.2
TORCH_THREADS = 4


def main():
    torch.set_num_threads(TORCH_THREADS)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)

    rows = [followup_model(folder) for folder in FOLLOWUP_MODELS]
    report = {
        "position": CHECK3_POSITION,
        "operator": CHECK3_OPERATOR,
        "mask_seeds": [0] + list(CHECK3_MASK_SEEDS),
        "results_dir": MASK_SEED_RESULTS_DIR,
        "models": rows,
        "git_commit": current_git_commit(),
        "written_at": utc_timestamp(),
    }
    write_report(report, OUTPUT_PATH)
    print(json.dumps(rows, indent=2))


def followup_model(folder):
    canonical_block = load_psbd_metrics(RESULTS_DIR, folder)["placements"][
        RECOMMENDED_PLACEMENT
    ]
    rate = canonical_block["adaptive_rate"]

    for mask_seed in CHECK3_MASK_SEEDS:
        run_cli(cli.sweep.main, sweep_argv(folder, rate, mask_seed))
    run_cli(
        cli.analyze.main,
        ["--checkpoint-folder", folder, "--results-dir", MASK_SEED_RESULTS_DIR],
    )
    reseeded_placements = load_psbd_metrics(MASK_SEED_RESULTS_DIR, folder)["placements"]

    by_seed = {"0": seed_reading(canonical_block, rate)}
    for mask_seed in CHECK3_MASK_SEEDS:
        block = reseeded_placements[f"{RECOMMENDED_PLACEMENT}_seed{mask_seed}"]
        by_seed[str(mask_seed)] = seed_reading(block, rate)

    row = {
        "folder": folder,
        "rate": rate,
        "by_seed": by_seed,
        "auroc_std": std_or_none([entry["auroc"] for entry in by_seed.values()]),
        "tpr_std": std_or_none([entry["tpr"] for entry in by_seed.values()]),
    }
    return row


def seed_reading(placement_block, rate):
    # AUROC does not depend on the quantile, so it is read off the 10% row.
    detections = {
        key: rate_row_detection(placement_block, rate, key) for key in QUANTILE_KEYS
    }
    reading = {
        "auroc": detections["q0.10"]["auroc"],
        "tpr": detections["q0.10"]["tpr"],
        "tpr_by_quantile": {key: detections[key]["tpr"] for key in QUANTILE_KEYS},
    }
    return reading


def sweep_argv(folder, rate, mask_seed):
    argv = [
        "--checkpoint-folder", folder,
        "--position", CHECK3_POSITION,
        "--operator", CHECK3_OPERATOR,
        "--rates", str(rate),
        "--mask-seed", str(mask_seed),
        "--results-dir", MASK_SEED_RESULTS_DIR,
        "--skip-existing",
    ]  # fmt: skip
    return argv


def run_cli(entry_point, argv):
    # Both CLIs parse sys.argv, so each call gets its own.
    saved = sys.argv
    sys.argv = [entry_point.__module__, *argv]
    try:
        entry_point()
    finally:
        sys.argv = saved


def write_report(report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()

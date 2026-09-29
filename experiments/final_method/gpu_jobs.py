"""Write the job list of the night's GPU queue from the caches on disk.

1 line per `cli.sweep` job, `<item> <folder> <sweep arguments>`, in the order the
driver runs them, and only for work not already cached, so a rerun after an
interruption resumes where the last one stopped. The driver calls this again
before each item, since an earlier item or another agent may have filled a cache.

    item_1_swin_band   Swin-S middle third, pre_residual blocks 9 to 16, full ladder,
                       on the Swin panel models that lack it
    item_2_k6          PSBD-TM at 6 passes at the k = 3 adaptive rate, on the ViT and
                       Swin panel models that hold neither a _k6 nor a _k20 cache there
    item_3_evaders     the middle band on the evasive checkpoints that clear the ASR bar

    .venv/bin/python -m experiments.final_method.gpu_jobs --item item_1_swin_band
"""

import argparse
import json
import os
import statistics

from defenses.decision import complete_rates
from experiments._paths import experiment_result_path
from experiments.cache_readouts.shared import (
    choose_rate,
    load_baselines,
    load_model_set,
    validation_shift_by_rate,
)
from scripts.paper.tab_swin import swin_cells

RESULTS_DIR = "results"
ANCHOR_POSITION, ANCHOR_OPERATOR = "before_attention_norm", "token_mask"
ANCHOR = f"{ANCHOR_POSITION}_{ANCHOR_OPERATOR}"
BAND = {"vit": (5, 8), "swin": (9, 16)}
# The full ladder every cached band was swept on.
FULL_LADDER = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
# On the panels the middle band reaches the 0.8 target at 0.5 to 0.9 on ViT and
# 0.9 to 0.99 on Swin. The evader ladders start 1 step below the lowest of those,
# and the readout flags a model whose lowest rate already reaches the target, so a
# model that would need a lower rate is named instead of misread.
EVADER_LADDER = {
    "vit": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    "swin": (0.8, 0.9, 0.95, 0.99),
}
K6 = 6
# Wall time per rate at k = 3 is read from the file times of 1 uncontended full
# ladder per architecture, the gap between successive rates' last split. The
# per-job overhead (model load, baseline check) is taken from the k = 20 runs of
# 2026-09-29 (scratch/k20_summary.txt, about 175 s for 20 passes at 1 rate against
# about 7 s per pass), so it is a stated estimate rather than a measurement.
TIMING_LADDERS = {
    "vit": "results/vit_cifar10_badnet_a2o_0_01/psbd/pre_residual_blocks_5_8",
    "swin": "results/swin_cifar100_badnet_a2o_0_1/psbd/pre_residual_blocks_9_16",
}
OVERHEAD_SECONDS = 35
WINDOW_HOURS = 13.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item",
        choices=("item_1_swin_band", "item_2_k6", "item_3_evaders", "plan"),
        required=True,
    )
    item = parser.parse_args().item
    if item == "plan":
        write_plan()
        return
    jobs = ITEMS[item]()
    for folder, arguments in jobs:
        print(item, folder, " ".join(arguments))


def seconds_per_rate(architecture):
    folder = TIMING_LADDERS[architecture]
    stamps = sorted(
        os.path.getmtime(os.path.join(folder, name))
        for name in os.listdir(folder)
        if name.endswith("_backdoor.pt")
    )
    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    # Rates added in a later session leave 1 gap of hours, the median ignores it.
    seconds = statistics.median(gaps)
    return seconds


def write_plan():
    per_rate = {arch: seconds_per_rate(arch) for arch in TIMING_LADDERS}
    plan = {
        "seconds_per_rate_k3": per_rate,
        "overhead_seconds": OVERHEAD_SECONDS,
        "items": {},
    }
    total = 0.0
    for item, builder in ITEMS.items():
        jobs = builder()
        rows = []
        for folder, arguments in jobs:
            architecture = folder.split("_")[0]
            rates = len(arguments[arguments.index("--rates") + 1 :])
            passes = (
                int(arguments[arguments.index("--forward-passes") + 1])
                if "--forward-passes" in arguments
                else 3
            )
            seconds = OVERHEAD_SECONDS + rates * per_rate[architecture] * passes / 3
            rows.append(
                {
                    "folder": folder,
                    "architecture": architecture,
                    "rates": rates,
                    "passes": passes,
                    "seconds": seconds,
                }
            )
        item_seconds = sum(r["seconds"] for r in rows)
        total += item_seconds
        plan["items"][item] = {
            "jobs": len(rows),
            "by_architecture": {
                arch: {
                    "jobs": sum(r["architecture"] == arch for r in rows),
                    "hours": sum(
                        r["seconds"] for r in rows if r["architecture"] == arch
                    )
                    / 3600,
                }
                for arch in TIMING_LADDERS
            },
            "hours": item_seconds / 3600,
            "cumulative_hours": total / 3600,
            "fits_window_alone": total / 3600 <= WINDOW_HOURS,
        }
    plan["total_hours"] = total / 3600
    plan["window_hours"] = WINDOW_HOURS
    path = experiment_result_path("final_method", "gpu_plan.json", RESULTS_DIR)
    with open(path, "w") as handle:
        json.dump(plan, handle, indent=2)
    print(f"wrote {path}")


def band_placement(architecture):
    first, last = BAND[architecture]
    placement = f"pre_residual_blocks_{first}_{last}"
    return placement


def band_arguments(architecture, rates):
    first, last = BAND[architecture]
    arguments = [
        "--position",
        "pre_residual",
        "--operator",
        "dropout",
        "--block-range",
        str(first),
        str(last),
        "--rates",
        *[str(r) for r in rates],
    ]
    return arguments


def missing_rates(folder, placement, rates):
    cached = set(complete_rates(os.path.join(RESULTS_DIR, folder, "psbd"), placement))
    missing = [rate for rate in rates if rate not in cached]
    return missing


def item_1():
    # A model whose PSBD-TM never reaches the adaptive target cannot be scored by
    # the final method under the adaptive rule, so its band is not swept.
    jobs = []
    for cell in swin_cells(RESULTS_DIR, "checkpoints"):
        psbd_dir = os.path.join(RESULTS_DIR, cell["folder_name"], "psbd")
        anchor_rate = choose_rate(
            validation_shift_by_rate(psbd_dir, ANCHOR, load_baselines(psbd_dir)),
            "adaptive",
        )
        if anchor_rate is None:
            continue
        rates = missing_rates(cell["folder_name"], band_placement("swin"), FULL_LADDER)
        if rates:
            jobs.append((cell["folder_name"], band_arguments("swin", rates)))
    return jobs


def item_2():
    folders = [m["folder_name"] for m in load_model_set("panel")] + [
        c["folder_name"] for c in swin_cells(RESULTS_DIR, "checkpoints")
    ]
    jobs = []
    for folder in folders:
        psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
        rate = choose_rate(
            validation_shift_by_rate(psbd_dir, ANCHOR, load_baselines(psbd_dir)),
            "adaptive",
        )
        if rate is None:
            continue
        held = any(
            rate in complete_rates(psbd_dir, f"{ANCHOR}{suffix}")
            for suffix in ("_k6", "_k20")
        )
        if not held:
            arguments = [
                "--position",
                ANCHOR_POSITION,
                "--operator",
                ANCHOR_OPERATOR,
                "--forward-passes",
                str(K6),
                "--rates",
                str(rate),
            ]
            jobs.append((folder, arguments))
    return jobs


def item_3():
    path = experiment_result_path(
        "final_method", "adaptive_attackers.json", RESULTS_DIR
    )
    with open(path) as handle:
        attackers = json.load(handle)["models"]
    jobs = []
    for row in attackers:
        if row["evader"]["asr_class"] != "clears":
            continue
        architecture = row["architecture"]
        rates = missing_rates(
            row["folder"], band_placement(architecture), EVADER_LADDER[architecture]
        )
        if rates:
            jobs.append((row["folder"], band_arguments(architecture, rates)))
    return jobs


ITEMS = {"item_1_swin_band": item_1, "item_2_k6": item_2, "item_3_evaders": item_3}


if __name__ == "__main__":
    main()

"""Whole-panel readings of the probe fusion record, the descriptive half of README.md.

The held-out test is `summarise.py`. This reads the same record on every model of the
current panel, the cells the ledger marks successful at the 2-point bar: single-probe
AUROC on all and on hard attacks, the paired gain of the listed unions over PSBD-TM
under `min_rank`, the late residual band on TaCT and PSBD-TM's rank among the singles on
hard attacks per poison rate. It writes `results/_experiments/probe_fusion/panel.json`.

    PYTHONPATH=. .venv/bin/python experiments/probe_fusion/panel.py
"""

import collections
import json
import os
import statistics

from experiments._paths import experiment_result_path
from experiments.probe_fusion.summarise import bootstrap_interval, implanted_cells

RECORD = experiment_result_path("probe_fusion", "probe_fusion.json")
COVERAGE = os.path.join("results", "coverage", "coverage.json")
HARD_ATTACKS = {"bpp", "wanet", "sig", "tact", "lc", "adaptive_blend"}
REFERENCE = "before_attention_norm_token_mask"
SINGLES = (REFERENCE, "both_sublayer_inputs_token_mask", "pre_residual_blocks_9_12")
UNIONS = (
    "n1_within_input_side",
    "c5_ban_pre_5_8",
    "c5_ban_pre_9_12",
    "c5_ban_pre_two_bands",
    "c3_ban_tm_gain",
)
RULE = "min_rank"


def main():
    with open(RECORD) as handle:
        record = json.load(handle)
    with open(COVERAGE) as handle:
        ledger = json.load(handle)
    cells = list(implanted_cells(record, ledger).values())
    hard = [cell for cell in cells if cell["attack"] in HARD_ATTACKS]
    tact = [cell for cell in cells if cell["attack"] == "tact"]

    singles = {
        name: {
            "all": mean_auroc(cells, name),
            "hard": mean_auroc(hard, name),
            "tact": mean_auroc(tact, name),
            "n_reaching_target": sum(
                cell["singles"][name]["reaches_target"] for cell in cells
            ),
        }
        for name in SINGLES
    }
    unions = {name: union_reading(cells, hard, name) for name in UNIONS}
    ranks = hard_ranks(hard)

    payload = {
        "n_cells": len(cells),
        "n_hard": len(hard),
        "n_tact": len(tact),
        "hard_attacks": sorted({cell["attack"] for cell in hard}),
        "singles": singles,
        "unions": unions,
        "reference_rank_on_hard_by_rate": ranks,
    }
    path = experiment_result_path("probe_fusion", "panel.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")


def mean_auroc(cells, name):
    values = [cell["singles"][name]["auroc"] for cell in cells]
    mean = statistics.mean(values)
    return mean


def union_reading(cells, hard, name):
    gains = [
        cell["combinations"][name][RULE]["auroc"] - cell["singles"][REFERENCE]["auroc"]
        for cell in cells
    ]
    low, high = bootstrap_interval(gains)
    reading = {
        "all": statistics.mean(
            cell["combinations"][name][RULE]["auroc"] for cell in cells
        ),
        "hard": statistics.mean(
            cell["combinations"][name][RULE]["auroc"] for cell in hard
        ),
        "gain": statistics.mean(gains),
        "ci95": [low, high],
    }
    return reading


def hard_ranks(hard):
    by_rate = collections.defaultdict(lambda: collections.defaultdict(list))
    for cell in hard:
        for placement, single in cell["singles"].items():
            by_rate[f"{cell['poison_rate']}"][placement].append(single["auroc"])
    ranks = {}
    for rate, table in sorted(by_rate.items()):
        ranked = sorted(table, key=lambda p: -statistics.mean(table[p]))
        ranks[rate] = {
            "rank": ranked.index(REFERENCE) + 1,
            "of": len(ranked),
            "n": len(table[REFERENCE]),
        }
    return ranks


if __name__ == "__main__":
    main()

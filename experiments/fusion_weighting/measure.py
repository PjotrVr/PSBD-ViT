"""PSBD-TM fused with a residual-dropout partner at every TM budget share, from the caches.

The fused score of a weighted minimum gives PSBD-TM the share w of the false-positive
budget and the partner 1 - w,

    u(x) = min(r_TM(x) / w, r_partner(x) / (1 - w)),

with r the percentile of a probe's fractional PSU within its own clean-validation
distribution. w = 0.5 is the plain minimum ("min") and w = 0.9 the pre-registered
weighted rule ("weighted 0.9/0.1"). Every fused score is thresholded at quantiles of
its own clean-validation distribution, so FPR is spent on clean data only. The
literal rule u(x) <= q, whose validation FPR the union bound caps at q, is read
beside it on the sweep shares. CPU only, caches only.

    .venv/bin/python -m experiments.fusion_weighting.measure --set vit_panel
    .venv/bin/python -m experiments.fusion_weighting.measure --set swin_panel
    .venv/bin/python -m experiments.fusion_weighting.measure --set backdoorbench
    .venv/bin/python -m experiments.fusion_weighting.measure --set training_set
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from defenses.cache import read_split_manifest
from defenses.decision import (
    RECOMMENDED_PLACEMENT,
    detection_report,
    pair_clean_to_backdoor,
    threshold_at_quantile,
)
from defenses.scores import to_rank
from experiments._paths import experiment_result_path
from experiments.cache_readouts.fusion_rules import fractional_psu
from experiments.cache_readouts.shared import (
    DEV_SET_PATH,
    choose_rate,
    load_baselines,
    load_model_set,
    load_passes,
    validation_shift_by_rate,
)
from experiments.training_set_detection.analyze import psbd_scores
from experiments.training_set_detection.common import load_part
from scripts.paper.tab_swin import swin_cells

SLUG = "fusion_weighting"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
SETS = ("vit_panel", "swin_panel", "backdoorbench", "training_set")
ANCHOR = RECOMMENDED_PLACEMENT

# The partner of each set. Swin-S reads blocks 17 to 24, the 2nd pre-registered
# Swin partner and the one that held (experiments/final_method/README.md). The ViT
# late band is the partner the pre-registration named, read beside the middle band
# the final method uses.
PARTNERS = {
    "vit_panel": {
        "band": "pre_residual_blocks_5_8",
        "late": "pre_residual_blocks_9_12",
    },
    "swin_panel": {"band": "pre_residual_blocks_17_24"},
    "backdoorbench": {"band": "pre_residual_blocks_5_8"},
}
# The Swin partner misses the adaptive target on 24 of 63 models. The report reads
# those at the rate nearest the target, so they are kept under their own label.
NEAREST_FALLBACK = {"swin_panel"}

# The sweep table's shares and a 0.01 grid between 0.5 and 0.99 for the curves.
SWEEP_SHARES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
FINE_SHARES = tuple(round(0.5 + 0.01 * step, 2) for step in range(50))
QUANTILES = (0.01, 0.05, 0.10, 0.20)
# Each probe alone on a 0.0001 grid up to 1% FPR, the ROC the allocation
# derivation reads at w q and (1 - w) q.
LOW_BUDGETS = tuple(round(0.0001 * step, 4) for step in range(1, 101))
BACKDOORBENCH_RECORDS = os.path.join(
    RESULTS_DIR, "_experiments", "backdoorbench_attacks", "models"
)
TRAINING_SET_SUMMARY = os.path.join(
    RESULTS_DIR, "_experiments", "training_set_detection", "summary.json"
)
# The login node is shared, so torch gets a few CPU threads and no more.
CPU_THREADS = 4


def main():
    args = parse_args()
    torch.set_num_threads(CPU_THREADS)
    started = time.perf_counter()

    if args.set == "training_set":
        models = training_set_models()
        rows = [measure_training_set_model(model) for model in models]
        partners = {"band": "pre_residual_blocks_5_8"}
    else:
        models = model_list(args.set)
        rows = [measure_cache_model(model, args.set) for model in models]
        partners = PARTNERS[args.set]

    payload = {
        "experiment": "fusion weighting, both rules and the TM share sweep",
        "model_set": args.set,
        "anchor": ANCHOR,
        "partners": partners,
        "sweep_shares": list(SWEEP_SHARES),
        "fine_shares": list(FINE_SHARES),
        "quantiles": list(QUANTILES),
        "low_budgets": list(LOW_BUDGETS),
        "threshold_reference": "clean_train"
        if args.set == "training_set"
        else "validation",
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    path = experiment_result_path(SLUG, f"readings_{args.set}.json", RESULTS_DIR)
    write_compact_json(payload, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=SETS, required=True)
    args = parser.parse_args()
    return args


def model_list(set_name):
    with open(DEV_SET_PATH) as handle:
        dev_folders = {entry["folder"] for entry in json.load(handle)["models"]}

    if set_name == "vit_panel":
        cells = load_model_set("panel")
    elif set_name == "swin_panel":
        cells = swin_cells(RESULTS_DIR, CHECKPOINTS_DIR)
    else:
        cells = backdoorbench_cells()

    models = [
        {
            "folder": cell["folder_name"],
            "dataset": cell["dataset"],
            "attack": cell["attack"],
            "poison_rate": cell["poison_rate"],
            "dev": cell["folder_name"] in dev_folders,
        }
        for cell in cells
    ]
    return models


def backdoorbench_cells():
    # A record without detection rates was never swept (a dead backdoor or a
    # checkpoint that was not read), so it has no cache. The backdoorbench_attacks
    # README drops no swept model for the 2-point bar, so neither does this set.
    cells = []
    for path in sorted(glob.glob(os.path.join(BACKDOORBENCH_RECORDS, "*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        if not (record.get("detection") or {}).get("rates"):
            continue
        cells.append(
            {
                "folder_name": f"bb_{record['folder']}",
                "dataset": record["dataset"],
                "attack": record["attack"],
                "poison_rate": record["poison_rate"],
            }
        )
    return cells


def measure_cache_model(model, set_name):
    psbd_dir = os.path.join(RESULTS_DIR, model["folder"], "psbd")
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)
    row = dict(model)

    anchor_rate = choose_rate(
        validation_shift_by_rate(psbd_dir, ANCHOR, baselines), "adaptive"
    )
    row["anchor_rate"] = anchor_rate
    row["partners"] = {}
    if anchor_rate is None:
        return row
    anchor_psu = fractional_psu(load_passes(psbd_dir, ANCHOR, anchor_rate, baselines))

    for name, placement in PARTNERS[set_name].items():
        shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
        rate = choose_rate(shift_by_rate, "adaptive") if shift_by_rate else None
        rate_rule = "adaptive"
        if rate is None and shift_by_rate and set_name in NEAREST_FALLBACK:
            rate = choose_rate(shift_by_rate, "nearest")
            rate_rule = "nearest"
        entry = {
            "placement": placement,
            "rate": rate,
            "rate_rule": rate_rule if rate is not None else None,
            "max_validation_shift": max(shift_by_rate.values())
            if shift_by_rate
            else None,
        }
        if rate is not None:
            entry["validation_shift"] = shift_by_rate[rate]
            partner_psu = fractional_psu(
                load_passes(psbd_dir, placement, rate, baselines)
            )
            entry.update(test_time_readings(anchor_psu, partner_psu, manifest))
        row["partners"][name] = entry
    return row


def test_time_readings(anchor_psu, partner_psu, manifest):
    anchor_ranks = probe_ranks(anchor_psu)
    partner_ranks = probe_ranks(partner_psu)
    for split in anchor_ranks:
        assert anchor_ranks[split].shape == partner_ranks[split].shape, split

    readings = {
        "tm_alone": evaluate_test_time(anchor_psu, manifest),
        "partner_alone": evaluate_test_time(partner_psu, manifest),
        "low_budget_tpr": {
            "tm_alone": low_budget_test_time(anchor_psu, manifest),
            "partner_alone": low_budget_test_time(partner_psu, manifest),
        },
        "sweep": {},
        "fine": {},
    }
    for share in SWEEP_SHARES:
        fused = fuse(anchor_ranks, partner_ranks, share)
        reading = evaluate_test_time(fused, manifest)
        reading["literal"] = literal_test_time(fused, manifest)
        readings["sweep"][f"{share:.2f}"] = reading
    for share in FINE_SHARES:
        fused = fuse(anchor_ranks, partner_ranks, share)
        readings["fine"][f"{share:.2f}"] = compact(evaluate_test_time(fused, manifest))
    return readings


def probe_ranks(psu_by_split):
    # Percentiles within each probe's own clean-validation distribution put the 2
    # probes on 1 scale without fitting anything to triggered data.
    ranks = {
        split: to_rank(psu, psu_by_split["validation"])  # (n_split,)
        for split, psu in psu_by_split.items()
    }
    return ranks


def fuse(anchor_ranks, partner_ranks, share):
    fused = {
        split: torch.minimum(
            anchor_ranks[split] / share, partner_ranks[split] / (1 - share)
        )  # (n_split,)
        for split in anchor_ranks
    }
    return fused


def evaluate_test_time(scores_by_split, manifest):
    # Low means poisoned. The clean split is paired down to the triggered split's
    # images as cli.analyze pairs it, and every threshold is the canon's quantile
    # of clean validation with the strict comparison detection_report uses.
    validation = scores_by_split["validation"]  # (n_validation,)
    clean_paired = pair_clean_to_backdoor(
        scores_by_split["clean"], manifest
    )  # (n_backdoor,)
    backdoor = scores_by_split["backdoor"]  # (n_backdoor,)

    reading = {
        "auroc": detection_report(validation, clean_paired, backdoor, QUANTILES[0])[
            "auroc"
        ],
        "tpr": {},
        "realized_fpr": {},
    }
    for quantile in QUANTILES:
        threshold = threshold_at_quantile(validation, quantile)
        reading["tpr"][f"q{quantile:.2f}"] = float(
            (backdoor < threshold).float().mean()
        )
        reading["realized_fpr"][f"q{quantile:.2f}"] = float(
            (clean_paired < threshold).float().mean()
        )
    return reading


def low_budget_test_time(scores_by_split, manifest):
    validation = scores_by_split["validation"]  # (n_validation,)
    backdoor = scores_by_split["backdoor"]  # (n_backdoor,)
    assert len(pair_clean_to_backdoor(scores_by_split["clean"], manifest)) == len(
        backdoor
    )
    tpr = [
        round(
            float(
                (backdoor < threshold_at_quantile(validation, budget)).float().mean()
            ),
            6,
        )
        for budget in LOW_BUDGETS
    ]
    return tpr


def literal_test_time(fused, manifest):
    # The literal rule flags u(x) <= q, the threshold the union bound is stated for.
    clean_paired = pair_clean_to_backdoor(fused["clean"], manifest)  # (n_backdoor,)
    literal = {
        f"q{quantile:.2f}": {
            "validation_fpr": float((fused["validation"] <= quantile).float().mean()),
            "realized_fpr": float((clean_paired <= quantile).float().mean()),
            "tpr": float((fused["backdoor"] <= quantile).float().mean()),
        }
        for quantile in QUANTILES
    }
    return literal


def compact(reading):
    short = {"auroc": round(reading["auroc"], 6)}
    short.update({key: round(value, 6) for key, value in reading["tpr"].items()})
    return short


def training_set_models():
    # The 6 pooled paper-mirror models: successful_2pt, reconstruction verified and
    # ViT, the "headline" list training_set_detection/analyze.py writes.
    with open(TRAINING_SET_SUMMARY) as handle:
        headline = json.load(handle)["sets"]["paper_mirror"]["headline"]
    models = []
    for folder in headline:
        with open(os.path.join(CHECKPOINTS_DIR, folder, "args.json")) as handle:
            training_args = json.load(handle)
        models.append(
            {
                "folder": folder,
                "dataset": training_args["dataset"],
                "attack": training_args["attack"],
                "poison_rate": training_args["poison_rate"],
                "dev": False,
            }
        )
    return models


def measure_training_set_model(model):
    folder = model["folder"]
    groups = load_part(folder, "groups")
    baseline = load_part(folder, "baseline")
    scores, rates, _ = psbd_scores(folder, baseline, groups["population"])
    anchor_psu = scores["ours"]["fractional"]["psbd_tm"]
    partner_psu = scores["ours"]["fractional"]["middle_band"]

    anchor_ranks = probe_ranks(anchor_psu)
    partner_ranks = probe_ranks(partner_psu)
    entry = {
        "placement": "pre_residual_blocks_5_8",
        "rate": rates["middle_band"]["ours"],
        "rate_rule": "adaptive",
        "tm_alone": evaluate_training_set(anchor_psu),
        "partner_alone": evaluate_training_set(partner_psu),
        "low_budget_tpr": {
            "tm_alone": low_budget_training_set(anchor_psu),
            "partner_alone": low_budget_training_set(partner_psu),
        },
        "sweep": {},
        "fine": {},
    }
    for share in SWEEP_SHARES:
        fused = fuse_groups(anchor_ranks, partner_ranks, share)
        entry["sweep"][f"{share:.2f}"] = evaluate_training_set(fused)
    for share in FINE_SHARES:
        fused = fuse_groups(anchor_ranks, partner_ranks, share)
        entry["fine"][f"{share:.2f}"] = compact(evaluate_training_set(fused))

    row = dict(model)
    row["anchor_rate"] = rates["psbd_tm"]["ours"]
    row["partners"] = {"band": entry}
    return row


def fuse_groups(anchor_ranks, partner_ranks, share):
    # Empty groups (a model without covers) stay empty.
    fused = {
        group: torch.minimum(
            anchor_ranks[group] / share, partner_ranks[group] / (1 - share)
        )  # (n_group,)
        for group in anchor_ranks
        if anchor_ranks[group].numel()
    }
    return fused


def low_budget_training_set(scores):
    clean = scores["clean_train"].float()  # (n_clean,)
    poisoned = scores["poisoned"].float()  # (n_poisoned,)
    tpr = [
        round(
            float((poisoned < threshold_at_quantile(clean, budget)).float().mean()), 6
        )
        for budget in LOW_BUDGETS
    ]
    return tpr


def evaluate_training_set(scores):
    # Thresholds at quantiles of the clean training images themselves, the fair
    # reading of training_set_detection when the training and validation clean
    # distributions differ, so the FPR on clean training images is the quantile.
    clean = scores["clean_train"].float()  # (n_clean,)
    poisoned = scores["poisoned"].float()  # (n_poisoned,)

    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(poisoned))])
    values = np.concatenate([-clean.numpy(), -poisoned.numpy()])
    reading = {
        "auroc": float(roc_auc_score(labels, values)),
        "tpr": {},
        "realized_fpr": {},
    }
    for quantile in QUANTILES:
        threshold = threshold_at_quantile(clean, quantile)
        reading["tpr"][f"q{quantile:.2f}"] = float(
            (poisoned < threshold).float().mean()
        )
        reading["realized_fpr"][f"q{quantile:.2f}"] = float(
            (clean < threshold).float().mean()
        )
    return reading


def rounded(value):
    # 6 decimals is below 1 input in 10000 of any split, and the readings of every
    # model at 58 shares would otherwise run to several megabytes.
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    return value


def write_compact_json(payload, path):
    with open(path, "w") as handle:
        json.dump(rounded(payload), handle, separators=(",", ":"))


if __name__ == "__main__":
    main()

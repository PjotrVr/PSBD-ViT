"""A single account of PSBD across operators and attacks: the critical rate, CPU only.

Each input has a critical rate p*, the smallest rate on a placement's ladder at
which most of its perturbed passes change its answer (`defenses.scores.critical_rate`
with flip_fraction 0.5, read from the sweep's cached per-pass argmax by
`defenses.decision.load_critical_rate_from_disk`). The account under test: PSBD at
any placement is a 2-sample test on p*. It separates when triggered p* is
stochastically larger than clean p*, and operators differ only in how far they
move the 2 distributions apart.

Predictions, fixed before the first read:

    P1  1 curve. Over every (model, placement) with a full ladder, the AUROC of the
        statistic at the adaptive rate is a monotone function of
        A* = P(p*_triggered > p*_clean), Spearman at least 0.8, median absolute
        difference at most 0.05.
    P2  Control. On the benign references probed with a trigger they never learned,
        A* is within 0.05 of 0.5 at every placement.
    P3  The rate. The adaptive rule does not place the rate at the clean median of
        p*. It places it where about 0.8 of held-out clean images have p* at or
        below it, since it asks for a clean shift ratio of 0.8.

A* is read in the pipeline's paired form: every triggered row against its own
clean twin (`defenses.decision.pair_clean_to_backdoor`), ties counted half. An
image whose answer survives every rate gets p* above the ladder, so the AUROC
reads it as the most robust. p* is only resolved to the ladder's steps, so A*
carries ties the per-rate statistic does not.

    PYTHONPATH=. python experiments/why_psbd_works/critical_rate.py
"""

import argparse
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from defenses.cache import read_split_manifest  # noqa: E402
from defenses.decision import (  # noqa: E402
    complete_rates,
    load_critical_rate_from_disk,
    pair_clean_to_backdoor,
)
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.why_psbd_works.measure import SLUG, auroc_low_is_positive  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    clearing_cells,
    excluded_folders,
    load_coverage,
    load_psbd_metrics,
)

# The placements behind the operator families of measure.py, plus the other
# single-site families the sweeps cached on most panel models.
PLACEMENTS = (
    "before_attention_norm_token_mask",
    "before_attention_norm",
    "before_attention_norm_channel_mask",
    "before_attention_norm_gaussian",
    "post_residual",
    "pre_residual",
    "before_attention_residual_token_mask",
    "before_mlp_token_mask",
    "before_mlp_gaussian",
    "after_embedding_token_mask",
    "mlp_neurons_channel_mask",
)
FLIP_FRACTION = 0.5
MIN_RATES = 5
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    excluded = excluded_folders(args.results_dir)
    models = [
        (cell["folder_name"], cell["attack"], cell["dataset"])
        for cell in clearing_cells(load_coverage(args.results_dir))
        if cell["folder_name"].startswith("vit_")
        and cell.get("successful_2pt")
        and cell["folder_name"] not in excluded
    ]
    models += [(f"vit_{d}_benign", "benign", d) for d in DATASETS]

    rows = []
    for folder, attack, dataset in models:
        for placement in PLACEMENTS:
            row = placement_row(args.results_dir, folder, placement)
            if row is not None:
                rows.append(
                    {"folder": folder, "attack": attack, "dataset": dataset, **row}
                )
        print(f"[ok] {folder}", flush=True)

    payload = {
        "flip_fraction": FLIP_FRACTION,
        "rows": rows,
        "predictions": check_predictions(rows),
    }
    out_path = experiment_result_path(SLUG, "critical_rate.json", args.output_root)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["predictions"], indent=2))


def placement_row(results_dir, folder, placement):
    report = load_psbd_metrics(results_dir, folder)
    block = (report or {}).get("placements", {}).get(placement)
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    rates = complete_rates(psbd_dir, placement)
    if block is None or len(rates) < MIN_RATES:
        return None
    rate = psbd_rate(block, "adaptive")
    if rate is None:
        return None
    adaptive = next(r for r in block["rates"] if r["rate"] == rate)

    critical = {
        split: load_critical_rate_from_disk(psbd_dir, placement, split, FLIP_FRACTION)
        for split in ("validation", "clean", "backdoor")
    }  # each (n,)
    if any(values is None for values in critical.values()):
        return None
    paired_clean = pair_clean_to_backdoor(
        critical["clean"], read_split_manifest(psbd_dir)
    )
    triggered = critical["backdoor"]  # (n_backdoor,)
    above_ladder = rates[-1]

    row = {
        "placement": placement,
        "rates": rates,
        "adaptive_rate": rate,
        "auroc_adaptive": adaptive["detection_psu_ratio"]["q0.25"]["auroc"],
        # High p* marks the triggered image, so its negation is low-is-positive.
        "a_star": auroc_low_is_positive(-triggered, -paired_clean),
        "clean_median": float(paired_clean.median()),
        "triggered_median": float(triggered.median()),
        "clean_never_flip": float((paired_clean > above_ladder).float().mean()),
        "triggered_never_flip": float((triggered > above_ladder).float().mean()),
        "validation_share_at_or_below_rate": float(
            (critical["validation"] <= rate).float().mean()
        ),
    }
    return row


def check_predictions(rows):
    backdoored = [
        r for r in rows if r["attack"] != "benign" and r["a_star"] is not None
    ]
    benign = [r for r in rows if r["attack"] == "benign" and r["a_star"] is not None]
    differences = [abs(r["auroc_adaptive"] - r["a_star"]) for r in backdoored]
    breaks = sorted(
        (
            {
                "folder": r["folder"],
                "placement": r["placement"],
                "auroc_adaptive": r["auroc_adaptive"],
                "a_star": r["a_star"],
            }
            for r in backdoored
            if abs(r["auroc_adaptive"] - r["a_star"]) > 0.1
        ),
        key=lambda b: -abs(b["auroc_adaptive"] - b["a_star"]),
    )
    shares = [r["validation_share_at_or_below_rate"] for r in rows]
    predictions = {
        "P1_pairs": len(backdoored),
        "P1_spearman": spearman(
            [r["a_star"] for r in backdoored], [r["auroc_adaptive"] for r in backdoored]
        ),
        "P1_median_abs_difference": statistics.median(differences),
        "P1_breaks_over_0.1": len(breaks),
        "P1_breaks": breaks,
        "P2_benign_rows": len(benign),
        "P2_max_distance_from_half": max(abs(r["a_star"] - 0.5) for r in benign),
        "P3_median_validation_share": statistics.median(shares),
        "P3_range": [min(shares), max(shares)],
    }
    predictions["P1_holds"] = (
        predictions["P1_spearman"] >= 0.8
        and predictions["P1_median_abs_difference"] <= 0.05
    )
    predictions["P2_holds"] = predictions["P2_max_distance_from_half"] <= 0.05
    return predictions


def spearman(first, second):
    a = torch.tensor(first).argsort().argsort().double()
    b = torch.tensor(second).argsort().argsort().double()
    value = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    return value


if __name__ == "__main__":
    main()

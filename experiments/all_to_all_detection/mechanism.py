"""Where triggered all-to-all predictions go under PSBD's perturbations, from the caches.

This is mechanism analysis and it uses attacker knowledge a defender does not have:
the true class of every test image and the (y + 1) mod K rotation our models were
trained with. No detection statistic in detect.py reads either.

For every image whose clean twin the model classifies correctly and whose triggered
twin the backdoor captures, and for every cached placement, it measures

    how often the triggered and the clean prediction shift, on the same images
    where shifted triggered predictions land: the source class y, the runner-up
        class of the unperturbed softmax, or elsewhere
    where shifted clean predictions land: the runner-up, the class 1 below the
        prediction (the offset control for the source) or elsewhere
    how concentrated the shifted destinations are, per image across passes and
        per predicted class across images
    whether an image's triggered fragility tracks its clean fragility (Spearman
        over images), the direct test that the backdoor path reads image content

The all-to-one BadNets models of the panel are the control. Their trigger is the
same patch and their shortcut does not depend on content.

    source .venv/bin/activate
    python experiments/all_to_all_detection/mechanism.py --datasets cifar10 gtsrb
    python experiments/all_to_all_detection/mechanism.py
"""

import argparse
import collections
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from defenses.decision import (  # noqa: E402
    PLACEMENT_MATCH_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    complete_rates,
)
from panel import (  # noqa: E402
    EXTRA_CACHE_ROOT,
    OUT_DIR,
    RESULTS_DIR,
    adaptive_rate,
    load_model_cache,
    main_panel_table,
    matched_rate,
    write_json,
)

CANONICAL_PLACEMENTS = (RECOMMENDED_PLACEMENT, PUBLISHED_PLACEMENT)
MIN_IMAGES = 50


def main():
    args = parse_args()
    if args.summarize_only:
        payload = json.load(open(os.path.join(OUT_DIR, args.out)))
        payload["summary"] = summarize(payload["records"])
        print(write_json(payload, args.out))
        return
    started = time.time()
    table = main_panel_table()

    all_to_all = [
        row["folder"]
        for row in table["all_to_all"]
        if row["relaxed_2pt"] or row["strict_5pt"]
        if row["variant"] is None and row["dataset"] in args.datasets
    ]
    all_to_one = [
        folder
        for architecture in ("vit", "swin")
        for folder in table["all_to_one"][architecture]
        if "_badnet_a2o_" in folder and folder.split("_")[1] in args.datasets
    ]

    records = []
    for folder in all_to_all + all_to_one:
        for placement in cached_placements(folder):
            for rule, rate in rule_rates(folder, placement).items():
                record = measure(folder, placement, rule, rate)
                if record is not None:
                    records.append(record)
        print(f"[ok] {folder}", flush=True)

    payload = {
        "datasets": args.datasets,
        "all_to_all_models": all_to_all,
        "all_to_one_models": all_to_one,
        "records": records,
        "summary": summarize(records),
        "seconds": round(time.time() - started, 1),
    }
    print(write_json(payload, args.out))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["cifar10", "gtsrb", "cifar100", "tiny"]
    )
    parser.add_argument("--out", default="mechanism.json")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


# The canonical caches and the ones run_gpu.sh swept for this experiment.
def cached_placements(folder):
    placements = set()
    for root in (RESULTS_DIR, EXTRA_CACHE_ROOT):
        psbd_dir = os.path.join(root, folder, "psbd")
        if not os.path.isdir(psbd_dir):
            continue
        placements |= {
            name
            for name in os.listdir(psbd_dir)
            if os.path.isdir(os.path.join(psbd_dir, name))
            and complete_rates(psbd_dir, name)
            and "pmodel" not in name
            and not name.endswith("_k20")
            and name != "head_profile"
        }
    ordered = sorted(placements)
    return ordered


# Every placement is read at the rate whose clean validation shift ratio sits
# nearest the canonical 0.6, the repository's device for comparing placements, and
# the 2 canonical placements are also read at the deployable adaptive rate.
def rule_rates(folder, placement):
    rates = {"matched_0.6": matched_rate(folder, placement, PLACEMENT_MATCH_TARGET)}
    if placement in CANONICAL_PLACEMENTS:
        rates["adaptive"] = adaptive_rate(folder, placement)
    rates = {rule: rate for rule, rate in rates.items() if rate is not None}
    return rates


def measure(folder, placement, rule, rate):
    cache = load_model_cache(folder, placement, rate)
    backdoor = cache["splits"]["backdoor"]
    clean = cache["splits"]["clean_paired"]
    num_classes = backdoor["probs"].shape[1]

    # Attacker knowledge, mechanism only: the true class of each image and the
    # label the attack wanted.
    source = clean["loader_labels"]  # (n,)
    attack_label = backdoor["loader_labels"]  # (n,)
    images = (backdoor["pred"] == attack_label) & (clean["pred"] == source)  # (n,)
    if int(images.sum()) < MIN_IMAGES:
        return None

    triggered = side_readout(backdoor, images, num_classes)
    clean_side = side_readout(clean, images, num_classes)

    triggered_destinations = backdoor["pass_argmax"][:, images]  # (k, m)
    triggered_shifted = triggered_destinations != backdoor["pred"][images][None]
    to_source = (triggered_destinations == source[images][None]) & triggered_shifted
    clean_destinations = clean["pass_argmax"][:, images]  # (k, m)
    clean_shifted = clean_destinations != clean["pred"][images][None]
    one_below = (clean["pred"][images] - 1) % num_classes  # (m,)
    clean_to_one_below = (clean_destinations == one_below[None]) & clean_shifted

    triggered_counts = triggered_shifted.sum(dim=0).float()  # (m,)
    clean_counts = clean_shifted.sum(dim=0).float()  # (m,)
    dependence = spearmanr(clean_counts.numpy(), triggered_counts.numpy())

    runner_up_is_source = (
        backdoor["probs"][images].topk(2, dim=1).indices[:, 1] == source[images]
    )  # (m,)

    record = {
        "folder": folder,
        "architecture": folder.split("_")[0],
        "dataset": folder.split("_")[1],
        "label_mode": "all_to_all" if "_a2a_" in folder else "all_to_one",
        "placement": placement,
        "rule": rule,
        "rate": rate,
        "num_classes": num_classes,
        "images": int(images.sum()),
        "passes": int(triggered_destinations.shape[0]),
        "triggered": triggered,
        "clean": clean_side,
        "triggered_shifts_to_source": share(to_source, triggered_shifted),
        "clean_shifts_to_one_below": share(clean_to_one_below, clean_shifted),
        "uniform_destination_chance": 1.0 / (num_classes - 1),
        "triggered_runner_up_is_source": float(runner_up_is_source.float().mean()),
        # A model whose triggered twins all shift in every pass has a constant
        # column and no rank correlation.
        "shift_count_spearman": None
        if math.isnan(dependence.statistic)
        else float(dependence.statistic),
    }
    return record


def side_readout(split, images, num_classes):
    destinations = split["pass_argmax"][:, images]  # (k, m)
    prediction = split["pred"][images]  # (m,)
    shifted = destinations != prediction[None]  # (k, m)
    runner_up = split["probs"][images].topk(2, dim=1).indices[:, 1]  # (m,)
    to_runner_up = (destinations == runner_up[None]) & shifted  # (k, m)

    shifted_destinations = destinations[shifted]  # (shifted passes,)
    histogram = torch.bincount(shifted_destinations, minlength=num_classes).float()
    top_share = float(histogram.max() / histogram.sum()) if histogram.sum() else None

    readout = {
        "shift_rate": float(shifted.float().mean()),
        "shifts_to_runner_up": share(to_runner_up, shifted),
        "top_destination_share": top_share,
        "same_destination_across_passes": same_destination_share(destinations, shifted),
        "per_class_top_destination_share": per_class_concentration(
            destinations, shifted, prediction, num_classes
        ),
    }
    return readout


def share(numerator_mask, denominator_mask):
    denominator = int(denominator_mask.sum())
    value = float(numerator_mask.sum()) / denominator if denominator else None
    return value


# Among images whose prediction shifted in at least 2 passes, the share whose
# shifted passes all name the same class.
def same_destination_share(destinations, shifted):
    eligible = shifted.sum(dim=0) >= 2  # (m,)
    if not bool(eligible.any()):
        return None
    agree = []
    for column in torch.nonzero(eligible).flatten().tolist():
        landed = destinations[shifted[:, column], column]  # (shifted passes,)
        agree.append(bool((landed == landed[0]).all()))
    value = sum(agree) / len(agree)
    return value


# For the images predicted c, the share of their shifted passes that land on the
# single most common destination, averaged over c weighted by shifted passes.
def per_class_concentration(destinations, shifted, prediction, num_classes):
    total_top = 0
    total = 0
    for predicted in torch.unique(prediction).tolist():
        rows = prediction == predicted  # (m,)
        landed = destinations[:, rows][shifted[:, rows]]  # (shifted passes,)
        if len(landed) == 0:
            continue
        total_top += int(torch.bincount(landed, minlength=num_classes).max())
        total += len(landed)
    value = total_top / total if total else None
    return value


SUMMARY_FIELDS = {
    "triggered_shift_rate": ("triggered", "shift_rate"),
    "clean_shift_rate": ("clean", "shift_rate"),
    "triggered_shifts_to_source": (None, "triggered_shifts_to_source"),
    "clean_shifts_to_one_below": (None, "clean_shifts_to_one_below"),
    "uniform_destination_chance": (None, "uniform_destination_chance"),
    "triggered_shifts_to_runner_up": ("triggered", "shifts_to_runner_up"),
    "clean_shifts_to_runner_up": ("clean", "shifts_to_runner_up"),
    "triggered_runner_up_is_source": (None, "triggered_runner_up_is_source"),
    "triggered_same_destination": ("triggered", "same_destination_across_passes"),
    "clean_same_destination": ("clean", "same_destination_across_passes"),
    "triggered_per_class_top_share": ("triggered", "per_class_top_destination_share"),
    "clean_per_class_top_share": ("clean", "per_class_top_destination_share"),
    "shift_count_spearman": (None, "shift_count_spearman"),
}


# Means over models per (architecture, label mode, placement, rule), each field
# over the models where it is defined, with that count beside it.
def summarize(records):
    grouped = collections.defaultdict(list)
    for record in records:
        key = "/".join(
            (
                record["architecture"],
                record["label_mode"],
                record["placement"],
                record["rule"],
            )
        )
        grouped[key].append(record)

    summary = {}
    for key, members in sorted(grouped.items()):
        block = {"n": len(members), "folders": [m["folder"] for m in members]}
        for name, (side, field) in SUMMARY_FIELDS.items():
            values = [(m[side] if side else m)[field] for m in members]
            values = [value for value in values if value is not None]
            block[name] = sum(values) / len(values) if values else None
            block[f"{name}_n"] = len(values)
        summary[key] = block
    return summary


if __name__ == "__main__":
    main()

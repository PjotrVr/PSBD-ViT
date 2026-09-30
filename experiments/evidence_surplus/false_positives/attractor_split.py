"""Split A3's false positives into attractor-class and other, from the A3 records and the cached sweep.

The attractor class of a model is the class that masked clean validation
predictions flip into most often at PSBD-TM's adaptive rate, read from the cached
per-pass argmax. A false positive belongs to it when the unperturbed prediction
is that class. The false positive rows are re-derived from the cache exactly as
surplus_factor/removal.py derives them, and checked against the record.

    python -m experiments.evidence_surplus.false_positives.attractor_split
"""

import json
import os

import torch

from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defenses.decision import RECOMMENDED_PLACEMENT
from defenses.scores import psu_ratio_from_cache
from experiments._paths import experiment_results_dir
from experiments.evidence_surplus.surplus_factor.removal import MODELS

SLUG = "evidence_surplus"
RESULTS = "results"
EDITS = ("unedited", "stream_own_class", "input_sufficient_tokens")
THRESHOLDS = ("0.01", "0.05")
NUM_CLASSES = {"cifar10": 10, "gtsrb": 43, "cifar100": 100, "tiny": 200}


def main():
    out_dir = os.path.join(experiment_results_dir(SLUG, RESULTS), "false_positives")
    rows = [analyse(model) for model in MODELS]
    pooled = pool_by_dataset(rows)
    with open(os.path.join(out_dir, "attractor_split.json"), "w") as f:
        json.dump({"models": rows, "pooled_by_dataset": pooled}, f, indent=1)
    print_table(rows)
    print(json.dumps(pooled, indent=1))


def analyse(model):
    with open(
        os.path.join(RESULTS, "_experiments", SLUG, "removal", f"{model}.json")
    ) as f:
        removal = json.load(f)
    rate = removal["adaptive_rate"]
    psbd_dir = os.path.join(RESULTS, model, "psbd")
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
    per_pass, per_pass_argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, RECOMMENDED_PLACEMENT, rate, "validation")
    )
    psu = psu_ratio_from_cache(probs, labels, per_pass)  # (n,)
    num_classes = probs.shape[1]

    fp_rows = torch.nonzero(psu <= removal["cached_thresholds"]["0.05"]).squeeze(1)
    assert fp_rows.numel() == removal["n_false_positive"]
    fp_at_001 = psu[fp_rows] <= removal["cached_thresholds"]["0.01"]
    assert fp_at_001.tolist() == removal["sets"]["false_positive"]["flagged_at_0.01"]

    attractor, attractor_all_passes, flip_counts = attractor_class(
        per_pass_argmax, labels, num_classes
    )
    flipped = per_pass_argmax.long() != labels.long().unsqueeze(0)  # (passes, n)

    # Images that never change their answer in any pass, the tail of the docs.
    never_flip = ~flipped.any(dim=0)  # (n,)
    tail_counts = torch.bincount(labels.long()[never_flip], minlength=num_classes)
    tail_class = int(tail_counts.argmax())

    in_attractor = labels.long()[fp_rows] == attractor  # (n_fp,)
    in_attractor_001 = in_attractor & fp_at_001

    edits = removal["sets"]["false_positive"]["edits"]
    split = {}
    for edit in EDITS:
        fresh = torch.tensor(edits[edit]["psu_ratio"])  # (n_fp,)
        assert fresh.numel() == fp_rows.numel()
        entry = {}
        for threshold in THRESHOLDS:
            flag = fresh <= removal["cached_thresholds"][threshold]  # (n_fp,)
            member = in_attractor & (fp_at_001 if threshold == "0.01" else True)
            other = ~in_attractor & (fp_at_001 if threshold == "0.01" else True)
            entry[threshold] = {
                "attractor": rate_of(flag, member),
                "other": rate_of(flag, other),
            }
        split[edit] = entry

    dataset = model.split("_")[1]
    record = {
        "model": model,
        "dataset": dataset,
        "num_classes": num_classes,
        "attractor_class": attractor,
        "attractor_class_all_passes": attractor_all_passes,
        "never_flip_tail_class": tail_class,
        "never_flip_tail_share": float(never_flip.float().mean()),
        "attractor_share_of_flips": float(flip_counts[attractor] / flip_counts.sum()),
        "attractor_share_of_clean_images": float(
            (labels.long() == attractor).float().mean()
        ),
        "n_false_positive": int(fp_rows.numel()),
        "n_attractor_false_positive": int(in_attractor.sum()),
        "attractor_share_of_false_positives_0.05": float(in_attractor.float().mean()),
        "n_false_positive_0.01": int(fp_at_001.sum()),
        "n_attractor_false_positive_0.01": int(in_attractor_001.sum()),
        "attractor_share_of_false_positives_0.01": float(
            in_attractor_001.sum() / fp_at_001.sum()
        ),
        "median_surplus_false_positive": removal["false_positive_median_surplus"],
        "median_surplus_non_flagged": removal["median_non_flagged_surplus"],
        "post_edit_flag_rate": split,
    }
    return record


def attractor_class(per_pass_argmax, labels, num_classes):
    # The attractor is where masking sends predictions, so only flipped passes count.
    flipped = per_pass_argmax.long() != labels.long().unsqueeze(0)  # (passes, n)
    flip_counts = torch.bincount(per_pass_argmax.long()[flipped], minlength=num_classes)
    all_counts = torch.bincount(per_pass_argmax.long().flatten(), minlength=num_classes)
    return int(flip_counts.argmax()), int(all_counts.argmax()), flip_counts


def pool_by_dataset(rows):
    # Pooling weights each false positive equally, so a model with 5 other
    # false positives does not count as much as a model with 51.
    pooled = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        entry = {"models": len(group)}
        for edit in EDITS:
            for part in ("attractor", "other"):
                cells = [r["post_edit_flag_rate"][edit]["0.05"][part] for r in group]
                n = sum(c["n"] for c in cells)
                flagged = sum(c["n"] * c["flag_rate"] for c in cells if c["n"])
                entry[f"{edit}_{part}"] = {
                    "n": n,
                    "flag_rate": flagged / n if n else None,
                }
        pooled[dataset] = entry
    return pooled


def rate_of(flag, member):
    count = int(member.sum())
    if count == 0:
        return {"n": 0, "flag_rate": None}
    value = float(flag[member].float().mean())
    return {"n": count, "flag_rate": value}


def print_table(rows):
    print(
        "model, classes, attractor, image share, tail class, fp share 0.05, fp share 0.01"
    )
    print("  then per edit at 0.05: attractor rate (n), other rate (n)")
    for r in rows:
        share = r["attractor_share_of_false_positives_0.05"]
        share_001 = r["attractor_share_of_false_positives_0.01"]
        print(
            r["model"],
            r["num_classes"],
            r["attractor_class"],
            f"{r['attractor_share_of_clean_images']:.3f}",
            r["never_flip_tail_class"],
            f"{share:.2f}",
            f"{share_001:.2f}",
        )
        for edit in EDITS:
            a = r["post_edit_flag_rate"][edit]["0.05"]["attractor"]
            o = r["post_edit_flag_rate"][edit]["0.05"]["other"]
            print(f"    {edit}: {fmt(a)} (n{a['n']}), {fmt(o)} (n{o['n']})")


def fmt(entry):
    value = entry["flag_rate"]
    return "--" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()

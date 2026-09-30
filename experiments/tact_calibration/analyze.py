"""CPU stage: calibrated PSBD-TM scores on an enlarged validation set.

Every model's 20-pass caches (run_passes.py) are re-split. The validation set grows
from the standard 2000 images to VALIDATION_SIZE of the same permutation, and the
added images leave the clean and triggered evaluation sets, so the 2 never overlap.
Every score below is read on that shrunken evaluation set with the first k passes.

    fractional PSU     s(x) = 1 - (1/k) sum_i P_c(x; p, theta_i') / P_c(x; theta)
    fallback class     r(x) = argmax_j sum_i 1[a_i(x) = j and a_i(x) != c]
    fallback score     u(x) = (n_r F_r(s) + m F(s)) / (n_r + m)
    class score        z(x) = (s - M_c) / D_c,
                       M_c = (n_c med_c + m med) / (n_c + m),
                       D_c = (n_c mad_c + m mad) / (n_c + m)
    OR rule            min over probes of F_probe(score_probe(x))

| symbol | meaning |
|---|---|
| c | the unperturbed argmax of x |
| a_i(x) | the argmax of perturbed pass i |
| r(x) | the most frequent class among the passes that left c, ties to the smallest class index, undefined when no pass left c |
| F, F_g | the fraction of clean validation scores below a value, over all validation images or those with key g |
| n_g | validation images with key g (r for the fallback score, c for the class score) |
| m | the shrinkage strength in images |
| med, mad | median and median absolute deviation of validation s, per class with subscript c |

An input with no flip is ranked by F alone. Low means poisoned for every score,
and every score is thresholded at a quantile of its own validation values.

    python -m experiments.tact_calibration.analyze tuning
    python -m experiments.tact_calibration.analyze confirm
"""

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch

from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import detection_report
from defenses.scores import psu_ratio_from_cache, to_rank
from experiments._paths import experiment_result_path
from experiments.cache_readouts.shared import paired_summary
from scripts.coverage_ledger import source_classes_of

SLUG = "tact_calibration"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
MODELS_ROOT = os.path.join(RESULTS_DIR, "_experiments", SLUG, "models")
PLAN_PATH = experiment_result_path(SLUG, "plan.json", RESULTS_DIR)
PREREGISTRATION = os.path.join("experiments", SLUG, "preregistration.json")

# CIFAR-100 and Tiny hold 50 and 25 validation images per class at 5000, and keep
# 5000 evaluation images. CIFAR-10 and GTSRB already hold 200 and about 47 per
# class at 2000, so 4000 doubles that and keeps 6000 and 8630 for evaluation.
VALIDATION_SIZE = {"cifar10": 4000, "cifar100": 5000, "gtsrb": 4000, "tiny": 5000}
ANCHOR = "before_attention_norm_token_mask_k20"
PARTNER = "pre_residual_blocks_5_8_k20"
PASS_COUNTS = (3, 5, 10, 20)
PRIMARY_K = 20
QUANTILES = (0.01, 0.05, 0.10)
SHRINKAGE_GRID = (5, 20, 50, 200)
DETECTORS = ("beatrix", "ted")
METHODS = (
    "plain",
    "fallback",
    "class",
    "or_fallback",
    "or_class",
    "final",
    "final_or_fallback",
    "final_or_class",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tuning", "confirm"))
    stage = parser.parse_args().stage
    started = time.perf_counter()

    with open(PLAN_PATH) as handle:
        plan = json.load(handle)["models"]
    if stage == "tuning":
        entries = [m for m in plan if m["group"] == "tuning"]
        grid = {"fallback": SHRINKAGE_GRID, "class": SHRINKAGE_GRID}
    else:
        # The tuning models are reread at the frozen m so the combined TaCT table
        # holds all of them, and are never pooled into a confirmation claim.
        frozen = read_committed_preregistration()
        entries = plan
        grid = {name: (frozen["shrinkage"][name],) for name in ("fallback", "class")}

    rows = []
    for entry in entries:
        if not cache_complete(entry["folder"]):
            print(f"[missing] {entry['folder']}")
            continue
        rows.append(measure(entry, grid))
        print(f"[ok] {entry['folder']}", flush=True)

    payload = {
        "stage": stage,
        "validation_size": VALIDATION_SIZE,
        "pass_counts": list(PASS_COUNTS),
        "grid": {name: list(values) for name, values in grid.items()},
        "models": rows,
        "summary": summarize(rows, grid),
        "wall_seconds": time.perf_counter() - started,
    }
    if stage == "tuning":
        payload["selection"] = select_shrinkage(rows)
    else:
        payload["preregistration_sha256"] = file_sha256(PREREGISTRATION)
    path = experiment_result_path(SLUG, f"{stage}.json", RESULTS_DIR)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def read_committed_preregistration():
    # The confirmation read is allowed only once the rule is frozen in a commit,
    # so the file on disk must equal the committed one.
    committed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", PREREGISTRATION],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert committed, f"{PREREGISTRATION} is not committed"
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", PREREGISTRATION]
    )
    assert unchanged.returncode == 0, f"{PREREGISTRATION} differs from HEAD"
    with open(PREREGISTRATION) as handle:
        frozen = json.load(handle)
    return frozen


def cache_complete(folder):
    psbd_dir = os.path.join(MODELS_ROOT, folder, "psbd")
    record_path = os.path.join(MODELS_ROOT, folder, "run_record.json")
    if not os.path.exists(record_path):
        return False
    with open(record_path) as handle:
        record = json.load(handle)
    rates = {p: record["placements"][p]["rate"] for p in record["placements"]}
    complete = all(rate is not None for rate in rates.values()) and os.path.isdir(
        psbd_dir
    )
    return complete


def measure(entry, grid):
    folder = entry["folder"]
    with open(os.path.join(MODELS_ROOT, folder, "run_record.json")) as handle:
        record = json.load(handle)
    psbd_dir = os.path.join(MODELS_ROOT, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    dataset = manifest["dataset"]
    rows_of = enlarged_rows(manifest, VALIDATION_SIZE[dataset])

    anchor_rate = record["placements"]["before_attention_norm_token_mask"]["rate"]
    partner_rate = record["placements"]["pre_residual_blocks_5_8"]["rate"]
    anchor = load_placement(psbd_dir, ANCHOR, anchor_rate, rows_of)
    partner = load_placement(psbd_dir, PARTNER, partner_rate, rows_of)
    sources = source_classes_of(
        CHECKPOINTS_DIR, {"folder_name": folder, "attack": attack_of(folder)}
    )

    row = {
        "folder": folder,
        "group": entry["group"],
        "in_panel": entry.get("in_panel", True),
        "dataset": dataset,
        "attack": attack_of(folder),
        "rates": {"anchor": anchor_rate, "partner": partner_rate},
        "split_sizes": {split: int(len(anchor[split]["labels"])) for split in anchor},
        "source_classes": list(sources) if sources else None,
        "target": manifest["probe_target_label"],
        "by_k": {},
    }
    if sources:
        row["premise"] = {
            "anchor": fallback_premise(anchor, sources),
            "partner": fallback_premise(partner, sources),
        }
    row["detectors"] = detector_readings(folder, rows_of, manifest)

    for k in PASS_COUNTS:
        anchor_psu = {split: psu_at(anchor[split], k) for split in anchor}
        partner_psu = {split: psu_at(partner[split], k) for split in partner}
        readings = {}
        for m_fallback in grid["fallback"]:
            for m_class in grid["class"]:
                scores = method_scores(
                    anchor, anchor_psu, partner_psu, k, m_fallback, m_class
                )
                for method, by_split in scores.items():
                    key = f"{method}/mf{m_fallback}/mc{m_class}"
                    readings[key] = evaluate(by_split, anchor, sources)
        row["by_k"][str(k)] = readings
    return row


def attack_of(folder):
    with open(os.path.join(CHECKPOINTS_DIR, folder, "args.json")) as handle:
        attack = json.load(handle)["attack"]
    return attack


def enlarged_rows(manifest, validation_size):
    # The standard split is perm[:2000] for validation and perm[2000:] for the
    # clean pool, in loader order. The first validation_size - 2000 clean rows are
    # therefore the next images of the same permutation, and they move to
    # validation together with every triggered copy of them.
    extra = validation_size - len(manifest["heldout_indices"])
    clean_indices = manifest["analysis_clean_indices"]
    backdoor_indices = manifest["analysis_backdoor_indices"]
    moved = set(clean_indices[:extra])

    backdoor_rows = [
        row for row, index in enumerate(backdoor_indices) if index not in moved
    ]
    row_of_clean = {index: row for row, index in enumerate(clean_indices)}
    rows_of = {
        "validation": ("validation", list(range(len(manifest["heldout_indices"])))),
        "validation_extra": ("clean", list(range(extra))),
        "clean": ("clean", list(range(extra, len(clean_indices)))),
        "backdoor": ("backdoor", backdoor_rows),
        "paired": (
            "clean",
            [row_of_clean[backdoor_indices[row]] for row in backdoor_rows],
        ),
    }

    validation_images = set(manifest["heldout_indices"]) | moved
    evaluation_images = set(clean_indices[extra:]) | {
        backdoor_indices[row] for row in backdoor_rows
    }
    assert len(validation_images) == validation_size
    assert validation_images.isdisjoint(evaluation_images), (
        "validation leaks into evaluation"
    )
    return rows_of


def load_placement(psbd_dir, placement, rate, rows_of):
    raw = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, loader_labels = load_baseline(baseline_path(psbd_dir, split))
        per_pass_probs, per_pass_argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )  # (passes, n_split) each
        assert per_pass_probs.shape == (20, labels.shape[0]), (placement, split)
        raw[split] = {
            "probs": probs,  # (n_split, num_classes)
            "labels": labels,  # (n_split,)
            "loader_labels": loader_labels,  # (n_split,)
            "per_pass_probs": per_pass_probs,  # (20, n_split)
            "per_pass_argmax": per_pass_argmax.long(),  # (20, n_split)
        }

    groups = {}
    for group, (split, rows) in rows_of.items():
        index = torch.tensor(rows, dtype=torch.long)
        groups[group] = {
            "probs": raw[split]["probs"][index],
            "labels": raw[split]["labels"][index],
            "loader_labels": raw[split]["loader_labels"][index],
            "per_pass_probs": raw[split]["per_pass_probs"][:, index],
            "per_pass_argmax": raw[split]["per_pass_argmax"][:, index],
        }

    # A triggered row's true class is its clean copy's label, which the paired
    # clean rows carry, since the backdoor loader serves the target instead.
    groups["backdoor"]["true_labels"] = groups["paired"]["loader_labels"]
    extra = groups.pop("validation_extra")
    groups["validation"] = {
        key: torch.cat([groups["validation"][key], extra[key]], dim=-1)
        if key.startswith("per_pass")
        else torch.cat([groups["validation"][key], extra[key]], dim=0)
        for key in extra
    }
    return groups


def psu_at(group, k):
    psu = psu_ratio_from_cache(
        group["probs"], group["labels"], group["per_pass_probs"][:k]
    )  # (n,)
    return psu


def fallback_class(group, k):
    argmax = group["per_pass_argmax"][:k]  # (k, n)
    flipped = argmax != group["labels"].view(1, -1)  # (k, n)
    num_classes = group["probs"].shape[1]

    # argmax over the counts returns the first maximal index, which is the tie
    # rule the docstring states: the smallest class index among the modes.
    counts = torch.zeros(num_classes, argmax.shape[1])  # (num_classes, n)
    counts.scatter_add_(0, argmax, flipped.float())
    has_flip = flipped.any(dim=0)  # (n,)
    reference = torch.where(
        has_flip, counts.argmax(dim=0), torch.full_like(has_flip, -1, dtype=torch.long)
    )
    return reference  # (n,), -1 where no pass flipped


def method_scores(anchor, anchor_psu, partner_psu, k, m_fallback, m_class):
    fallback_keys = {split: fallback_class(anchor[split], k) for split in anchor}
    predicted = {split: anchor[split]["labels"] for split in anchor}
    fallback = shrunk_percentile(anchor_psu, fallback_keys, m_fallback)
    class_z = shrunk_robust_z(anchor_psu, predicted, m_class)

    def ranks(scores):
        ranked = {
            split: to_rank(scores[split], scores["validation"]) for split in scores
        }
        return ranked

    plain_rank, partner_rank = ranks(anchor_psu), ranks(partner_psu)
    fallback_rank, class_rank = ranks(fallback), ranks(class_z)

    def minimum(*members):
        combined = {
            split: torch.stack([member[split] for member in members]).min(dim=0).values
            for split in members[0]
        }
        return combined

    scores = {
        "plain": anchor_psu,
        "fallback": fallback,
        "class": class_z,
        "or_fallback": minimum(plain_rank, fallback_rank),
        "or_class": minimum(plain_rank, class_rank),
        "final": minimum(plain_rank, partner_rank),
        "final_or_fallback": minimum(plain_rank, partner_rank, fallback_rank),
        "final_or_class": minimum(plain_rank, partner_rank, class_rank),
    }
    return scores


def shrunk_percentile(psu, keys, m):
    validation = psu["validation"]  # (n_validation,)
    validation_keys = keys["validation"]  # (n_validation,)
    calibrated = {}
    for split in psu:
        scores, split_keys = psu[split], keys[split]  # (n,), (n,)
        global_rank = to_rank(scores, validation)  # (n,)
        out = global_rank.clone()
        for key in split_keys.unique().tolist():
            if key < 0:
                continue
            members = validation[validation_keys == key]  # (n_key,)
            where = split_keys == key
            n_key = members.numel()
            if n_key == 0:
                continue
            class_rank = to_rank(scores[where], members)
            out[where] = (n_key * class_rank + m * global_rank[where]) / (n_key + m)
        calibrated[split] = out
    return calibrated


def shrunk_robust_z(psu, keys, m):
    validation = psu["validation"]  # (n_validation,)
    validation_keys = keys["validation"]  # (n_validation,)
    global_median = validation.median()
    global_mad = (validation - global_median).abs().median()
    calibrated = {}
    for split in psu:
        scores, split_keys = psu[split], keys[split]  # (n,), (n,)
        out = torch.empty_like(scores)
        for key in split_keys.unique().tolist():
            members = validation[validation_keys == key]  # (n_key,)
            where = split_keys == key
            n_key = members.numel()
            if n_key:
                class_median = members.median()
                class_mad = (members - class_median).abs().median()
            else:
                class_median, class_mad = global_median, global_mad
            location = (n_key * class_median + m * global_median) / (n_key + m)
            scale = ((n_key * class_mad + m * global_mad) / (n_key + m)).clamp_min(1e-6)
            out[where] = (scores[where] - location) / scale
        calibrated[split] = out
    return calibrated


def evaluate(scores, anchor, sources):
    validation, clean = (
        scores["validation"],
        scores["clean"],
    )  # (n_validation,), (n_clean,)
    backdoor, paired = scores["backdoor"], scores["paired"]  # (n_backdoor,) each
    reading = {}
    for quantile in QUANTILES:
        report = detection_report(validation, paired, backdoor, quantile)
        reading["auroc"] = report["auroc"]
        tag = f"q{quantile:.2f}"
        reading[f"{tag}:tpr"] = report["tpr"]
        reading[f"{tag}:paired_fpr"] = report["fpr"]
        reading[f"{tag}:fpr"] = float((clean < report["threshold"]).float().mean())
        if sources:
            source_clean = torch.isin(
                anchor["clean"]["loader_labels"], torch.tensor(sources)
            )
            reading[f"{tag}:source_fpr"] = float(
                (clean[source_clean] < report["threshold"]).float().mean()
            )
    return reading


def fallback_premise(groups, sources):
    # Where the perturbed passes of a triggered input go when they leave the
    # target: its own source class, the attractor (the most common fallback of
    # clean validation images) or elsewhere.
    triggered = groups["backdoor"]
    reference = fallback_class(triggered, PRIMARY_K)  # (n_backdoor,)
    validation_reference = fallback_class(
        groups["validation"], PRIMARY_K
    )  # (n_validation,)
    flipped_validation = validation_reference[validation_reference >= 0]
    attractor = (
        int(flipped_validation.mode().values) if flipped_validation.numel() else -1
    )
    true_labels = triggered["true_labels"]  # (n_backdoor,)
    is_source_true = torch.isin(true_labels, torch.tensor(sources))
    assert bool(is_source_true.all()), "a triggered TaCT row outside its source classes"

    shifted = triggered["per_pass_argmax"] != triggered["labels"].view(1, -1)  # (20, n)
    premise = {
        "n_triggered": int(reference.numel()),
        "no_flip": float((reference < 0).float().mean()),
        "fallback_is_own_source": float((reference == true_labels).float().mean()),
        "attractor_class": attractor,
        "attractor_is_a_source": attractor in sources,
        "fallback_is_attractor": float((reference == attractor).float().mean()),
        "attractor_share_clean_validation": float(
            (validation_reference == attractor).float().mean()
        ),
        "triggered_shift_ratio": float(shifted.float().mean()),
        "validation_shift_ratio": float(
            (
                groups["validation"]["per_pass_argmax"]
                != groups["validation"]["labels"].view(1, -1)
            )
            .float()
            .mean()
        ),
        "triggered_predicted_target": float(
            (triggered["labels"] == triggered["loader_labels"]).float().mean()
        ),
    }
    return premise


def detector_readings(folder, rows_of, manifest):
    readings = {}
    for detector in DETECTORS:
        directory = os.path.join(RESULTS_DIR, folder, "detectors")
        paths = {
            split: os.path.join(directory, f"{detector}_scores_{split}.pt")
            for split in ("validation", "clean", "backdoor")
        }
        if not all(os.path.exists(path) for path in paths.values()):
            continue
        raw = {split: torch.load(path).float() for split, path in paths.items()}
        assert raw["validation"].shape[0] == len(manifest["heldout_indices"])
        assert raw["clean"].shape[0] == len(manifest["analysis_clean_indices"])
        assert raw["backdoor"].shape[0] == len(manifest["analysis_backdoor_indices"])
        scores = {
            group: raw[split][torch.tensor(rows, dtype=torch.long)]
            for group, (split, rows) in rows_of.items()
        }
        scores["validation"] = torch.cat(
            [scores["validation"], scores.pop("validation_extra")]
        )
        reading = {}
        for quantile in QUANTILES:
            report = detection_report(
                scores["validation"], scores["paired"], scores["backdoor"], quantile
            )
            reading["auroc"] = report["auroc"]
            reading[f"q{quantile:.2f}:tpr"] = report["tpr"]
            reading[f"q{quantile:.2f}:fpr"] = float(
                (scores["clean"] < report["threshold"]).float().mean()
            )
        readings[detector] = reading
    return readings


def reading_key(method, grid_point):
    key = f"{method}/mf{grid_point[0]}/mc{grid_point[1]}"
    return key


def summarize(rows, grid):
    fields = (
        ["auroc"]
        + [f"q{q:.2f}:tpr" for q in QUANTILES]
        + [f"q{q:.2f}:fpr" for q in QUANTILES]
    )
    points = [(mf, mc) for mf in grid["fallback"] for mc in grid["class"]]
    subsets = {
        "tuning": [r for r in rows if r["group"] == "tuning"],
        "holdout": [r for r in rows if r["group"] == "holdout"],
        "tact_panel": [
            r for r in rows if r["attack"] == "tact" and r["group"] != "holdout"
        ],
        "dev_non_tact": [r for r in rows if r["group"] == "dev" and r["in_panel"]],
        "panel_non_tact": [
            r
            for r in rows
            if r["attack"] != "tact" and r["in_panel"] and r["group"] != "holdout"
        ],
    }
    summary = {}
    for name, subset in subsets.items():
        if not subset:
            continue
        block = {"n": len(subset), "folders": [r["folder"] for r in subset]}
        for k in PASS_COUNTS:
            for point in points:
                reference = reading_key("plain", point)
                for method in METHODS:
                    key = reading_key(method, point)
                    block[f"k{k}/{key}"] = {
                        field: paired_summary(
                            [r["by_k"][str(k)][key][field] for r in subset],
                            [r["by_k"][str(k)][reference][field] for r in subset],
                        )
                        for field in fields
                    }
        summary[name] = block
    return summary


def select_shrinkage(rows):
    # m for each calibrated score alone, by its mean TPR over the 3 headline FPRs
    # on the 4 tuning models at k = 20. Ties go to the larger m, the one closer to
    # the global calibration.
    selection = {}
    for name, position in (("fallback", 0), ("class", 1)):
        scored = {}
        for m in SHRINKAGE_GRID:
            point = (m, SHRINKAGE_GRID[0]) if position == 0 else (SHRINKAGE_GRID[0], m)
            key = reading_key(name, point)
            values = [
                r["by_k"][str(PRIMARY_K)][key][f"q{q:.2f}:tpr"]
                for r in rows
                for q in QUANTILES
            ]
            scored[m] = float(np.mean(values))
        chosen = max(sorted(scored, reverse=True), key=lambda m: round(scored[m], 3))
        selection[name] = {"scores": scored, "chosen": chosen}
    return selection


def file_sha256(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return digest


if __name__ == "__main__":
    main()

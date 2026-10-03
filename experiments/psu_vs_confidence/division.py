"""What dividing PSU by the starting confidence does, rendered into README.md.

Fractional PSU divides the paper's absolute drop by the starting confidence P_c, the
no-perturbation probability of the predicted class. This script reads, on the headline
panel (`experiments.cache_readouts.shared.load_model_set("panel")`) with PSBD-TM
(`before_attention_norm_token_mask`) at the adaptive 0.8 rate, how P_c is distributed on
clean validation, paired clean and triggered images, how much of P_c survives masking,
how the 2 scores rank images, which images change side at thresholds at the 0.01, 0.05
and 0.10 quantiles of clean validation (low means poisoned) and how the clean validation
flags spread over true classes. It then reads the training-set setting of Li et al. from
the raw tensors of `experiments/training_set_detection` (PSBD-TM at the ladder's
`our_rate`, thresholds at quantiles of the clean training images). CPU only, cached
tensors only. It writes `results/_experiments/psu_vs_confidence/division.json` and
replaces the block between the division markers of README.md.

    PYTHONPATH=. .venv/bin/python experiments/psu_vs_confidence/division.py
"""

import glob
import json
import os
import statistics

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from defenses.cache import read_split_manifest
from defenses.decision import pair_clean_to_backdoor
from defenses.scores import psu_from_cache, psu_ratio_from_cache
from experiments._paths import experiment_result_path
from experiments.cache_readouts.shared import (
    choose_rate,
    load_baselines,
    load_model_set,
    load_passes,
    psbd_dir_of,
    validation_shift_by_rate,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
README = os.path.join(REPO_ROOT, "experiments", "psu_vs_confidence", "README.md")
TRAINING_SET_RAW = os.path.join(
    REPO_ROOT, "results", "_experiments", "training_set_detection", "raw"
)
PLACEMENT = "before_attention_norm_token_mask"
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
DISPLAY_NAMES = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "gtsrb": "GTSRB",
    "tiny": "Tiny",
}
QUANTILES = (0.01, 0.05, 0.10)
CONFIDENCE_EDGES = (0.5, 0.9, 0.99)
SCORES = ("absolute", "fractional")
# A training-set TPR change above this size is named in the README as an exception.
TRAINING_SET_GAIN_FLOOR = 0.05
BEGIN = "<!-- division:begin -->"
END = "<!-- division:end -->"


def main():
    panel_rows = [
        row for row in map(measure_panel_model, load_model_set("panel")) if row
    ]
    panel_summary = {
        group: summarize([r for r in panel_rows if group in ("all", r["dataset"])])
        for group in DATASETS + ("all",)
    }
    training_rows = [measure_training_set_model(d) for d in training_set_folders()]

    payload = {
        "placement": PLACEMENT,
        "rate_rule": "adaptive 0.8",
        "quantiles": QUANTILES,
        "summary_note": "mean and median over models of each per-model value",
        "panel_summary": panel_summary,
        "panel_models": panel_rows,
        "training_set_models": training_rows,
    }
    path = write_payload(payload)
    render(payload)
    print(f"{len(panel_rows)} panel models, {len(training_rows)} training-set models")
    print(f"wrote {path} and {README}")


def measure_panel_model(model):
    psbd_dir = psbd_dir_of(model)
    baselines = load_baselines(psbd_dir)
    shift_by_rate = validation_shift_by_rate(psbd_dir, PLACEMENT, baselines)
    rate = choose_rate(shift_by_rate, "adaptive")
    if rate is None:
        return None

    manifest = read_split_manifest(psbd_dir)
    passes = load_passes(psbd_dir, PLACEMENT, rate, baselines)
    validation = score_split(passes["validation"])
    clean = score_split(passes["clean"])
    backdoor = score_split(passes["backdoor"])

    # The clean test split is paired to the triggered images by source index, so the
    # clean side of every TPR and AUROC reads the same images the trigger was put on.
    paired_clean = {
        name: pair_clean_to_backdoor(values, manifest) for name, values in clean.items()
    }  # each (n_backdoor,)

    true_labels = baselines["validation"][2]  # (n_validation,) loader labels
    assert true_labels.shape == validation["confidence"].shape

    row = {
        "folder": model["folder_name"],
        "dataset": model["dataset"],
        "attack": model["attack"],
        "poison_rate": model["poison_rate"],
        "rate": rate,
        "confidence": {
            "validation": confidence_profile(validation["confidence"]),
            "clean": confidence_profile(paired_clean["confidence"]),
            "backdoor": confidence_profile(backdoor["confidence"]),
        },
        "surviving_median": {
            "validation": validation["surviving"].median().item(),
            "backdoor": backdoor["surviving"].median().item(),
        },
        "auroc": {
            name: low_means_poisoned_auroc(paired_clean[name], backdoor[name])
            for name in SCORES
        },
        "spearman_abs_frac_pooled": pooled_rank_correlation(validation, backdoor),
        "spearman_with_confidence_val": {
            name: float(
                spearmanr(validation[name].numpy(), validation["confidence"].numpy())[0]
            )
            for name in SCORES
        },
        "budgets": {
            f"q{q:.2f}": budget_readout(validation, paired_clean, backdoor, q)
            for q in QUANTILES
        },
        "per_class": {
            name: {
                f"q{q:.2f}": class_spread(validation[name], true_labels, q)
                for q in QUANTILES
            }
            for name in SCORES
        },
    }
    return row


def score_split(split):
    probs = split["baseline_probs"]  # (n, num_classes)
    labels = split["baseline_labels"]  # (n,) unperturbed argmax
    per_pass = split["per_pass_probs"]  # (passes, n) probability of that argmax

    confidence = probs.gather(1, labels.view(-1, 1).long()).squeeze(1).float()  # (n,)
    surviving = per_pass.float().mean(dim=0)  # (n,)
    absolute = psu_from_cache(probs, labels, per_pass)  # (n,)
    fractional = psu_ratio_from_cache(probs, labels, per_pass)  # (n,)
    assert absolute.shape == fractional.shape == confidence.shape

    scores = {
        "confidence": confidence,
        "surviving": surviving,
        "absolute": absolute,
        "fractional": fractional,
    }
    return scores


def confidence_profile(confidence):
    profile = {"median": confidence.median().item()}
    for edge in CONFIDENCE_EDGES:
        profile[f"share_below_{edge}"] = (confidence < edge).float().mean().item()
    return profile


def low_means_poisoned_auroc(clean, backdoor):
    # Backdoor is the positive class and a low score means poisoned, so the scores
    # enter negated.
    scores = -np.concatenate([clean.numpy(), backdoor.numpy()])  # (n_clean + n_bd,)
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    value = float(roc_auc_score(labels, scores))
    return value


def pooled_rank_correlation(validation, backdoor):
    pooled_absolute = torch.cat([validation["absolute"], backdoor["absolute"]])
    pooled_fractional = torch.cat([validation["fractional"], backdoor["fractional"]])
    rho = float(spearmanr(pooled_absolute.numpy(), pooled_fractional.numpy())[0])
    return rho


def budget_readout(validation, clean, backdoor, quantile):
    flagged = {}
    readout = {}
    for name in SCORES:
        threshold = torch.quantile(validation[name], quantile)
        flagged[name] = {
            split: scores[name] <= threshold
            for split, scores in (
                ("validation", validation),
                ("clean", clean),
                ("backdoor", backdoor),
            )
        }  # each (n_split,) bool
        flagged_validation = flagged[name]["validation"]
        readout[name] = {
            "tpr": flagged[name]["backdoor"].float().mean().item(),
            "fpr_clean": flagged[name]["clean"].float().mean().item(),
            "median_confidence_of_flagged_val": validation["confidence"][
                flagged_validation
            ]
            .median()
            .item(),
        }

    absolute, fractional = flagged["absolute"], flagged["fractional"]
    kept = absolute["validation"] & fractional["validation"]
    dropped = absolute["validation"] & ~fractional["validation"]
    added = fractional["validation"] & ~absolute["validation"]
    gained = fractional["backdoor"] & ~absolute["backdoor"]
    lost = absolute["backdoor"] & ~fractional["backdoor"]

    readout["tpr_gain"] = readout["fractional"]["tpr"] - readout["absolute"]["tpr"]
    readout["val_flags_kept"] = kept.sum().item() / max(
        1, absolute["validation"].sum().item()
    )
    readout["val_dropped_median_confidence"] = masked_median(
        validation["confidence"], dropped
    )
    readout["val_added_median_confidence"] = masked_median(
        validation["confidence"], added
    )
    readout["backdoor_gained"] = gained.float().mean().item()
    readout["backdoor_lost"] = lost.float().mean().item()
    readout["backdoor_gained_median_confidence"] = masked_median(
        backdoor["confidence"], gained
    )
    return readout


def masked_median(values, mask):
    if not mask.any():
        return None
    median = values[mask].median().item()
    return median


def class_spread(scores, true_labels, quantile):
    threshold = torch.quantile(scores, quantile)
    flagged = scores <= threshold  # (n,)
    classes = true_labels.unique()  # (C,)
    counts = torch.stack([(true_labels == c).sum() for c in classes]).float()  # (C,)
    flags = torch.stack([flagged[true_labels == c].sum() for c in classes]).float()

    # Index of dispersion of the per-class flag counts against a binomial spread at
    # the global flag rate, 1 when the flags fall on classes in proportion to their
    # size.
    flag_rate = flagged.float().mean()
    expected = counts * flag_rate  # (C,)
    variance = counts * flag_rate * (1 - flag_rate)  # (C,)
    chi_square = (((flags - expected) ** 2) / variance.clamp_min(1e-9)).sum()
    dispersion = chi_square / (len(classes) - 1)

    worst_count = max(1, len(classes) // 10)
    worst_flags = flags.sort(descending=True).values[:worst_count].sum()
    worst_tenth_share = worst_flags / flags.sum().clamp_min(1)

    spread = {
        "dispersion": dispersion.item(),
        "worst_tenth_share": worst_tenth_share.item(),
        "classes_with_a_flag": (flags > 0).float().mean().item(),
        "validation_images_per_class": counts.mean().item(),
    }
    return spread


def training_set_folders():
    folders = []
    for folder in sorted(glob.glob(os.path.join(TRAINING_SET_RAW, "vit_*"))):
        ladder_path = os.path.join(folder, "ladder_psbd_tm.pt")
        if not os.path.exists(ladder_path):
            continue
        ladder = torch.load(ladder_path, map_location="cpu", weights_only=False)
        passes_path = os.path.join(folder, f"passes_psbd_tm_{ladder['our_rate']}.pt")
        if os.path.exists(passes_path):
            folders.append(folder)
    return folders


def measure_training_set_model(folder):
    ladder = torch.load(
        os.path.join(folder, "ladder_psbd_tm.pt"),
        map_location="cpu",
        weights_only=False,
    )
    rate = ladder["our_rate"]
    baseline = torch.load(
        os.path.join(folder, "baseline.pt"), map_location="cpu", weights_only=False
    )
    passes = torch.load(
        os.path.join(folder, f"passes_psbd_tm_{rate}.pt"),
        map_location="cpu",
        weights_only=False,
    )

    row = {
        "folder": os.path.basename(folder),
        "rate": rate,
        "confidence": {},
        "tpr": {},
    }
    scores = {}
    for group in ("validation", "clean_train", "poisoned"):
        split = {
            "baseline_probs": baseline[group]["probs"],  # (n, num_classes)
            "baseline_labels": passes[group]["baseline_labels"],  # (n,)
            "per_pass_probs": passes[group]["per_pass_probs"],  # (passes, n)
        }
        scores[group] = score_split(split)
        row["confidence"][group] = confidence_profile(scores[group]["confidence"])

    # Li et al. set the threshold on the training set itself, so the clean training
    # images take the role clean validation has on the panel.
    for q in QUANTILES:
        row["tpr"][f"q{q:.2f}"] = {}
        for name in SCORES:
            threshold = torch.quantile(scores["clean_train"][name], q)
            tpr = (scores["poisoned"][name] <= threshold).float().mean().item()
            row["tpr"][f"q{q:.2f}"][name] = tpr
        tpr_pair = row["tpr"][f"q{q:.2f}"]
        tpr_pair["gain"] = tpr_pair["fractional"] - tpr_pair["absolute"]
    return row


def summarize(rows):
    measured = [strip_labels(row) for row in rows]
    summary = {
        "n": len(rows),
        "mean": reduce_tree(measured, statistics.mean),
        "median": reduce_tree(measured, statistics.median),
    }
    return summary


def strip_labels(row):
    labels = ("folder", "dataset", "attack", "poison_rate", "rate")
    measured = {key: value for key, value in row.items() if key not in labels}
    return measured


def reduce_tree(trees, reducer):
    first = trees[0]
    if isinstance(first, dict):
        reduced = {key: reduce_tree([t[key] for t in trees], reducer) for key in first}
        return reduced
    present = [value for value in trees if value is not None]
    reduced = reducer(present) if present else None
    return reduced


def write_payload(payload):
    path = experiment_result_path(
        "psu_vs_confidence", "division.json", os.path.join(REPO_ROOT, "results")
    )
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def render(payload):
    block = "\n".join(render_lines(payload))
    with open(README) as handle:
        text = handle.read()
    start = text.index(BEGIN)
    stop = text.index(END) + len(END)
    text = text[:start] + block + text[stop:]
    with open(README, "w") as handle:
        handle.write(text)


def percent(value):
    text = f"{100 * value:.0f}%"
    return text


def join_words(words):
    if len(words) == 1:
        return words[0]
    text = ", ".join(words[:-1]) + " and " + words[-1]
    return text


def render_lines(payload):
    summary = payload["panel_summary"]
    means = {group: summary[group]["mean"] for group in summary}
    pooled = means["all"]
    n_models = summary["all"]["n"]
    q1 = "q0.01"

    lines = [
        BEGIN,
        "<!-- Everything down to division:end is rendered by division.py from "
        "results/_experiments/psu_vs_confidence/division.json. -->",
        "",
        "## What the division does",
        "",
        "Fractional PSU divides the absolute drop by the starting confidence P_c, the "
        "unperturbed probability of the predicted class. What that changes is read on "
        f"the {n_models}-model panel with PSBD-TM (`{PLACEMENT}`) at the adaptive 0.8 "
        "rate, from the cached tensors only. Every number below is a mean over models "
        "of a per-model value.",
        "",
        "    PYTHONPATH=. .venv/bin/python experiments/psu_vs_confidence/division.py",
        "",
    ]
    lines += denominator_lines(means)
    lines += budget_lines(means, pooled, q1)
    lines += ranking_lines(means, pooled)
    lines += per_class_lines(means, pooled, q1)
    lines += training_set_lines(payload["training_set_models"])
    lines += [END]
    return lines


def denominator_lines(means):
    validation_low = {
        d: means[d]["confidence"]["validation"]["share_below_0.9"] for d in DATASETS
    }
    backdoor_low = [
        means[d]["confidence"]["backdoor"]["share_below_0.9"] for d in DATASETS
    ]
    by_low_share = sorted(DATASETS, key=lambda d: validation_low[d])
    pooled = means["all"]

    lines = [
        "### The denominator",
        "",
        "| dataset | P_c below 0.5 (validation) | below 0.9 (validation) "
        "| below 0.99 (validation) | below 0.9 (paired clean) | below 0.9 (triggered) "
        "| surviving share (validation) | surviving share (triggered) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in DATASETS + ("all",):
        confidence = means[group]["confidence"]
        surviving = means[group]["surviving_median"]
        lines.append(
            f"| {group} | {confidence['validation']['share_below_0.5']:.3f} "
            f"| {confidence['validation']['share_below_0.9']:.3f} "
            f"| {confidence['validation']['share_below_0.99']:.3f} "
            f"| {confidence['clean']['share_below_0.9']:.3f} "
            f"| {confidence['backdoor']['share_below_0.9']:.3f} "
            f"| {surviving['validation']:.3f} | {surviving['backdoor']:.3f} |"
        )
    lines += [
        "",
        "The surviving share is the median over images of the probability of the "
        "predicted class averaged over the masked passes. P_c is close to 1 for most "
        "images. It falls below 0.9 on "
        + join_words(
            [
                f"{percent(validation_low[d])} of clean validation images on "
                f"{DISPLAY_NAMES[d]}"
                if d == by_low_share[0]
                else f"{percent(validation_low[d])} on {DISPLAY_NAMES[d]}"
                for d in by_low_share
            ]
        )
        + f", against {percent(min(backdoor_low))} to {percent(max(backdoor_low))} "
        "of triggered images.",
        "",
        "Masking leaves a median of "
        f"{pooled['surviving_median']['validation']:.3f} of the clean probability "
        f"against {pooled['surviving_median']['backdoor']:.3f} of the triggered one. "
        "The absolute PSU of a clean image is therefore close to its starting "
        "confidence, so a clean image the model was unsure of drops little in absolute "
        "terms and lands in the low tail that the threshold reads as suspicious. On "
        "validation the Spearman correlation with P_c is "
        f"{pooled['spearman_with_confidence_val']['absolute']:+.2f} for absolute PSU "
        f"and {pooled['spearman_with_confidence_val']['fractional']:+.2f} for "
        "fractional PSU.",
        "",
    ]
    return lines


def budget_lines(means, pooled, q1):
    lines = [
        "### Flags at a fixed false positive rate",
        "",
        "Thresholds at the 0.01, 0.05 and 0.10 quantiles of clean validation, low means "
        "poisoned. Gained and lost are shares of triggered images flagged by only the "
        "fractional or only the absolute form. Kept is the share of the absolute "
        "form's validation flags the fractional form also raises. The P_c columns are "
        "medians over the flagged, dropped or added validation images.",
        "",
        "| dataset | q | TPR absolute | TPR fractional | gain | gained | lost | kept "
        "| P_c flagged absolute | P_c flagged fractional | P_c dropped | P_c added |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in DATASETS + ("all",):
        for key, budget in means[group]["budgets"].items():
            lines.append(
                f"| {group} | {key[1:]} | {budget['absolute']['tpr']:.3f} "
                f"| {budget['fractional']['tpr']:.3f} | {budget['tpr_gain']:+.3f} "
                f"| {budget['backdoor_gained']:.3f} | {budget['backdoor_lost']:.3f} "
                f"| {budget['val_flags_kept']:.3f} "
                f"| {budget['absolute']['median_confidence_of_flagged_val']:.3f} "
                f"| {budget['fractional']['median_confidence_of_flagged_val']:.3f} "
                f"| {budget['val_dropped_median_confidence']:.3f} "
                f"| {budget['val_added_median_confidence']:.3f} |"
            )

    pooled_q1 = pooled["budgets"][q1]
    gains = {
        d: {k: b["tpr_gain"] for k, b in means[d]["budgets"].items()} for d in DATASETS
    }
    largest = max(DATASETS, key=lambda d: gains[d][q1])
    smallest = min(DATASETS, key=lambda d: abs(gains[d][q1]))
    lines += [
        "",
        "At 1% FPR the division replaces "
        f"{percent(1 - pooled_q1['val_flags_kept'])} of the validation flags. The flags "
        "it drops have a median P_c of "
        f"{pooled_q1['val_dropped_median_confidence']:.2f}, the ones it adds "
        f"{pooled_q1['val_added_median_confidence']:.3f}, so low-confidence clean images "
        "leave the suspicious tail and confident clean images whose prediction survives "
        "masking take their place. TPR moves from "
        f"{pooled_q1['absolute']['tpr']:.3f} to {pooled_q1['fractional']['tpr']:.3f}, "
        f"{pooled_q1['backdoor_gained']:.3f} of triggered images gained against "
        f"{pooled_q1['backdoor_lost']:.3f} lost.",
        "",
        f"The gain is largest on {DISPLAY_NAMES[largest]} ("
        + ", ".join(
            f"{gains[largest][k]:+.3f} at {percent(float(k[1:]))}"
            for k in gains[largest]
        )
        + f" FPR) and about 0 on {DISPLAY_NAMES[smallest]} ("
        + ", ".join(f"{gains[smallest][k]:+.3f}" for k in gains[smallest])
        + ").",
        "",
    ]
    lines += low_share_against_gain_lines(means, q1)
    return lines


def low_share_against_gain_lines(means, q1):
    tiny, cifar100 = means["tiny"], means["cifar100"]
    tiny_low = tiny["confidence"]["validation"]["share_below_0.9"]
    cifar100_low = cifar100["confidence"]["validation"]["share_below_0.9"]
    tiny_gain = tiny["budgets"][q1]["tpr_gain"]
    cifar100_gain = cifar100["budgets"][q1]["tpr_gain"]
    if not (tiny_low > cifar100_low and tiny_gain < cifar100_gain):
        return []

    tiny_flag_confidence = tiny["budgets"][q1]["absolute"][
        "median_confidence_of_flagged_val"
    ]
    cifar100_flag_confidence = cifar100["budgets"][q1]["absolute"][
        "median_confidence_of_flagged_val"
    ]
    tiny_kept = tiny["budgets"][q1]["val_flags_kept"]
    cifar100_kept = cifar100["budgets"][q1]["val_flags_kept"]
    lines = [
        "The gain does not track the low-confidence share strictly. Tiny has more "
        f"clean validation images below 0.9 than CIFAR-100 ({percent(tiny_low)} against "
        f"{percent(cifar100_low)}) and a smaller gain at 1% FPR ({tiny_gain:+.3f} "
        f"against {cifar100_gain:+.3f}).",
    ]
    if tiny_flag_confidence > 0.99 and tiny_kept > cifar100_kept:
        lines[-1] += (
            " The data shows where the difference sits. On Tiny the absolute form's 1% "
            f"flags already have a median P_c of {tiny_flag_confidence:.3f} and the "
            f"division keeps {percent(tiny_kept)} of them, against "
            f"{cifar100_flag_confidence:.3f} and {percent(cifar100_kept)} on CIFAR-100, "
            "so at this budget most of Tiny's flags are already confident robust images "
            "and few low-confidence flags are left for the division to replace."
        )
    else:
        lines[-1] += " Why is open."
    lines.append("")
    return lines


def ranking_lines(means, pooled):
    lines = [
        "### Ranking",
        "",
        "| dataset | AUROC absolute | AUROC fractional | Spearman absolute against "
        "fractional | Spearman absolute with P_c | Spearman fractional with P_c |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in DATASETS + ("all",):
        group_means = means[group]
        lines.append(
            f"| {group} | {group_means['auroc']['absolute']:.3f} "
            f"| {group_means['auroc']['fractional']:.3f} "
            f"| {group_means['spearman_abs_frac_pooled']:.3f} "
            f"| {group_means['spearman_with_confidence_val']['absolute']:+.2f} "
            f"| {group_means['spearman_with_confidence_val']['fractional']:+.2f} |"
        )
    lines += [
        "",
        "AUROC barely moves, from "
        f"{pooled['auroc']['absolute']:.3f} to {pooled['auroc']['fractional']:.3f}, "
        "because the division hardly reorders images. The rank correlation of the 2 "
        "scores over validation and triggered images pooled is "
        f"{pooled['spearman_abs_frac_pooled']:.3f}. The images it does reorder sit in "
        "the low tail, which is where a threshold at a small FPR is set.",
        "",
    ]
    return lines


def per_class_lines(means, pooled, q1):
    lines = [
        "### Per-class spread of the clean flags",
        "",
        "Per true class of clean validation, the index of dispersion of the flag counts "
        "(1 when flags fall on classes in proportion to their size) and the share of all "
        "flags held by the worst tenth of classes.",
        "",
        "| dataset | images per class | q | dispersion absolute | dispersion fractional "
        "| worst tenth absolute | worst tenth fractional |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in DATASETS + ("all",):
        spread = means[group]["per_class"]
        for key in spread["absolute"]:
            absolute, fractional = spread["absolute"][key], spread["fractional"][key]
            lines.append(
                f"| {group} | {absolute['validation_images_per_class']:.0f} | {key[1:]} "
                f"| {absolute['dispersion']:.2f} | {fractional['dispersion']:.2f} "
                f"| {absolute['worst_tenth_share']:.2f} "
                f"| {fractional['worst_tenth_share']:.2f} |"
            )

    pooled_absolute = pooled["per_class"]["absolute"][q1]
    pooled_fractional = pooled["per_class"]["fractional"][q1]
    per_class = {
        d: means[d]["per_class"]["absolute"][q1]["validation_images_per_class"]
        for d in ("cifar100", "tiny")
    }
    lines += [
        "",
        "The division concentrates the clean false positives in fewer classes. At 1% FPR "
        "the worst tenth of classes hold "
        f"{pooled_absolute['worst_tenth_share']:.2f} of the flags under absolute PSU and "
        f"{pooled_fractional['worst_tenth_share']:.2f} under fractional PSU, and the "
        f"dispersion rises from {pooled_absolute['dispersion']:.2f} to "
        f"{pooled_fractional['dispersion']:.2f}. So the division gives no calibration "
        "across classes. "
        f"CIFAR-100 and Tiny have {per_class['cifar100']:.0f} and {per_class['tiny']:.0f} "
        "validation images per class, so their per-class numbers are noisy.",
        "",
    ]
    return lines


def training_set_lines(rows):
    lines = [
        "### The training-set setting",
        "",
        "Li et al. detect poisoned images inside the training set and set the threshold "
        "on it. The raw tensors of `experiments/training_set_detection` hold PSBD-TM at "
        "the ladder's own rate on these models, read here with thresholds at quantiles "
        "of the clean training images.",
        "",
        "| model | P_c below 0.9 (validation) | (clean train) | (poisoned) "
        "| TPR 1% abs | frac | TPR 5% abs | frac | TPR 10% abs | frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    exceptions = []
    for row in rows:
        confidence = row["confidence"]
        cells = [
            f"`{row['folder']}`",
            f"{confidence['validation']['share_below_0.9']:.3f}",
            f"{confidence['clean_train']['share_below_0.9']:.3f}",
            f"{confidence['poisoned']['share_below_0.9']:.3f}",
        ]
        for key, tpr in row["tpr"].items():
            cells += [f"{tpr['absolute']:.3f}", f"{tpr['fractional']:.3f}"]
            if abs(tpr["gain"]) > TRAINING_SET_GAIN_FLOOR:
                exceptions.append((row["folder"], key, tpr["gain"]))
    exceptions_by_folder = {}
    for folder, key, gain in exceptions:
        exceptions_by_folder.setdefault(folder, []).append(
            f"{percent(float(key[1:]))} ({gain:+.3f})"
        )
        lines.append("| " + " | ".join(cells) + " |")

    tiny_rows = [row for row in rows if row["folder"].startswith("vit_tiny_")]
    tiny_validation = statistics.mean(
        r["confidence"]["validation"]["share_below_0.9"] for r in tiny_rows
    )
    tiny_train = statistics.mean(
        r["confidence"]["clean_train"]["share_below_0.9"] for r in tiny_rows
    )
    if exceptions_by_folder:
        exception_text = ", except " + join_words(
            [
                f"`{folder}` at " + join_words(budgets)
                for folder, budgets in exceptions_by_folder.items()
            ]
        )
    else:
        exception_text = ""
    lines += [
        "",
        "Clean training images are more confident than validation images. "
        f"On the {len(tiny_rows)} Tiny models P_c falls below 0.9 on "
        f"{percent(tiny_validation)} of validation and {percent(tiny_train)} of clean "
        "training images. With few low-confidence clean images there is little for the "
        "division to move, and the fractional TPR is within "
        f"{TRAINING_SET_GAIN_FLOOR} of the absolute one everywhere{exception_text}. This "
        "is consistent with Li et al. not needing the division in their setting.",
        "",
    ]
    return lines


if __name__ == "__main__":
    main()

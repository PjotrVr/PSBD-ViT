"""CPU stage: turn the raw training-set scores into per-model readings and verdicts.

Reads results/_experiments/training_set_detection/raw/<folder>/ written by score.py,
writes models/<folder>.json beside it and summary.json with the pooled means and
the verdict of every prediction in PREDICTIONS.md. The test-time readings of the
same models are recomputed from the cli.sweep caches with the same code, so the 2
settings are compared on equal terms.

    cd <main checkout>
    PYTHONPATH=<worktree> .venv/bin/python <worktree>/experiments/training_set_detection/analyze.py
"""

import argparse
import json
import os
import sys
import time

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(EXPERIMENT_DIR)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from defenses.cache import read_split_manifest  # noqa: E402
from defenses.decision import threshold_at_quantile  # noqa: E402
from defenses.scores import psu_from_cache, psu_ratio_from_cache, to_rank  # noqa: E402
from detectors.spectral_signatures import (  # noqa: E402
    flag_top,
    removal_count,
    spectral_signature_scores,
)
from experiments.cache_readouts.shared import (  # noqa: E402
    choose_rate,
    evaluate_scores,
    load_baselines,
    load_passes,
    paired_summary,
    validation_shift_by_rate,
)
from experiments.training_set_detection import common  # noqa: E402
from experiments.training_set_detection.common import (  # noqa: E402
    CD_L_SUBSET,
    PAPER_MIRROR,
    PLACEMENTS,
    RATE_RULES,
    RECORDS_DIR,
    RESULTS_DIR,
    dev_set,
    has_part,
    load_part,
    passes_part,
    placements_for,
)
from experiments.training_set_detection.score import li_rate_rule  # noqa: E402
from scripts.paper._common import clearing_cells, load_coverage  # noqa: E402

NOMINAL_QUANTILES = (0.01, 0.05, 0.10, 0.25)
REALIZED_QUANTILES = (0.01, 0.05, 0.10)
PAPER_QUANTILE = 0.25
PSU_FORMS = ("fractional", "absolute")
# R2 allows the training-set attack success to fall this far below the recorded ASR.
ASR_SLACK = 0.05
# R3: the untriggered image of a poisoned index must sit at least this far from
# the training loss toward the validation loss, and the 2 must differ by at least
# MEMBERSHIP_MIN_GAP nats for the check to say anything.
MEMBERSHIP_MIN_RATIO = 0.5
MEMBERSHIP_MIN_GAP = 0.05
CLEAN_LABEL_MODES = ("clean_label", "clean_label_multi")
METHODS = ("final_min", "psbd_tm", "psbd_rd")
DETECTORS = ("strip", "cd_l")


def main():
    args = parse_args()
    started = time.perf_counter()
    records_dir = RECORDS_DIR
    if args.smoke:
        common.RAW_ROOT = os.path.join("scratch", common.SLUG, "smoke_raw")
        records_dir = os.path.join("scratch", common.SLUG, "smoke_records")
    panel = {cell["folder_name"] for cell in clearing_cells(load_coverage(RESULTS_DIR))}
    models_dir = os.path.join(records_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    rows = {}
    for folder in common.model_queue():
        if not model_complete(folder):
            print(f"[pending] {folder}")
            continue
        row = measure_model(folder, folder in panel)
        write_json(row, os.path.join(models_dir, f"{folder}.json"))
        rows[folder] = row
        print(f"[ok] {folder}")

    summary = {
        "experiment": "training-set detection, PSBD placements in Li et al.'s setting",
        "sets": {
            "paper_mirror": summarize_set(
                rows, [f for f in PAPER_MIRROR if "vit" in f]
            ),
            "development": summarize_set(rows, list(dev_set())),
        },
        "resnet": {
            folder: rows[folder]["paper_rule"]
            for folder in PAPER_MIRROR
            if folder.startswith("resnet18") and folder in rows
        },
        "pending": [f for f in common.model_queue() if f not in rows],
        "wall_seconds": time.perf_counter() - started,
    }
    summary["verdicts"] = judge(summary, rows)
    write_json(summary, os.path.join(records_dir, "summary.json"))
    print(f"wrote {records_dir}/summary.json in {summary['wall_seconds']:.0f} s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="read the smoke parts under scratch/"
    )
    args = parser.parse_args()
    return args


def model_complete(folder):
    needed = ["groups", "baseline"]
    needed += [f"ladder_{p}" for p in placements_for(folder)]
    complete = all(has_part(folder, part) for part in needed)
    return complete


def measure_model(folder, successful_2pt):
    groups = load_part(folder, "groups")
    baseline = load_part(folder, "baseline")
    population = groups["population"]

    checks = reconstruction_checks(groups, baseline)
    row = {
        "folder": folder,
        "dataset": groups["dataset"],
        "attack": groups["attack"],
        "label_mode": groups["label_mode"],
        "architecture": "resnet18" if folder.startswith("resnet18") else "vit",
        "successful_2pt": successful_2pt,
        "population": population,
        "scored": {name: len(order) for name, order in groups["orders"].items()},
        "reconstruction": checks,
        "validation_baseline_check": baseline["validation_check"],
    }
    row["headline"] = bool(
        successful_2pt
        and checks["verdict"] == "verified"
        and row["architecture"] == "vit"
    )

    scores, rates, checks_by_part = psbd_scores(folder, baseline, population)
    row["rates"] = rates
    row["validation_pass_checks"] = checks_by_part
    row["readings"] = {
        rule: {
            form: {
                method: readout(scores[rule][form][method], population)
                for method in scores[rule][form]
            }
            for form in PSU_FORMS
        }
        for rule in RATE_RULES
    }
    row["gap"] = {
        rule: {
            form: {
                method: train_validation_gap(scores[rule][form][method])
                for method in scores[rule][form]
            }
            for form in PSU_FORMS
        }
        for rule in RATE_RULES
    }

    # PSBD read on the CD-L subset, so every method of P6 is scored on the same
    # images.
    row["common_subset"] = {
        method: readout(
            prefix(scores["ours"]["fractional"][method], CD_L_SUBSET), population
        )
        for method in scores["ours"]["fractional"]
    }
    row["detectors"] = {}
    row["detector_validation_checks"] = {}
    for name in DETECTORS:
        if not has_part(folder, name):
            continue
        part = load_part(folder, name)
        row["detectors"][name] = readout(subset_groups(part), population)
        row["common_subset"][name] = readout(
            prefix(subset_groups(part), CD_L_SUBSET), population
        )
        row["detector_validation_checks"][name] = part.get(
            "validation_check", part.get("validation_source")
        )
    row["spectral_signatures"] = spectral_signatures_reading(groups, baseline)
    row["paper_rule"] = paper_rule_table(row)
    row["test_time"] = test_time_readings(folder)
    return row


def reconstruction_checks(groups, baseline):
    recorded = groups["recorded_counts"]
    population = groups["population"]
    r1 = {
        "poisoned": population["poisoned"],
        "recorded_poisoned": recorded["n_poisoned"],
        "cover": population["cover"],
        "recorded_cover": recorded["n_cover"],
    }
    r1["passed"] = all(
        recorded_count is None or recorded_count == count
        for count, recorded_count in (
            (population["poisoned"], recorded["n_poisoned"]),
            (population["cover"], recorded["n_cover"]),
        )
    )

    predicted = baseline["poisoned"]["labels"]  # (n_poisoned,)
    success = float((predicted == groups["success_labels"]).float().mean())
    recorded_asr = groups["recorded_asr"]
    r2 = {
        "training_attack_success": success,
        "recorded_asr": recorded_asr,
        "passed": recorded_asr is None or success >= recorded_asr - ASR_SLACK,
    }

    losses = {
        "poisoned_clean": cross_entropy(
            baseline["poisoned_clean"]["probs"], groups["true_labels"]["poisoned_clean"]
        ),
        "clean_train": cross_entropy(
            baseline["clean_train"]["probs"], groups["true_labels"]["clean_train"]
        ),
        "validation": cross_entropy(
            baseline["validation"]["probs"], baseline["validation"]["loader_labels"]
        ),
    }
    spread = losses["validation"] - losses["clean_train"]
    ratio = (
        (losses["poisoned_clean"] - losses["clean_train"]) / spread if spread else None
    )
    r3 = {"mean_cross_entropy": losses, "ratio": ratio}
    clean_label = groups["label_mode"] in CLEAN_LABEL_MODES
    if clean_label:
        r3["status"] = (
            "not applicable, the poisoned images were trained with their true label"
        )
    elif spread < MEMBERSHIP_MIN_GAP:
        r3["status"] = "inconclusive, training and validation loss too close"
    else:
        r3["status"] = "passed" if ratio >= MEMBERSHIP_MIN_RATIO else "failed"

    if not (r1["passed"] and r2["passed"]):
        verdict = "failed"
    elif clean_label:
        verdict = "count and recorded seed only"
    elif r3["status"] == "passed":
        verdict = "verified"
    else:
        verdict = "ambiguous" if r3["status"].startswith("inconclusive") else "failed"

    checks = {
        "seed_recorded": groups["seed_recorded"],
        "seed_used": groups["seed_used"],
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "verdict": verdict,
    }
    return checks


def cross_entropy(probs, labels):
    chosen = probs.gather(1, labels.view(-1, 1).long()).squeeze(1)  # (n,)
    loss = float(-chosen.clamp_min(1e-12).log().mean())
    return loss


def psbd_scores(folder, baseline, population):
    """Per group PSU for every method, rate rule and PSU form, plus the rates."""
    rates = {}
    checks = {}
    per_placement = {rule: {form: {} for form in PSU_FORMS} for rule in RATE_RULES}
    for placement in placements_for(folder):
        ladder = load_part(folder, f"ladder_{placement}")
        rule_rates = {
            "ours": ladder["our_rate"],
            "li": li_rate_rule(ladder, population),
        }
        rates[placement] = {
            **rule_rates,
            "li_candidates": ladder["candidates"],
            "li_gap": li_gaps(ladder, population),
        }
        for rule, rate in rule_rates.items():
            if rate is None:
                continue
            passes = load_part(folder, passes_part(placement, rate))
            checks[f"{placement}@{rate:g}"] = passes["validation_check"]
            for form in PSU_FORMS:
                per_placement[rule][form][placement] = group_psu(passes, baseline, form)

    scores = {rule: {form: {} for form in PSU_FORMS} for rule in RATE_RULES}
    for rule in RATE_RULES:
        for form in PSU_FORMS:
            available = per_placement[rule][form]
            for placement, psu in available.items():
                scores[rule][form][placement] = psu
            if "psbd_tm" in available and "middle_band" in available:
                scores[rule][form]["final_min"] = min_rank_fusion(
                    available["psbd_tm"], available["middle_band"]
                )
    return scores, rates, checks


def li_gaps(ladder, population):
    gaps = {}
    total = sum(population.values())
    for rate in ladder["candidates"]:
        measured = ladder["per_rate"][rate]
        training_shift = 0.0
        for group, count in population.items():
            if count == 0:
                continue
            argmax = measured[group]["per_pass_argmax"]  # (passes, n_subset)
            labels = measured[group]["baseline_labels"]  # (n_subset,)
            shift = float((argmax.long() != labels.view(1, -1).long()).float().mean())
            training_shift += shift * count / total
        gaps[f"{rate:g}"] = {
            "validation_shift": ladder["validation_shift"][rate],
            "training_shift": training_shift,
        }
    return gaps


def group_psu(passes, baseline, form):
    psu_function = psu_ratio_from_cache if form == "fractional" else psu_from_cache
    psu = {}
    for group in ("poisoned", "cover", "clean_train", "validation"):
        if passes.get(group) is None:
            psu[group] = torch.empty(0)
            continue
        psu[group] = psu_function(
            baseline[group]["probs"],
            baseline[group]["labels"],
            passes[group]["per_pass_probs"],
        )  # (n_group,)
    return psu


def min_rank_fusion(tm, band):
    # The final method: each probe's percentile within its own clean-validation
    # distribution, then the smaller of the 2 (experiments.cache_readouts.fusion_rules).
    fused = {}
    for group in tm:
        if tm[group].numel() == 0:
            fused[group] = torch.empty(0)
            continue
        assert tm[group].shape == band[group].shape, group
        fused[group] = torch.minimum(
            to_rank(tm[group], tm["validation"]),
            to_rank(band[group], band["validation"]),
        )  # (n_group,)
    return fused


def subset_groups(part):
    groups = {
        group: part.get(group, torch.empty(0)).float()
        for group in ("poisoned", "cover", "clean_train", "validation")
    }
    return groups


def prefix(scores, sizes):
    cut = {
        group: values[: sizes[group]] if group in sizes else values
        for group, values in scores.items()
    }
    return cut


def readout(scores, population):
    """Every reading of 1 method on 1 model. Low scores mean poisoned throughout."""
    validation = scores["validation"]  # (n_validation,)
    clean = scores["clean_train"]  # (n_clean,)
    poisoned = scores["poisoned"]  # (n_poisoned,)
    cover = scores["cover"]  # (n_cover,)

    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(poisoned))])
    values = np.concatenate([-clean.numpy(), -poisoned.numpy()])
    reading = {
        "n": {
            "poisoned": len(poisoned),
            "clean_train": len(clean),
            "cover": len(cover),
        },
        "auroc": float(roc_auc_score(labels, values)),
        "nominal": {},
        "realized": {},
    }

    # Li et al.'s code counts covers as negatives. Their share of the negatives is
    # their share of the training population, not of this sample.
    negatives = population["clean_train"] + population["cover"]
    cover_weight = population["cover"] / negatives if negatives else 0.0
    for quantile in NOMINAL_QUANTILES:
        threshold = threshold_at_quantile(validation, quantile)
        fpr_clean = float((clean < threshold).float().mean())
        fpr_cover = float((cover < threshold).float().mean()) if len(cover) else None
        reading["nominal"][f"q{quantile:.2f}"] = {
            "threshold": threshold,
            "tpr": float((poisoned < threshold).float().mean()),
            "fpr_clean_train": fpr_clean,
            "flag_rate_cover": fpr_cover,
            "fpr_paper_convention": fpr_clean * (1 - cover_weight)
            + (fpr_cover or 0.0) * cover_weight,
        }
    for quantile in REALIZED_QUANTILES:
        threshold = threshold_at_quantile(clean, quantile)
        reading["realized"][f"q{quantile:.2f}"] = {
            "threshold": threshold,
            "tpr": float((poisoned < threshold).float().mean()),
            "fpr_clean_train": float((clean < threshold).float().mean()),
        }
    return reading


def train_validation_gap(scores):
    clean = scores["clean_train"].numpy()
    validation = scores["validation"].numpy()
    gap = {
        "median_clean_train": float(np.median(clean)),
        "median_validation": float(np.median(validation)),
        "median_difference": float(np.median(clean) - np.median(validation)),
        "mean_difference": float(clean.mean() - validation.mean()),
        "ks_statistic": float(ks_2samp(clean, validation).statistic),
        "ks_pvalue": float(ks_2samp(clean, validation).pvalue),
    }
    return gap


def spectral_signatures_reading(groups, baseline):
    """The reference cleanser's TPR and FPR over the whole training set.

    Only the target label holds poisoned rows, and all of them plus every clean
    row of that label were scored, so its removals are exact. Every other label
    loses min(int(1.5 * epsilon * N), n // 2) clean rows whichever they are,
    which fixes the false positives outside the target label without features.
    """
    poisoned_features = baseline["poisoned"]["features"]  # (n_poisoned, dim)
    target_features = baseline["target_clean"]["features"]  # (n_target_clean, dim)
    features = torch.cat([poisoned_features, target_features])  # (n_label, dim)
    is_poisoned = torch.cat(
        [
            torch.ones(len(poisoned_features), dtype=torch.bool),
            torch.zeros(len(target_features), dtype=torch.bool),
        ]
    )  # (n_label,)

    counts = groups["training_label_counts"]
    target = groups["target_label"]
    dataset_size = groups["n_train"]
    epsilon = groups["realized_poison_rate"]
    assert len(features) == counts[target], (len(features), counts[target])

    scores = spectral_signature_scores(features)  # (n_label,)
    flagged = flag_top(scores, removal_count(len(features), epsilon, dataset_size))
    other_labels_removed = sum(
        removal_count(count, epsilon, dataset_size)
        for label, count in enumerate(counts)
        if label != target
    )
    n_poisoned = int(is_poisoned.sum())
    negatives = dataset_size - n_poisoned
    reading = {
        "epsilon": epsilon,
        "tpr": float((flagged & is_poisoned).sum()) / n_poisoned,
        "fpr_paper_convention": float(
            (flagged & ~is_poisoned).sum() + other_labels_removed
        )
        / negatives,
        "auroc_within_target_label": float(
            roc_auc_score(is_poisoned.numpy(), scores.numpy())
        ),
    }
    return reading


def paper_rule_table(row):
    # Li et al.'s own configuration: absolute PSU, their rate rule, T at the 25th
    # percentile of clean validation. Every other combination sits beside it.
    table = {}
    for rule in RATE_RULES:
        for form in PSU_FORMS:
            for method, reading in row["readings"][rule][form].items():
                cell = reading["nominal"][f"q{PAPER_QUANTILE:.2f}"]
                table[f"{method}|{rule}|{form}"] = {
                    "tpr": cell["tpr"],
                    "fpr_clean_train": cell["fpr_clean_train"],
                    "fpr_paper_convention": cell["fpr_paper_convention"],
                }
    return table


def test_time_readings(folder):
    """The same methods on the paired test splits, from the cli.sweep caches."""
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)
    quantiles = REALIZED_QUANTILES + (PAPER_QUANTILE,)

    psu = {}
    for placement in placements_for(folder):
        cache = PLACEMENTS[placement]["cache"]
        rate = choose_rate(
            validation_shift_by_rate(psbd_dir, cache, baselines), "adaptive"
        )
        if rate is None:
            continue
        passes = load_passes(psbd_dir, cache, rate, baselines)
        psu[placement] = {
            split: psu_ratio_from_cache(
                passes[split]["baseline_probs"],
                passes[split]["baseline_labels"],
                passes[split]["per_pass_probs"],
            )
            for split in passes
        }
    if "psbd_tm" in psu and "middle_band" in psu:
        psu["final_min"] = {
            split: torch.minimum(
                to_rank(psu["psbd_tm"][split], psu["psbd_tm"]["validation"]),
                to_rank(psu["middle_band"][split], psu["middle_band"]["validation"]),
            )
            for split in psu["psbd_tm"]
        }

    readings = {
        method: flatten_evaluation(evaluate_scores(scores, manifest, quantiles))
        for method, scores in psu.items()
    }
    for name in DETECTORS:
        record_path = os.path.join(
            RESULTS_DIR, folder, "detectors", f"{name}_metrics.json"
        )
        if not os.path.exists(record_path):
            continue
        with open(record_path) as handle:
            detection = json.load(handle)["detection"]
        readings[name] = {
            "auroc": detection["q0.01"]["auroc"],
            **{
                f"q{q:.2f}:tpr": detection[f"q{q:.2f}"]["tpr"]
                for q in quantiles
                if f"q{q:.2f}" in detection
            },
        }
    return readings


def flatten_evaluation(evaluation):
    flat = {"auroc": evaluation["auroc"]}
    for key, cell in evaluation["at_fpr"].items():
        flat[f"{key}:tpr"] = cell["tpr"]
        flat[f"{key}:fpr"] = cell["realized_fpr"]
    return flat


def summarize_set(rows, folders):
    present = [rows[f] for f in folders if f in rows]
    headline = [row for row in present if row["headline"]]
    summary = {
        "folders": folders,
        "scored": [row["folder"] for row in present],
        "headline": [row["folder"] for row in headline],
        "excluded": {
            row["folder"]: {
                "successful_2pt": row["successful_2pt"],
                "reconstruction": row["reconstruction"]["verdict"],
            }
            for row in present
            if not row["headline"]
        },
        "means": {},
    }
    if not headline:
        return summary

    for rule in RATE_RULES:
        for form in PSU_FORMS:
            key = f"{rule}|{form}"
            summary["means"][key] = {
                method: mean_reading(
                    [row["readings"][rule][form].get(method) for row in headline]
                )
                for method in METHODS
            }
    summary["means"]["common_subset"] = {
        method: mean_reading([row["common_subset"].get(method) for row in headline])
        for method in METHODS + DETECTORS
    }
    summary["means"]["test_time"] = {
        method: mean_flat([row["test_time"].get(method) for row in headline])
        for method in METHODS + DETECTORS
    }
    summary["paired"] = paired_gains(headline)
    return summary


def mean_reading(readings):
    if any(r is None for r in readings):
        return None
    mean = {
        "n": len(readings),
        "auroc": float(np.mean([r["auroc"] for r in readings])),
        "nominal": {
            key: {
                field: float(
                    np.mean([r["nominal"][key][field] or 0.0 for r in readings])
                )
                for field in ("tpr", "fpr_clean_train", "fpr_paper_convention")
            }
            for key in readings[0]["nominal"]
        },
        "realized": {
            key: {"tpr": float(np.mean([r["realized"][key]["tpr"] for r in readings]))}
            for key in readings[0]["realized"]
        },
    }
    return mean


def mean_flat(readings):
    if any(r is None for r in readings) or not readings:
        return None
    keys = set.intersection(*(set(r) for r in readings))
    mean = {key: float(np.mean([r[key] for r in readings])) for key in sorted(keys)}
    mean["n"] = len(readings)
    return mean


def paired_gains(headline):
    def auroc(row, rule, method):
        return row["readings"][rule]["fractional"][method]["auroc"]

    gains = {
        "train_tm_minus_rd_ours": paired_summary(
            [auroc(r, "ours", "psbd_tm") for r in headline],
            [auroc(r, "ours", "psbd_rd") for r in headline],
        ),
        "train_tm_minus_rd_li": paired_summary(
            [auroc(r, "li", "psbd_tm") for r in headline],
            [auroc(r, "li", "psbd_rd") for r in headline],
        ),
        "train_final_minus_tm_ours": paired_summary(
            [auroc(r, "ours", "final_min") for r in headline],
            [auroc(r, "ours", "psbd_tm") for r in headline],
        ),
        "test_tm_minus_rd": paired_summary(
            [r["test_time"]["psbd_tm"]["auroc"] for r in headline],
            [r["test_time"]["psbd_rd"]["auroc"] for r in headline],
        ),
    }
    for quantile in ("q0.01", "q0.05", "q0.10"):
        gains[f"train_tm_minus_rd_ours_realized_{quantile}"] = paired_summary(
            [
                r["readings"]["ours"]["fractional"]["psbd_tm"]["realized"][quantile][
                    "tpr"
                ]
                for r in headline
            ],
            [
                r["readings"]["ours"]["fractional"]["psbd_rd"]["realized"][quantile][
                    "tpr"
                ]
                for r in headline
            ],
        )
    return gains


def judge(summary, rows):
    verdicts = {}
    for set_name, block in summary["sets"].items():
        if not block["headline"]:
            verdicts[set_name] = "no headline model scored yet"
            continue
        headline = [rows[f] for f in block["headline"]]
        verdicts[set_name] = judge_set(block, headline)
    verdicts["P5"] = judge_resnet(rows)
    verdicts["P7"] = judge_paper_table(rows)
    return verdicts


def judge_set(block, headline):
    ours = block["means"]["ours|fractional"]
    li = block["means"]["li|fractional"]
    tm_auroc = ours["psbd_tm"]["auroc"]
    rd_auroc = ours["psbd_rd"]["auroc"]
    final_auroc = ours["final_min"]["auroc"]
    leads = [
        ours["psbd_tm"]["realized"][q]["tpr"] > ours["psbd_rd"]["realized"][q]["tpr"]
        for q in ("q0.01", "q0.05", "q0.10")
    ]
    p1 = tm_auroc > rd_auroc and sum(leads) >= 2 and final_auroc >= tm_auroc - 0.005

    train_gain = block["paired"]["train_tm_minus_rd_ours"]["mean_difference"]
    test_gain = block["paired"]["test_tm_minus_rd"]["mean_difference"]
    p2 = train_gain >= 0 and test_gain is not None and train_gain >= 0.5 * test_gain

    p3 = li["psbd_tm"]["auroc"] > li["psbd_rd"]["auroc"]

    majority = len(headline) / 2
    below = sum(
        r["gap"]["ours"]["fractional"]["psbd_tm"]["median_difference"] < 0
        for r in headline
    )
    inflated = sum(
        r["readings"]["ours"]["fractional"]["psbd_tm"]["nominal"]["q0.25"][
            "fpr_clean_train"
        ]
        > 0.25
        for r in headline
    )
    absolute_lower = sum(
        r["readings"]["ours"]["absolute"]["psbd_tm"]["nominal"]["q0.25"][
            "fpr_clean_train"
        ]
        < r["readings"]["ours"]["fractional"]["psbd_tm"]["nominal"]["q0.25"][
            "fpr_clean_train"
        ]
        for r in headline
    )
    p4 = {
        "fractional_median_below": f"{below} of {len(headline)}",
        "fractional_fpr_above_0.25": f"{inflated} of {len(headline)}",
        "absolute_fpr_lower_than_fractional": f"{absolute_lower} of {len(headline)}",
        "holds_fractional": below > majority and inflated > majority,
        "holds_absolute": absolute_lower > majority,
    }

    train_order = order_of(
        block["means"]["common_subset"], ("psbd_tm", "strip", "cd_l")
    )
    test_order = order_of(block["means"]["test_time"], ("psbd_tm", "strip", "cd_l"))
    p6 = {
        "train_order": train_order,
        "test_order": test_order,
        "holds": train_order == test_order,
    }

    verdicts = {
        "P1": {
            "holds": p1,
            "auroc": {
                "final_min": final_auroc,
                "psbd_tm": tm_auroc,
                "psbd_rd": rd_auroc,
            },
            "tm_leads_rd_at_realized_1_5_10": leads,
        },
        "P2": {"holds": p2, "train_gain": train_gain, "test_gain": test_gain},
        "P3": {
            "holds": p3,
            "li_tm": li["psbd_tm"]["auroc"],
            "li_rd": li["psbd_rd"]["auroc"],
        },
        "P4": p4,
        "P6": p6,
    }
    return verdicts


def order_of(means, methods):
    available = {m: means[m]["auroc"] for m in methods if means.get(m)}
    order = sorted(available, key=lambda m: -available[m])
    return order


def judge_resnet(rows):
    cells = {}
    for folder in PAPER_MIRROR:
        if not folder.startswith("resnet18") or folder not in rows:
            continue
        cell = rows[folder]["paper_rule"]["psbd_rd|li|absolute"]
        cells[folder] = {
            **cell,
            "holds": cell["tpr"] >= 0.8 and cell["fpr_clean_train"] <= 0.30,
        }
    verdict = {
        "models": cells,
        "holds": bool(cells) and all(c["holds"] for c in cells.values()),
    }
    return verdict


def judge_paper_table(rows):
    vit = [f for f in PAPER_MIRROR if f.startswith("vit") and f in rows]
    reached = {
        f: rows[f]["paper_rule"]["psbd_tm|li|absolute"]["tpr"] >= 0.8 for f in vit
    }
    verdict = {
        "reached": reached,
        "count": f"{sum(reached.values())} of {len(vit)}",
        "holds": len(vit) == 9 and sum(reached.values()) >= 7,
    }
    return verdict


def write_json(payload, path):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

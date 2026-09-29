"""Model sets, cache loading and scoring shared by the 3 cache readouts.

Every score is read the way cli.analyze reads it: defenses.cache for the tensors,
defenses.scores for PSU and the shift ratio, defenses.decision for the adaptive
rate, the pairing and the quantile threshold. Nothing here restates the canon.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


from data.splits import SPLITS  # noqa: E402
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defenses.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    PRIMARY_DATASETS,
    complete_rates,
    detection_report,
    pair_clean_to_backdoor,
    select_rate_adaptively,
    select_rate_at_matched_shift,
    threshold_diagnostics,
)
from defenses.scores import shift_ratio  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.probe_union.measure import select_models  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    SECOND_SUCCESS,
    bootstrap_ci,
    clearing_cells,
    load_coverage,
    load_psbd_metrics,
)

SLUG = "cache_readouts"
RESULTS_DIR = "results"
DEV_SET_PATH = os.path.join(REPO_ROOT, "experiments", SLUG, "dev_set.json")
FIGURES_DIR = os.path.join(REPO_ROOT, "experiments", SLUG, "figures")
MODEL_SETS = ("dev", "holdout", "panel")

# The order every table lists attacks in, patch triggers first because E1 is
# about them.
ATTACK_ORDER = (
    "badnet_a2o",
    "tact",
    "blend",
    "lf",
    "bpp",
    "wanet",
    "sig",
    "lc",
    "adaptive_blend",
)
PATCH_ATTACKS = ("badnet_a2o", "tact")
GLOBAL_ATTACKS = ("blend", "lf", "bpp")


def load_model_set(name):
    coverage = load_coverage(RESULTS_DIR)
    cells = {cell["folder_name"]: cell for cell in coverage["cells"]}
    five_point = {
        cell["folder_name"] for cell in clearing_cells(coverage, SECOND_SUCCESS)
    }

    if name == "dev":
        with open(DEV_SET_PATH) as handle:
            declared = json.load(handle)["models"]
        models = []
        for entry in declared:
            cell = dict(cells[entry["folder"]])
            # The label in dev_set.json is a claim about the ledger, so a
            # ledger rebuild that changes it must stop the run.
            assert cell["successful_2pt"] == entry["declared_successful_2pt"], entry
            cell["report"] = load_psbd_metrics(RESULTS_DIR, entry["folder"])
            cell["role"] = entry["role"]
            models.append(cell)
    else:
        models = select_models(RESULTS_DIR)
        if name == "holdout":
            models = [m for m in models if m["dataset"] in PRIMARY_DATASETS]

    for model in models:
        model["successful_5pt"] = model["folder_name"] in five_point
    return models


def model_label(model):
    label = {
        "folder": model["folder_name"],
        "dataset": model["dataset"],
        "attack": model["attack"],
        "poison_rate": model["poison_rate"],
        "successful_2pt": model["successful_2pt"],
        "successful_5pt": model["successful_5pt"],
    }
    return label


def psbd_dir_of(model):
    path = os.path.join(RESULTS_DIR, model["folder_name"], "psbd")
    return path


def load_baselines(psbd_dir):
    baselines = {
        split: load_baseline(baseline_path(psbd_dir, split)) for split in SPLITS
    }
    return baselines


def validation_shift_by_rate(psbd_dir, placement, baselines):
    _, validation_labels, _ = baselines["validation"]
    shift_by_rate = {}
    for rate in complete_rates(psbd_dir, placement):
        _, per_pass_argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )  # (passes, n_validation)
        shift_by_rate[rate] = shift_ratio(validation_labels, per_pass_argmax)
    return shift_by_rate


def choose_rate(shift_by_rate, rule):
    # "adaptive" is the paper's rule. "nearest" is the fallback for a placement
    # whose ladder never reaches the target: the rate closest to it, still
    # chosen on clean validation only, and always reported under its own label.
    if rule == "adaptive":
        rate = select_rate_adaptively(shift_by_rate, ADAPTIVE_SHIFT_TARGET)
    else:
        rate = select_rate_at_matched_shift(shift_by_rate, ADAPTIVE_SHIFT_TARGET)
    return rate


def load_passes(psbd_dir, placement, rate, baselines):
    passes = {}
    for split in SPLITS:
        baseline_probs, baseline_labels, _ = baselines[split]
        per_pass_probs, per_pass_argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )  # (passes, n_split) each
        assert per_pass_probs.shape[1] == baseline_labels.shape[0], (placement, split)
        passes[split] = {
            "baseline_probs": baseline_probs,  # (n_split, num_classes)
            "baseline_labels": baseline_labels,  # (n_split,)
            "per_pass_probs": per_pass_probs,  # (passes, n_split)
            "per_pass_argmax": per_pass_argmax,  # (passes, n_split)
        }
    return passes


def evaluate_scores(scores_by_split, manifest, quantiles):
    # Low means poisoned for every score handed in here, which is PSBD's sign.
    # The clean split is paired down to the triggered split's images exactly as
    # cli.analyze pairs it, so FPR is realized on the same image population.
    validation = scores_by_split["validation"]  # (n_validation,)
    clean_paired = pair_clean_to_backdoor(
        scores_by_split["clean"], manifest
    )  # (n_backdoor,)
    backdoor = scores_by_split["backdoor"]  # (n_backdoor,)

    by_quantile = {}
    auroc = None
    for quantile in quantiles:
        report = detection_report(validation, clean_paired, backdoor, quantile)
        ties = threshold_diagnostics(validation, clean_paired, backdoor, quantile)
        auroc = report["auroc"]
        by_quantile[f"q{quantile:.2f}"] = {
            "threshold": report["threshold"],
            "tpr": report["tpr"],
            "realized_fpr": report["fpr"],
            "tie_share_at_threshold": ties["tie_share_at_threshold"],
        }
    evaluation = {"auroc": auroc, "at_fpr": by_quantile}
    return evaluation


def paired_summary(values, reference):
    differences = [value - base for value, base in zip(values, reference)]
    low, high = bootstrap_ci(differences, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    summary = {
        "n": len(differences),
        "mean": sum(values) / len(values) if values else None,
        "reference_mean": sum(reference) / len(reference) if reference else None,
        "mean_difference": sum(differences) / len(differences) if differences else None,
        "ci95": [low, high],
        "n_higher": sum(d > 0 for d in differences),
        "n_lower": sum(d < 0 for d in differences),
    }
    return summary


def ordered_attacks(attacks):
    ordered = [a for a in ATTACK_ORDER if a in attacks]
    ordered += sorted(set(attacks) - set(ordered))
    return ordered


def output_path(experiment, model_set):
    path = experiment_result_path(SLUG, f"{experiment}_{model_set}.json", RESULTS_DIR)
    return path


def write_json(payload, path):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

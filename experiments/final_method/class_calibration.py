"""Calibration by predicted class, tested on the whole ViT panel.

The defender knows each input's predicted class, and a triggered input is
predicted as the target. So each score can be calibrated against the clean
validation images the model predicts as the same class, instead of against all
classes. The TaCT anomaly (anomalies.py) suggests why that could matter: the clean
source-class images of a source-specific attack sit far above the all-class
validation distribution. 2 forms need no knowledge of the attack.

    percentile   F_c(s) = (n_c F^_c(s) + m F^(s)) / (n_c + m)
    z-score      z_c(s) = (s - mu'_c) / sigma'_c,
                 mu'_c = (n_c mu_c + m mu) / (n_c + m),
                 sigma'^2_c = (n_c sigma^2_c + m sigma^2) / (n_c + m)

| symbol | meaning |
|---|---|
| s | a probe's fractional PSU, low means poisoned |
| c | the unperturbed predicted class of the input |
| n_c | clean validation images predicted as c |
| F^_c, mu_c, sigma_c | their empirical CDF, mean and standard deviation |
| F^, mu, sigma | the same over the whole validation split |
| m | the shrinkage strength in images, so a class with n_c = 0 falls back to the global calibration and a class with n_c much larger than m keeps its own |

Each form is applied to PSBD-TM alone and to the final method, the minimum and
the average of the 2 calibrated scores of PSBD-TM and the middle band. Every
score is thresholded at a quantile of its own clean-validation values. m is fixed
on the development set first (`--stage dev`), written to
`preregistration_classcal.json`, and only then are the other panel models read
once (`--stage rest`). `--stage panel` rereads all 54 for the combined table.

    .venv/bin/python -m experiments.final_method.class_calibration --stage dev
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from defenses.cache import read_split_manifest
from defenses.decision import pair_clean_to_backdoor
from experiments._paths import experiment_result_path
from experiments.cache_readouts.fusion_rules import fractional_psu
from experiments.cache_readouts.shared import (
    REPO_ROOT,
    choose_rate,
    load_baselines,
    load_model_set,
    load_passes,
    ordered_attacks,
    paired_summary,
    validation_shift_by_rate,
    write_json,
)
from scripts.coverage_ledger import source_classes_of

SLUG = "final_method"
RESULTS_DIR = "results"
ANCHOR = "before_attention_norm_token_mask"
PARTNER = "pre_residual_blocks_5_8"
QUANTILES = (0.01, 0.05, 0.10)
HEADLINE = ("q0.01", "q0.05", "q0.10")
SHRINKAGE_GRID = (5, 20, 50, 200)
PREREGISTRATION = os.path.join(
    REPO_ROOT, "experiments", SLUG, "preregistration_classcal.json"
)
METHODS = ("psbd_tm", "final_min", "final_average")
FORMS = ("global", "class_percentile", "class_z")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "rest", "panel"), required=True)
    stage = parser.parse_args().stage
    started = time.perf_counter()

    dev_models = [m for m in load_model_set("dev") if m["successful_2pt"]]
    dev_folders = {m["folder_name"] for m in dev_models}
    if stage == "dev":
        models = dev_models
        grid = SHRINKAGE_GRID
    else:
        with open(PREREGISTRATION) as handle:
            frozen = json.load(handle)
        grid = (frozen["shrinkage"]["class_percentile"], frozen["shrinkage"]["class_z"])
        panel = load_model_set("panel")
        models = (
            panel
            if stage == "panel"
            else [m for m in panel if m["folder_name"] not in dev_folders]
        )

    rows = [measure(model, grid) for model in models]
    payload = {
        "experiment": "calibration by predicted class",
        "stage": stage,
        "shrinkage_grid": list(grid),
        "summary": {f"m={m}": summarize(rows, m) for m in grid},
        "tact_placement": [
            tact_placement(model) for model in models if model["attack"] == "tact"
        ],
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    if stage == "dev":
        payload["selection"] = select_shrinkage(payload["summary"])
    path = experiment_result_path(SLUG, f"class_calibration_{stage}.json", RESULTS_DIR)
    write_json(payload, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def adaptive_psu(psbd_dir, placement, baselines):
    rate = choose_rate(
        validation_shift_by_rate(psbd_dir, placement, baselines), "adaptive"
    )
    psu = fractional_psu(load_passes(psbd_dir, placement, rate, baselines))
    return psu


def measure(model, grid):
    folder = model["folder_name"]
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    manifest = read_split_manifest(psbd_dir)
    predicted = {
        split: baselines[split][1].long() for split in baselines
    }  # (n_split,) each
    anchor = adaptive_psu(psbd_dir, ANCHOR, baselines)
    partner = adaptive_psu(psbd_dir, PARTNER, baselines)

    # The validation images the model predicts as the attacker's target are the
    # ones a triggered input is calibrated against under this rule.
    target = model.get("target_label")
    as_target = predicted["validation"] == target  # (n_validation,)
    row = {
        "folder": folder,
        "attack": model["attack"],
        "dataset": model["dataset"],
        "target_class_validation": {
            "n": int(as_target.sum()),
            "median_psu": float(anchor["validation"][as_target].median())
            if as_target.any()
            else None,
            "std_psu": float(anchor["validation"][as_target].std(unbiased=False))
            if as_target.any()
            else None,
            "global_median_psu": float(anchor["validation"].median()),
            "global_std_psu": float(anchor["validation"].std(unbiased=False)),
            "triggered_median_psu": float(anchor["backdoor"].median()),
        },
        "by_m": {},
    }
    for m in grid:
        readings = {}
        for form in FORMS:
            anchor_cal = calibrate(anchor, predicted, form, m)
            partner_cal = calibrate(partner, predicted, form, m)
            for method in METHODS:
                scores = combine(method, form, anchor_cal, partner_cal, anchor, partner)
                readings[f"{form}/{method}"] = evaluate(scores, manifest)
        row["by_m"][f"m={m}"] = readings
    return row


def calibrate(psu, predicted, form, m):
    validation = psu["validation"]  # (n_validation,)
    validation_classes = predicted["validation"]  # (n_validation,)
    if form == "global":
        sorted_all = validation.sort().values
        calibrated = {
            split: torch.searchsorted(sorted_all, psu[split].contiguous()).float()
            / len(sorted_all)
            for split in psu
        }
        return calibrated

    global_sorted = validation.sort().values
    global_mean, global_std = validation.mean(), validation.std(unbiased=False)
    calibrated = {}
    for split in psu:
        scores = psu[split]  # (n,)
        classes = predicted[split]  # (n,)
        out = torch.empty_like(scores)
        global_rank = torch.searchsorted(
            global_sorted, scores.contiguous()
        ).float() / len(global_sorted)
        for label in classes.unique():
            members = validation[validation_classes == label]  # (n_c,)
            n_c = members.numel()
            where = classes == label
            if form == "class_percentile":
                class_rank = (
                    torch.searchsorted(
                        members.sort().values, scores[where].contiguous()
                    ).float()
                    / n_c
                    if n_c
                    else torch.zeros(int(where.sum()))
                )
                out[where] = (n_c * class_rank + m * global_rank[where]) / (n_c + m)
            else:
                class_mean = members.mean() if n_c else global_mean
                class_var = members.var(unbiased=False) if n_c else global_std**2
                mean = (n_c * class_mean + m * global_mean) / (n_c + m)
                std = (
                    ((n_c * class_var + m * global_std**2) / (n_c + m))
                    .sqrt()
                    .clamp_min(1e-6)
                )
                out[where] = (scores[where] - mean) / std
        calibrated[split] = out
    return calibrated


def combine(method, form, anchor_cal, partner_cal, anchor, partner):
    # The global form reproduces the rules already reported: PSBD-TM alone on its
    # raw fractional PSU, the minimum of percentiles and the average of raw PSU.
    if method == "psbd_tm":
        scores = anchor if form == "global" else anchor_cal
    elif method == "final_min":
        scores = {s: torch.minimum(anchor_cal[s], partner_cal[s]) for s in anchor_cal}
    elif form == "global":
        scores = {s: (anchor[s] + partner[s]) / 2 for s in anchor}
    else:
        scores = {s: (anchor_cal[s] + partner_cal[s]) / 2 for s in anchor_cal}
    return scores


def evaluate(scores, manifest):
    from sklearn.metrics import roc_auc_score

    validation = scores["validation"].numpy()
    clean = pair_clean_to_backdoor(scores["clean"], manifest)  # (n_backdoor,)
    backdoor = scores["backdoor"]  # (n_backdoor,)
    reading = {}
    for quantile in QUANTILES:
        threshold = float(np.quantile(validation, quantile))
        reading[f"q{quantile:.2f}:tpr"] = float((backdoor < threshold).float().mean())
        reading[f"q{quantile:.2f}:realized_fpr"] = float(
            (clean < threshold).float().mean()
        )
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    reading["auroc"] = float(
        roc_auc_score(labels, np.concatenate([-clean.numpy(), -backdoor.numpy()]))
    )
    return reading


def summarize(rows, m):
    key = f"m={m}"
    fields = ["auroc"] + [
        f"{q}:{kind}" for q in HEADLINE for kind in ("tpr", "realized_fpr")
    ]

    def block(subset):
        result = {"n": len(subset)}
        for form in FORMS:
            for method in METHODS:
                name = f"{form}/{method}"
                reference = f"global/{method}"
                result[name] = {
                    field: paired_summary(
                        [r["by_m"][key][name][field] for r in subset],
                        [r["by_m"][key][reference][field] for r in subset],
                    )
                    for field in fields
                }
        return result

    summary = {"all": block(rows)}
    summary["by_attack"] = {
        attack: block([r for r in rows if r["attack"] == attack])
        for attack in ordered_attacks({r["attack"] for r in rows})
    }
    summary["by_dataset"] = {
        dataset: block([r for r in rows if r["dataset"] == dataset])
        for dataset in sorted({r["dataset"] for r in rows})
    }
    return summary


def select_shrinkage(summary):
    # The selection reads PSBD-TM alone, the mean of its TPR over the 3 headline
    # FPRs, per form. A tie goes to the larger m, the one closer to the global
    # calibration every published number uses.
    selection = {}
    for form in FORMS[1:]:
        scored = {
            m: sum(
                summary[f"m={m}"]["all"][f"{form}/psbd_tm"][f"{q}:tpr"]["mean"]
                for q in HEADLINE
            )
            / 3
            for m in SHRINKAGE_GRID
        }
        best = max(sorted(scored, reverse=True), key=lambda m: round(scored[m], 4))
        selection[form] = {"scores": scored, "chosen": best}
    return selection


def tact_placement(model):
    # Where source-class clean, target-class clean and triggered inputs sit
    # against the global PSBD-TM threshold.
    folder = model["folder_name"]
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    anchor = adaptive_psu(psbd_dir, ANCHOR, baselines)
    sources = source_classes_of(
        "checkpoints", {"folder_name": folder, "attack": "tact"}
    )
    _, _, clean_labels = baselines["clean"]
    target = model["target_label"]
    source_clean = anchor["clean"][sum(clean_labels == s for s in sources).bool()]
    target_clean = anchor["clean"][clean_labels == target]
    groups = {
        "source_clean": source_clean,
        "target_clean": target_clean,
        "triggered": anchor["backdoor"],
        "validation": anchor["validation"],
    }
    placement = {
        "folder": folder,
        "source_classes": list(sources),
        "target": target,
        "by_fpr": {},
    }
    placement["median"] = {
        name: float(values.median()) for name, values in groups.items()
    }
    for quantile in QUANTILES:
        threshold = float(np.quantile(anchor["validation"].numpy(), quantile))
        placement["by_fpr"][f"q{quantile:.2f}"] = {
            "threshold": threshold,
            **{
                f"{name}_below": float((values < threshold).float().mean())
                for name, values in groups.items()
                if name != "validation"
            },
        }
    return placement


if __name__ == "__main__":
    main()

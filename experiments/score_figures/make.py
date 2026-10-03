"""Score histograms and low-FPR ROC curves for any combination of PSBD probes.

Reads the stage-1 caches only (CPU). Every probe is read at its adaptive 0.8 rate,
every score is thresholded at quantiles of its own clean-validation distribution
and TPR is read on the triggered split, the realized FPR on the paired clean test
split. What is drawn is set in config.py.

    .venv/bin/python -m experiments.score_figures.make
    .venv/bin/python -m experiments.score_figures.make --set backdoorbench
    .venv/bin/python -m experiments.score_figures.make --set vit_panel --models a,b
    .venv/bin/python -m experiments.score_figures.make --models vit_tiny_wanet_0_05
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import time

import numpy as np
import torch

from defenses.cache import read_split_manifest
from defenses.decision import ADAPTIVE_SHIFT_TARGET, pair_clean_to_backdoor
from defenses.scores import psu_ratio_from_cache, to_rank
from experiments._paths import experiment_results_dir
from experiments.cache_readouts.shared import (
    choose_rate,
    evaluate_scores,
    load_baselines,
    load_model_set,
    load_passes,
    validation_shift_by_rate,
)
from experiments.score_figures import config
from experiments.score_figures.figures import (
    ATTACK_NAMES,
    combination_label,
    draw_histogram,
    draw_mean_roc,
    draw_model_roc,
    model_title,
)
from experiments.score_figures.pictures import PICTURE_NAMES, run_pictures
from scripts.paper.tab_swin import swin_cells

SLUG = "score_figures"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
BACKDOORBENCH_RECORDS = os.path.join(
    RESULTS_DIR, "_experiments", "backdoorbench_attacks", "models"
)
# The login node is shared, so torch gets a few CPU threads and no more.
CPU_THREADS = 4
SPLITS = ("validation", "clean", "backdoor")
SINGLE_RULE = "psu"
CUSTOM_SET = "custom"
SCORES = "scores"
SET_TITLES = {
    "vit_panel": "ViT-B/16 panel",
    "swin_panel": "Swin-S panel",
    "backdoorbench": "BackdoorBench ViT-B/16 checkpoints",
}


def main():
    args = parse_args()
    torch.set_num_threads(CPU_THREADS)
    started = time.perf_counter()

    root = experiment_results_dir(SLUG, RESULTS_DIR)
    set_names = chosen_sets(args)
    if SCORES in args.figures:
        for set_name in set_names:
            models = resolve_set(set_name, args.models)
            set_dir = os.path.join(root, set_name)
            for model in models:
                numbers, scores = score_model(model)
                write_model_outputs(set_dir, model, numbers, scores)
                print(f"{set_name} {model['folder']} done", flush=True)

            all_numbers = read_set_numbers(set_dir)
            write_set_outputs(set_dir, set_name, all_numbers)
            print(f"{set_name}: {len(all_numbers)} models in {set_dir}", flush=True)

    pictures = [name for name in args.figures if name != SCORES]
    run_pictures(pictures, set_names, root)

    print(f"finished in {time.perf_counter() - started:.0f} s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=sorted(config.MODEL_SETS))
    parser.add_argument(
        "--models",
        help="comma-separated results/ folder names, a filter on --set or, alone, "
        "an explicit set written under 'custom'",
    )
    parser.add_argument(
        "--figures",
        default=",".join((SCORES,) + PICTURE_NAMES),
        help="comma-separated figure kinds to draw, 'scores' being the histograms, "
        "ROC curves and numbers every other kind reads, the rest switched on in "
        f"config.py: {', '.join(PICTURE_NAMES)}",
    )
    args = parser.parse_args()
    args.models = args.models.split(",") if args.models else None
    args.figures = args.figures.split(",")
    unknown = sorted(set(args.figures) - {SCORES, *PICTURE_NAMES})
    assert not unknown, f"unknown figure kinds {unknown}"
    return args


def chosen_sets(args):
    if args.set:
        return [args.set]
    if args.models:
        return [CUSTOM_SET]
    return list(config.MODEL_SETS)


def resolve_set(set_name, only_folders):
    if set_name == CUSTOM_SET:
        models = [explicit_model(folder) for folder in only_folders]
        return models

    source = config.MODEL_SETS[set_name]
    if isinstance(source, list):
        models = [explicit_model(folder) for folder in source]
    elif source == "vit_panel":
        models = [ledger_model(cell, "vit") for cell in load_model_set("panel")]
    elif source == "swin_panel":
        models = [
            ledger_model(cell, "swin")
            for cell in swin_cells(RESULTS_DIR, CHECKPOINTS_DIR)
        ]
    elif source == "backdoorbench":
        models = backdoorbench_models()
    else:
        raise ValueError(f"unknown model set source {source!r}")

    if only_folders:
        known = {model["folder"] for model in models}
        unknown = sorted(set(only_folders) - known)
        assert not unknown, f"{unknown} are not in set {set_name}"
        models = [model for model in models if model["folder"] in only_folders]
    return models


def ledger_model(cell, architecture):
    model = {
        "folder": cell["folder_name"],
        "architecture": architecture,
        "dataset": cell["dataset"],
        "attack": cell["attack"],
        "poison_rate": cell["poison_rate"],
    }
    return model


def backdoorbench_models():
    # A record without detection rates was never swept (a dead backdoor or a
    # checkpoint without its trigger generator), so it has no cache to read.
    models = []
    for path in sorted(glob.glob(os.path.join(BACKDOORBENCH_RECORDS, "*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        if not (record.get("detection") or {}).get("rates"):
            continue
        models.append(
            {
                "folder": f"bb_{record['folder']}",
                "architecture": "vit",
                "dataset": record["dataset"],
                "attack": record["attack"],
                "poison_rate": record["poison_rate"],
            }
        )
    return models


def explicit_model(folder):
    if folder.startswith("bb_"):
        record_path = os.path.join(BACKDOORBENCH_RECORDS, f"{folder[3:]}.json")
    else:
        record_path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
    record = {}
    if os.path.exists(record_path):
        with open(record_path) as handle:
            record = json.load(handle)

    model = {
        "folder": folder,
        "architecture": "swin" if folder.startswith("swin_") else "vit",
        "dataset": record.get("dataset"),
        "attack": record.get("attack"),
        "poison_rate": record.get("poison_rate"),
    }
    return model


def score_model(model):
    psbd_dir = os.path.join(RESULTS_DIR, model["folder"], "psbd")
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)

    probe_names = sorted({name for combo in config.COMBINATIONS for name in combo})
    probe_placements = config.PROBES[model["architecture"]]
    probes = {
        name: read_probe(psbd_dir, probe_placements.get(name), baselines)
        for name in probe_names
    }

    numbers = dict(model)
    numbers["budgets"] = list(config.BUDGETS)
    numbers["roc_fpr_grid"] = rounded(roc_grid())
    numbers["probes"] = {
        name: {key: value for key, value in probe.items() if key != "psu"}
        for name, probe in probes.items()
    }
    numbers["combinations"] = {}
    scores = {}
    for combination in config.COMBINATIONS:
        key = "+".join(combination)
        skipped = {name: probes[name]["skipped"] for name in combination}
        skipped = {name: reason for name, reason in skipped.items() if reason}
        if skipped:
            numbers["combinations"][key] = {"skipped": skipped}
            continue

        psus = [probes[name]["psu"] for name in combination]
        rules = [SINGLE_RULE] if len(combination) == 1 else config.FUSION_RULES
        numbers["combinations"][key] = {}
        for rule in rules:
            scores_by_split = combine(psus, rule)
            numbers["combinations"][key][rule] = evaluate_rule(
                scores_by_split, manifest
            )
            scores[(key, rule)] = scores_by_split
    return numbers, scores


def read_probe(psbd_dir, placement, baselines):
    # A probe that cannot be read at its adaptive rate is skipped with the
    # reason, never replaced by another rate or placement.
    probe = {"placement": placement, "rate": None, "skipped": None}
    if placement is None:
        probe["skipped"] = "probe not defined for this architecture in config.PROBES"
        return probe
    shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
    if not shift_by_rate:
        probe["skipped"] = "placement not cached"
        return probe

    rate = choose_rate(shift_by_rate, "adaptive")
    probe["rates_cached"] = sorted(shift_by_rate)
    probe["max_validation_shift"] = max(shift_by_rate.values())
    if rate is None:
        probe["skipped"] = (
            f"ladder never reaches the {ADAPTIVE_SHIFT_TARGET} shift target"
        )
        return probe

    passes = load_passes(psbd_dir, placement, rate, baselines)
    probe["rate"] = rate
    probe["validation_shift"] = shift_by_rate[rate]
    probe["psu"] = {
        split: psu_ratio_from_cache(
            passes[split]["baseline_probs"],
            passes[split]["baseline_labels"],
            passes[split]["per_pass_probs"],
        )  # (n_split,)
        for split in SPLITS
    }
    return probe


def combine(psus, rule):
    if rule == SINGLE_RULE:
        return psus[0]

    shares = fusion_shares(len(psus), rule)  # (k, 1)
    fused = {}
    for split in SPLITS:
        # Percentiles within each probe's own clean-validation distribution put
        # the probes on 1 scale without fitting anything to triggered data.
        ranks = torch.stack(
            [to_rank(psu[split], psu["validation"]) for psu in psus]
        )  # (k, n_split)
        assert ranks.shape[0] == len(psus), ranks.shape
        fused[split] = (ranks / shares).min(dim=0).values  # (n_split,)
    return fused


def fusion_shares(probe_count, rule):
    if rule == "min":
        shares = [1.0] * probe_count
    elif rule == "weighted":
        rest = (1.0 - config.WEIGHTED_FIRST_SHARE) / (probe_count - 1)
        shares = [config.WEIGHTED_FIRST_SHARE] + [rest] * (probe_count - 1)
    else:
        raise ValueError(f"unknown fusion rule {rule!r}")
    shares_column = torch.tensor(shares, dtype=torch.float32).unsqueeze(1)  # (k, 1)
    return shares_column


def evaluate_rule(scores_by_split, manifest):
    evaluation = evaluate_scores(scores_by_split, manifest, config.BUDGETS)
    validation = scores_by_split["validation"].float().numpy()  # (n_validation,)
    clean_paired = (
        pair_clean_to_backdoor(scores_by_split["clean"], manifest).float().numpy()
    )  # (n_backdoor,)
    backdoor = scores_by_split["backdoor"].float().numpy()  # (n_backdoor,)

    evaluation["roc"] = roc_points(validation, clean_paired, backdoor)
    evaluation["histogram"] = histogram(validation, backdoor)
    return evaluation


def roc_grid():
    grid = np.logspace(np.log10(config.ROC_LOWEST_FPR), 0.0, config.ROC_POINTS)
    return grid


def roc_points(validation, clean_paired, backdoor):
    # The same threshold rule as defenses.decision.detection_report, a linear
    # quantile of clean validation with a strict "below" test, at every grid
    # budget instead of 4.
    thresholds = np.quantile(validation, roc_grid())  # (points,)
    tpr = (backdoor[None, :] < thresholds[:, None]).mean(axis=1)  # (points,)
    realized_fpr = (clean_paired[None, :] < thresholds[:, None]).mean(
        axis=1
    )  # (points,)
    points = {"tpr": rounded(tpr), "realized_fpr": rounded(realized_fpr)}
    return points


def histogram(validation, backdoor):
    low = float(min(validation.min(), backdoor.min()))
    high = float(np.quantile(validation, config.HISTOGRAM_UPPER_QUANTILE))
    if high <= low:
        high = float(validation.max())
    edges = np.linspace(low, high, config.HISTOGRAM_BINS + 1)  # (bins + 1,)

    validation_counts, _ = np.histogram(validation, bins=edges)  # (bins,)
    backdoor_counts, _ = np.histogram(backdoor, bins=edges)  # (bins,)
    counts = {
        "bin_edges": rounded(edges, 6),
        "validation_counts": validation_counts.tolist(),
        "backdoor_counts": backdoor_counts.tolist(),
        "validation_above_range": int((validation > high).sum()),
        "backdoor_above_range": int((backdoor > high).sum()),
        "validation_total": int(len(validation)),
        "backdoor_total": int(len(backdoor)),
    }
    return counts


def rounded(values, digits=5):
    values = [round(float(value), digits) for value in values]
    return values


def write_model_outputs(set_dir, model, numbers, scores):
    model_dir = os.path.join(set_dir, model["folder"])
    os.makedirs(model_dir, exist_ok=True)
    write_json(numbers, os.path.join(model_dir, "numbers.json"))

    title = model_title(numbers)
    curves = []
    for (key, rule), scores_by_split in scores.items():
        evaluation = numbers["combinations"][key][rule]
        curves.append((key, rule, evaluation["roc"]["tpr"]))
        thresholds = {
            budget: evaluation["at_fpr"][f"q{budget:.2f}"]["threshold"]
            for budget in config.BUDGETS
        }
        draw_histogram(
            os.path.join(model_dir, f"hist_{key.replace('+', '-')}_{rule}.png"),
            f"{title}\n{curve_label(key, rule, numbers)}",
            score_axis_label(key, rule, numbers["architecture"]),
            scores_by_split["validation"].float().numpy(),
            scores_by_split["backdoor"].float().numpy(),
            evaluation["histogram"]["bin_edges"],
            thresholds,
        )
    draw_model_roc(
        os.path.join(model_dir, "roc.png"),
        title,
        numbers["roc_fpr_grid"],
        [
            (key, rule, curve_label(key, rule, numbers, fused_rates=False), tpr)
            for key, rule, tpr in curves
        ],
        combination_order(),
    )


def curve_label(key, rule, numbers, fused_rates=True):
    architecture = numbers["architecture"]
    rates = ", ".join(
        f"{config.PROBE_SHORT_NAMES[architecture].get(name, name)} rate "
        f"{numbers['probes'][name]['rate']:g}"
        for name in key.split("+")
    )
    if rule == SINGLE_RULE:
        return f"{combination_label(architecture, key, rule)}, rate {numbers['probes'][key]['rate']:g}"
    # The ROC legend lists every probe alone with its rate, so its fused entries
    # leave the rates out to keep the legend no wider than the plot.
    if not fused_rates:
        return combination_label(architecture, key, rule)
    label = f"{combination_label(architecture, key, rule)} ({rates})"
    return label


def score_axis_label(key, rule, architecture):
    if rule == SINGLE_RULE:
        name = config.PROBE_SHORT_NAMES[architecture].get(key, key)
        return f"{name} fractional PSU (lower is more suspicious)"
    if rule == "weighted":
        return "fused score, min of percentile / share (lower is more suspicious)"
    return "fused score, min of clean-validation percentiles (lower is more suspicious)"


def combination_order():
    order = ["+".join(combination) for combination in config.COMBINATIONS]
    return order


def read_set_numbers(set_dir):
    # Every model directory under the set is aggregated, so a run filtered with
    # --models refreshes those models and keeps the others' numbers.
    all_numbers = []
    for path in sorted(glob.glob(os.path.join(set_dir, "*", "numbers.json"))):
        with open(path) as handle:
            all_numbers.append(json.load(handle))
    return all_numbers


def write_set_outputs(set_dir, set_name, all_numbers):
    rows = model_rows(set_name, all_numbers)
    write_csv(rows, os.path.join(set_dir, "models.csv"))
    write_csv(mean_rows(set_name, rows), os.path.join(set_dir, "means.csv"))

    mean_curves = mean_roc_curves(all_numbers)
    architecture = all_numbers[0]["architecture"]
    labels = {
        ("+".join(combination), rule): combination_label(
            architecture, "+".join(combination), rule
        )
        for combination in config.COMBINATIONS
        for rule in ([SINGLE_RULE] if len(combination) == 1 else config.FUSION_RULES)
    }
    set_title = SET_TITLES.get(set_name, set_name)
    draw_mean_roc(
        os.path.join(set_dir, "mean_roc.png"),
        f"{set_title}: mean TPR over models",
        roc_grid().tolist(),
        mean_curves,
        combination_order(),
        labels,
    )
    for attack in sorted({numbers["attack"] for numbers in all_numbers}):
        attack_numbers = [n for n in all_numbers if n["attack"] == attack]
        draw_mean_roc(
            os.path.join(set_dir, "mean_roc_by_attack", f"{attack}.png"),
            f"{set_title}, {ATTACK_NAMES.get(attack, attack)}: mean TPR over models",
            roc_grid().tolist(),
            mean_roc_curves(attack_numbers),
            combination_order(),
            labels,
        )

    index = {
        "set": set_name,
        "generator": "experiments/score_figures/make.py",
        "git_commit": git_commit(),
        "adaptive_shift_target": ADAPTIVE_SHIFT_TARGET,
        "config": config_snapshot(),
        "roc_fpr_grid": rounded(roc_grid()),
        "mean_roc": [
            {"combination": key, "rule": rule, "n_models": n, "mean_tpr": rounded(tpr)}
            for key, rule, n, tpr in mean_curves
        ],
        "models": [model_entry(numbers) for numbers in all_numbers],
    }
    write_json(index, os.path.join(set_dir, "index.json"))


def model_rows(set_name, all_numbers):
    rows = []
    for numbers in all_numbers:
        base = {
            "set": set_name,
            "folder": numbers["folder"],
            "architecture": numbers["architecture"],
            "dataset": numbers["dataset"],
            "attack": numbers["attack"],
            "poison_rate": numbers["poison_rate"],
        }
        for key, by_rule in numbers["combinations"].items():
            rates = " ".join(
                f"{name}={numbers['probes'][name]['rate']}" for name in key.split("+")
            )
            if "skipped" in by_rule:
                reasons = " | ".join(
                    f"{name}: {reason}" for name, reason in by_rule["skipped"].items()
                )
                rows.append(
                    {
                        **base,
                        "combination": key,
                        "rule": "",
                        "rates": rates,
                        "skipped": reasons,
                    }
                )
                continue
            for rule, evaluation in by_rule.items():
                row = {
                    **base,
                    "combination": key,
                    "rule": rule,
                    "rates": rates,
                    "skipped": "",
                }
                row.update(headline_fields(evaluation))
                rows.append(row)
    return rows


def headline_fields(evaluation):
    fields = {}
    for budget in config.BUDGETS:
        at_budget = evaluation["at_fpr"][f"q{budget:.2f}"]
        fields[f"tpr_{budget:g}"] = round(at_budget["tpr"], 4)
        fields[f"fpr_{budget:g}"] = round(at_budget["realized_fpr"], 4)
    fields["auroc"] = round(evaluation["auroc"], 4)
    return fields


def mean_rows(set_name, rows):
    scored = [row for row in rows if not row["skipped"]]
    metric_names = [f"tpr_{b:g}" for b in config.BUDGETS]
    metric_names += [f"fpr_{b:g}" for b in config.BUDGETS] + ["auroc"]

    groupings = [
        ("all", lambda row: "all"),
        ("dataset", lambda row: row["dataset"]),
        ("attack", lambda row: row["attack"]),
    ]
    means = []
    for group_kind, group_of in groupings:
        groups = {}
        for row in scored:
            groups.setdefault(
                (group_of(row), row["combination"], row["rule"]), []
            ).append(row)
        for (group, key, rule), members in sorted(groups.items()):
            mean_row = {
                "set": set_name,
                "group_kind": group_kind,
                "group": group,
                "combination": key,
                "rule": rule,
                "n_models": len(members),
            }
            for name in metric_names:
                mean_row[name] = round(float(np.mean([m[name] for m in members])), 4)
            means.append(mean_row)
    return means


def mean_roc_curves(all_numbers):
    curves = {}
    for numbers in all_numbers:
        for key, by_rule in numbers["combinations"].items():
            if "skipped" in by_rule:
                continue
            for rule, evaluation in by_rule.items():
                curves.setdefault((key, rule), []).append(evaluation["roc"]["tpr"])

    mean_curves = []
    for (key, rule), tprs in curves.items():
        mean_tpr = np.mean(np.array(tprs), axis=0)  # (points,)
        mean_curves.append((key, rule, len(tprs), mean_tpr.tolist()))
    return mean_curves


def model_entry(numbers):
    entry = {
        "folder": numbers["folder"],
        "dataset": numbers["dataset"],
        "attack": numbers["attack"],
        "poison_rate": numbers["poison_rate"],
        "rates": {name: probe["rate"] for name, probe in numbers["probes"].items()},
        "skipped_probes": {
            name: probe["skipped"]
            for name, probe in numbers["probes"].items()
            if probe["skipped"]
        },
        "numbers": f"{numbers['folder']}/numbers.json",
    }
    return entry


def config_snapshot():
    snapshot = {
        "probes": config.PROBES,
        "combinations": config.COMBINATIONS,
        "fusion_rules": config.FUSION_RULES,
        "weighted_first_share": config.WEIGHTED_FIRST_SHARE,
        "budgets": list(config.BUDGETS),
        "histogram_upper_quantile": config.HISTOGRAM_UPPER_QUANTILE,
    }
    return snapshot


def git_commit():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return commit


def write_json(payload, path):
    # Number lists are kept on 1 line so a numbers.json stays readable and small.
    text = json.dumps(payload, indent=1)
    text = re.sub(
        r"\[\s*(-?[\d.e+-]+(?:,\s*-?[\d.e+-]+)*)\s*\]",
        lambda match: "[" + ", ".join(match.group(1).split()).replace(",,", ",") + "]",
        text,
    )
    with open(path, "w") as handle:
        handle.write(text)


def write_csv(rows, path):
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

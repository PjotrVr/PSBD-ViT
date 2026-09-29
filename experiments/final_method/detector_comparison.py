"""The final method against the 11 competitor detectors on the ViT panel.

The final method is PSBD-TM fused with residual dropout in blocks 5 to 8, under
the plain minimum of clean-validation percentiles and under the plain average of
fractional PSU. Its per-model readings come from the fusion readout
(`fusion_rules_panel.json`, middle band at the adaptive rate). Every competitor
reading comes from `results/<folder>/detectors/<name>_metrics.json`, scored by
`cli.baselines` on the same splits and quantiles. Each paired interval is the
per-model difference from the best competitor at that FPR, the detector with the
highest mean TPR there.

    .venv/bin/python -m experiments.final_method.detector_comparison
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from detectors import DETECTOR_NAMES  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.cache_readouts.shared import (  # noqa: E402
    REPO_ROOT,
    ordered_attacks,
    output_path,
    paired_summary,
    write_json,
)
from scripts.paper._common import OKABE_ITO, figure_sidecar, save_figure  # noqa: E402

SLUG = "final_method"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(REPO_ROOT, "experiments", SLUG, "figures")
HEADLINE = ("q0.01", "q0.05", "q0.10")
PARTNER = "middle_band"
OURS = {
    "psbd_tm": "tm_alone",
    "final_min": "min_rank",
    "final_average": "mean_psu",
}


def main():
    started = time.perf_counter()
    fusion_path = output_path("fusion_rules", "panel")
    with open(fusion_path) as handle:
        fusion = json.load(handle)

    readings = {}
    missing = {}
    for row in fusion["models"]:
        folder = row["folder"]
        partner = row["partners"]["adaptive"][PARTNER]
        assert partner["rate"] is not None, folder
        per_method = {
            name: reading_from_rule(partner["rules"][rule])
            for name, rule in OURS.items()
        }
        for detector in DETECTOR_NAMES:
            record = read_detector(folder, detector)
            if record is None:
                missing.setdefault(detector, []).append(folder)
                continue
            per_method[detector] = reading_from_detector(record)
        readings[folder] = {
            "attack": row["attack"],
            "dataset": row["dataset"],
            **per_method,
        }

    methods = list(OURS) + list(DETECTOR_NAMES)
    summary = summarize(readings, methods)

    payload = {
        "experiment": "final method against the competitor detectors, ViT panel",
        "fusion_source": fusion_path,
        "partner": PARTNER,
        "methods": methods,
        "missing_detector_records": missing,
        "summary": summary,
        "models": readings,
        "wall_seconds": time.perf_counter() - started,
    }
    path = experiment_result_path(SLUG, "detector_comparison.json", RESULTS_DIR)
    write_json(payload, path)
    plot(summary, methods, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def read_detector(folder, detector):
    path = os.path.join(RESULTS_DIR, folder, "detectors", f"{detector}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        record = json.load(handle)
    if record.get("status") != "scored":
        return None
    return record


def reading_from_rule(evaluation):
    reading = {"auroc": evaluation["auroc"]}
    for quantile in HEADLINE:
        reading[f"{quantile}:tpr"] = evaluation["at_fpr"][quantile]["tpr"]
        reading[f"{quantile}:fpr"] = evaluation["at_fpr"][quantile]["realized_fpr"]
    return reading


def reading_from_detector(record):
    detection = record["detection"]
    reading = {"auroc": detection["q0.01"]["auroc"]}
    for quantile in HEADLINE:
        reading[f"{quantile}:tpr"] = detection[quantile]["tpr"]
        reading[f"{quantile}:fpr"] = detection[quantile]["fpr"]
    return reading


def summarize(readings, methods):
    fields = ["auroc"] + [f"{q}:{kind}" for q in HEADLINE for kind in ("tpr", "fpr")]
    folders = list(readings)

    def mean_of(method, field, subset):
        values = [readings[f][method][field] for f in subset if method in readings[f]]
        result = sum(values) / len(values) if values else None
        return result

    overall = {
        method: {
            "n": sum(method in readings[f] for f in folders),
            **{field: mean_of(method, field, folders) for field in fields},
        }
        for method in methods
    }

    # The best competitor is chosen per FPR on the full panel, then every one of
    # ours is compared with it model by model.
    competitors = [m for m in methods if m not in OURS]
    best = {}
    for field in [f"{q}:tpr" for q in HEADLINE] + ["auroc"]:
        champion = max(competitors, key=lambda m: overall[m][field] or -1)
        best[field] = champion
    paired = {}
    for method in OURS:
        paired[method] = {}
        for field, champion in best.items():
            shared = [f for f in folders if champion in readings[f]]
            values = [readings[f][method][field] for f in shared]
            reference = [readings[f][champion][field] for f in shared]
            paired[method][field] = {
                "against": champion,
                **paired_summary(values, reference),
            }

    attacks = ordered_attacks({readings[f]["attack"] for f in folders})
    by_attack = {}
    for attack in attacks:
        subset = [f for f in folders if readings[f]["attack"] == attack]
        by_attack[attack] = {
            method: {
                "n": sum(method in readings[f] for f in subset),
                **{field: mean_of(method, field, subset) for field in fields},
            }
            for method in methods
        }

    summary = {
        "overall": overall,
        "best_competitor": best,
        "paired": paired,
        "by_attack": by_attack,
    }
    return summary


def plot(summary, methods, json_path):
    overall = summary["overall"]
    figure, axis = plt.subplots(figsize=(14, 5))
    width = 0.8 / len(HEADLINE)
    plotted = {}
    for index, quantile in enumerate(HEADLINE):
        heights = [overall[m][f"{quantile}:tpr"] or 0 for m in methods]
        positions = [
            i + (index - len(HEADLINE) / 2 + 0.5) * width for i in range(len(methods))
        ]
        axis.bar(
            positions,
            heights,
            width,
            label=f"TPR at {float(quantile[1:]):.0%} FPR",
            color=OKABE_ITO[index],
        )
        plotted[quantile] = dict(zip(methods, heights))
    axis.set_xticks(range(len(methods)))
    axis.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("mean TPR over the ViT panel")
    axis.legend(fontsize=8)
    axis.set_title("Final method and the competitor detectors, ViT panel")

    figure_path = os.path.join(FIGURES_DIR, "detector_comparison.png")
    save_figure(figure, figure_path)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/final_method/detector_comparison.py",
        [json_path],
        plotted,
    )


if __name__ == "__main__":
    main()

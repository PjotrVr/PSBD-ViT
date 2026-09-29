"""Verdicts of preregistration_classcal.json on the confirmation read.

.venv/bin/python -m experiments.final_method.judge_classcal
"""

import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments._paths import experiment_result_path  # noqa: E402
from experiments.cache_readouts.shared import REPO_ROOT, write_json  # noqa: E402
from experiments.final_method.class_calibration import (  # noqa: E402
    FORMS,
    HEADLINE,
    METHODS,
    PREREGISTRATION,
)
from scripts.paper._common import OKABE_ITO, figure_sidecar, save_figure  # noqa: E402

SLUG = "final_method"


def main():
    with open(PREREGISTRATION, "rb") as handle:
        raw = handle.read()
    preregistration = json.loads(raw)
    path = experiment_result_path(SLUG, "class_calibration_rest.json")
    with open(path) as handle:
        rest = json.load(handle)
    m = preregistration["shrinkage"]["class_z"]
    summary = rest["summary"][f"m={m}"]

    results = [
        raises(summary, "CC-z-tm", ["class_z/psbd_tm"]),
        raises(summary, "CC-z-final", ["class_z/final_min", "class_z/final_average"]),
        no_harm(summary, preregistration["predictions"][2]),
        fpr_kept(summary, preregistration["predictions"][3]),
    ]
    verdicts = {
        "preregistration_sha256": hashlib.sha256(raw).hexdigest(),
        "results": results,
    }
    out = experiment_result_path(SLUG, "class_calibration_verdicts.json")
    write_json(verdicts, out)
    for result in results:
        print(result["id"], result["verdict"])
    plot(m)


def plot(m):
    path = experiment_result_path(SLUG, "class_calibration_panel.json")
    with open(path) as handle:
        overall = json.load(handle)["summary"][f"m={m}"]["all"]
    n_models = overall["n"]
    figure, axes = plt.subplots(
        1, len(METHODS), figsize=(15, 4.5), sharey=True, squeeze=False
    )
    plotted = {}
    width = 0.8 / len(FORMS)
    for index, method in enumerate(METHODS):
        axis = axes[0][index]
        for form_index, form in enumerate(FORMS):
            heights = [
                overall[f"{form}/{method}"][f"{q}:tpr"]["mean"] for q in HEADLINE
            ]
            positions = [
                i + (form_index - len(FORMS) / 2 + 0.5) * width
                for i in range(len(HEADLINE))
            ]
            axis.bar(positions, heights, width, label=form, color=OKABE_ITO[form_index])
            plotted[f"{method}/{form}"] = dict(zip(HEADLINE, heights))
        axis.set_xticks(range(len(HEADLINE)))
        axis.set_xticklabels([f"TPR at {float(q[1:]):.0%} FPR" for q in HEADLINE])
        axis.set_title(method)
        axis.set_ylim(0, 1)
    axes[0][0].set_ylabel(f"mean TPR, {n_models} ViT panel models")
    axes[0][0].legend(fontsize=8)
    figure.suptitle(
        f"Calibration by predicted class against global calibration, m = {m}"
    )
    figure_path = os.path.join(
        REPO_ROOT, "experiments", SLUG, "figures", "class_calibration.png"
    )
    save_figure(figure, figure_path)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/final_method/judge_classcal.py",
        [path],
        plotted,
    )


def raises(summary, name, keys):
    readings = {
        f"{key}/{q}": summary["all"][key][f"{q}:tpr"] for key in keys for q in HEADLINE
    }
    above = all(r["mean_difference"] > 0 for r in readings.values())
    excluded = all(r["ci95"][0] > 0 for r in readings.values())
    verdict = "held" if above and excluded else ("inconclusive" if above else "failed")
    result = {
        "id": name,
        "verdict": verdict,
        "readings": {
            k: {"mean_difference": r["mean_difference"], "ci95": r["ci95"]}
            for k, r in readings.items()
        },
    }
    return result


def no_harm(summary, prediction):
    per_attack = {
        attack: group[prediction["key"]]["q0.01:tpr"]["mean_difference"]
        for attack, group in summary["by_attack"].items()
    }
    verdict = (
        "held"
        if all(v >= prediction["bound"] for v in per_attack.values())
        else "failed"
    )
    result = {"id": prediction["id"], "verdict": verdict, "per_attack": per_attack}
    return result


def fpr_kept(summary, prediction):
    realized = {
        q: summary["all"][prediction["key"]][f"{q}:realized_fpr"]["mean"]
        for q in HEADLINE
    }
    kept = all(
        abs(value - float(q[1:])) <= prediction["tolerance"]
        for q, value in realized.items()
    )
    result = {
        "id": prediction["id"],
        "verdict": "held" if kept else "failed",
        "realized": realized,
    }
    return result


if __name__ == "__main__":
    main()

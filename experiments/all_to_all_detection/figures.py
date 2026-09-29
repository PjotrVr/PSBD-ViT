"""The 4 figures of README.md, each with a JSON sidecar of the numbers it draws.

fragility      shift rate of triggered and clean twins per placement
destinations   where shifted triggered predictions land, against chance
tokens         where triggered predictions go when tokens are masked (ViT)
detectors      mean AUROC with its bootstrap interval, per statistic and population

source .venv/bin/activate
python experiments/all_to_all_detection/figures.py
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from panel import OUT_DIR  # noqa: E402
from scripts.paper._common import OKABE_ITO, figure_sidecar  # noqa: E402

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
GENERATOR = "experiments/all_to_all_detection/figures.py"
MECHANISM_PLACEMENTS = {
    "vit": (
        ("before_attention_norm_token_mask", "adaptive", "PSBD-TM"),
        ("post_residual", "adaptive", "PSBD-RD"),
        ("pre_residual_blocks_1_4", "matched_0.6", "dropout 1-4"),
        ("pre_residual_blocks_5_8", "matched_0.6", "dropout 5-8"),
        ("pre_residual_blocks_9_12", "matched_0.6", "dropout 9-12"),
    ),
    "swin": (
        ("before_attention_norm_token_mask", "adaptive", "PSBD-TM"),
        ("post_residual", "adaptive", "PSBD-RD"),
        ("pre_residual_blocks_1_8", "matched_0.6", "dropout 1-8"),
        ("pre_residual_blocks_9_16", "matched_0.6", "dropout 9-16"),
        ("pre_residual_blocks_17_24", "matched_0.6", "dropout 17-24"),
    ),
}
TOKEN_CONDITIONS = (
    ("baseline", "none"),
    ("random_all_12", "random, 1-12"),
    ("trigger_blocks_all_12", "trigger, 1-12"),
    ("trigger_blocks_blocks_1_4", "trigger, 1-4"),
    ("trigger_blocks_blocks_5_8", "trigger, 5-8"),
    ("trigger_blocks_blocks_9_12", "trigger, 9-12"),
    ("content_keep_0.6", "content 60%"),
    ("content_keep_0.3", "content 30%"),
    ("content_keep_0.1", "content 10%"),
    ("content_keep_0.0", "content 0%"),
)
TOKEN_OUTCOMES = (
    ("triggered_on_attack_label", "attack label"),
    ("triggered_on_source", "source class"),
    ("triggered_elsewhere", "elsewhere"),
)
DETECTOR_POPULATIONS = (
    ("all_to_all_strict_2pt", "all-to-all, ASR bar"),
    ("all_to_all_relaxed_2pt", "all-to-all, relaxed bar"),
    ("benign", "benign"),
    ("all_to_one", "all-to-one panel"),
)


def main():
    mechanism = read_json("mechanism.json")
    detection = read_json("detection.json")
    tokens = read_json("tokens.json")

    plot_fragility(mechanism["summary"])
    plot_destinations(mechanism["summary"])
    plot_tokens(tokens["models"])
    plot_detectors(detection)


def read_json(name):
    payload = json.load(open(os.path.join(OUT_DIR, name)))
    return payload


def mechanism_value(summary, architecture, label_mode, placement, rule, field):
    block = summary.get(f"{architecture}/{label_mode}/{placement}/{rule}")
    value = block.get(field) if block else None
    return value


def plot_fragility(summary):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    series = (
        ("all_to_all", "triggered_shift_rate", "all-to-all, triggered"),
        ("all_to_all", "clean_shift_rate", "all-to-all, clean twin"),
        ("all_to_one", "triggered_shift_rate", "all-to-one, triggered"),
        ("all_to_one", "clean_shift_rate", "all-to-one, clean twin"),
    )
    plotted = {}
    for axis, (architecture, placements) in zip(axes, MECHANISM_PLACEMENTS.items()):
        positions = np.arange(len(placements))
        width = 0.2
        for index, (label_mode, field, label) in enumerate(series):
            values = [
                mechanism_value(
                    summary, architecture, label_mode, placement, rule, field
                )
                for placement, rule, _ in placements
            ]
            plotted[f"{architecture}/{label}"] = values
            heights = [np.nan if value is None else value for value in values]
            axis.bar(
                positions + (index - 1.5) * width,
                heights,
                width,
                label=label,
                color=OKABE_ITO[index],
            )
        axis.set_xticks(positions, [name for _, _, name in placements], rotation=20)
        axis.set_title("ViT-B/16" if architecture == "vit" else "Swin-S")
        axis.set_ylabel("share of passes whose prediction moved")
    axes[0].legend()
    plotted["placements"] = {
        architecture: [list(entry) for entry in placements]
        for architecture, placements in MECHANISM_PLACEMENTS.items()
    }
    save("fragility", figure, ["mechanism.json"], plotted)


def plot_destinations(summary):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    series = (
        ("all_to_all", "triggered_shifts_to_source", "all-to-all, triggered to source"),
        ("all_to_one", "triggered_shifts_to_source", "all-to-one, triggered to source"),
        ("all_to_all", "clean_shifts_to_one_below", "all-to-all, clean to class below"),
        ("all_to_all", "uniform_destination_chance", "uniform chance"),
    )
    plotted = {}
    for axis, (architecture, placements) in zip(axes, MECHANISM_PLACEMENTS.items()):
        positions = np.arange(len(placements))
        width = 0.2
        for index, (label_mode, field, label) in enumerate(series):
            values = [
                mechanism_value(
                    summary, architecture, label_mode, placement, rule, field
                )
                for placement, rule, _ in placements
            ]
            plotted[f"{architecture}/{label}"] = values
            heights = [np.nan if value is None else value for value in values]
            axis.bar(
                positions + (index - 1.5) * width,
                heights,
                width,
                label=label,
                color=OKABE_ITO[index],
            )
        axis.set_xticks(positions, [name for _, _, name in placements], rotation=20)
        axis.set_title("ViT-B/16" if architecture == "vit" else "Swin-S")
        axis.set_ylabel("share of shifted passes")
    axes[0].legend()
    save("destinations", figure, ["mechanism.json"], plotted)


def plot_tokens(models):
    groups = {
        "all-to-all": [m for m in models if m["label_mode"] == "all_to_all"],
        "all-to-one": [m for m in models if m["label_mode"] == "all_to_one"],
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    plotted = {}
    for axis, (group, members) in zip(axes, groups.items()):
        positions = np.arange(len(TOKEN_CONDITIONS))
        bottom = np.zeros(len(TOKEN_CONDITIONS))
        for index, (field, label) in enumerate(TOKEN_OUTCOMES):
            values = [
                float(np.mean([m["conditions"][condition][field] for m in members]))
                if members
                else np.nan
                for condition, _ in TOKEN_CONDITIONS
            ]
            plotted[f"{group}/{label}"] = values
            axis.bar(
                positions, values, bottom=bottom, label=label, color=OKABE_ITO[index]
            )
            bottom = bottom + np.nan_to_num(values)
        axis.set_xticks(positions, [name for _, name in TOKEN_CONDITIONS], rotation=45)
        axis.set_title(f"{group}, {len(members)} ViT models")
        axis.set_ylabel("share of captured triggered images")
        plotted[f"{group}/models"] = [m["folder"] for m in members]
    axes[0].legend()
    plotted["conditions"] = [list(entry) for entry in TOKEN_CONDITIONS]
    save("tokens", figure, ["tokens.json"], plotted)


def plot_detectors(detection):
    names = detection["statistics"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    plotted = {}
    for axis, architecture in zip(axes, ("vit", "swin")):
        rows = np.arange(len(names))
        for index, (population, label) in enumerate(DETECTOR_POPULATIONS):
            block = detection["summary"].get(f"{architecture}/{population}/all", {})
            means = [
                block[name]["mean_auroc"] if name in block else np.nan for name in names
            ]
            lows = [
                block[name]["mean_auroc_ci"][0] if name in block else np.nan
                for name in names
            ]
            highs = [
                block[name]["mean_auroc_ci"][1] if name in block else np.nan
                for name in names
            ]
            counts = [block[name]["n"] if name in block else 0 for name in names]
            plotted[f"{architecture}/{population}"] = {
                "mean_auroc": means,
                "ci_low": lows,
                "ci_high": highs,
                "n": counts,
            }
            offset = (index - 1.5) * 0.18
            errors = np.array([np.subtract(means, lows), np.subtract(highs, means)])
            axis.errorbar(
                means,
                rows + offset,
                xerr=errors,
                fmt="o",
                label=label,
                color=OKABE_ITO[index],
                markersize=4,
            )
        axis.axvline(0.5, color="grey", linewidth=0.8)
        axis.set_yticks(rows, names)
        axis.set_xlim(0, 1)
        axis.set_xlabel("mean AUROC, low score flagged")
        axis.set_title("ViT-B/16" if architecture == "vit" else "Swin-S")
    axes[0].legend()
    plotted["statistics"] = names
    save("detectors", figure, ["detection.json"], plotted)


def save(name, figure, inputs, plotted):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=130)
    plt.close(figure)
    sources = [
        os.path.join("results/_experiments/all_to_all_detection", i) for i in inputs
    ]
    figure_sidecar(
        os.path.join(FIGURE_DIR, f"{name}.json"), GENERATOR, sources, plotted
    )


if __name__ == "__main__":
    main()

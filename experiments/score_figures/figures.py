"""The per-model ROC and histogram, the set mean ROC and the labels every figure shares."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.score_figures import config  # noqa: E402
from scripts.paper._common import OKABE_ITO  # noqa: E402

# Single probes are dashed and fusions solid. A fused combination draws both
# rules in its own color, the weighted rule marked with dots.
RULE_STYLES = {"psu": "--", "min": "-", "weighted": "-"}
RULE_MARKERS = {"psu": None, "min": None, "weighted": "o"}
THRESHOLD_STYLES = (":", "--", "-.", "-")
# Okabe-Ito without its yellow, which is unreadable as a thin line on white.
LINE_COLORS = tuple(color for color in OKABE_ITO if color != "#F0E442")
ATTACK_NAMES = {
    "badnet_a2o": "BadNets",
    "blend": "Blend",
    "lf": "LF",
    "bpp": "BPP",
    "wanet": "WaNet",
    "tact": "TaCT",
    "lc": "LC",
    "sig": "SIG",
    "adaptive_blend": "Adaptive-Blend",
    "blind": "Blind",
    "ssba": "SSBA",
    "trojannn": "TrojanNN",
    "inputaware": "Input-Aware",
}
DATASET_NAMES = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "gtsrb": "GTSRB",
    "tiny": "Tiny ImageNet",
}


def apply_style():
    plt.rcParams.update(
        {
            "font.size": config.FONT_SIZE,
            "axes.titlesize": config.TITLE_SIZE,
            "axes.labelsize": config.FONT_SIZE,
            "xtick.labelsize": config.FONT_SIZE - 1,
            "ytick.labelsize": config.FONT_SIZE - 1,
            "legend.fontsize": config.LEGEND_SIZE,
            "figure.titlesize": config.TITLE_SIZE,
            "lines.linewidth": config.LINE_WIDTH,
        }
    )


apply_style()


def probe_label(architecture, name):
    label = config.PROBE_LABELS[architecture].get(name, name)
    return label


def combination_label(architecture, key, rule):
    names = key.split("+")
    if rule == "psu":
        return probe_label(architecture, names[0])
    short = " + ".join(config.PROBE_SHORT_NAMES[architecture].get(n, n) for n in names)
    if rule == "weighted":
        share = config.WEIGHTED_FIRST_SHARE
        rest = (1 - share) / (len(names) - 1)
        return f"{short}, weighted {share:g}/{rest:.2g}"
    return f"{short}, plain min"


def model_title(numbers):
    attack = ATTACK_NAMES.get(numbers["attack"], numbers["attack"])
    dataset = DATASET_NAMES.get(numbers["dataset"], numbers["dataset"])
    rate = numbers["poison_rate"]
    rate_text = f"{rate * 100:g}% poisoning" if rate is not None else "rate unknown"
    architecture = "Swin-S" if numbers["architecture"] == "swin" else "ViT-B/16"
    title = f"{architecture}, {attack} on {dataset}, {rate_text}"
    return title


def draw_model_roc(path, title, fpr_grid, curves, combination_order):
    figure, axis = plt.subplots(figsize=(9, 7))
    for key, rule, label, tpr in curves:
        axis.plot(
            fpr_grid,
            tpr,
            color=combination_color(key, combination_order),
            ls=RULE_STYLES[rule],
            marker=RULE_MARKERS[rule],
            markevery=8,
            ms=5,
            label=label,
        )
    style_roc_axis(axis, title)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    save(figure, path)


def draw_mean_roc(path, title, fpr_grid, mean_curves, combination_order, labels):
    figure, axis = plt.subplots(figsize=(9, 7))
    ordered = sorted(
        mean_curves, key=lambda curve: (combination_order.index(curve[0]), curve[1])
    )
    for key, rule, n_models, mean_tpr in ordered:
        axis.plot(
            fpr_grid,
            mean_tpr,
            color=combination_color(key, combination_order),
            ls=RULE_STYLES[rule],
            marker=RULE_MARKERS[rule],
            markevery=8,
            ms=5,
            label=f"{labels[(key, rule)]} ({n_models} models)",
        )
    style_roc_axis(axis, title)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    save(figure, path)


def style_roc_axis(axis, title):
    for budget in config.BUDGETS:
        axis.axvline(budget, color="0.6", ls="--", lw=1)
        axis.text(budget, 0.02, f" {budget * 100:g}%", color="0.3")
    axis.set_xscale("log")
    axis.set_xlim(config.ROC_LOWEST_FPR, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title(title)
    axis.set_xlabel("false-positive budget (clean-validation quantile, log scale)")
    axis.set_ylabel("TPR, share of triggered inputs flagged")
    axis.grid(alpha=0.3)


def draw_histogram(path, title, x_label, validation, backdoor, edges, thresholds):
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.hist(
        validation,
        bins=edges,
        density=True,
        alpha=0.55,
        color=OKABE_ITO[0],
        label="clean validation",
    )
    axis.hist(
        backdoor,
        bins=edges,
        density=True,
        alpha=0.55,
        color=OKABE_ITO[1],
        label="triggered",
    )
    for (budget, threshold), style in zip(thresholds.items(), THRESHOLD_STYLES):
        axis.axvline(
            threshold,
            color="k",
            ls=style,
            lw=1.6,
            label=f"threshold at {budget * 100:g}% FPR",
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("density (log scale)")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    save(figure, path)


def combination_color(key, combination_order):
    color = LINE_COLORS[combination_order.index(key) % len(LINE_COLORS)]
    return color


def save(figure, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(figure)

"""The 3 figure kinds make.py draws: a per-model ROC, a per-model histogram and a set mean ROC."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.score_figures import config  # noqa: E402
from scripts.paper._common import OKABE_ITO  # noqa: E402

# A fused combination draws its rules in the color of the combination, so the
# rule is told apart by the line style.
RULE_STYLES = {"psu": "-", "min": "-", "weighted": "--"}
THRESHOLD_STYLES = (":", "--", "-.", "-")
# Okabe-Ito without its yellow, which is unreadable as a thin line on white.
LINE_COLORS = tuple(color for color in OKABE_ITO if color != "#F0E442")


def draw_model_roc(path, title, fpr_grid, curves, combination_order):
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for key, rule, label, tpr in curves:
        axis.plot(
            fpr_grid,
            tpr,
            color=combination_color(key, combination_order),
            ls=RULE_STYLES[rule],
            lw=1.8,
            label=label,
        )
    style_roc_axis(axis, title)
    save(figure, path)


def draw_mean_roc(path, title, fpr_grid, mean_curves, combination_order):
    figure, axis = plt.subplots(figsize=(8, 5.5))
    ordered = sorted(
        mean_curves, key=lambda curve: (combination_order.index(curve[0]), curve[1])
    )
    for key, rule, n_models, mean_tpr in ordered:
        label = key if rule == "psu" else f"{key}, {rule}"
        axis.plot(
            fpr_grid,
            mean_tpr,
            color=combination_color(key, combination_order),
            ls=RULE_STYLES[rule],
            lw=1.8,
            label=f"{label} ({n_models} models)",
        )
    style_roc_axis(axis, title)
    save(figure, path)


def style_roc_axis(axis, title):
    for budget in config.BUDGETS:
        axis.axvline(budget, color="0.6", ls="--", lw=0.8)
        axis.text(budget, 0.02, f" {budget * 100:g}%", color="0.4", fontsize=8)
    axis.set_xscale("log")
    axis.set_xlim(config.ROC_LOWEST_FPR, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("false-positive rate (clean validation quantile, log scale)")
    axis.set_ylabel("TPR")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=7)


def draw_histogram(path, title, x_label, validation, backdoor, edges, thresholds):
    figure, axis = plt.subplots(figsize=(7, 4.5))
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
            lw=1,
            label=f"threshold at {budget * 100:g}% FPR",
        )
    axis.set_title(title, fontsize=9)
    axis.set_xlabel(x_label, fontsize=8)
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=7)
    save(figure, path)


def combination_color(key, combination_order):
    color = LINE_COLORS[combination_order.index(key) % len(LINE_COLORS)]
    return color


def save(figure, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=config.DPI)
    plt.close(figure)

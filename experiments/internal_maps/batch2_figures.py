"""CPU stage of the second batch: figures and the survival model from batch2.py's records.

.venv/bin/python -m experiments.internal_maps.batch2_figures
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from defenses.decision import RECOMMENDED_PLACEMENT  # noqa: E402
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.cache_readouts.shared import load_baselines, load_passes  # noqa: E402
from experiments.internal_maps import config  # noqa: E402
from experiments.internal_maps.measure import write_json  # noqa: E402

SLUG = "internal_maps"
LATE_BLOCKS = slice(8, 12)
BUDGETS = (0.01, 0.05, 0.10)
# The cache's PSU reads 3 passes per image, so the prediction draws 3 too.
CACHE_PASSES = 3
SIMULATED_DRAWS = 20
TRIGGER_SIZES = (1, 4, 9, 16)
FIT_RATE = "0.5"


def main():
    root = experiment_results_dir(SLUG)
    vit_dir = os.path.join(root, "vit")
    confidence = read_records(vit_dir, config.CONFIDENCE_MODELS, "confidence.json")
    veto = read_records(vit_dir, config.VETO_MODELS, "veto.json")
    survival = read_records(vit_dir, config.SURVIVAL_MODELS, "survival.json")

    for record in confidence:
        draw_confidence(
            os.path.join(vit_dir, record["folder"], "confidence_gain.png"), record
        )
    for record in veto:
        draw_veto(os.path.join(vit_dir, record["folder"], "tact_veto.png"), record)
    survival_summary = {}
    for record in survival:
        analysis = analyze_survival(record)
        survival_summary[record["folder"]] = analysis
        draw_survival(
            os.path.join(vit_dir, record["folder"], "survival_against_kept.png"),
            record,
            analysis,
        )
    if survival:
        write_json(os.path.join(root, "survival_numbers.json"), survival_summary)
    write_json(
        os.path.join(root, "batch2_numbers.json"),
        {
            "confidence": {r["folder"]: confidence_summary(r) for r in confidence},
            "veto": {r["folder"]: veto_summary(r) for r in veto},
            "survival": {
                folder: {k: v for k, v in a.items() if k != "binned"}
                for folder, a in survival_summary.items()
            },
        },
    )
    print(
        f"[ok] {len(confidence)} confidence, {len(veto)} veto, {len(survival)} survival"
    )


def read_records(directory, folders, name):
    records = []
    for folder in folders:
        path = os.path.join(directory, folder, name)
        if os.path.exists(path):
            with open(path) as handle:
                records.append(json.load(handle))
    return records


def confidence_summary(record):
    summary = {
        "negative_psu_validation_count": record["negative_psu_validation_count"],
        "validation_count": record["validation_count"],
        "rate": record["rate"],
    }
    for name, group in record["groups"].items():
        if not group["count"]:
            summary[name] = {"count": 0}
            continue
        against = group["against_attention"]
        summary[name] = {
            "count": group["count"],
            "mean_psu": mean(group["psu"]),
            "mean_base_probability": mean(group["base_probability"]),
            "mean_largest_single_rise": mean(group["largest_single_rise"]),
            "mean_total_rise": mean(against["total_rise"]),
            "mean_total_drop": mean(against["total_drop"]),
            "rise_share_on_top_attention_quarter": mean(
                against["rise_share_on_top_attention_quarter"]
            ),
            "rise_weighted_attention_rank": mean(
                against["rise_weighted_attention_rank"]
            ),
            "drop_weighted_attention_rank": mean(
                against["drop_weighted_attention_rank"]
            ),
            "rise_share_on_border": mean(against["rise_share_on_border"]),
            "border_area_share": against["border_area_share"],
            "joint_top_k": group["joint_top_k"],
        }
    return summary


def veto_summary(record):
    refused = record["refused"]
    summary = {
        "source_classes": record["source_classes"],
        "non_source_stamped": record["non_source_stamped"],
        "non_source_sent_to_target": record["non_source_sent_to_target"],
        "refused_count": record["refused_count"],
        "mean_largest_single_rise": mean(refused["largest_single_rise"]),
        "median_units_for_half_rise": float(np.median(refused["units_for_half_rise"])),
        "mean_rise_share_on_trigger": mean(refused["rise_share_on_trigger"]),
        "joint_top_k": refused["joint_top_k"],
        "unstamped_joint_top_k": refused["unstamped_joint_top_k"],
        "source_mean_largest_single_drop": mean(
            record["source"]["largest_single_drop"]
        ),
        "content_hidden_probe": record["content_hidden_probe"],
    }
    return summary


# Survival as a function of the share of trigger tokens left visible, per pass.
# "Kept" is the mean over the last 4 blocks of the visible share of the trigger's
# tokens, since the class token reads a patch trigger in blocks 9 to 12
# (why_token_masking_works part A). The all-block mean is stored beside it.
def analyze_survival(record):
    trigger_tokens = record["trigger_tokens"]
    rows = []
    for rate, reading in record["by_rate"].items():
        visible = np.array(reading["visible_per_block"], dtype=float)  # (k, n, 12)
        kept_late = visible[:, :, LATE_BLOCKS].mean(axis=2) / trigger_tokens  # (k, n)
        kept_all = visible.mean(axis=2) / trigger_tokens  # (k, n)
        survived = np.array(reading["survived"], dtype=float)  # (k, n)
        ratio = np.clip(np.array(reading["target_ratio"], dtype=float), 0, None)
        rows.append((float(rate), kept_late, kept_all, survived, ratio))

    binned = {}
    for rate, kept_late, _, survived, _ in rows:
        values = np.round(kept_late.flatten(), 4)
        binned[str(rate)] = [
            [
                float(v),
                float(survived.flatten()[values == v].mean()),
                int((values == v).sum()),
            ]
            for v in np.unique(values)
        ]

    all_kept = np.concatenate([r[1].flatten() for r in rows])
    all_survived = np.concatenate([r[3].flatten() for r in rows])
    survival_fit = fit_logistic(all_kept, all_survived)
    fit_row = [r for r in rows if str(r[0]) == FIT_RATE][0]
    ratio_fit = fit_logistic(fit_row[1].flatten(), fit_row[4].flatten())

    prediction = {}
    for rate, kept_late, _, survived, ratio in rows:
        measured = measured_tpr(record["folder"], rate, record["target_label"])
        predicted = predicted_tpr(record["folder"], rate, trigger_tokens, ratio_fit)
        prediction[str(rate)] = {
            "measured_tpr_cache": measured,
            "predicted_tpr": predicted,
            "survival_rate": float(survived.mean()),
            "mean_kept_late": float(kept_late.mean()),
            "measured_mean_ratio": float(ratio.mean()),
        }
    by_size = {
        str(size): {
            str(rate): predicted_tpr(record["folder"], rate, size, ratio_fit)
            for rate, *_ in rows
        }
        for size in TRIGGER_SIZES
    }
    analysis = {
        "trigger_tokens": trigger_tokens,
        "kept_definition": "mean over blocks 9 to 12 of the visible share of trigger tokens",
        "survival_logistic": survival_fit,
        "ratio_logistic_fit_rate": FIT_RATE,
        "ratio_logistic": ratio_fit,
        "by_rate": prediction,
        "predicted_tpr_by_trigger_size": by_size,
        "binned": binned,
    }
    return analysis


def logistic(x, intercept, slope):
    value = 1.0 / (1.0 + np.exp(-(intercept + slope * x)))
    return value


def fit_logistic(kept, outcome):
    if outcome.min() == outcome.max():
        constant = {"intercept": None, "slope": None, "constant": float(outcome.mean())}
        return constant
    parameters, _ = curve_fit(logistic, kept, outcome, p0=(-2.0, 6.0), maxfev=20000)
    fitted = {"intercept": float(parameters[0]), "slope": float(parameters[1])}
    return fitted


def evaluate_fit(fit, x):
    if fit.get("constant") is not None:
        return np.full_like(np.asarray(x, dtype=float), fit["constant"])
    values = logistic(np.asarray(x, dtype=float), fit["intercept"], fit["slope"])
    return values


# The cache's own TPR at this rate: fractional PSU of the hits against the
# quantile thresholds of the clean validation scores.
def measured_tpr(folder, rate, target):
    psbd_dir = os.path.join("results", folder, "psbd")
    baselines = load_baselines(psbd_dir)
    passes = load_passes(psbd_dir, RECOMMENDED_PLACEMENT, rate, baselines)
    scores = {
        split: psu_ratio_from_cache(
            passes[split]["baseline_probs"],
            passes[split]["baseline_labels"],
            passes[split]["per_pass_probs"],
        )
        for split in ("validation", "backdoor")
    }
    hits = passes["backdoor"]["baseline_labels"] == target
    reading = {}
    for budget in BUDGETS:
        threshold = torch.quantile(scores["validation"], budget)
        reading[f"q{budget:.2f}"] = float(
            (scores["backdoor"][hits] <= threshold).float().mean()
        )
    return reading


# The predicted TPR when survival depends only on how many trigger tokens stay
# visible. Every pass draws each late block's visible trigger tokens as
# Binomial(m, 1 - rate), independently, maps the kept share through the ratio
# curve fitted at rate 0.5, and PSU is 1 minus the mean over 3 passes, as in the
# cache. The thresholds are the cache's clean validation quantiles at that rate.
def predicted_tpr(folder, rate, trigger_tokens, ratio_fit):
    psbd_dir = os.path.join("results", folder, "psbd")
    baselines = load_baselines(psbd_dir)
    passes = load_passes(psbd_dir, RECOMMENDED_PLACEMENT, rate, baselines)
    validation = psu_ratio_from_cache(
        passes["validation"]["baseline_probs"],
        passes["validation"]["baseline_labels"],
        passes["validation"]["per_pass_probs"],
    )
    generator = np.random.default_rng(int(rate * 1000) + trigger_tokens)
    draws = generator.binomial(
        trigger_tokens, 1.0 - rate, size=(SIMULATED_DRAWS * 1000, CACHE_PASSES, 4)
    )  # (images, passes, late blocks)
    kept = draws.mean(axis=2) / trigger_tokens  # (images, passes)
    psu = 1.0 - evaluate_fit(ratio_fit, kept).mean(axis=1)  # (images,)
    reading = {}
    for budget in BUDGETS:
        threshold = float(torch.quantile(validation, budget))
        reading[f"q{budget:.2f}"] = float((psu <= threshold).mean())
    return reading


def draw_survival(path, record, analysis):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axis = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(analysis["binned"])))
    for color, (rate, points) in zip(colors, analysis["binned"].items()):
        kept = [p[0] for p in points]
        survived = [p[1] for p in points]
        sizes = [10 + 200 * p[2] / max(q[2] for q in points) for p in points]
        axis.scatter(
            kept, survived, s=sizes, color=color, alpha=0.7, label=f"rate {rate}"
        )
    grid = np.linspace(0, 1, 101)
    axis.plot(
        grid,
        evaluate_fit(analysis["survival_logistic"], grid),
        color="black",
        lw=1.5,
        label="logistic fit, all rates",
    )
    axis.set_xlabel("kept: visible share of trigger tokens, mean over blocks 9 to 12")
    axis.set_ylabel("P(triggered prediction survives the pass)")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    axis.set_title(
        f"{record['folder']}, {record['trigger_tokens']} trigger tokens", fontsize=10
    )

    axis = axes[1]
    rates = sorted(float(r) for r in analysis["by_rate"])
    for budget, style in zip(BUDGETS, (":", "--", "-")):
        key = f"q{budget:.2f}"
        measured = [
            analysis["by_rate"][str(r)]["measured_tpr_cache"][key] for r in rates
        ]
        predicted = [analysis["by_rate"][str(r)]["predicted_tpr"][key] for r in rates]
        axis.plot(
            rates,
            measured,
            color="#D55E00",
            ls=style,
            marker="o",
            label=f"measured, {budget:.0%} FPR",
        )
        axis.plot(
            rates,
            predicted,
            color="#0072B2",
            ls=style,
            marker="s",
            label=f"predicted, {budget:.0%} FPR",
        )
    axis.set_xlabel("PSBD-TM rate")
    axis.set_ylabel("TPR")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7.5)
    axis.set_title(
        f"TPR from kept share alone (ratio curve fitted at rate {FIT_RATE})",
        fontsize=10,
    )
    figure.tight_layout()
    save(figure, path)


def draw_confidence(path, record):
    groups = [
        (name, record["groups"][name])
        for name in record["groups"]
        if record["groups"][name]["count"]
    ]
    shown = config.MAPS_DRAWN
    figure, axes = plt.subplots(
        2 * len(groups), shown, figsize=(2.5 * shown, 5.4 * len(groups))
    )
    for index, (name, group) in enumerate(groups):
        pixels = record["example_pixels"][name]
        for column in range(shown):
            image_axis = axes[2 * index][column]
            map_axis = axes[2 * index + 1][column]
            if column >= len(pixels):
                image_axis.axis("off")
                map_axis.axis("off")
                continue
            image_axis.imshow(np.array(pixels[column], dtype=np.uint8))
            image_axis.set_title(
                f"{name.replace('_', ' ')}\ntrue {group['labels'][column]}, pred {group['predicted'][column]}"
                f"\nP_c {group['base_probability'][column]:.2f}, PSU {group['psu'][column]:+.2f}",
                fontsize=7.5,
            )
            change = np.array(group["change_maps"][column])
            limit = max(np.abs(change).max(), 1e-4)
            image = map_axis.imshow(change, cmap="RdBu_r", vmin=-limit, vmax=limit)
            attention = np.array(group["attention_maps"][column])
            map_axis.contour(
                attention >= np.quantile(attention, 0.75),
                levels=[0.5],
                colors="k",
                linewidths=0.8,
            )
            map_axis.set_title(
                "change in P_c, 1 token hidden\n(contour: top quarter of class-token attention)",
                fontsize=7,
            )
            plt.colorbar(image, ax=map_axis, fraction=0.046)
            for axis in (image_axis, map_axis):
                axis.set_xticks([])
                axis.set_yticks([])
    summary = confidence_summary(record)
    lines = []
    for name, _ in groups:
        s = summary[name]
        joint = s["joint_top_k"]
        lines.append(
            f"{name}: {s['count']} images, mean P_c {s['mean_base_probability']:.2f},"
            f" rise share on top attention quarter {s['rise_share_on_top_attention_quarter']:.2f},"
            f" on the border ring {s['rise_share_on_border']:.2f} (area {s['border_area_share']:.2f}),"
            f" top 10 rising hidden: P_c {joint['10']['top']['probability']:.2f}"
            f" (10 random: {joint['10']['random']['probability']:.2f})"
        )
    figure.suptitle(
        f"{record['folder']}: which tokens raise the prediction when hidden (PSBD-TM rate {record['rate']})",
        fontsize=10,
    )
    figure.text(0.5, 0.005, "\n".join(lines), ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(figure, path)


def draw_veto(path, record):
    refused = record["refused"]
    shown = min(config.MAPS_DRAWN, len(refused["change_maps"]))
    figure = plt.figure(figsize=(2.6 * (shown + 2), 8.4))
    grid_spec = figure.add_gridspec(3, shown + 2)
    for column in range(shown):
        axis = figure.add_subplot(grid_spec[0, column])
        axis.imshow(
            np.array(record["example_pixels"]["stamped"][column], dtype=np.uint8)
        )
        axis.set_title(
            f"stamped non-source\nP(target) {refused['base_target_probability'][column]:.3f}",
            fontsize=7.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis = figure.add_subplot(grid_spec[1, column])
        change = np.array(refused["change_maps"][column])
        limit = max(np.abs(change).max(), 1e-4)
        image = axis.imshow(change, cmap="RdBu_r", vmin=-limit, vmax=limit)
        mark_units(axis, record["trigger_units"])
        axis.set_title("change in P(target),\n1 token hidden", fontsize=7.5)
        axis.set_xticks([])
        axis.set_yticks([])
        plt.colorbar(image, ax=axis, fraction=0.046)
    for row, (title, values) in enumerate(
        (
            ("mean over refused images", refused["change_maps_mean"]),
            ("source images, mean", record["source"]["change_maps_mean"]),
        )
    ):
        axis = figure.add_subplot(grid_spec[row, shown : shown + 2])
        values = np.array(values)
        limit = max(np.abs(values).max(), 1e-4)
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
        mark_units(axis, record["trigger_units"])
        axis.set_title(f"change in P(target), 1 token hidden,\n{title}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        plt.colorbar(image, ax=axis, fraction=0.046)

    axis = figure.add_subplot(grid_spec[2, :])
    counts = [int(k) for k in refused["joint_top_k"]]
    for key, label, color in (
        ("joint_top_k", "stamped, own top-k rising tokens hidden", "#D55E00"),
        ("unstamped_joint_top_k", "unstamped twin, the same tokens hidden", "#0072B2"),
    ):
        top = [refused[key][str(k)]["top"]["argmax_is_class"] for k in counts]
        random = [refused[key][str(k)]["random"]["argmax_is_class"] for k in counts]
        axis.plot(counts, top, color=color, marker="o", label=label)
        axis.plot(
            counts,
            random,
            color=color,
            ls=":",
            marker="x",
            label=label.split(",")[0] + ", k random tokens hidden",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("tokens hidden in every block (k)")
    axis.set_ylabel("share sent to the target")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7.5)
    probe = record["content_hidden_probe"]
    axis.set_title(
        f"{record['non_source_stamped']} stamped non-source images, share sent to the target"
        f" {record['non_source_sent_to_target']:.2f}, {record['refused_count']} refused ones mapped."
        f" Content-hidden probe: AUROC {probe['auroc']:.3f}, TPR at 1/5/10% FPR"
        f" {probe['q0.01']['tpr']:.2f}/{probe['q0.05']['tpr']:.2f}/{probe['q0.10']['tpr']:.2f}",
        fontsize=9,
    )
    figure.suptitle(
        f"{record['folder']}: the veto of a stamped non-source image (sources {record['source_classes']}, target {record['target_label']})",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save(figure, path)


def mark_units(axis, units):
    for unit in units:
        row, column = divmod(unit, 14)
        axis.add_patch(
            plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, ec="lime", lw=1)
        )


def mean(values):
    average = float(np.mean(values))
    return average


def save(figure, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=config.DPI)
    plt.close(figure)


if __name__ == "__main__":
    main()

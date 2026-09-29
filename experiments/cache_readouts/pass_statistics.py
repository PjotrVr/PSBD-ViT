"""X19 and N17: which statistic over PSBD's perturbed passes separates best.

Reads per_pass_probs and per_pass_argmax of PSBD-TM and PSBD-RD at the adaptive
rate and scores every image with 6 statistics, all oriented so that low means
poisoned. The mean fractional PSU is the headline statistic and must reproduce
psbd_metrics.json exactly, which is the control of the whole readout.

    .venv/bin/python -m experiments.cache_readouts.pass_statistics --set dev
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defenses.cache import read_split_manifest  # noqa: E402
from defenses.decision import (  # noqa: E402
    HEADLINE_QUANTILE,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments.cache_readouts.shared import (  # noqa: E402
    FIGURES_DIR,
    MODEL_SETS,
    choose_rate,
    evaluate_scores,
    load_baselines,
    load_model_set,
    load_passes,
    model_label,
    ordered_attacks,
    output_path,
    paired_summary,
    psbd_dir_of,
    validation_shift_by_rate,
    write_json,
)
from scripts.paper._common import OKABE_ITO, figure_sidecar, rate_row, save_figure  # noqa: E402

PLACEMENTS = {"psbd_tm": RECOMMENDED_PLACEMENT, "psbd_rd": PUBLISHED_PLACEMENT}
STATISTICS = (
    "mean_psu",
    "worst_pass",
    "best_pass",
    "shift_count",
    "shift_count_tiebroken",
    "spread",
)
REFERENCE_STATISTIC = "mean_psu"
# 1%, 5% and 10% FPR are the operating points the task asks for. The headline
# 0.25 rides along only so the control can compare against the stored record.
QUANTILES = (0.01, 0.05, 0.10, HEADLINE_QUANTILE)
REPORTED_QUANTILES = ("q0.01", "q0.05", "q0.10")
# Below this many passes a worst or best pass and a spread are not defined.
MIN_PASSES = 2


def main():
    args = parse_args()
    started = time.perf_counter()

    models = load_model_set(args.set)
    rows = [measure_model(model) for model in models]
    summary = summarize(rows)

    payload = {
        "experiment": "X19 and N17, pass statistics",
        "model_set": args.set,
        "statistics": list(STATISTICS),
        "reference_statistic": REFERENCE_STATISTIC,
        "placements": PLACEMENTS,
        "quantiles": list(QUANTILES),
        "control": control_summary(rows),
        "summary": summary,
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    path = output_path("pass_statistics", args.set)
    write_json(payload, path)
    plot(summary, args.set, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=MODEL_SETS, required=True)
    args = parser.parse_args()
    return args


def measure_model(model):
    psbd_dir = psbd_dir_of(model)
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)

    row = model_label(model)
    row["placements"] = {}
    for short_name, placement in PLACEMENTS.items():
        shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
        rate = choose_rate(shift_by_rate, "adaptive")
        if rate is None:
            row["placements"][short_name] = {"rate": None}
            continue

        passes = load_passes(psbd_dir, placement, rate, baselines)
        scores = {split: pass_statistics(passes[split]) for split in passes}
        evaluations = {
            statistic: evaluate_scores(
                {split: scores[split][statistic] for split in scores},
                manifest,
                QUANTILES,
            )
            for statistic in STATISTICS
        }
        row["placements"][short_name] = {
            "rate": rate,
            "passes": int(passes["validation"]["per_pass_probs"].shape[0]),
            "validation_shift": shift_by_rate[rate],
            "statistics": evaluations,
            "control": control_against_record(model, placement, rate, evaluations),
        }
    return row


def pass_statistics(split_passes):
    baseline_probs = split_passes["baseline_probs"]  # (n, num_classes)
    baseline_labels = split_passes["baseline_labels"]  # (n,)
    per_pass_probs = split_passes["per_pass_probs"].float()  # (passes, n)
    per_pass_argmax = split_passes["per_pass_argmax"].long()  # (passes, n)
    assert per_pass_probs.shape[0] >= MIN_PASSES, per_pass_probs.shape

    mean_psu = psu_ratio_from_cache(
        baseline_probs, baseline_labels, per_pass_probs
    )  # (n,)

    # The same clamp psu_ratio_from_cache applies, so every statistic divides by
    # the identical starting confidence and differs only in how it pools passes.
    tracked = baseline_probs.gather(1, baseline_labels.view(-1, 1).long()).squeeze(1)
    tracked = tracked.float().clamp_min(1e-6)  # (n,)
    retained = per_pass_probs / tracked.view(1, -1)  # (passes, n)

    # The worst pass is the largest fractional drop, the best the smallest. A
    # triggered patch input that survives every pass keeps both low.
    worst_pass = 1.0 - retained.min(dim=0).values  # (n,)
    best_pass = 1.0 - retained.max(dim=0).values  # (n,)

    shifted = per_pass_argmax != baseline_labels.view(1, -1).long()  # (passes, n)
    shift_count = shifted.float().sum(dim=0)  # (n,)

    # With 3 passes the count takes 4 values, so a quantile threshold lands on a
    # tie and flags nothing. Adding the mean PSU scaled into (-0.25, 0.25) keeps
    # the count's order and breaks ties inside a count by the soft score.
    shift_count_tiebroken = shift_count + mean_psu.clamp(-1.0, 1.0) / 4.0  # (n,)

    # The sign is fixed before any triggered score is read: under the OR account
    # a triggered input keeps its label on every pass, so its passes agree and
    # its spread is low.
    spread = retained.std(dim=0, unbiased=False)  # (n,)

    statistics = {
        "mean_psu": mean_psu,
        "worst_pass": worst_pass,
        "best_pass": best_pass,
        "shift_count": shift_count,
        "shift_count_tiebroken": shift_count_tiebroken,
        "spread": spread,
    }
    return statistics


def control_against_record(model, placement, rate, evaluations):
    block = model["report"]["placements"][placement]
    record_row = rate_row(block, rate)
    recorded = record_row["detection_psu_ratio"]
    computed = evaluations[REFERENCE_STATISTIC]

    comparisons = {"auroc": (computed["auroc"], recorded["q0.25"]["auroc"])}
    for key in REPORTED_QUANTILES:
        comparisons[f"tpr_{key}"] = (
            computed["at_fpr"][key]["tpr"],
            recorded[key]["tpr"],
        )
        comparisons[f"fpr_{key}"] = (
            computed["at_fpr"][key]["realized_fpr"],
            recorded[key]["fpr"],
        )
    control = {
        "recorded_adaptive_rate": block.get("adaptive_rate"),
        "rate_matches_record": block.get("adaptive_rate") == rate,
        "exact": all(a == b for a, b in comparisons.values()),
        "pairs": comparisons,
    }
    return control


def control_summary(rows):
    checks = [
        row["placements"][name]["control"]
        for row in rows
        for name in PLACEMENTS
        if row["placements"][name].get("rate") is not None
    ]
    summary = {
        "n_checked": len(checks),
        "n_exact": sum(check["exact"] for check in checks),
        "n_rate_matches": sum(check["rate_matches_record"] for check in checks),
    }
    return summary


def summarize(rows):
    # Only successful backdoors are pooled. A dev model below the 2-point bar
    # stays in the per-model rows under its label.
    pooled = [row for row in rows if row["successful_2pt"]]
    summary = {
        "pooled_folders": [row["folder"] for row in pooled],
        "unpooled_folders": [
            row["folder"] for row in rows if not row["successful_2pt"]
        ],
    }
    for name in PLACEMENTS:
        covered = [
            row for row in pooled if row["placements"][name].get("rate") is not None
        ]
        summary[name] = {
            "n_models": len(covered),
            "missing": [row["folder"] for row in pooled if row not in covered],
            "all": summarize_group(covered, name),
            "by_attack": {
                attack: summarize_group(
                    [row for row in covered if row["attack"] == attack], name
                )
                for attack in ordered_attacks({row["attack"] for row in covered})
            },
        }
    return summary


def summarize_group(rows, name):
    def metric(row, statistic, field):
        evaluation = row["placements"][name]["statistics"][statistic]
        if field == "auroc":
            return evaluation["auroc"]
        quantile, kind = field.split(":")
        value = evaluation["at_fpr"][quantile][kind]
        return value

    fields = ["auroc"] + [
        f"{q}:{kind}" for q in REPORTED_QUANTILES for kind in ("tpr", "realized_fpr")
    ]
    group = {"n": len(rows), "folders": [row["folder"] for row in rows]}
    for statistic in STATISTICS:
        group[statistic] = {}
        for field in fields:
            values = [metric(row, statistic, field) for row in rows]
            reference = [metric(row, REFERENCE_STATISTIC, field) for row in rows]
            group[statistic][field] = paired_summary(values, reference)
    return group


def plot(summary, model_set, json_path):
    placements = list(PLACEMENTS)
    fields = ("q0.01:tpr", "q0.05:tpr", "q0.10:tpr")
    figure, axes = plt.subplots(
        len(placements), len(fields), figsize=(19, 7), sharey=True, squeeze=False
    )
    plotted = {}
    width = 0.8 / len(STATISTICS)
    for row_index, name in enumerate(placements):
        by_attack = summary[name]["by_attack"]
        attacks = list(by_attack)
        for column_index, field in enumerate(fields):
            axis = axes[row_index][column_index]
            for stat_index, statistic in enumerate(STATISTICS):
                heights = [by_attack[a][statistic][field]["mean"] for a in attacks]
                positions = [
                    i + (stat_index - len(STATISTICS) / 2 + 0.5) * width
                    for i in range(len(attacks))
                ]
                axis.bar(
                    positions,
                    heights,
                    width,
                    label=statistic,
                    color=OKABE_ITO[stat_index % len(OKABE_ITO)],
                )
                plotted[f"{name}/{field}/{statistic}"] = dict(zip(attacks, heights))
            axis.set_xticks(range(len(attacks)))
            axis.set_xticklabels(
                [f"{a}\n(n={by_attack[a]['n']})" for a in attacks], fontsize=8
            )
            nominal = field.split(":")[0][1:]
            axis.set_title(f"{name}, TPR at {float(nominal):.0%} nominal FPR")
            axis.set_ylim(0, 1)
    axes[0][0].set_ylabel("mean TPR")
    axes[1][0].set_ylabel("mean TPR")
    axes[0][0].legend(fontsize=7)
    figure.suptitle(f"Pass statistics at the adaptive rate, {model_set} set")

    figure_path = f"{FIGURES_DIR}/pass_statistics_{model_set}.png"
    save_figure(figure, figure_path)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/cache_readouts/pass_statistics.py",
        [json_path],
        plotted,
    )


if __name__ == "__main__":
    main()

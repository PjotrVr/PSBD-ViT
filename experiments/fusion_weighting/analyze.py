"""Paired summaries of both fusion rules, the share sweep, share selection and the figure.

Reads the readings_<set>.json files measure.py wrote and writes summary.json, the
sweep figure and its sidecar. CPU only, seconds.

    .venv/bin/python -m experiments.fusion_weighting.analyze
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from experiments._paths import experiment_result_path, experiment_results_dir  # noqa: E402
from experiments.cache_readouts.shared import ordered_attacks, paired_summary  # noqa: E402
from scripts.paper._common import OKABE_ITO, figure_sidecar  # noqa: E402

SLUG = "fusion_weighting"
RESULTS_DIR = "results"
REPORT_FIGURE = os.path.join("tmp", "report", "figures", "fusion_share_sweep.pdf")
# Type size of the share sweep, sized for a landscape page of the report.
SWEEP_FONT_SIZE = 15

# Each summarized group: the readings file, the partner and which partner rate
# rules it pools. The Swin reading with the nearest-rate models is the report's
# 63-model population and stays labeled.
GROUPS = {
    "vit_band": ("vit_panel", "band", ("adaptive",)),
    "vit_late": ("vit_panel", "late", ("adaptive",)),
    "swin_band": ("swin_panel", "band", ("adaptive",)),
    "swin_band_with_nearest": ("swin_panel", "band", ("adaptive", "nearest")),
    "backdoorbench": ("backdoorbench", "band", ("adaptive",)),
    "training_set": ("training_set", "band", ("adaptive",)),
}
SWEEP_GROUPS = ("vit_band", "swin_band", "backdoorbench")
SELECTION_GROUP = "vit_band"
MIN_SHARE = "0.50"
WEIGHTED_SHARE = "0.90"
RULES = {"tm": None, "min": MIN_SHARE, "weighted": WEIGHTED_SHARE}
BUDGETS = ("q0.01", "q0.05", "q0.10", "q0.20")
HEADLINE_BUDGETS = ("q0.01", "q0.05", "q0.10")
# A model "loses" when its TPR at a budget falls by more than this against PSBD-TM
# alone, the same 5 points the clean-accuracy bar of the literature allows.
LOSS_MARGIN = 0.05
# The fine-grid shares whose mean TPR at 1% FPR lies within this of the best one
# form the plateau. 0.01 is 1 triggered input in 100, the smallest change the
# report discusses.
PLATEAU_TOLERANCE = 0.01
SELECTION_FIELD = "q0.01"


def main():
    readings = {
        name: load_readings(name)
        for name in ("vit_panel", "swin_panel", "backdoorbench", "training_set")
    }
    groups = {
        name: group_rows(readings[source], partner, rate_rules)
        for name, (source, partner, rate_rules) in GROUPS.items()
    }
    sweep_shares = readings["vit_panel"]["sweep_shares"]
    fine_shares = readings["vit_panel"]["fine_shares"]

    summary = {
        "experiment": "fusion weighting",
        "rules": {"min": f"share {MIN_SHARE}", "weighted": f"share {WEIGHTED_SHARE}"},
        "loss_margin": LOSS_MARGIN,
        "plateau_tolerance": PLATEAU_TOLERANCE,
        "groups": {},
    }
    for name, rows in groups.items():
        source, partner, rate_rules = GROUPS[name]
        block = {
            "source": f"readings_{source}.json",
            "partner": partner,
            "placement": rows[0]["partner"]["placement"] if rows else None,
            "partner_rate_rules": list(rate_rules),
            "n": len(rows),
            "folders": [row["folder"] for row in rows],
            "missing": missing_models(readings[source], partner, rate_rules),
            "rules": rule_summary(rows),
            "losses": loss_counts(rows),
            "by_attack": breakdown(rows, "attack"),
            "by_dataset": breakdown(rows, "dataset"),
            "per_model": per_model_rows(rows),
            "sweep": sweep_summary(rows, sweep_shares),
            "fine": fine_summary(rows, fine_shares),
            "partner_carried": partner_carried(rows),
            "allocation": allocation(
                rows, fine_shares, readings[source]["low_budgets"]
            ),
            "trade_off": trade_off(rows),
        }
        if name != "training_set":
            block["literal_union_bound"] = literal_bound(rows, sweep_shares)
        summary["groups"][name] = block

    summary["selection"] = {
        "leave_one_dataset_out": leave_one_dataset_out(
            groups[SELECTION_GROUP], sweep_shares, fine_shares
        ),
        "dev_set": dev_selection(groups[SELECTION_GROUP], sweep_shares, fine_shares),
    }
    # The pre-registration chose its shares on the late band, so both selections
    # are repeated there, and the Swin-S fold shows whether the ViT pick transfers.
    summary["selection"]["leave_one_dataset_out_late"] = leave_one_dataset_out(
        groups["vit_late"], sweep_shares, fine_shares
    )
    summary["selection"]["dev_set_late"] = dev_selection(
        groups["vit_late"], sweep_shares, fine_shares
    )
    summary["selection"]["leave_one_dataset_out_swin"] = leave_one_dataset_out(
        groups["swin_band"], sweep_shares, fine_shares
    )

    path = experiment_result_path(SLUG, "summary.json", RESULTS_DIR)
    write_json(summary, path)
    plot_sweep(summary, path)
    print(f"wrote {path}")


def load_readings(name):
    path = experiment_result_path(SLUG, f"readings_{name}.json", RESULTS_DIR)
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def group_rows(readings, partner, rate_rules):
    rows = []
    for model in readings["models"]:
        entry = model.get("partners", {}).get(partner)
        if model["anchor_rate"] is None or not entry or entry["rate"] is None:
            continue
        if entry["rate_rule"] not in rate_rules:
            continue
        rows.append({**model, "partner": entry})
    return rows


def missing_models(readings, partner, rate_rules):
    missing = []
    for model in readings["models"]:
        entry = model.get("partners", {}).get(partner) or {}
        if model["anchor_rate"] is None:
            missing.append({"folder": model["folder"], "why": "PSBD-TM has no rate"})
        elif entry.get("rate") is None or entry.get("rate_rule") not in rate_rules:
            missing.append(
                {
                    "folder": model["folder"],
                    "why": "partner misses the adaptive target",
                    "max_validation_shift": entry.get("max_validation_shift"),
                }
            )
    return missing


def reading_of(row, rule):
    share = RULES.get(rule, rule)
    if share is None:
        reading = row["partner"]["tm_alone"]
    elif share == "partner":
        reading = row["partner"]["partner_alone"]
    else:
        reading = row["partner"]["sweep"][share]
    return reading


def value_of(row, rule, field):
    reading = reading_of(row, rule)
    if field == "auroc":
        value = reading["auroc"]
    else:
        kind, budget = field.split(":")
        value = reading[kind][budget]
    return value


def fields():
    names = [f"tpr:{budget}" for budget in BUDGETS]
    names += [f"realized_fpr:{budget}" for budget in BUDGETS]
    names += ["auroc"]
    return names


def rule_summary(rows):
    comparisons = {
        "min_minus_tm": ("min", "tm"),
        "weighted_minus_tm": ("weighted", "tm"),
        "weighted_minus_min": ("weighted", "min"),
    }
    summary = {
        "means": {
            rule: {
                field: mean([value_of(r, rule, field) for r in rows])
                for field in fields()
            }
            for rule in RULES
        },
        "paired": {},
    }
    for name, (rule, reference) in comparisons.items():
        summary["paired"][name] = {
            field: paired_summary(
                [value_of(r, rule, field) for r in rows],
                [value_of(r, reference, field) for r in rows],
            )
            for field in fields()
            if not field.startswith("realized_fpr")
        }
    return summary


def loss_counts(rows):
    losses = {}
    for rule in ("min", "weighted"):
        losses[rule] = {}
        for field in [f"tpr:{budget}" for budget in BUDGETS] + ["auroc"]:
            differences = [
                (value_of(r, rule, field) - value_of(r, "tm", field), r["folder"])
                for r in rows
            ]
            losers = sorted(d for d in differences if d[0] < -LOSS_MARGIN)
            worst = min(differences) if differences else (None, None)
            losses[rule][field] = {
                "count": len(losers),
                "worst": worst[0],
                "worst_folder": worst[1],
                "losers": [{"folder": f, "difference": d} for d, f in losers],
            }
    return losses


def breakdown(rows, key):
    values = {row[key] for row in rows}
    keys = ordered_attacks(values) if key == "attack" else sorted(values)
    table = {}
    for value in keys:
        subset = [row for row in rows if row[key] == value]
        table[value] = {
            "n": len(subset),
            "means": {
                rule: {
                    field: mean([value_of(r, rule, field) for r in subset])
                    for field in fields()
                }
                for rule in RULES
            },
            "weighted_minus_min": {
                field: paired_summary(
                    [value_of(r, "weighted", field) for r in subset],
                    [value_of(r, "min", field) for r in subset],
                )
                for field in [f"tpr:{budget}" for budget in BUDGETS] + ["auroc"]
            },
            "losses": {
                rule: {
                    budget: sum(
                        value_of(r, rule, f"tpr:{budget}")
                        - value_of(r, "tm", f"tpr:{budget}")
                        < -LOSS_MARGIN
                        for r in subset
                    )
                    for budget in BUDGETS
                }
                for rule in ("min", "weighted")
            },
        }
    return table


def per_model_rows(rows):
    table = []
    for row in rows:
        entry = {
            "folder": row["folder"],
            "dataset": row["dataset"],
            "attack": row["attack"],
            "poison_rate": row["poison_rate"],
            "dev": row.get("dev", False),
            "anchor_rate": row["anchor_rate"],
            "partner_rate": row["partner"]["rate"],
            "partner_rate_rule": row["partner"]["rate_rule"],
        }
        for rule in list(RULES) + ["partner"]:
            entry[rule] = {
                field: value_of(row, rule, field)
                for field in fields()
                if not field.startswith("realized_fpr") or rule != "partner"
            }
        table.append(entry)
    return table


def sweep_summary(rows, shares):
    sweep = {}
    for share in shares:
        key = f"{share:.2f}"
        sweep[key] = {
            "means": {
                field: mean([value_of(r, key, field) for r in rows])
                for field in fields()
            },
            "minus_tm": {
                field: paired_summary(
                    [value_of(r, key, field) for r in rows],
                    [value_of(r, "tm", field) for r in rows],
                )
                for field in ["tpr:q0.01", "tpr:q0.05", "tpr:q0.10", "auroc"]
            },
            "losses": {
                budget: sum(
                    value_of(r, key, f"tpr:{budget}")
                    - value_of(r, "tm", f"tpr:{budget}")
                    < -LOSS_MARGIN
                    for r in rows
                )
                for budget in BUDGETS
            },
        }
    return sweep


def fine_summary(rows, shares):
    curves = {"shares": shares, "tm": {}, "means": {}, "losses": {}}
    for budget in BUDGETS:
        curves["tm"][budget] = mean([value_of(r, "tm", f"tpr:{budget}") for r in rows])
    curves["tm"]["auroc"] = mean([value_of(r, "tm", "auroc") for r in rows])
    for name in list(BUDGETS) + ["auroc"]:
        curves["means"][name] = [
            mean([r["partner"]["fine"][f"{share:.2f}"][name] for r in rows])
            for share in shares
        ]
    for budget in HEADLINE_BUDGETS:
        curves["losses"][budget] = [
            sum(
                r["partner"]["fine"][f"{share:.2f}"][budget]
                - value_of(r, "tm", f"tpr:{budget}")
                < -LOSS_MARGIN
                for r in rows
            )
            for share in shares
        ]
    curves["plateau"] = plateau(shares, curves["means"][SELECTION_FIELD])
    return curves


def plateau(shares, curve):
    best = max(curve)
    best_share = shares[curve.index(best)]
    within = [
        share
        for share, value in zip(shares, curve)
        if value >= best - PLATEAU_TOLERANCE
    ]
    weighted_value = curve[shares.index(float(WEIGHTED_SHARE))]
    result = {
        "best_share": best_share,
        "best_value": best,
        "within_tolerance": [min(within), max(within)],
        "within_tolerance_contiguous": len(within)
        == round((max(within) - min(within)) / 0.01) + 1,
        "weighted_value": weighted_value,
        "weighted_gap_to_best": best - weighted_value,
        "min_value": curve[0],
    }
    return result


def partner_carried(rows):
    # The share pi of models the partner carries at the strictest budget, the pi of
    # the allocation derivation in the README.
    carried = [
        row["folder"]
        for row in rows
        if value_of(row, "partner", f"tpr:{SELECTION_FIELD}")
        - value_of(row, "tm", f"tpr:{SELECTION_FIELD}")
        > LOSS_MARGIN
    ]
    result = {
        "n": len(rows),
        "count": len(carried),
        "share": len(carried) / len(rows) if rows else None,
        "folders": carried,
    }
    return result


def allocation(rows, shares, low_budgets):
    # The union's TPR at budget q is at least the larger of the 2 probes' own TPR
    # at w q and (1 - w) q, read off each probe's ROC alone. The share maximizing
    # that bound is where the marginal TPR per unit of budget is equal for the 2
    # probes, so it shows where the optimum sits and why it moves.
    budget_index = {round(budget, 4): index for index, budget in enumerate(low_budgets)}
    nominal = float(SELECTION_FIELD[1:])

    def tpr_at(row, probe, budget):
        tpr = row["partner"]["low_budget_tpr"][probe][budget_index[round(budget, 4)]]
        return tpr

    bound = [
        mean(
            [
                max(
                    tpr_at(r, "tm_alone", share * nominal),
                    tpr_at(r, "partner_alone", (1 - share) * nominal),
                )
                for r in rows
            ]
        )
        for share in shares
    ]
    tm_curve = [mean([tpr_at(r, "tm_alone", b) for r in rows]) for b in low_budgets]
    partner_curve = [
        mean([tpr_at(r, "partner_alone", b) for r in rows]) for b in low_budgets
    ]
    half = budget_index[round(nominal / 2, 4)]
    tenth = budget_index[round(nominal / 10, 4)]
    full = budget_index[round(nominal, 4)]
    result = {
        "max_of_probes_bound": bound,
        "bound_best_share": shares[bound.index(max(bound))],
        "low_budgets": low_budgets,
        "tm_alone_mean_tpr": tm_curve,
        "partner_alone_mean_tpr": partner_curve,
        "tm_tpr_lost_halving_budget": tm_curve[full] - tm_curve[half],
        "partner_tpr_at_half_budget": partner_curve[half],
        "partner_tpr_at_tenth_budget": partner_curve[tenth],
    }
    return result


def trade_off(rows):
    # Per attack, then per dataset and attack for the sets whose attacks repeat
    # across datasets, weighted minus min at the headline budgets.
    table = {}
    for attack in ordered_attacks({row["attack"] for row in rows}):
        for dataset in sorted(
            {row["dataset"] for row in rows if row["attack"] == attack}
        ):
            subset = [
                r for r in rows if r["attack"] == attack and r["dataset"] == dataset
            ]
            table[f"{dataset}/{attack}"] = {
                "n": len(subset),
                "weighted_minus_min": {
                    budget: mean(
                        [
                            value_of(r, "weighted", f"tpr:{budget}")
                            - value_of(r, "min", f"tpr:{budget}")
                            for r in subset
                        ]
                    )
                    for budget in HEADLINE_BUDGETS
                },
                "tm": {
                    budget: mean([value_of(r, "tm", f"tpr:{budget}") for r in subset])
                    for budget in HEADLINE_BUDGETS
                },
                "min": {
                    budget: mean([value_of(r, "min", f"tpr:{budget}") for r in subset])
                    for budget in HEADLINE_BUDGETS
                },
                "weighted": {
                    budget: mean(
                        [value_of(r, "weighted", f"tpr:{budget}") for r in subset]
                    )
                    for budget in HEADLINE_BUDGETS
                },
            }
    return table


def literal_bound(rows, shares):
    bound = {}
    for share in shares:
        key = f"{share:.2f}"
        bound[key] = {}
        for budget in BUDGETS:
            nominal = float(budget[1:])
            validation = [
                r["partner"]["sweep"][key]["literal"][budget]["validation_fpr"]
                for r in rows
            ]
            realized = [
                r["partner"]["sweep"][key]["literal"][budget]["realized_fpr"]
                for r in rows
            ]
            bound[key][budget] = {
                "nominal": nominal,
                "validation_fpr_mean": mean(validation),
                "validation_fpr_max": max(validation),
                "realized_fpr_mean": mean(realized),
                "realized_fpr_max": max(realized),
                "tpr_mean": mean(
                    [r["partner"]["sweep"][key]["literal"][budget]["tpr"] for r in rows]
                ),
            }
    return bound


def pick_share_by_losses(rows, shares, fine):
    # The pre-registration's own criterion, the share that loses least: fewest
    # models losing more than LOSS_MARGIN at 1% FPR, then the higher mean TPR, and
    # a remaining tie goes to the smaller share.
    def key(share):
        losses = sum(
            share_tpr(r, share, fine) - value_of(r, "tm", f"tpr:{SELECTION_FIELD}")
            < -LOSS_MARGIN
            for r in rows
        )
        tpr = mean([share_tpr(r, share, fine) for r in rows])
        return (-losses, tpr)

    keys = [key(share) for share in shares]
    picked = shares[keys.index(max(keys))]
    return picked


def pick_share(rows, shares, fine):
    # The share with the highest mean TPR at 1% FPR. A tie goes to the smaller
    # share, the one closer to the plain minimum, so a tie never favours 0.9.
    means = [mean([share_tpr(r, share, fine) for r in rows]) for share in shares]
    best = max(means)
    picked = shares[means.index(best)]
    return picked, dict(zip([f"{s:.2f}" for s in shares], means))


def share_tpr(row, share, fine):
    key = f"{share:.2f}"
    if fine:
        value = row["partner"]["fine"][key][SELECTION_FIELD]
    else:
        value = row["partner"]["sweep"][key]["tpr"][SELECTION_FIELD]
    return value


def held_out_reading(rows, picked, fine):
    reading = {
        "n": len(rows),
        "picked_tpr": mean([share_tpr(r, picked, fine) for r in rows]),
        "weighted_tpr": mean(
            [value_of(r, "weighted", f"tpr:{SELECTION_FIELD}") for r in rows]
        ),
        "min_tpr": mean([value_of(r, "min", f"tpr:{SELECTION_FIELD}") for r in rows]),
        "tm_tpr": mean([value_of(r, "tm", f"tpr:{SELECTION_FIELD}") for r in rows]),
        "picked_losses": sum(
            share_tpr(r, picked, fine) - value_of(r, "tm", f"tpr:{SELECTION_FIELD}")
            < -LOSS_MARGIN
            for r in rows
        ),
    }
    for rule in ("min", "weighted"):
        reading[f"{rule}_losses"] = sum(
            value_of(r, rule, f"tpr:{SELECTION_FIELD}")
            - value_of(r, "tm", f"tpr:{SELECTION_FIELD}")
            < -LOSS_MARGIN
            for r in rows
        )
    return reading


def leave_one_dataset_out(rows, sweep_shares, fine_shares):
    folds = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        fit = [r for r in rows if r["dataset"] != dataset]
        held = [r for r in rows if r["dataset"] == dataset]
        fold = {"n_fit": len(fit), "n_held_out": len(held)}
        for grid_name, shares, fine in (
            ("sweep", sweep_shares, False),
            ("fine", fine_shares, True),
        ):
            picked, fit_means = pick_share(fit, shares, fine)
            picked_by_losses = pick_share_by_losses(fit, shares, fine)
            fold[grid_name] = {
                "picked": picked,
                "distance_to_0.9": abs(picked - float(WEIGHTED_SHARE)),
                "fit_means": fit_means,
                "held_out": held_out_reading(held, picked, fine),
                "picked_by_losses": picked_by_losses,
                "held_out_by_losses": held_out_reading(held, picked_by_losses, fine),
            }
        folds[dataset] = fold
    return folds


def dev_selection(rows, sweep_shares, fine_shares):
    dev = [r for r in rows if r.get("dev")]
    rest = [r for r in rows if not r.get("dev")]
    selection = {
        "n_dev": len(dev),
        "dev_folders": [r["folder"] for r in dev],
        "n_rest": len(rest),
    }
    for grid_name, shares, fine in (
        ("sweep", sweep_shares, False),
        ("fine", fine_shares, True),
    ):
        picked, fit_means = pick_share(dev, shares, fine)
        picked_by_losses = pick_share_by_losses(dev, shares, fine)
        selection[grid_name] = {
            "picked": picked,
            "distance_to_0.9": abs(picked - float(WEIGHTED_SHARE)),
            "fit_means": fit_means,
            "held_out": held_out_reading(rest, picked, fine),
            "picked_by_losses": picked_by_losses,
            "held_out_by_losses": held_out_reading(rest, picked_by_losses, fine),
        }
    return selection


def mean(values):
    result = sum(values) / len(values) if values else None
    return result


def plot_sweep(summary, json_path):
    titles = {
        "vit_band": "ViT-B/16 panel, blocks 5 to 8",
        "swin_band": "Swin-S panel, blocks 17 to 24",
        "backdoorbench": "BackdoorBench, blocks 5 to 8",
    }
    plt.rcParams.update(
        {"font.size": SWEEP_FONT_SIZE, "axes.titlesize": SWEEP_FONT_SIZE + 1}
    )
    figure, axes = plt.subplots(2, len(SWEEP_GROUPS), figsize=(16, 10), sharex=True)
    plotted = {}
    for column, name in enumerate(SWEEP_GROUPS):
        block = summary["groups"][name]
        fine = block["fine"]
        shares = fine["shares"]
        top, bottom = axes[0][column], axes[1][column]
        plotted[name] = {"n": block["n"], "shares": shares}
        for index, field in enumerate(list(HEADLINE_BUDGETS) + ["auroc"]):
            label = (
                "AUROC" if field == "auroc" else f"TPR at {float(field[1:]):.0%} FPR"
            )
            color = OKABE_ITO[index]
            top.plot(shares, fine["means"][field], color=color, lw=2.4, label=label)
            top.axhline(fine["tm"][field], color=color, linestyle=":", linewidth=1)
            plotted[name][field] = fine["means"][field]
            plotted[name][f"tm_alone:{field}"] = fine["tm"][field]
        for index, budget in enumerate(HEADLINE_BUDGETS):
            bottom.plot(
                shares,
                fine["losses"][budget],
                color=OKABE_ITO[index],
                label=f"at {float(budget[1:]):.0%} FPR",
            )
            plotted[name][f"losses:{budget}"] = fine["losses"][budget]
        for axis in (top, bottom):
            axis.axvline(
                float(WEIGHTED_SHARE), color="grey", linestyle="--", linewidth=1
            )
        bottom.yaxis.set_major_locator(MaxNLocator(integer=True))
        top.set_title(f"{titles[name]} (n={block['n']})")
        bottom.set_xlabel("PSBD-TM share of the false-positive budget, w")
    axes[0][0].set_ylabel("mean over models (dotted, PSBD-TM alone)")
    axes[1][0].set_ylabel(f"models losing more than {LOSS_MARGIN:.2f} TPR")
    axes[0][0].legend()
    axes[1][0].legend()

    directory = experiment_results_dir(SLUG, RESULTS_DIR)
    png_path = os.path.join(directory, "share_sweep.png")
    pdf_path = os.path.join(directory, "share_sweep.pdf")
    figure.tight_layout()
    figure.savefig(png_path, dpi=120)
    figure.savefig(pdf_path)
    os.makedirs(os.path.dirname(REPORT_FIGURE), exist_ok=True)
    figure.savefig(REPORT_FIGURE)
    plt.close(figure)
    figure_sidecar(
        os.path.join(directory, "share_sweep.json"),
        "experiments/fusion_weighting/analyze.py",
        [json_path],
        plotted,
    )


def sanitized(value):
    # bootstrap_ci returns NaN below 3 values, which strict JSON cannot hold.
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(item) for item in value]
    return value


def write_json(payload, path):
    with open(path, "w") as handle:
        json.dump(sanitized(payload), handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

"""Verdicts of the pre-registered predictions, read from confirm.json.

Every threshold and rule is read from preregistration.json, whose sha256 confirm.json
stored when it ran, so a verdict cannot be computed against an edited rule.

    python -m experiments.tact_calibration.judge
"""

import json

import numpy as np

from experiments._paths import experiment_result_path
from experiments.cache_readouts.shared import paired_summary
from experiments.tact_calibration.analyze import PREREGISTRATION, file_sha256

SLUG = "tact_calibration"
RESULTS_DIR = "results"
QUANTILE_TAGS = ("q0.01", "q0.05", "q0.10")


def main():
    with open(PREREGISTRATION) as handle:
        frozen = json.load(handle)
    with open(experiment_result_path(SLUG, "confirm.json", RESULTS_DIR)) as handle:
        confirm = json.load(handle)
    assert confirm["preregistration_sha256"] == file_sha256(PREREGISTRATION)

    point = f"mf{frozen['shrinkage']['fallback']}/mc{frozen['shrinkage']['class']}"
    rows = confirm["models"]
    holdout = [r for r in rows if r["group"] == "holdout"]
    dev = [r for r in rows if r["group"] == "dev" and r["in_panel"]]
    panel_non_tact = [
        r
        for r in rows
        if r["attack"] != "tact" and r["in_panel"] and r["group"] != "holdout"
    ]

    def reading(row, method, k, field):
        value = row["by_k"][str(k)][f"{method}/{point}"][field]
        return value

    verdicts = {}
    rules = frozen["rules"]

    premise = rules["premise"]
    own = [r["premise"]["anchor"]["fallback_is_own_source"] for r in holdout]
    attractor = [r["premise"]["anchor"]["fallback_is_attractor"] for r in holdout]
    verdicts["P1"] = {
        "own_source_mean": float(np.mean(own)) if own else None,
        "attractor_mean": float(np.mean(attractor)) if attractor else None,
        "per_model": {
            r["folder"]: {
                "own_source": r["premise"]["anchor"]["fallback_is_own_source"],
                "attractor": r["premise"]["anchor"]["fallback_is_attractor"],
            }
            for r in holdout
        },
        "holds": bool(own) and float(np.mean(own)) < premise["own_source_below"],
    }

    plain = rules["plain_low"]
    for name, method in (("P2_tm", "plain"), ("P2_final", "final")):
        for k in plain["pass_counts"]:
            values = {r["folder"]: reading(r, method, k, "q0.10:tpr") for r in holdout}
            low = sum(v < plain["tpr_below"] for v in values.values())
            verdicts[f"{name}_k{k}"] = {
                "tpr_at_10pct": values,
                "n_low": low,
                "n_models": len(values),
                "holds": len(values) == 6 and low >= plain["at_least"],
            }

    gain = rules["holdout_gain"]
    for name, method, reference in (
        ("P3", "or_fallback", "plain"),
        ("P3_final", "final_or_fallback", "final"),
        ("P4", "or_class", "plain"),
        ("P4_final", "final_or_class", "final"),
    ):
        summary = compare(holdout, method, reference, gain["pass_count"], reading)
        tenth = summary["q0.10:tpr"]
        verdicts[name] = {
            "summary": summary,
            "holds": tenth["mean_difference"] is not None
            and tenth["mean_difference"] >= gain["mean_gain_at_least"]
            and tenth["ci95"][0] is not None
            and tenth["ci95"][0] > 0,
        }

    harm = rules["no_harm"]
    for subset_name, subset in (("dev", dev), ("panel", panel_non_tact)):
        for method, reference in (
            ("or_fallback", "plain"),
            ("or_class", "plain"),
            ("final_or_fallback", "final"),
            ("final_or_class", "final"),
        ):
            summary = compare(subset, method, reference, harm["pass_count"], reading)
            worst = min(
                summary[f"{tag}:tpr"]["mean_difference"] for tag in QUANTILE_TAGS
            )
            verdicts[f"P5_{subset_name}_{method}"] = {
                "summary": summary,
                "worst_mean_difference": worst,
                "holds": worst >= harm["mean_loss_no_worse_than"],
            }

    fpr = rules["fpr_control"]
    for method in ("or_fallback", "or_class", "final_or_fallback", "final_or_class"):
        realized = {
            tag: float(
                np.mean(
                    [
                        reading(r, method, fpr["pass_count"], f"{tag}:fpr")
                        for r in panel_non_tact
                    ]
                )
            )
            for tag in QUANTILE_TAGS
        }
        within = all(
            abs(realized[tag] - float(tag[1:])) <= fpr["tolerance"][tag]
            for tag in QUANTILE_TAGS
        )
        verdicts[f"P6_{method}"] = {"realized_fpr": realized, "holds": within}

    passes = rules["pass_count"]
    by_k = {}
    for k in passes["candidates"]:
        by_k[str(k)] = compare(
            panel_non_tact, "plain", "plain", k, reading, reference_k=20
        )
    recommended = next(
        (
            k
            for k in passes["candidates"]
            if all(
                by_k[str(k)][field]["mean_difference"] >= -passes["within"]
                for field in ("auroc",) + tuple(f"{tag}:tpr" for tag in QUANTILE_TAGS)
            )
        ),
        None,
    )
    k3 = by_k["3"]["auroc"]
    verdicts["P7"] = {
        "by_k_vs_20": by_k,
        "recommended_k": recommended,
        "predicted_k": passes["predicted"],
        "holds": recommended == passes["predicted"],
    }
    verdicts["P7_auroc_gain"] = {
        "k20_minus_k3": {
            "mean": -k3["mean_difference"],
            "ci95": [-k3["ci95"][1], -k3["ci95"][0]],
        },
        "holds": k3["ci95"][1] is not None and k3["ci95"][1] < 0,
    }

    payload = {
        "preregistration_sha256": confirm["preregistration_sha256"],
        "n_models": {
            "holdout": len(holdout),
            "dev": len(dev),
            "panel_non_tact": len(panel_non_tact),
        },
        "verdicts": verdicts,
    }
    path = experiment_result_path(SLUG, "verdicts.json", RESULTS_DIR)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    for name, verdict in verdicts.items():
        print(name, "holds" if verdict["holds"] else "FAILS")
    print(f"wrote {path}")


def compare(rows, method, reference, k, reading, reference_k=None):
    reference_k = k if reference_k is None else reference_k
    fields = ("auroc",) + tuple(f"{tag}:tpr" for tag in QUANTILE_TAGS)
    summary = {
        field: paired_summary(
            [reading(r, method, k, field) for r in rows],
            [reading(r, reference, reference_k, field) for r in rows],
        )
        for field in fields
    }
    return summary


if __name__ == "__main__":
    main()

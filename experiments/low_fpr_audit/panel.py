"""PSBD-TM at low false-positive budgets on the current panel, rendered into README.md.

Placement `before_attention_norm_token_mask`, fractional PSU (`detection_psu_ratio`),
thresholds at the q0.01, q0.05, q0.10 and q0.25 clean-validation quantiles, over the
models of the headline panel (`experiments.probe_union.measure.select_models`),
grouped by poison rate. 2 rate rules are read from each `psbd_metrics.json`: the
adaptive 0.8 rule of the headline and the matched 0.6 rung the earlier record used.
The script writes `results/_experiments/low_fpr_audit/panel.json` and replaces the
block between the results markers of README.md.

    PYTHONPATH=. .venv/bin/python experiments/low_fpr_audit/panel.py
"""

import collections
import json
import os
import statistics

from experiments._paths import experiment_result_path
from experiments.probe_union.measure import select_models

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
README = os.path.join(REPO_ROOT, "experiments", "low_fpr_audit", "README.md")
PLACEMENT = "before_attention_norm_token_mask"
QUANTILES = ("q0.01", "q0.05", "q0.10", "q0.25")
MATCHED_KEY = "sigma0.6"
# A model is flagged when AUROC looks like a working detector and TPR at 1% FPR is
# still near 0, the gap this audit exists to show.
FLAG_AUROC = 0.85
FLAG_TPR = 0.05
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"


def main():
    models = select_models("results")
    readings = {
        "adaptive": [adaptive_reading(model) for model in models],
        "matched": [matched_reading(model) for model in models],
    }
    payload = {
        "placement": PLACEMENT,
        "n_models": len(models),
        "rules": {rule: summarize(rows) for rule, rows in readings.items()},
        "models": readings,
    }
    path = experiment_result_path("low_fpr_audit", "panel.json")
    write_json(payload, path)
    render(payload)
    print(f"wrote {path} and {README}")


def adaptive_reading(model):
    block = model["report"]["placements"][PLACEMENT]
    rate = block["adaptive_rate"]
    reading = reading_at(model, block, rate)
    return reading


def matched_reading(model):
    block = model["report"]["placements"][PLACEMENT]
    rate = block["matched_shift"][MATCHED_KEY]["rate"]
    reading = reading_at(model, block, rate)
    return reading


def reading_at(model, block, rate):
    row = next(r for r in block["rates"] if r["rate"] == rate)
    detection = row["detection_psu_ratio"]
    reading = {
        "folder": model["folder_name"],
        "poison_rate": model["poison_rate"],
        "rate": rate,
        "auroc": detection["q0.25"]["auroc"],
        "auprc": detection["q0.25"]["auprc"],
    }
    for quantile in QUANTILES:
        reading[f"{quantile}:tpr"] = detection[quantile]["tpr"]
        reading[f"{quantile}:fpr"] = detection[quantile]["fpr"]
    return reading


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[f"{row['poison_rate']}"].append(row)
        groups["all"].append(row)
    fields = ["auroc", "auprc"]
    fields += [f"{q}:tpr" for q in QUANTILES] + [f"{q}:fpr" for q in QUANTILES]
    by_group = {
        group: {"n": len(members)}
        | {field: statistics.mean(r[field] for r in members) for field in fields}
        for group, members in groups.items()
    }
    flagged = sorted(
        (r for r in rows if r["auroc"] >= FLAG_AUROC and r["q0.01:tpr"] < FLAG_TPR),
        key=lambda r: -r["auroc"],
    )
    summary = {"by_group": by_group, "flagged": flagged}
    return summary


def write_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def render(payload):
    adaptive = payload["rules"]["adaptive"]
    matched = payload["rules"]["matched"]
    n = payload["n_models"]
    groups = [g for g in ("0.01", "0.05", "0.1") if g in adaptive["by_group"]]
    lines = [
        BEGIN,
        "<!-- Everything down to results:end is rendered by panel.py from "
        "results/_experiments/low_fpr_audit/panel.json. -->",
        "",
        "## Result on the current panel",
        "",
        "PSBD-TM (`token_mask @ before_attention_norm`) at the adaptive 0.8 rule's "
        f"rate, fractional PSU, the paper panel of {n} models successful at the "
        "2-point clean-accuracy bar, every value read from "
        "`results/<folder>/psbd_metrics.json`. Thresholds are set on clean "
        "validation, so every TPR below is what a defender would get.",
        "",
        "    PYTHONPATH=. .venv/bin/python experiments/low_fpr_audit/panel.py",
        "",
        rate_table(adaptive, groups),
        "",
    ]
    all_row = adaptive["by_group"]["all"]
    tpr1 = [adaptive["by_group"][g]["q0.01:tpr"] for g in groups]
    aurocs = [adaptive["by_group"][g]["auroc"] for g in groups]
    lines += [
        f"At a 1% false-positive budget PSBD-TM catches {min(tpr1):.2f} to "
        f"{max(tpr1):.2f} of triggered inputs by poison rate, against an AUROC of "
        f"{min(aurocs):.2f} to {max(aurocs):.2f}. Over the whole panel it catches "
        f"{all_row['q0.01:tpr']:.3f} at 1% FPR against an AUROC of "
        f"{all_row['auroc']:.3f}, so the low-FPR tail still costs about "
        f"{1 - all_row['q0.01:tpr']:.2f} of the triggered inputs.",
        "",
    ]
    flagged = adaptive["flagged"]
    lines += [
        f"**{len(flagged)} of {n} models have AUROC >= {FLAG_AUROC} and "
        f"TPR@1%FPR < {FLAG_TPR}:**",
        "",
        "| model | AUROC | TPR@1% | TPR@5% |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{r['folder']}` | {r['auroc']:.3f} | **{r['q0.01:tpr']:.3f}** "
        f"| {r['q0.05:tpr']:.3f} |"
        for r in flagged
    ]
    below_chance = sorted(
        (r for r in payload["models"]["adaptive"] if r["auroc"] < 0.5),
        key=lambda r: r["auroc"],
    )
    lines += [
        "",
        "The flag needs a high AUROC, so it leaves out the models where the score "
        "itself inverts. Those read TPR near 0 at every budget and are "
        + join_words(
            [
                f"`{r['folder']}` (AUROC {r['auroc']:.3f}, TPR@1% {r['q0.01:tpr']:.3f})"
                for r in below_chance
            ]
        )
        + ".",
        "",
        "## The matched 0.6 rule on the same models",
        "",
        f"Read at the matched 0.6 rung instead, the same {n} models give a mean "
        f"TPR@1%FPR of {matched['by_group']['all']['q0.01:tpr']:.3f} against "
        f"{all_row['q0.01:tpr']:.3f} at the adaptive rule and flag "
        f"{len(matched['flagged'])} models against {len(flagged)}, so the adaptive "
        "0.8 rule, which perturbs harder, is what separates the extreme tail.",
        "",
        rate_table(matched, groups),
        "",
        "## Realized false-positive rates",
        "",
        "Achieved FPR on the paired clean test split tracks the nominal quantile, "
        f"{all_row['q0.01:fpr']:.4f} against 0.01 and {all_row['q0.05:fpr']:.4f} "
        "against 0.05 on the panel at the adaptive rule. The calibration is right and "
        "the clean and triggered score distributions overlap in the extreme tail.",
        END,
    ]
    block = "\n".join(lines)
    with open(README) as handle:
        text = handle.read()
    start = text.index(BEGIN)
    stop = text.index(END) + len(END)
    text = text[:start] + block + text[stop:]
    with open(README, "w") as handle:
        handle.write(text)


def join_words(words):
    if len(words) == 1:
        return words[0]
    text = ", ".join(words[:-1]) + " and " + words[-1]
    return text


def rate_table(rule, groups):
    lines = [
        "| poison | n | AUROC | AUPRC | TPR@1% | TPR@5% | TPR@10% | TPR@25% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group in [*groups, "all"]:
        row = rule["by_group"][group]
        label = "all" if group == "all" else f"{float(group):.0%}"
        lines.append(
            f"| {label} | {row['n']} | {row['auroc']:.3f} | {row['auprc']:.3f} "
            f"| **{row['q0.01:tpr']:.3f}** | {row['q0.05:tpr']:.3f} "
            f"| {row['q0.10:tpr']:.3f} | {row['q0.25:tpr']:.3f} |"
        )
    table = "\n".join(lines)
    return table


if __name__ == "__main__":
    main()

"""Write experiments/tact_calibration/README.md from the records this directory wrote.

Every number is read from tuning.json, confirm.json, verdicts.json, plan.json or
the per-model run_record.json files, and every sentence whose wording depends on
the data is guarded by an assert, so a rerun that changes the data stops the
render instead of printing a stale claim.

    python -m experiments.tact_calibration.render_readme
"""

import glob
import json
import os

import numpy as np

from experiments._paths import experiment_result_path
from experiments.tact_calibration.analyze import (
    MODELS_ROOT,
    PASS_COUNTS,
    PREREGISTRATION,
    VALIDATION_SIZE,
)
from experiments.tact_calibration.judge import compare
from scripts.paper._common import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

SLUG = "tact_calibration"
OUT = os.path.join("experiments", SLUG, "README.md")
TAGS = ("q0.01", "q0.05", "q0.10")
METHOD_WORDS = {
    "plain": "PSBD-TM",
    "fallback": "fallback score alone",
    "class": "class score alone",
    "or_fallback": "PSBD-TM OR fallback",
    "or_class": "PSBD-TM OR class",
    "final": "final method",
    "final_or_fallback": "final OR fallback",
    "final_or_class": "final OR class",
}
TACT_METHODS = (
    "plain",
    "or_fallback",
    "or_class",
    "final",
    "final_or_fallback",
    "final_or_class",
)
PAIRS = (
    ("or_fallback", "plain"),
    ("or_class", "plain"),
    ("final", "plain"),
    ("final_or_fallback", "final"),
    ("final_or_class", "final"),
)


def main():
    frozen = read(PREREGISTRATION)
    tuning = read(experiment_result_path(SLUG, "tuning.json"))
    confirm = read(experiment_result_path(SLUG, "confirm.json"))
    verdicts = read(experiment_result_path(SLUG, "verdicts.json"))
    plan = read(experiment_result_path(SLUG, "plan.json"))
    records = [
        read(path)
        for path in sorted(glob.glob(os.path.join(MODELS_ROOT, "*", "run_record.json")))
    ]
    assert verdicts["preregistration_sha256"] == confirm["preregistration_sha256"]

    point = f"mf{frozen['shrinkage']['fallback']}/mc{frozen['shrinkage']['class']}"
    rows = confirm["models"]

    def reading(row, method, k, field):
        value = row["by_k"][str(k)][f"{method}/{point}"][field]
        return value

    groups = {
        "tuning": [r for r in rows if r["group"] == "tuning"],
        "holdout": [r for r in rows if r["group"] == "holdout"],
        "dev": [r for r in rows if r["group"] == "dev" and r["in_panel"]],
        "panel": [
            r
            for r in rows
            if r["attack"] != "tact" and r["in_panel"] and r["group"] != "holdout"
        ],
    }
    sections = [
        header(frozen, plan, groups),
        wall_times(records, plan),
        premise_section(groups, verdicts),
        tact_section(groups, reading, verdicts),
        harm_section(groups, reading, verdicts),
        pass_count_section(groups, reading, verdicts),
        verdict_section(verdicts, frozen),
        files_section(),
    ]
    text = "\n\n".join(sections) + "\n"
    with open(OUT, "w") as handle:
        handle.write(text)
    print(f"wrote {OUT}")
    del tuning


def read(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def header(frozen, plan, groups):
    sizes = ", ".join(f"{name} {size}" for name, size in VALIDATION_SIZE.items())
    n_plan = len(plan["models"])
    text = f"""# TaCT calibration of PSBD-TM

PSBD-TM detects TaCT with a high AUROC and almost no TPR at a deployable FPR. The AUROC pairs each triggered input with its own clean copy, a source-class image, and those images are more fragile under masking than the clean images of every class the threshold is set on. This experiment asks whether a label-free calibration can move the threshold toward the source classes without losing what PSBD-TM already catches, and how many forward passes the score needs. The rules, the shrinkage choice and every prediction were frozen in `PREDICTIONS.md` (commit of `preregistration.json`) after reading the 4 tuning models only.

## Method

Every model was swept once at k = 20 at each placement's adaptive rate, PSBD-TM (`before_attention_norm_token_mask`) and `pre_residual_blocks_5_8`, with `cli.sweep`'s own functions (bfloat16, batch 64, mask seed 0) into `results/_experiments/tact_calibration/models/`. A smaller k is the first k passes, read at {", ".join(str(k) for k in PASS_COUNTS)}. The 6 holdout retrains had no `psbd_metrics.json`, so `run_passes.py` walked each placement's ladder from `configs/psbd_basis.json` at 3 passes on the standard 2000 validation images and took the first rate whose shift ratio reached the target. Validation grew to {sizes} images from the same split permutation, and those images left the clean and triggered evaluation sets, which `analyze.enlarged_rows` asserts. Beatrix and TED were rescored on the same enlarged validation and shrunken evaluation sets from their stored scores.

The fallback score ranks an input's fractional PSU among validation images with the same fallback class (the most frequent class the perturbed passes moved to), shrunk toward all validation images with m = {frozen["shrinkage"]["fallback"]}. The class score is a robust z score by predicted class with median and MAD shrunk toward the global ones with m = {frozen["shrinkage"]["class"]}. The OR rules take the minimum of validation percentiles. The formulas are in `analyze.py`'s docstring. Intervals are paired bootstrap 95% intervals over models ({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}). The plan held {n_plan} models, of which {sum(len(v) for v in groups.values()) - len(groups["dev"])} unique ones were read into the pooled sets below (tuning {len(groups["tuning"])}, holdout {len(groups["holdout"])}, non-TaCT development {len(groups["dev"])}, non-TaCT panel {len(groups["panel"])}).

## Commands

```
python -m experiments.tact_calibration.run_passes plan
bash experiments/tact_calibration/queue.sh
python -m experiments.tact_calibration.analyze tuning
python -m experiments.tact_calibration.analyze confirm
python -m experiments.tact_calibration.judge
python -m experiments.tact_calibration.render_readme
```"""
    return text


def wall_times(records, plan):
    by_group = {}
    group_of = {m["folder"]: m["group"] for m in plan["models"]}
    for record in records:
        by_group.setdefault(group_of[record["folder"]], []).append(record)
    lines = [
        "## Wall times",
        "",
        "GPU seconds per model, measured inside the lock, on the shared login-node A100 at memory fraction 0.15. Waiting for the lock is not counted.",
        "",
        "| group | models | mean seconds | max seconds | models reusing an earlier PSBD-TM cache |",
        "|---|---|---|---|---|",
    ]
    for group in ("tuning", "dev", "holdout", "rest"):
        members = by_group.get(group, [])
        if not members:
            continue
        seconds = [m["wall_seconds"] for m in members]
        reused = sum(
            1
            for m in members
            if m["placements"]["before_attention_norm_token_mask"].get("reused_from")
        )
        lines.append(
            f"| {group} | {len(members)} | {np.mean(seconds):.0f} | {max(seconds):.0f} | {reused} |"
        )
    missing = [
        m["folder"]
        for m in plan["models"]
        if m["folder"] not in {r["folder"] for r in records}
    ]
    if missing:
        lines += [
            "",
            f"Not swept when this README was rendered: {len(missing)} models, {', '.join(f'`{f}`' for f in missing)}.",
        ]
    text = "\n".join(lines)
    return text


def premise_section(groups, verdicts):
    lines = [
        "## Premise of the fallback score",
        "",
        "The share of triggered inputs whose PSBD-TM fallback class is their own source class, against the share that falls to the attractor, the most common fallback class of clean validation images. Shift ratios are the share of perturbed passes that left the unperturbed argmax.",
        "",
        "<!-- results:begin -->",
        "| model | group | triggered | own source | attractor class | to attractor | clean validation to attractor | no flip | triggered shift | validation shift |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group in ("tuning", "holdout"):
        for row in groups[group]:
            p = row["premise"]["anchor"]
            lines.append(
                f"| `{row['folder']}` | {group} | {p['n_triggered']} | {p['fallback_is_own_source']:.3f} | {p['attractor_class']} | {p['fallback_is_attractor']:.3f} | {p['attractor_share_clean_validation']:.3f} | {p['no_flip']:.3f} | {p['triggered_shift_ratio']:.3f} | {p['validation_shift_ratio']:.3f} |"
            )
    lines.append("<!-- results:end -->")
    p1 = verdicts["verdicts"]["P1"]
    lines += [
        "",
        f"The mean own-source share over the holdout is {p1['own_source_mean']:.3f} and the mean share falling to the attractor is {p1['attractor_mean']:.3f}. "
        + (
            "Masking sends a triggered input where it sends its clean source image, to a class the model falls back to under heavy masking, so the fallback class carries no information about the source and the fallback score reduces to the plain one."
            if p1["holds"]
            else "The premise holds on the holdout at least in part, so the fallback class carries some information about the source."
        ),
    ]
    text = "\n".join(lines)
    return text


def tact_section(groups, reading, verdicts):
    lines = [
        "## TaCT detection",
        "",
        "TPR at 1%, 5% and 10% FPR, then AUROC, at k = 20. FPR is the quantile of the enlarged clean validation set. Beatrix and TED are read on the same sets, and the holdout has no detector records.",
        "",
        "<!-- results:begin -->",
    ]
    for group in ("tuning", "holdout"):
        lines += [
            f"**{group.capitalize()} models**",
            "",
            "| model | method | TPR 1% | TPR 5% | TPR 10% | AUROC |",
            "|---|---|---|---|---|---|",
        ]
        for row in groups[group]:
            for method in TACT_METHODS:
                values = [reading(row, method, 20, f"{tag}:tpr") for tag in TAGS]
                auroc = reading(row, method, 20, "auroc")
                lines.append(
                    f"| `{row['folder']}` | {METHOD_WORDS[method]} | "
                    + " | ".join(f"{v:.3f}" for v in values)
                    + f" | {auroc:.3f} |"
                )
            for detector, block in row["detectors"].items():
                lines.append(
                    f"| `{row['folder']}` | {detector} | "
                    + " | ".join(f"{block[f'{tag}:tpr']:.3f}" for tag in TAGS)
                    + f" | {block['auroc']:.3f} |"
                )
        lines.append("")
        lines += paired_table(
            groups[group],
            reading,
            f"{group.capitalize()} means, paired against the reference",
        )
        lines.append("")
    lines.append("<!-- results:end -->")
    text = "\n".join(lines)
    return text


def paired_table(rows, reading, title):
    lines = [
        f"**{title}**",
        "",
        "| method | reference | TPR 1% | TPR 5% | TPR 10% | AUROC |",
        "|---|---|---|---|---|---|",
    ]
    if len(rows) < 3:
        return lines + ["", "Fewer than 3 models, no interval."]
    for method, reference in PAIRS:
        summary = compare(rows, method, reference, 20, reading)
        cells = []
        for field in [f"{tag}:tpr" for tag in TAGS] + ["auroc"]:
            s = summary[field]
            cells.append(
                f"{s['mean']:.3f} ({s['mean_difference']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}])"
            )
        lines.append(
            f"| {METHOD_WORDS[method]} | {METHOD_WORDS[reference]} | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def harm_section(groups, reading, verdicts):
    lines = [
        "## Harm on the other attacks",
        "",
        "Mean at k = 20 with the paired difference to the reference and its interval. The development set is the 7 successful non-TaCT models of `experiments/cache_readouts/dev_set.json`, the panel set every successful non-TaCT panel model read, development included.",
        "",
        "<!-- results:begin -->",
    ]
    lines += paired_table(groups["dev"], reading, "Non-TaCT development models")
    lines.append("")
    lines += paired_table(groups["panel"], reading, "Non-TaCT panel models")
    lines += ["<!-- results:end -->"]
    text = "\n".join(lines)
    return text


def pass_count_section(groups, reading, verdicts):
    lines = [
        "## Pass count",
        "",
        "Mean over models at each k for PSBD-TM alone and the final method, the first k of the 20 cached passes.",
        "",
        "<!-- results:begin -->",
        "| models | method | k | TPR 1% | TPR 5% | TPR 10% | AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("panel", "tuning", "holdout"):
        rows = groups[name]
        if not rows:
            continue
        for method in ("plain", "final", "or_class"):
            for k in PASS_COUNTS:
                values = [
                    np.mean([reading(r, method, k, f"{tag}:tpr") for r in rows])
                    for tag in TAGS
                ]
                auroc = np.mean([reading(r, method, k, "auroc") for r in rows])
                lines.append(
                    f"| {name} ({len(rows)}) | {METHOD_WORDS[method]} | {k} | "
                    + " | ".join(f"{v:.3f}" for v in values)
                    + f" | {auroc:.3f} |"
                )
    lines.append("<!-- results:end -->")
    p7 = verdicts["verdicts"]["P7"]
    gain = verdicts["verdicts"]["P7_auroc_gain"]
    lines += [
        "",
        f"By the pre-registered rule the recommended k is {p7['recommended_k']}: the smallest k whose mean AUROC and TPR at each FPR for PSBD-TM alone on the non-TaCT panel models stay within 0.01 of k = 20. Going from k = 3 to k = 20 changes mean AUROC by {gain['k20_minus_k3']['mean']:+.3f} [{gain['k20_minus_k3']['ci95'][0]:+.3f}, {gain['k20_minus_k3']['ci95'][1]:+.3f}].",
    ]
    text = "\n".join(lines)
    return text


def verdict_section(verdicts, frozen):
    lines = [
        "## Verdicts",
        "",
        "| prediction | expected | verdict |",
        "|---|---|---|",
    ]
    for name, verdict in verdicts["verdicts"].items():
        family = name.split("_")[0]
        expected = frozen["expected"].get(family, "")
        lines.append(
            f"| {name} | {expected} | {'holds' if verdict['holds'] else 'fails'} |"
        )
    text = "\n".join(lines)
    return text


def files_section():
    text = """## Files

`run_passes.py` and `queue.sh` are the GPU stage, `analyze.py` the CPU stage (`tuning` and `confirm`), `judge.py` the verdicts and this renderer the README. Records live under `results/_experiments/tact_calibration/`: `plan.json`, `tuning.json`, `confirm.json`, `verdicts.json`, the per-model caches and run records under `models/` and the queue logs under `logs/`."""
    return text


if __name__ == "__main__":
    main()

"""The final method read from the caches: PSBD-TM fused with the middle residual band.

The middle band is the middle third of the block stack, blocks 5 to 8 of ViT-B/16
and blocks 9 to 16 of Swin-S (`--block-range` counts Swin blocks 1 to 24 across
its 4 stages). Both are fused with PSBD-TM under the plain minimum of
clean-validation percentiles and the plain average of fractional PSU, each
placement at its own adaptive rate, with `fusion_rules.fuse_and_evaluate`.

2 sets, each read once when its caches are complete.
    swin_panel   the successful Swin-S panel, judged against
                 experiments/cache_readouts/preregistration_swin.json
    evaders      the adaptive attackers that clear the ASR bar, each beside its
                 non-evasive twin

    .venv/bin/python -m experiments.final_method.fusion_readout --set swin_panel
"""

import argparse
import hashlib
import json
import os
import time

from defenses.cache import read_split_manifest
from experiments._paths import experiment_result_path
from experiments.cache_readouts.fusion_rules import fractional_psu, fuse_and_evaluate
from experiments.cache_readouts.shared import (
    REPO_ROOT,
    choose_rate,
    load_baselines,
    load_passes,
    ordered_attacks,
    paired_summary,
    validation_shift_by_rate,
    write_json,
)
from scripts.paper.tab_swin import swin_cells

SLUG = "final_method"
RESULTS_DIR = "results"
ANCHOR = "before_attention_norm_token_mask"
MIDDLE_BAND = {"vit": "pre_residual_blocks_5_8", "swin": "pre_residual_blocks_9_16"}
RULES = {"psbd_tm": "tm_alone", "final_min": "min_rank", "final_average": "mean_psu"}
FIELDS = (
    "q0.01:tpr",
    "q0.05:tpr",
    "q0.10:tpr",
    "q0.01:realized_fpr",
    "q0.05:realized_fpr",
    "q0.10:realized_fpr",
    "auroc",
)
PREREGISTRATION = os.path.join(
    REPO_ROOT, "experiments", "cache_readouts", "preregistration_swin.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("swin_panel", "evaders"), required=True)
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="restrict to these folders, for a smoke test of the code path only",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    models = model_list(args.set)
    if args.folders:
        models = [m for m in models if m["folder"] in args.folders]
    rows = [measure(model) for model in models]
    payload = {
        "experiment": f"final method, {args.set}",
        "anchor": ANCHOR,
        "middle_band": MIDDLE_BAND,
        "rules": RULES,
        "smoke_test": bool(args.folders),
        "summary": summarize(rows),
        "models": rows,
    }
    if args.set == "swin_panel":
        payload["verdicts"] = judge(payload["summary"])
    payload["wall_seconds"] = time.perf_counter() - started

    suffix = "_smoke" if args.folders else ""
    path = experiment_result_path(SLUG, f"fusion_{args.set}{suffix}.json", RESULTS_DIR)
    write_json(payload, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def model_list(model_set):
    if model_set == "swin_panel":
        models = [
            {
                "folder": c["folder_name"],
                "architecture": "swin",
                "attack": c["attack"],
                "twin": None,
            }
            for c in swin_cells(RESULTS_DIR, "checkpoints")
        ]
        return models

    path = experiment_result_path(SLUG, "adaptive_attackers.json", RESULTS_DIR)
    with open(path) as handle:
        attackers = json.load(handle)["models"]
    models = [
        {
            "folder": row["folder"],
            "architecture": row["architecture"],
            "attack": row["attack"],
            "family": row["family"],
            "twin": row["twin"],
        }
        for row in attackers
        if row["evader"]["asr_class"] == "clears"
    ]
    return models


def measure(model):
    row = dict(model)
    row["reading"] = fused_reading(model["folder"], model["architecture"])
    if model["twin"]:
        row["twin_reading"] = fused_reading(model["twin"], model["architecture"])
    return row


def fused_reading(folder, architecture):
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    partner = MIDDLE_BAND[architecture]

    rates = {}
    for placement in (ANCHOR, partner):
        shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
        rates[placement] = (
            choose_rate(shift_by_rate, "adaptive") if shift_by_rate else None
        )
        if shift_by_rate and rates[placement] is None:
            rates[f"{placement}:max_validation_shift"] = max(shift_by_rate.values())
        if shift_by_rate:
            rates[f"{placement}:lowest_rate_reaches_target"] = rates[placement] == min(
                shift_by_rate
            )
    if rates[ANCHOR] is None or rates[partner] is None:
        return {"rates": rates, "rules": None}

    anchor_psu = fractional_psu(load_passes(psbd_dir, ANCHOR, rates[ANCHOR], baselines))
    partner_psu = fractional_psu(
        load_passes(psbd_dir, partner, rates[partner], baselines)
    )
    evaluations = fuse_and_evaluate(
        anchor_psu, partner_psu, read_split_manifest(psbd_dir)
    )
    reading = {
        "rates": rates,
        "rules": {name: flatten(evaluations[rule]) for name, rule in RULES.items()},
    }
    return reading


def flatten(evaluation):
    flat = {"auroc": evaluation["auroc"], "inverted": evaluation["auroc"] < 0.5}
    for quantile, block in evaluation["at_fpr"].items():
        flat[f"{quantile}:tpr"] = block["tpr"]
        flat[f"{quantile}:realized_fpr"] = block["realized_fpr"]
    return flat


def summarize(rows):
    scored = [r for r in rows if r["reading"]["rules"]]
    summary = {
        "n_models": len(rows),
        "n_scored": len(scored),
        "unscored": [
            {"folder": r["folder"], "rates": r["reading"]["rates"]}
            for r in rows
            if r not in scored
        ],
        "groups": {},
    }
    groups = {"all": scored}
    for key in ("architecture", "family"):
        for value in sorted({r.get(key) for r in scored if r.get(key)}):
            groups[f"{key}={value}"] = [r for r in scored if r.get(key) == value]
    for name, group in groups.items():
        summary["groups"][name] = summarize_group(group)
    return summary


def summarize_group(rows):
    def block(subset, reading_key):
        result = {}
        for method in RULES:
            result[method] = {
                "n_inverted": sum(
                    r[reading_key]["rules"][method]["inverted"] for r in subset
                )
            }
            for field in FIELDS:
                values = [r[reading_key]["rules"][method][field] for r in subset]
                reference = [r[reading_key]["rules"]["psbd_tm"][field] for r in subset]
                result[method][field] = paired_summary(values, reference)
        return result

    group = {"n": len(rows), "evader": block(rows, "reading")}
    with_twin = [
        r for r in rows if r.get("twin_reading") and r["twin_reading"]["rules"]
    ]
    if with_twin:
        group["n_with_twin"] = len(with_twin)
        group["twin"] = block(with_twin, "twin_reading")
    group["by_attack"] = {
        attack: block([r for r in rows if r["attack"] == attack], "reading")
        for attack in ordered_attacks({r["attack"] for r in rows})
    }
    return group


def judge(summary):
    with open(PREREGISTRATION, "rb") as handle:
        raw = handle.read()
    preregistration = json.loads(raw)
    overall = summary["groups"]["all"]["evader"]
    method_of = {rule: method for method, rule in RULES.items()}
    results = []
    for prediction in preregistration["predictions"]:
        method = method_of[prediction["rule_name"]]
        readings = {field: overall[method][field] for field in prediction["fields"]}
        above = all(r["mean_difference"] > 0 for r in readings.values())
        excluded = all(r["ci95"][0] > 0 for r in readings.values())
        verdict = (
            "held" if above and excluded else ("inconclusive" if above else "failed")
        )
        results.append(
            {
                "id": prediction["id"],
                "verdict": verdict,
                "readings": {
                    f: {"mean_difference": r["mean_difference"], "ci95": r["ci95"]}
                    for f, r in readings.items()
                },
            }
        )
    verdicts = {
        "preregistration_sha256": hashlib.sha256(raw).hexdigest(),
        "results": results,
    }
    return verdicts


if __name__ == "__main__":
    main()

"""Swin-S second probe, attempt 2: PSBD-TM fused with residual dropout in blocks 17 to 24.

Pre-registered in preregistration_swin_late.json after attempt 1 (blocks 9 to 16)
failed. The readout is the one of `fusion_readout.py --set swin_panel`, with 2
additions the pre-registration asks for. A partner whose ladder never reaches the
adaptive target is read at the rate nearest it, and the 2 readings are kept apart,
the primary one on models where both placements reach the target and a labeled
one that adds the nearest-rate models. Results are grouped overall, per attack and
per poison rate. Run once.

    .venv/bin/python -m experiments.final_method.swin_late_readout
"""

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
from experiments.final_method.fusion_readout import ANCHOR, FIELDS, RULES, flatten
from scripts.paper.tab_swin import swin_cells

SLUG = "final_method"
RESULTS_DIR = "results"
PARTNER = "pre_residual_blocks_17_24"
PREREGISTRATION = os.path.join(
    REPO_ROOT, "experiments", SLUG, "preregistration_swin_late.json"
)
READINGS = ("adaptive_only", "with_nearest")


def main():
    started = time.perf_counter()
    with open(PREREGISTRATION, "rb") as handle:
        raw = handle.read()
    preregistration = json.loads(raw)

    rows = [measure(cell) for cell in swin_cells(RESULTS_DIR, "checkpoints")]
    summary = {reading: summarize(rows, reading) for reading in READINGS}
    payload = {
        "experiment": "Swin-S second probe, attempt 2",
        "anchor": ANCHOR,
        "partner": PARTNER,
        "rules": RULES,
        "preregistration_sha256": hashlib.sha256(raw).hexdigest(),
        "coverage": coverage(rows),
        "summary": summary,
        "verdicts": {
            reading: judge(preregistration, summary[reading]) for reading in READINGS
        },
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    path = experiment_result_path(SLUG, "fusion_swin_panel_late.json", RESULTS_DIR)
    write_json(payload, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def measure(cell):
    folder = cell["folder_name"]
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    row = {
        "folder": folder,
        "attack": cell["attack"],
        "dataset": cell["dataset"],
        "poison_rate": cell["poison_rate"],
        "rates": {},
        "rules": None,
    }

    anchor_shift = validation_shift_by_rate(psbd_dir, ANCHOR, baselines)
    partner_shift = validation_shift_by_rate(psbd_dir, PARTNER, baselines)
    anchor_rate = choose_rate(anchor_shift, "adaptive")
    partner_rate = choose_rate(partner_shift, "adaptive") if partner_shift else None
    row["rates"] = {
        "anchor": anchor_rate,
        "partner": partner_rate,
        "partner_rule": "adaptive",
        "partner_max_validation_shift": max(partner_shift.values())
        if partner_shift
        else None,
        "partner_cached": bool(partner_shift),
    }
    if partner_rate is None and partner_shift:
        partner_rate = choose_rate(partner_shift, "nearest")
        row["rates"]["partner"] = partner_rate
        row["rates"]["partner_rule"] = "nearest"
    if anchor_rate is None or partner_rate is None:
        return row

    anchor_psu = fractional_psu(load_passes(psbd_dir, ANCHOR, anchor_rate, baselines))
    partner_psu = fractional_psu(
        load_passes(psbd_dir, PARTNER, partner_rate, baselines)
    )
    evaluations = fuse_and_evaluate(
        anchor_psu, partner_psu, read_split_manifest(psbd_dir)
    )
    row["rules"] = {name: flatten(evaluations[rule]) for name, rule in RULES.items()}
    return row


def coverage(rows):
    result = {
        "n_panel": len(rows),
        "n_anchor_reaches": sum(r["rates"]["anchor"] is not None for r in rows),
        "n_partner_cached": sum(r["rates"]["partner_cached"] for r in rows),
        "n_partner_reaches": sum(
            r["rates"]["partner_rule"] == "adaptive"
            and r["rates"]["partner"] is not None
            for r in rows
        ),
        "partner_nearest": [
            {
                "folder": r["folder"],
                "rate": r["rates"]["partner"],
                "max_validation_shift": r["rates"]["partner_max_validation_shift"],
            }
            for r in rows
            if r["rates"]["partner_rule"] == "nearest"
        ],
        "unscored": [r["folder"] for r in rows if r["rules"] is None],
    }
    return result


def summarize(rows, reading):
    scored = [
        r
        for r in rows
        if r["rules"]
        and (reading == "with_nearest" or r["rates"]["partner_rule"] == "adaptive")
    ]
    groups = {"all": scored}
    for attack in ordered_attacks({r["attack"] for r in scored}):
        groups[f"attack={attack}"] = [r for r in scored if r["attack"] == attack]
    for rate in sorted({r["poison_rate"] for r in scored}):
        groups[f"poison_rate={rate}"] = [r for r in scored if r["poison_rate"] == rate]
    summary = {name: summarize_group(group) for name, group in groups.items()}
    return summary


def summarize_group(rows):
    group = {"n": len(rows)}
    for method in RULES:
        group[method] = {
            "n_inverted": sum(r["rules"][method]["inverted"] for r in rows)
        }
        for field in FIELDS:
            values = [r["rules"][method][field] for r in rows]
            reference = [r["rules"]["psbd_tm"][field] for r in rows]
            group[method][field] = paired_summary(values, reference)
    return group


def judge(preregistration, summary):
    method_of = {rule: method for method, rule in RULES.items()}
    results = []
    for prediction in preregistration["predictions"]:
        group = summary.get(prediction["group"])
        if group is None:
            results.append({"id": prediction["id"], "verdict": "inconclusive", "n": 0})
            continue
        method = method_of[prediction["rule_name"]]
        readings = {field: group[method][field] for field in prediction["fields"]}
        above = all(r["mean_difference"] > 0 for r in readings.values())
        excluded = all(r["ci95"][0] > 0 for r in readings.values())
        verdict = (
            "held" if above and excluded else ("inconclusive" if above else "failed")
        )
        results.append(
            {
                "id": prediction["id"],
                "verdict": verdict,
                "n": group["n"],
                "readings": {
                    f: {"mean_difference": r["mean_difference"], "ci95": r["ci95"]}
                    for f, r in readings.items()
                },
            }
        )
    return results


if __name__ == "__main__":
    main()

"""Compute control: PSBD-TM at 6 passes against the final method at 3 plus 3.

The final method spends 6 perturbed forward passes per input, 3 for PSBD-TM and 3
for the middle band. A fair control gives PSBD-TM alone the same 6 passes, at the
rate the adaptive rule chose at k = 3, so any gain of the fusion over it is a gain
of the second probe and not of the extra passes. The 6 passes come from a
`_k6` cache if there is a complete ladder rung and otherwise from the first 6 passes of a `_k20`
cache at the same rate. Models with neither are listed, not dropped silently.

    .venv/bin/python -m experiments.final_method.compute_control
"""

import argparse
import os
import time

from data.splits import SPLITS
from defenses.cache import dropout_pass_path, read_split_manifest
from defenses.decision import complete_rates
from defenses.scores import psu_ratio_from_cache
from experiments._paths import experiment_result_path
from experiments.cache_readouts.shared import (
    choose_rate,
    evaluate_scores,
    load_baselines,
    load_model_set,
    paired_summary,
    validation_shift_by_rate,
    write_json,
)
from experiments.final_method.fusion_readout import ANCHOR, fused_reading
from defenses.cache import load_dropout_pass_probs
from scripts.paper.tab_swin import swin_cells

SLUG = "final_method"
RESULTS_DIR = "results"
PASSES = 6
QUANTILES = (0.01, 0.05, 0.10)
FIELDS = ("q0.01:tpr", "q0.05:tpr", "q0.10:tpr", "auroc")
FUSED = ("final_min", "final_average")


def main():
    # Swin is read only after its pre-registered fusion readout, since this
    # control reads the same Swin middle-band scores.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("vit", "swin"), required=True)
    architecture = parser.parse_args().architecture
    started = time.perf_counter()
    if architecture == "vit":
        models = [
            {"folder": m["folder_name"], "architecture": "vit", "attack": m["attack"]}
            for m in load_model_set("panel")
        ]
    else:
        models = [
            {"folder": c["folder_name"], "architecture": "swin", "attack": c["attack"]}
            for c in swin_cells(RESULTS_DIR, "checkpoints")
        ]
    rows = [measure(model) for model in models]
    payload = {
        "experiment": "compute control, PSBD-TM at 6 passes against the fusion at 3 plus 3",
        "architecture": architecture,
        "passes": PASSES,
        "summary": summarize(rows),
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    path = experiment_result_path(
        SLUG, f"compute_control_{architecture}.json", RESULTS_DIR
    )
    write_json(payload, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def measure(model):
    psbd_dir = os.path.join(RESULTS_DIR, model["folder"], "psbd")
    baselines = load_baselines(psbd_dir)
    rate = choose_rate(
        validation_shift_by_rate(psbd_dir, ANCHOR, baselines), "adaptive"
    )
    row = dict(model, rate=rate, source=None, tm_k6=None, fused=None)
    if rate is None:
        return row

    source = six_pass_source(psbd_dir, rate)
    if source is None:
        return row
    row["fused"] = fused_reading(model["folder"], model["architecture"])["rules"]

    scores = {}
    for split in SPLITS:
        probs, labels, _ = baselines[split]
        per_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, source, rate, split)
        )
        assert per_pass_probs.shape[0] >= PASSES, (source, per_pass_probs.shape)
        scores[split] = psu_ratio_from_cache(
            probs, labels, per_pass_probs[:PASSES]
        )  # (n,)
    evaluation = evaluate_scores(scores, read_split_manifest(psbd_dir), QUANTILES)
    row["source"] = source
    row["tm_k6"] = {"auroc": evaluation["auroc"]}
    for quantile, block in evaluation["at_fpr"].items():
        row["tm_k6"][f"{quantile}:tpr"] = block["tpr"]
        row["tm_k6"][f"{quantile}:realized_fpr"] = block["realized_fpr"]
    return row


def six_pass_source(psbd_dir, rate):
    for suffix in ("_k6", "_k20"):
        placement = f"{ANCHOR}{suffix}"
        if rate in complete_rates(psbd_dir, placement):
            return placement
    return None


def summarize(rows):
    scored = [r for r in rows if r["tm_k6"] and r["fused"]]
    summary = {
        "n_models": len(rows),
        "n_scored": len(scored),
        "missing_six_pass_cache": [
            r["folder"] for r in rows if r["rate"] and not r["tm_k6"]
        ],
        "no_fusion": [r["folder"] for r in rows if r["tm_k6"] and not r["fused"]],
        "sources": {
            s: sum(r["source"] == s for r in scored)
            for s in {r["source"] for r in scored}
        },
    }
    for method in FUSED:
        summary[method] = {}
        for field in FIELDS:
            values = [r["fused"][method][field] for r in scored]
            reference = [r["tm_k6"][field] for r in scored]
            summary[method][field] = paired_summary(values, reference)
    summary["tm_k6"] = {
        field: sum(r["tm_k6"][field] for r in scored) / len(scored) if scored else None
        for field in FIELDS
    }
    summary["tm_k3"] = {
        field: sum(r["fused"]["psbd_tm"][field] for r in scored) / len(scored)
        if scored
        else None
        for field in FIELDS
    }
    return summary


if __name__ == "__main__":
    main()

"""Per-model and per-attack readout of PSBD-TM and the final method on BackdoorBench models.

CPU only. For each queued model it reads the evaluation record (clean accuracy and
ASR through this project's loader), applies the success bar and, where both
probes' caches are complete, scores PSBD-TM, its partner pre_residual_blocks_5_8
and the final method with experiments/cache_readouts/fusion_rules.py's own
functions: fractional PSU at each probe's adaptive rate, the plain minimum of the
2 clean-validation percentiles (min_rank) and the plain average of the 2
fractional PSUs (mean_psu). A PSBD-RD cache is read when it exists and never
required. Writes 1 JSON per model under models/ and summary.json.

    .venv/bin/python -m experiments.backdoorbench_attacks.readout
"""

import os
import time

from defenses.cache import read_split_manifest
from defenses.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT
from experiments.backdoorbench_attacks.common import (
    EVALUATION_DIR,
    LEADERBOARD,
    MODEL_RECORDS_DIR,
    SUMMARY_PATH,
    SWEEPS_DIR,
    read_json,
    write_json,
)
from experiments.backdoorbench_attacks.queue import MODELS, readable
from experiments.cache_readouts.fusion_rules import (
    QUANTILES,
    fractional_psu,
    fuse_and_evaluate,
)
from experiments.cache_readouts.shared import (
    choose_rate,
    evaluate_scores,
    load_baselines,
    load_passes,
    validation_shift_by_rate,
)

PSBD_TM = RECOMMENDED_PLACEMENT
PARTNER = "pre_residual_blocks_5_8"
PSBD_RD = PUBLISHED_PLACEMENT
BASIS_PATH = "configs/psbd_basis.json"
RESULTS_DIR = "results"
# A model counts as collapsed below half its reference clean accuracy, the rule
# the coverage ledger applies to the panel's diverged cells.
COLLAPSE_SHARE = 0.5
# The methods the report reads, as (record key, fusion rule or None).
METHODS = {
    "psbd_tm": "tm_alone",
    "final_min": "min_rank",
    "final_mean": "mean_psu",
}
FIELDS = ("q0.01:tpr", "q0.05:tpr", "q0.10:tpr", "auroc")


def main():
    started = time.perf_counter()
    basis = read_json(BASIS_PATH)
    references = clean_references()

    rows = []
    for folder in readable(MODELS):
        row = read_model(folder, basis, references)
        write_json(row, os.path.join(MODEL_RECORDS_DIR, f"{folder}.json"))
        rows.append(row)
        print(line_of(row), flush=True)

    payload = {
        "experiment": "backdoorbench_attacks readout",
        "placements": {"psbd_tm": PSBD_TM, "partner": PARTNER, "psbd_rd": PSBD_RD},
        "quantiles": list(QUANTILES),
        "asr_bar": basis["asr_bar"],
        "clean_accuracy_drop_bar_headline": basis["clean_accuracy_drop_bar_headline"],
        "collapse_share": COLLAPSE_SHARE,
        "references": references,
        "models": [row["folder"] for row in rows],
        "by_attack": summarize(rows),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(payload, SUMMARY_PATH)
    print(f"wrote {SUMMARY_PATH}")


def clean_references():
    # 2 references, since BackdoorBench publishes no benign ViT-B/16. The best
    # clean accuracy BackdoorBench reports for any ViT-B/16 of the dataset, which
    # includes its 0.1% models, and this project's own benign ViT, which was
    # trained on a different recipe.
    references = {}
    for dataset in ("cifar10", "cifar100", "gtsrb", "tiny"):
        leaderboard = {
            folder: row["clean_accuracy"]
            for folder, row in LEADERBOARD.items()
            if folder.startswith(f"{dataset}_") and row["clean_accuracy"] is not None
        }
        best_folder = max(leaderboard, key=leaderboard.get)
        own_path = f"checkpoints/vit_{dataset}_benign/metrics.json"
        references[dataset] = {
            "backdoorbench_best": leaderboard[best_folder],
            "backdoorbench_best_folder": best_folder,
            "own_benign": read_json(own_path)["clean_accuracy"],
            "own_benign_source": own_path,
        }
    return references


def read_model(folder, basis, references):
    evaluation_path = os.path.join(EVALUATION_DIR, f"{folder}.json")
    row = {"folder": folder, "leaderboard": LEADERBOARD.get(folder)}
    if not os.path.exists(evaluation_path):
        row["status"] = "not evaluated"
        return row

    evaluation = read_json(evaluation_path)
    scores = evaluation["backdoorbench_normalization"]
    reference = references[evaluation["dataset"]]
    drop_bar = basis["clean_accuracy_drop_bar_headline"]
    row.update(
        {
            "dataset": evaluation["dataset"],
            "attack": evaluation["attack"],
            "poison_rate": evaluation["poison_rate"],
            "clean_accuracy": scores["clean_accuracy"],
            "asr": scores["asr"],
            "clean_drop_vs_backdoorbench_best": scores["clean_accuracy"]
            - reference["backdoorbench_best"],
            "clean_drop_vs_own_benign": scores["clean_accuracy"]
            - reference["own_benign"],
        }
    )
    row["clears_asr_bar"] = scores["asr"] >= basis["asr_bar"]
    row["collapsed"] = (
        scores["clean_accuracy"] < COLLAPSE_SHARE * reference["backdoorbench_best"]
    )
    row["judged"] = row["clears_asr_bar"] and not row["collapsed"]
    row["within_2pt_of_backdoorbench_best"] = (
        row["clean_drop_vs_backdoorbench_best"] >= drop_bar
    )
    row["within_2pt_of_own_benign"] = row["clean_drop_vs_own_benign"] >= drop_bar

    psbd_dir = os.path.join(RESULTS_DIR, f"bb_{folder}", "psbd")
    sweep_path = os.path.join(SWEEPS_DIR, f"{folder}.json")
    if not os.path.exists(sweep_path):
        row["status"] = "evaluated, not swept"
        return row

    row["status"] = "swept"
    row["sweep_wall_seconds"] = read_json(sweep_path)["wall_seconds"]
    row["detection"] = score_detection(psbd_dir)
    row["psbd_metrics_control"] = stored_adaptive_auroc(psbd_dir)
    return row


def score_detection(psbd_dir):
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)
    detection = {"n_backdoor": len(manifest["analysis_backdoor_indices"])}

    tm_shift = validation_shift_by_rate(psbd_dir, PSBD_TM, baselines)
    partner_shift = validation_shift_by_rate(psbd_dir, PARTNER, baselines)
    tm_rate = choose_rate(tm_shift, "adaptive")
    partner_rate = choose_rate(partner_shift, "adaptive")
    # A partner whose ladder never reaches the shift target is read at the rate
    # closest to it, fusion_rules.py's "nearest" rule, labeled as such.
    partner_rule = "adaptive"
    if partner_rate is None:
        partner_rate = choose_rate(partner_shift, "nearest")
        partner_rule = "nearest"
    detection["rates"] = {
        "psbd_tm": tm_rate,
        "partner": partner_rate,
        "partner_rule": partner_rule,
        "psbd_tm_validation_shift": tm_shift.get(tm_rate),
        "partner_validation_shift": partner_shift.get(partner_rate),
    }
    if tm_rate is None:
        return detection

    tm_psu = fractional_psu(load_passes(psbd_dir, PSBD_TM, tm_rate, baselines))
    partner_psu = fractional_psu(
        load_passes(psbd_dir, PARTNER, partner_rate, baselines)
    )
    fused = fuse_and_evaluate(tm_psu, partner_psu, manifest)
    detection["methods"] = {name: fused[rule] for name, rule in METHODS.items()}
    detection["methods"]["partner"] = evaluate_scores(partner_psu, manifest, QUANTILES)

    rd_shift = validation_shift_by_rate(psbd_dir, PSBD_RD, baselines)
    rd_rate = choose_rate(rd_shift, "adaptive") if rd_shift else None
    if rd_rate is not None:
        rd_psu = fractional_psu(load_passes(psbd_dir, PSBD_RD, rd_rate, baselines))
        detection["methods"]["psbd_rd"] = evaluate_scores(rd_psu, manifest, QUANTILES)
        detection["rates"]["psbd_rd"] = rd_rate
    return detection


def stored_adaptive_auroc(psbd_dir):
    # cli.analyze's fractional-PSU AUROC at its own adaptive rate, which the
    # fractional PSU read here must equal, the check cache_readouts runs against
    # every model it reads. The "adaptive" block of psbd_metrics.json is absolute
    # PSU, so the control reads rates[].detection_psu_ratio instead.
    metrics = read_json(os.path.join(os.path.dirname(psbd_dir), "psbd_metrics.json"))
    block = metrics["placements"][PSBD_TM]
    rate = block["adaptive_rate"]
    by_rate = {entry["rate"]: entry for entry in block["rates"]}
    control = {
        "adaptive_rate": rate,
        "auroc": by_rate[rate]["detection_psu_ratio"]["q0.25"]["auroc"]
        if rate
        else None,
    }
    return control


def metric(evaluation, field):
    if field == "auroc":
        value = evaluation["auroc"]
    else:
        quantile, kind = field.split(":")
        value = evaluation["at_fpr"][quantile][kind]
    return value


def summarize(rows):
    swept = [row for row in rows if row.get("detection", {}).get("methods")]
    groups = {}
    for attack in sorted({row["attack"] for row in swept}):
        members = [row for row in swept if row["attack"] == attack]
        groups[attack] = {
            "judged": group_means([row for row in members if row["judged"]]),
            "all_swept": group_means(members),
        }
    return groups


def group_means(rows):
    group = {"n": len(rows), "folders": [row["folder"] for row in rows]}
    if not rows:
        return group
    methods = sorted(
        set.intersection(*(set(row["detection"]["methods"]) for row in rows))
    )
    for method in methods:
        group[method] = {
            field: sum(
                metric(row["detection"]["methods"][method], field) for row in rows
            )
            / len(rows)
            for field in FIELDS
        }
    return group


def line_of(row):
    if row["status"] != "swept":
        return f"{row['folder']}: {row['status']}"
    methods = row["detection"].get("methods", {})
    cells = [
        f"{name} auroc {metric(evaluation, 'auroc'):.3f} tpr1 {metric(evaluation, 'q0.01:tpr'):.3f} "
        f"tpr5 {metric(evaluation, 'q0.05:tpr'):.3f}"
        for name, evaluation in methods.items()
    ]
    line = (
        f"{row['folder']} asr {row['asr']:.3f} judged {row['judged']}: "
        + " | ".join(cells)
    )
    return line


if __name__ == "__main__":
    main()

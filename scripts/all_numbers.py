"""Every detection number in the repository as 1 table, 1 row per (model, defense).

The paper reads a few means off the coverage ledger and the per-model records.
This table holds every reading those means are taken over, so any number in the
paper can be recomputed with a filter and a mean, and any model the paper leaves
out can be looked up with the reason it was left out.

Models. Every ViT panel cell of the 4 paper datasets the coverage ledger lists,
implanted or not, the Swin cells the same panel rule selects
(scripts.paper._common.swin_coverage, the population scripts/paper/tab_swin.py
reads) and the benign reference of each dataset and architecture.
Each model carries its ledger verdicts: asr_class, diverged, source_mapped and
the 2 success verdicts successful_2pt and successful_5pt. SIG models carry an
audit_note and are kept, since their trigger amplitude is under audit.

Defenses. Every PSBD placement cached for a model, at the adaptive rule (the
smallest rate whose clean-validation shift ratio reaches 0.8). PSBD-TM
(before_attention_norm_token_mask) and PSBD-RD (post_residual) also at the
matched rule (the rate whose shift ratio sits nearest 0.6). Every competitor
detector registered in detectors.DETECTOR_NAMES. PSBD readings come from the
chosen rate's detection_psu_ratio block through cli.compare_detectors.psbd_values,
the reader every paper generator uses. Detector readings come through
cli.compare_detectors.detector_values, which drops failed and smoke records.

Metrics. AUROC is the one-sided area with triggered images as the positive class
and a low score as poisoned evidence, never flipped. TPR and FPR are read at the
threshold set at the 0.10, 0.20 and 0.25 quantile of the 2000 clean validation
scores, where FPR is the realized rate on the paired clean test images. See
docs/metrics.md for the formulas.

    PYTHONPATH=. python scripts/all_numbers.py
"""

import argparse
import csv
import json
import os

from cli.compare_detectors import detector_values, psbd_rate, psbd_values
from defenses.decision import (
    PLACEMENT_MATCH_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from detectors import DETECTOR_NAMES
from scripts.coverage_ledger import (
    SUCCESS_BARS,
    clean_accuracy_of,
    retarget_declaration,
)
from scripts.paper._common import (
    HEADLINE_KEY,
    PANEL_DATASETS,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    rate_row,
    swin_coverage,
)
from utils.provenance import current_git_commit, utc_timestamp

DEFAULT_OUT_DIR = os.path.join("results", "all_numbers")
# The 2 false-positive budgets the paper reports, and the headline quantile.
QUANTILE_KEYS = ("q0.10", "q0.20", "q0.25")
# The 2 placements with a name of their own and the reader-facing names they go by.
# The suffix a matched-rule reading of a named placement carries, built from the
# target so a change of target renames the rows rather than mislabeling them.
MATCHED_SUFFIX = f"matched {PLACEMENT_MATCH_TARGET:g}"
NAMED_PLACEMENTS = {
    RECOMMENDED_PLACEMENT: "PSBD-TM",
    PUBLISHED_PLACEMENT: "PSBD-RD",
}
MODEL_FIELDS = (
    "architecture",
    "folder_name",
    "kind",
    "in_panel",
    "dataset",
    "attack",
    "label_mode",
    "poison_rate",
    "realized_poison_rate",
    "target_label",
    "asr",
    "clean_accuracy",
    "clean_accuracy_benign",
    "clean_accuracy_drop",
    "asr_class",
    "diverged",
    "source_mapped",
    "source_class_accuracy",
    *SUCCESS_BARS,
    "audit_note",
)
READING_FIELDS = (
    "defense",
    "family",
    "placement",
    "rule",
    "status",
    "rate",
    "validation_shift_ratio",
    "auroc",
    "auroc_absolute_psu",
    *(f"{field}_{key}" for key in QUANTILE_KEYS for field in ("tpr", "fpr")),
    "n_backdoor",
)
ROW_FIELDS = MODEL_FIELDS + READING_FIELDS
# Attacks whose readings are kept but flagged until an audit reports. SIG models
# trained before the trigger amplitude changed learned a fainter sinusoid than
# attacks/sig.py now builds, so a cache can hold triggered images of either
# amplitude, and vit_cifar10_sig_0_1's PSBD cache and detector records already
# disagree on its triggered images.
UNDER_AUDIT = {
    "sig": "SIG trigger amplitude under audit: caches may mix the trained and the "
    "current amplitude",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--declaration", default=os.path.join("configs", "psbd_basis.json")
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def model_record(cell: dict, architecture: str, kind: str, in_panel: bool) -> dict:
    """A ledger cell cut down to the fields every row repeats."""
    record = {field: cell.get(field) for field in MODEL_FIELDS}
    record["architecture"] = architecture
    record["kind"] = kind
    record["in_panel"] = in_panel
    record["audit_note"] = UNDER_AUDIT.get(cell.get("attack"))
    return record


def vit_models(results_dir: str) -> list[dict]:
    """Every ViT panel cell of the paper datasets, as the written ledger lists it.

    Raises when the ledger predates the success verdicts, since every row carries
    them and a missing verdict would read as a failed model.
    """
    coverage = load_coverage(results_dir)
    missing = [
        cell["folder_name"]
        for cell in coverage["cells"]
        if any(field not in cell for field in SUCCESS_BARS)
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} ledger cells carry no success verdict, "
            "run scripts/coverage_ledger.py first"
        )
    models = [
        model_record(cell, "vit", "attack", in_panel=True) for cell in coverage["cells"]
    ]
    return models


def swin_models(ledger: dict) -> list[dict]:
    """Every Swin panel cell of the paper datasets, the ViT panel rule applied to Swin."""
    models = [
        model_record(cell, "swin", "attack", in_panel=True)
        for cell in ledger["cells"]
        if cell["dataset"] in PANEL_DATASETS
    ]
    return models


def benign_models(
    results_dir: str, checkpoints_dir: str, declaration: dict
) -> list[dict]:
    """The benign reference of every paper dataset on both architectures.

    Their triggered split is clean images stamped with a trigger the model never
    learned, so a detector's AUROC there is its false-alarm behavior, never a
    detection.
    """
    models = []
    for architecture in ("vit", "swin"):
        references = retarget_declaration(declaration, architecture)["benign_reference"]
        for dataset in PANEL_DATASETS:
            folder = references[dataset]
            accuracy = clean_accuracy_of(checkpoints_dir, results_dir, folder)
            cell = {
                "folder_name": folder,
                "dataset": dataset,
                "attack": "benign",
                "clean_accuracy": accuracy,
                "clean_accuracy_benign": accuracy,
                "clean_accuracy_drop": 0.0 if accuracy is not None else None,
                "asr_class": "benign",
                **{field: False for field in SUCCESS_BARS},
            }
            models.append(model_record(cell, architecture, "benign", in_panel=False))
    return models


def quantile_fields(values: dict | None) -> dict:
    """AUROC plus TPR and realized FPR at every reported quantile, from a ladder block."""
    fields = {}
    headline = (values or {}).get(HEADLINE_KEY) or {}
    fields["auroc"] = headline.get("auroc")
    for key in QUANTILE_KEYS:
        block = (values or {}).get(key) or {}
        fields[f"tpr_{key}"] = block.get("tpr")
        fields[f"fpr_{key}"] = block.get("fpr")
    return fields


def psbd_reading(report: dict, placement: str, rule: str) -> dict:
    """1 PSBD placement at 1 rate rule: the chosen rate, its shift ratio and its metrics.

    A rule that chose no rate (the ladder never reached the shift target) gives a
    row with status shift_target_unreached and no metrics, so the gap is visible.
    """
    block = report["placements"][placement]
    values = psbd_values(report, placement, rule)
    rate = psbd_rate(block, rule)
    chosen = rate_row(block, rate) if rate is not None else None
    name = NAMED_PLACEMENTS.get(placement, f"PSBD {placement}")
    reading = {
        "defense": name if rule == "adaptive" else f"{name} {MATCHED_SUFFIX}",
        "family": "psbd",
        "placement": placement,
        "rule": rule,
        "status": "scored" if values is not None else "shift_target_unreached",
        "rate": rate,
        "validation_shift_ratio": (chosen or {})
        .get("shift_ratio", {})
        .get("validation"),
        "auroc_absolute_psu": ((chosen or {}).get("detection") or {})
        .get(HEADLINE_KEY, {})
        .get("auroc"),
        "n_backdoor": (values or {}).get("_n_backdoor"),
        **quantile_fields(values),
    }
    return reading


def psbd_readings(results_dir: str, folder: str) -> list[dict]:
    """Every cached placement at the adaptive rule, the 2 named ones also at the matched rule."""
    report = load_psbd_metrics(results_dir, folder)
    if report is None:
        return []
    readings = []
    for placement in sorted(report["placements"]):
        readings.append(psbd_reading(report, placement, "adaptive"))
        if placement in NAMED_PLACEMENTS:
            readings.append(psbd_reading(report, placement, "matched"))
    return readings


def detector_readings(results_dir: str, folder: str) -> list[dict]:
    """Every registered competitor detector with a record on this model.

    A failed record and a smoke record (scored on a max_samples subset) are kept
    as rows with that status and no metrics, the way the detector tables skip them.
    """
    readings = []
    for name in DETECTOR_NAMES:
        notes = {"failed": [], "smoke": []}
        values = detector_values(results_dir, folder, name, notes)
        if values is None and not notes["failed"] and not notes["smoke"]:
            continue
        status = "scored"
        if notes["failed"]:
            status = "failed"
        elif notes["smoke"]:
            status = "smoke"
        readings.append(
            {
                "defense": name,
                "family": "detector",
                "placement": None,
                "rule": None,
                "status": status,
                "rate": None,
                "validation_shift_ratio": None,
                "auroc_absolute_psu": None,
                "n_backdoor": (values or {}).get("_n_backdoor"),
                **quantile_fields(values),
            }
        )
    return readings


def model_rows(results_dir: str, model: dict) -> list[dict]:
    """1 row per defense reading on this model, each carrying the model's fields."""
    folder = model["folder_name"]
    readings = psbd_readings(results_dir, folder) + detector_readings(
        results_dir, folder
    )
    rows = [{**model, **reading} for reading in readings]
    return rows


def write_outputs(
    out_dir: str, models: list[dict], rows: list[dict], declaration: dict
) -> None:
    """all_numbers.csv (the rows) and all_numbers.json (the rows, the models, the bars)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_numbers.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generator": "scripts/all_numbers.py",
        "written_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "bars": {
            "asr_bar": declaration["asr_bar"],
            **{field: declaration[key] for field, key in SUCCESS_BARS.items()},
        },
        "models": models,
        "rows": rows,
    }
    with open(os.path.join(out_dir, "all_numbers.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def summary_lines(models: list[dict], rows: list[dict]) -> list[str]:
    """What was written, per architecture: models, success counts and rows."""
    lines = []
    for architecture in ("vit", "swin"):
        attacks = [
            model
            for model in models
            if model["architecture"] == architecture and model["in_panel"]
        ]
        clears = sum(model["asr_class"] == "clears" for model in attacks)
        success = {
            field: sum(bool(model[field]) for model in attacks)
            for field in SUCCESS_BARS
        }
        n_rows = sum(row["architecture"] == architecture for row in rows)
        lines.append(
            f"{architecture}: {len(attacks)} panel models, {clears} clear, "
            + ", ".join(f"{count} {field}" for field, count in success.items())
            + f", {n_rows} rows"
        )
    return lines


def main() -> None:
    args = parse_args()
    declaration = load_declaration(args.declaration)

    swin = swin_coverage(args.results_dir, args.checkpoints_dir, args.declaration)

    models = (
        vit_models(args.results_dir)
        + swin_models(swin)
        + benign_models(args.results_dir, args.checkpoints_dir, declaration)
    )
    rows = [row for model in models for row in model_rows(args.results_dir, model)]
    write_outputs(args.out_dir, models, rows, declaration)

    print(f"[ok] {args.out_dir}/all_numbers.csv and all_numbers.json")
    for line in summary_lines(models, rows):
        print(f"     {line}")


if __name__ == "__main__":
    main()

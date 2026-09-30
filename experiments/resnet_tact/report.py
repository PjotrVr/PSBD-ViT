"""Reads the ResNet-18 TaCT test off its artefacts, judges PREDICTIONS.md and renders the README.

Writes 1 JSON per model and summary.json under results/_experiments/resnet_tact/ and
replaces the block between the results markers of experiments/resnet_tact/README.md.
The checkpoints and PSBD caches live in the main checkout, so their roots are flags.

    PYTHONPATH=. python experiments/resnet_tact/report.py \\
        --checkpoints-dir /lustre/home/pstika/projects/PSBD-ViT/checkpoints \\
        --results-dir /lustre/home/pstika/projects/PSBD-ViT/results
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scripts.coverage_ledger import source_class_accuracy  # noqa: E402
from scripts.paper._common import clearing_cells, load_coverage  # noqa: E402

TACT = (
    "resnet18_gtsrb_tact_0_05_src6",
    "resnet18_gtsrb_tact_0_1_src12",
    "resnet18_cifar10_tact_0_01",
    "resnet18_cifar10_tact_0_05_src3",
)
CONTROLS = (
    "resnet18_gtsrb_badnet_a2o_0_1",
    "resnet18_gtsrb_blend_0_1",
    "resnet18_cifar10_badnet_a2o_0_1",
)
BENIGN = {"gtsrb": "resnet18_gtsrb_benign", "cifar10": "resnet18_cifar10_benign"}

# The bars of PREDICTIONS.md.
ASR_BAR = 0.85
CLEAN_DROP_BAR = 0.02
SOURCE_MAPPED_ACCURACY = 0.5
FAILS_BELOW = 0.2
CONTROL_ABOVE = 0.8
DETECTED_AT = 0.5
TACT_NEEDED = 3
TACT_REFUTING = 2
NON_SOURCE_AT_MOST = 0.2
BADNET_NON_SOURCE_AT_LEAST = 0.9
BLANK_EXCESS_AT_MOST = 0.2

PLACEMENT = "post_residual"
VIT_TM = "before_attention_norm_token_mask"
QUANTILES = ("q0.01", "q0.05", "q0.10")
HEADLINE_QUANTILE = "q0.10"
STATISTICS = {"absolute": "detection", "fractional": "detection_psu_ratio"}
DETECTORS = ("ted", "beatrix")

SLUG = "resnet_tact"
README = os.path.join(REPO_ROOT, "experiments", SLUG, "README.md")
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output-dir", default=os.path.join(REPO_ROOT, "results", "_experiments", SLUG)
    )
    parser.add_argument(
        "--queue-log",
        default="/lustre/home/pstika/projects/PSBD-ViT/scratch/resnet_tact/queue.log",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stage_minutes = read_stage_minutes(args.queue_log)

    records = [
        model_record(folder, role, args, stage_minutes)
        for role, folders in (("tact", TACT), ("control", CONTROLS))
        for folder in folders
    ]
    records = [record for record in records if record is not None]
    for record in records:
        write_json(os.path.join(args.output_dir, f"{record['folder']}.json"), record)

    vit_rows = vit_tact_reference(args.results_dir)
    verdicts = judge(records)
    summary = {"models": records, "vit_tact_panel": vit_rows, "verdicts": verdicts}
    write_json(os.path.join(args.output_dir, "summary.json"), summary)

    render_readme(README, results_markdown(records, vit_rows, verdicts))
    print(json.dumps(verdicts, indent=2))


def model_record(folder, role, args, stage_minutes):
    metadata = read_json(os.path.join(args.checkpoints_dir, folder, "args.json"))
    if metadata is None:
        return None
    dataset = metadata["dataset"]
    benign = read_json(os.path.join(args.checkpoints_dir, BENIGN[dataset], "args.json"))
    benign_accuracy = benign["clean_accuracy"] if benign else None
    cell = {"folder_name": folder, "attack": metadata["attack"]}
    source_accuracy = source_class_accuracy(
        args.checkpoints_dir, args.results_dir, cell
    )

    record = {
        "folder": folder,
        "role": role,
        "dataset": dataset,
        "attack": metadata["attack"],
        "poison_rate": metadata["poison_rate"],
        "source_classes": (metadata.get("attack_config_overrides") or {}).get(
            "source_classes", [1] if metadata["attack"] == "tact" else None
        ),
        "epochs": metadata["epochs"],
        "git_commit": metadata["git_commit"],
        "asr": metadata["asr"],
        "clean_accuracy": metadata["clean_accuracy"],
        "benign_clean_accuracy": benign_accuracy,
        "clean_accuracy_drop": None
        if benign_accuracy is None
        else benign_accuracy - metadata["clean_accuracy"],
        "source_class_accuracy": source_accuracy,
        "wall_minutes": {
            "train": training_minutes(metadata),
            **{
                stage: minutes
                for stage, minutes in stage_minutes.get(folder, {}).items()
                if stage != "train"
            },
        },
        "psbd": psbd_reading(
            os.path.join(args.results_dir, folder, "psbd_metrics.json"), PLACEMENT
        ),
        "sufficiency": read_json(
            os.path.join(args.output_dir, "sufficiency", f"{folder}.json")
        ),
        "detectors": detector_readings(args.results_dir, folder),
    }
    record["successful"] = is_successful(record)
    return record


def training_minutes(metadata):
    started = metadata.get("trained_started_at")
    ended = metadata.get("trained_ended_at")
    if not started or not ended:
        return None
    span = datetime.fromisoformat(ended) - datetime.fromisoformat(started)
    minutes = round(span.total_seconds() / 60, 1)
    return minutes


# The queue log holds 1 "done <stage>.<folder> in N min" line per finished stage,
# which is the only record of the sweep and sufficiency wall times.
def read_stage_minutes(path):
    minutes = {}
    if not os.path.exists(path):
        return minutes
    pattern = re.compile(r"done (\w+)\.(\S+) in (\d+) min")
    with open(path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                stage, folder, value = match.groups()
                minutes.setdefault(folder, {})[stage] = int(value)
    return minutes


def psbd_reading(path, placement):
    metrics = read_json(path)
    if metrics is None or placement not in metrics["placements"]:
        return None
    block = metrics["placements"][placement]
    rate = block["adaptive_rate"]
    row = next(r for r in block["rates"] if r["rate"] == rate)
    reading = {
        "placement": placement,
        "adaptive_rate": rate,
        "clean_validation_shift_ratio": row["shift_ratio"]["validation"],
        "n_clean": metrics["split_sizes"]["clean"],
        "n_backdoor": metrics["split_sizes"]["backdoor"],
    }
    for name, key in STATISTICS.items():
        detection = row[key]
        reading[name] = {
            "auroc": detection["q0.25"]["auroc"],
            **{
                quantile: {
                    "tpr": detection[quantile]["tpr"],
                    "fpr": detection[quantile]["fpr"],
                }
                for quantile in QUANTILES
            },
        }
    return reading


def detector_readings(results_dir, folder):
    readings = {}
    for name in DETECTORS:
        report = read_json(
            os.path.join(results_dir, folder, "detectors", f"{name}_metrics.json")
        )
        if report is None or report.get("status") != "scored":
            continue
        detection = report["detection"]
        readings[name] = {
            "auroc": detection["q0.25"]["auroc"],
            **{
                quantile: {
                    "tpr": detection[quantile]["tpr"],
                    "fpr": detection[quantile]["fpr"],
                }
                for quantile in QUANTILES
            },
        }
    return readings


def is_successful(record):
    if record["asr"] < ASR_BAR:
        return False
    if record["clean_accuracy_drop"] is None:
        return None
    if record["clean_accuracy_drop"] > CLEAN_DROP_BAR:
        return False
    source_accuracy = record["source_class_accuracy"]
    if record["attack"] == "tact" and source_accuracy is None:
        return None
    if record["attack"] == "tact" and source_accuracy < SOURCE_MAPPED_ACCURACY:
        return False
    return True


# The ViT TaCT models of the paper panel at PSBD-TM and at the ViT post_residual
# dropout (PSBD-RD), both at the adaptive rate, beside their trigger readings.
def vit_tact_reference(results_dir):
    rows = []
    for cell in clearing_cells(load_coverage(results_dir)):
        if cell["attack"] != "tact":
            continue
        folder = cell["folder_name"]
        metrics_path = os.path.join(results_dir, folder, "psbd_metrics.json")
        if psbd_reading(metrics_path, VIT_TM) is None:
            continue
        sufficiency = read_json(
            os.path.join(
                REPO_ROOT,
                "results",
                "_experiments",
                "evidence_surplus",
                "sufficiency",
                f"{folder}.json",
            )
        )
        rows.append(
            {
                "folder": folder,
                "psbd_tm": psbd_reading(metrics_path, VIT_TM),
                "psbd_rd": psbd_reading(metrics_path, PLACEMENT),
                "non_source_stamped_on_target": sufficiency["classes"][
                    "non_source_stamped_on_target"
                ]
                if sufficiency
                else None,
            }
        )
    return rows


def headline_tpr(record, statistic):
    if record["psbd"] is None:
        return None
    tpr = record["psbd"][statistic][HEADLINE_QUANTILE]["tpr"]
    return tpr


def judge(records):
    verdicts = {
        name: judge_detection(records, name) for name in ("absolute", "fractional")
    }
    verdicts["P2"] = judge_mechanism(records)
    verdicts["P3"] = judge_link(records)
    return verdicts


def judge_detection(records, statistic):
    tact = [
        r
        for r in records
        if r["role"] == "tact" and r["successful"] and r["psbd"] is not None
    ]
    controls = [r for r in records if r["role"] == "control" and r["psbd"]]
    badnets = [r for r in controls if r["attack"] == "badnet_a2o"]
    control_tprs = {r["folder"]: headline_tpr(r, statistic) for r in controls}
    tact_tprs = {r["folder"]: headline_tpr(r, statistic) for r in tact}

    failing = sum(tpr < FAILS_BELOW for tpr in tact_tprs.values())
    detected = sum(tpr >= DETECTED_AT for tpr in tact_tprs.values())
    badnets_hold = all(headline_tpr(r, statistic) > CONTROL_ABOVE for r in badnets)
    controls_hold = all(tpr > CONTROL_ABOVE for tpr in control_tprs.values())

    if len(tact) < TACT_NEEDED:
        verdict = "no verdict, fewer than 3 successful TaCT models"
    elif not badnets or not badnets_hold:
        verdict = "void, a BadNets control misses its bar"
    elif detected >= TACT_REFUTING:
        verdict = "refuted"
    elif failing >= TACT_NEEDED:
        verdict = "supported" if controls_hold else "supported on the BadNets controls"
    else:
        verdict = "mixed"
    judged = {
        "statistic": statistic,
        "tact_tpr_at_10_fpr": tact_tprs,
        "control_tpr_at_10_fpr": control_tprs,
        "tact_failing": failing,
        "tact_detected": detected,
        "tact_judged": len(tact),
        "controls_hold": controls_hold,
        "verdict": verdict,
    }
    return judged


def judge_mechanism(records):
    tact = [
        r
        for r in records
        if r["role"] == "tact" and r["successful"] and r["sufficiency"]
    ]
    badnets = [
        r
        for r in records
        if r["role"] == "control" and r["attack"] == "badnet_a2o" and r["sufficiency"]
    ]
    if len(tact) < TACT_NEEDED or not badnets:
        return {"verdict": "no verdict, readings missing"}
    non_source = {
        r["folder"]: r["sufficiency"]["classes"]["non_source_stamped_on_target"]
        for r in tact
    }
    blank = {r["folder"]: r["sufficiency"]["blank"]["excess"] for r in tact}
    badnet_non_source = {
        r["folder"]: r["sufficiency"]["classes"]["non_source_stamped_on_target"]
        for r in badnets
    }
    holds = (
        all(v <= NON_SOURCE_AT_MOST for v in non_source.values())
        and all(v <= BLANK_EXCESS_AT_MOST for v in blank.values())
        and all(v >= BADNET_NON_SOURCE_AT_LEAST for v in badnet_non_source.values())
    )
    judged = {
        "tact_non_source_stamped_on_target": non_source,
        "tact_blank_excess": blank,
        "badnet_non_source_stamped_on_target": badnet_non_source,
        "verdict": "supported" if holds else "refuted",
    }
    return judged


def judge_link(records):
    tact = [
        r
        for r in records
        if r["role"] == "tact" and r["successful"] and r["sufficiency"] and r["psbd"]
    ]
    detected = [r for r in tact if headline_tpr(r, "absolute") >= DETECTED_AT]
    if not detected:
        return {"verdict": "not tested, no TaCT model is detected"}
    most_transferable = max(
        tact, key=lambda r: r["sufficiency"]["classes"]["non_source_stamped_on_target"]
    )
    holds = all(r["folder"] == most_transferable["folder"] for r in detected)
    judged = {
        "detected": [r["folder"] for r in detected],
        "most_transferable": most_transferable["folder"],
        "verdict": "supported" if holds else "refuted",
    }
    return judged


def fmt(value, digits=3):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = f"{value:.{digits}f}"
    return text


def results_markdown(records, vit_rows, verdicts):
    lines = [
        BEGIN,
        "<!-- Everything down to results:end is rendered by report.py from "
        "results/_experiments/resnet_tact/. -->",
        "",
        "## Models trained",
        "",
        "Clean accuracy and ASR are the final values `cli.train_backdoor` wrote to "
        "`args.json`. The benign reference is the ResNet-18 benign model of the same "
        "dataset and recipe, and the source-class accuracy is clean accuracy on the "
        "source classes of the PSBD analysis split.",
        "",
        "| model | source classes | ASR | clean acc | benign | drop | source-class acc "
        "| successful | train min |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in records:
        sources = r["source_classes"]
        lines.append(
            f"| `{r['folder']}` | {format_sources(sources)} | {fmt(r['asr'])} "
            f"| {fmt(r['clean_accuracy'])} | {fmt(r['benign_clean_accuracy'])} "
            f"| {fmt(r['clean_accuracy_drop'])} | {fmt(r['source_class_accuracy'])} "
            f"| {fmt(r['successful'])} | {fmt(r['wall_minutes']['train'], 1)} |"
        )

    for statistic, title in (
        ("absolute", "Original PSBD, absolute PSU"),
        ("fractional", "Fractional PSU, the secondary reading"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "Dropout at `post_residual`, 3 passes, the adaptive rate of the 0.8 rule. "
            "TPR at the 0.01, 0.05 and 0.10 quantiles of the clean validation scores, "
            "with the realized FPR on the paired clean test split at the 0.10 "
            "quantile, then the AUROC.",
            "",
            "| model | rate | TPR@1% | TPR@5% | TPR@10% | FPR at q0.10 | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in records:
            reading = r["psbd"]
            if reading is None:
                lines.append(f"| `{r['folder']}` | -- | -- | -- | -- | -- | -- |")
                continue
            detection = reading[statistic]
            lines.append(
                f"| `{r['folder']}` | {reading['adaptive_rate']:g} "
                + " ".join(f"| {fmt(detection[q]['tpr'])}" for q in QUANTILES)
                + f" | {fmt(detection['q0.10']['fpr'])} | {fmt(detection['auroc'])} |"
            )

    lines += [
        "",
        "## Trigger sufficiency",
        "",
        "`experiments/why_psbd_works/sufficiency.py` on each model. The class "
        "reading stamps the trigger on test images of every class but the target "
        "and splits the share sent to the target by source and non-source class. "
        "The blank reading stamps it on content-free carriers against the same "
        "carriers unstamped.",
        "",
        "| model | non-source stamped to target | source stamped to target "
        "| clean to target | blank excess |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in records:
        reading = r["sufficiency"]
        if reading is None:
            lines.append(f"| `{r['folder']}` | -- | -- | -- | -- |")
            continue
        classes = reading["classes"]
        lines.append(
            f"| `{r['folder']}` | {fmt(classes['non_source_stamped_on_target'])} "
            f"| {fmt(classes['source_stamped_on_target'])} "
            f"| {fmt(classes['clean_on_target'])} | {fmt(reading['blank']['excess'])} |"
        )

    scored = [r for r in records if r["detectors"]]
    if scored:
        lines += [
            "",
            "## Competitor detectors on the TaCT models",
            "",
            "| model | detector | TPR@1% | TPR@5% | TPR@10% | AUROC |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in scored:
            for name, detection in r["detectors"].items():
                lines.append(
                    f"| `{r['folder']}` | {name} "
                    + " ".join(f"| {fmt(detection[q]['tpr'])}" for q in QUANTILES)
                    + f" | {fmt(detection['auroc'])} |"
                )

    lines += [
        "",
        "## ViT TaCT models of the paper panel",
        "",
        "The same fractional readings for the ViT TaCT models of the panel, at "
        "PSBD-TM and at the ViT `post_residual` dropout (PSBD-RD), with the "
        "non-source class reading of `experiments/evidence_surplus/`, for the ViT TaCT models of the panel that have a PSBD-TM cache.",
        "",
        "| model | PSBD-TM TPR@1% | TPR@5% | TPR@10% | AUROC | PSBD-RD TPR@10% "
        "| AUROC | non-source stamped to target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in vit_rows:
        tm = row["psbd_tm"]["fractional"] if row["psbd_tm"] else None
        rd = row["psbd_rd"]["fractional"] if row["psbd_rd"] else None
        tm_cells = (
            " ".join(f"| {fmt(tm[q]['tpr'])}" for q in QUANTILES)
            + f" | {fmt(tm['auroc'])}"
            if tm
            else "| -- | -- | -- | --"
        )
        rd_cells = (
            f"| {fmt(rd['q0.10']['tpr'])} | {fmt(rd['auroc'])}" if rd else "| -- | --"
        )
        lines.append(
            f"| `{row['folder']}` {tm_cells} {rd_cells} "
            f"| {fmt(row['non_source_stamped_on_target'])} |"
        )

    lines += ["", "## Verdicts", ""]
    for name in ("absolute", "fractional"):
        judged = verdicts[name]
        label = "P1" if name == "absolute" else "P1 on fractional PSU"
        lines.append(
            f"- **{label}**: {judged['verdict']}. {judged['tact_failing']} of "
            f"{judged['tact_judged']} successful TaCT models read TPR at 10% FPR "
            f"below {FAILS_BELOW} and {judged['tact_detected']} read {DETECTED_AT} "
            f"or more. Every control reads above {CONTROL_ABOVE} "
            f"({fmt(judged['controls_hold'])})."
        )
    lines.append(f"- **P2**: {verdicts['P2']['verdict']}.")
    lines.append(f"- **P3**: {verdicts['P3']['verdict']}.")
    lines += ["", END]
    markdown = "\n".join(lines)
    return markdown


def format_sources(sources):
    if not sources:
        return "--"
    if len(sources) == 1:
        return str(sources[0])
    text = f"{sources[0]}..{sources[-1]}"
    return text


def render_readme(path, block):
    with open(path) as handle:
        text = handle.read()
    start, stop = text.index(BEGIN), text.index(END) + len(END)
    rendered = text[:start] + block + text[stop:]
    with open(path, "w") as handle:
        handle.write(rendered)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        content = json.load(handle)
    return content


def write_json(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(content, handle, indent=2)


if __name__ == "__main__":
    main()

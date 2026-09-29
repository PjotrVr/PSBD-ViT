"""Stage 2 of the novel probes: scores, references, unions and verdict inputs, on CPU.

Each probe's fractional PSU is read at its primary rate from the stage-1 passes and
compared on the same rows against 2 references. Reference 1 is PSBD-TM alone and
reference 2 is the plain minimum of the PSBD-TM and `pre_residual_blocks_5_8`
clean-validation percentiles. On a backdoored model both are read from the
stage-1 caches (the exact passes cli.analyze scores) restricted to the rows stage 1
used. On a benign model they are recomputed by stage 1, since the caches hold only
the default probe trigger.

    .venv/bin/python -m experiments.novel_probes.evaluate --set dev
    .venv/bin/python -m experiments.novel_probes.evaluate --set benign
"""

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from defenses.cache import read_split_manifest  # noqa: E402
from defenses.decision import (  # noqa: E402
    RECOMMENDED_PLACEMENT,
    detection_report,
    pair_clean_to_backdoor,
    threshold_diagnostics,
)
from defenses.scores import psu_ratio_from_cache, to_rank  # noqa: E402
from experiments.cache_readouts.shared import (  # noqa: E402
    choose_rate,
    load_baselines,
    load_model_set,
    load_passes,
    paired_summary,
    psbd_dir_of,
    validation_shift_by_rate,
)
from experiments.novel_probes.measure import (  # noqa: E402
    BENIGN_FOLDERS,
    SLUG,
    passes_path,
)
from experiments._paths import experiment_result_path  # noqa: E402
from scripts.paper._common import OKABE_ITO, figure_sidecar, save_figure  # noqa: E402

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
QUANTILES = (0.01, 0.05, 0.10, 0.25)
HEADLINE_QUANTILES = ("q0.01", "q0.05", "q0.10")
CACHED_PLACEMENTS = {
    "psbd_tm": RECOMMENDED_PLACEMENT,
    "middle_band": "pre_residual_blocks_5_8",
    "late_band": "pre_residual_blocks_9_12",
    "mlp_channel_mask": "mlp_neurons_channel_mask",
}
REFERENCE_PROBES = ("psbd_tm", "middle_band")
# A benign model has no backdoor, so every reading there should sit near chance.
# The tolerance is fixed here, before any benign reading exists.
BENIGN_TOLERANCE = 0.1


def main():
    args = parse_args()
    started = time.perf_counter()

    if args.set == "benign":
        rows = [
            row
            for folder in BENIGN_FOLDERS
            for row in measure_benign(folder, args.smoke)
        ]
        summary = summarize_benign(rows)
    else:
        models = load_model_set(args.set)
        rows = [measure_backdoored(model, args.smoke) for model in models]
        rows = [row for row in rows if row is not None]
        summary = summarize(rows)

    payload = {
        "experiment": "novel probes",
        "model_set": args.set,
        "smoke": args.smoke,
        "quantiles": list(QUANTILES),
        "reference_1": "psbd_tm alone",
        "reference_2": "min rank of psbd_tm and pre_residual_blocks_5_8",
        "benign_tolerance": BENIGN_TOLERANCE,
        "summary": summary,
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    suffix = "_smoke" if args.smoke else ""
    path = experiment_result_path(SLUG, f"{args.set}{suffix}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    if args.set == "benign":
        plot_benign(summary, rows, path, suffix)
    else:
        plot_backdoored(summary, args.set, path, suffix)
    print(f"wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("dev", "benign", "holdout"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    return args


def measure_backdoored(model, smoke):
    folder = model["folder_name"]
    path = passes_path(folder, smoke)
    if not os.path.exists(path):
        print(f"no stage-1 passes for {folder}")
        return None
    record = torch.load(path, weights_only=False)
    subject = record["subjects"]["own"]
    validation_count = len(record["baseline"]["validation"]["labels"])

    cached = cached_references(folder, subject, validation_count)
    fresh = fresh_scores(record, "own")

    row = {
        "folder": folder,
        "attack": model["attack"],
        "dataset": model["dataset"],
        "poison_rate": model["poison_rate"],
        "successful_2pt": model["successful_2pt"],
        "n_backdoor": len(subject["backdoor_rows"]),
        "runner_check": runner_check(
            record, subject, cached, fresh.pop("psbd_tm", None)
        ),
        "rates": {name: reading["rate"] for name, reading in cached.items()},
        "rate_rules": {name: reading["rule"] for name, reading in cached.items()},
    }
    row.update(read_all(cached, fresh))
    return row


def measure_benign(folder, smoke):
    path = passes_path(folder, smoke)
    if not os.path.exists(path):
        print(f"no stage-1 passes for {folder}")
        return []
    record = torch.load(path, weights_only=False)
    rows = []
    for key, subject in record["subjects"].items():
        fresh = fresh_scores(record, key)
        references = {name: fresh.pop(name) for name in REFERENCE_PROBES}
        row = {
            "folder": folder,
            "probe_attack": subject["probe_attack"],
            "dataset": record["dataset"],
            "n_backdoor": len(subject["backdoor_rows"]),
        }
        row.update(read_all(references, fresh))
        rows.append(row)
    return rows


def cached_references(folder, subject, validation_count):
    model = {"folder_name": folder}
    psbd_dir = psbd_dir_of(model)
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)
    count = len(subject["backdoor_rows"])
    assert subject["backdoor_rows"] == list(range(count))

    references = {}
    for name, placement in CACHED_PLACEMENTS.items():
        shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
        rate = choose_rate(shift_by_rate, "adaptive")
        rule = "adaptive"
        if rate is None:
            rate = choose_rate(shift_by_rate, "nearest")
            rule = "nearest"
        passes = load_passes(psbd_dir, placement, rate, baselines)
        psu = {
            split: psu_ratio_from_cache(
                passes[split]["baseline_probs"],
                passes[split]["baseline_labels"],
                passes[split]["per_pass_probs"],
            )  # (n_split,)
            for split in passes
        }
        references[name] = {
            "rate": rate,
            "rule": rule,
            "validation": psu["validation"][:validation_count],
            "clean": pair_clean_to_backdoor(psu["clean"], manifest)[:count],
            "backdoor": psu["backdoor"][:count],
            "baseline_labels": {
                "validation": passes["validation"]["baseline_labels"][
                    :validation_count
                ],
                "clean": pair_clean_to_backdoor(
                    passes["clean"]["baseline_labels"], manifest
                )[:count],
                "backdoor": passes["backdoor"]["baseline_labels"][:count],
            },
        }
    return references


def fresh_scores(record, key):
    subject = record["subjects"][key]
    baselines = {
        "validation": record["baseline"]["validation"],
        "clean": subject["baseline"]["clean"],
        "backdoor": subject["baseline"]["backdoor"],
    }
    scores = {}
    for name, measured in record["probes"].items():
        for reading in ("primary", "canonical"):
            rate = measured["chosen"][reading]
            if rate is None or (reading == "canonical" and measured["sign"] == "psbd"):
                continue
            passes = {
                "validation": measured["validation"][rate],
                "clean": measured["tests"][key][rate]["clean"],
                "backdoor": measured["tests"][key][rate]["backdoor"],
            }
            psu = {
                split: psu_ratio_from_cache(
                    baselines[split]["probs"],
                    baselines[split]["labels"],
                    passes[split]["probs"],
                )  # (n_split,)
                for split in passes
            }
            # A fragile probe flags inputs that move more than clean ones, so its
            # primary score is negated to keep "low means poisoned" for every
            # reading. Its canonical reading keeps PSBD's sign, as a secondary.
            flip = measured["sign"] == "fragile" and reading == "primary"
            label = name if reading == "primary" else f"{name}@canonical"
            scores[label] = {
                "rate": rate,
                "rule": rule_of(measured, reading),
                "validation_shift": measured["validation_shift"][rate],
                "validation": -psu["validation"] if flip else psu["validation"],
                "clean": -psu["clean"] if flip else psu["clean"],
                "backdoor": -psu["backdoor"] if flip else psu["backdoor"],
            }
    return scores


def rule_of(measured, reading):
    chosen = measured["chosen"]
    if reading == "canonical" or measured["sign"] == "psbd":
        rule = "adaptive" if chosen["adaptive"] is not None else "nearest"
        return rule
    rule = "stability" if chosen["stability"] is not None else "smallest_rung"
    return rule


def runner_check(record, subject, cached, fresh_tm):
    # The fresh PSBD-TM of stage 1 and the cached one differ only in the mask
    # draws and the batch composition, so their rates and AUROCs should agree,
    # and the unperturbed predictions should agree almost everywhere.
    tm = cached["psbd_tm"]
    agreement = {
        split: float(
            (baseline["labels"] == tm["baseline_labels"][split]).float().mean()
        )
        for split, baseline in (
            ("validation", record["baseline"]["validation"]),
            ("clean", subject["baseline"]["clean"]),
            ("backdoor", subject["baseline"]["backdoor"]),
        )
    }
    check = {"baseline_argmax_agreement": agreement}
    if fresh_tm is not None:
        check["fresh_tm_rate"] = fresh_tm["rate"]
        check["cached_tm_rate"] = tm["rate"]
        check["fresh_tm_auroc"] = evaluate(fresh_tm)["auroc"]
        check["cached_tm_auroc"] = evaluate(tm)["auroc"]
    return check


def read_all(references, probes):
    tm, middle = references["psbd_tm"], references["middle_band"]
    readings = {
        "references": {
            "tm_alone": evaluate(tm),
            "union_tm_middle": evaluate(union([tm, middle])),
        },
        "probes": {},
    }
    if "late_band" in references:
        readings["references"]["union_tm_late"] = evaluate(
            union([tm, references["late_band"]])
        )
    if "mlp_channel_mask" in references:
        readings["references"]["mlp_channel_mask"] = evaluate(
            references["mlp_channel_mask"]
        )

    for name, scores in probes.items():
        readings["probes"][name] = {
            "rate": scores["rate"],
            "rule": scores["rule"],
            "validation_shift": scores["validation_shift"],
            "standalone": evaluate(scores),
            "union_tm": evaluate(union([tm, scores])),
            "union_tm_middle": evaluate(union([tm, middle, scores])),
        }
    return readings


def union(members):
    # The plain minimum of each member's percentile in its own clean-validation
    # distribution, the rule of experiments/cache_readouts (min_rank).
    fused = {}
    for split in ("validation", "clean", "backdoor"):
        ranks = torch.stack(
            [to_rank(member[split], member["validation"]) for member in members]
        )  # (members, n_split)
        fused[split] = ranks.min(dim=0).values  # (n_split,)
    return fused


def evaluate(scores):
    validation = scores["validation"].float()  # (n_validation,)
    clean = scores["clean"].float()  # (n_pairs,)
    backdoor = scores["backdoor"].float()  # (n_pairs,)
    assert clean.shape == backdoor.shape, (clean.shape, backdoor.shape)

    by_quantile = {}
    auroc = None
    for quantile in QUANTILES:
        report = detection_report(validation, clean, backdoor, quantile)
        ties = threshold_diagnostics(validation, clean, backdoor, quantile)
        auroc = report["auroc"]
        by_quantile[f"q{quantile:.2f}"] = {
            "tpr": report["tpr"],
            "realized_fpr": report["fpr"],
            "tie_share_at_threshold": ties["tie_share_at_threshold"],
        }
    evaluation = {"auroc": auroc, "at_fpr": by_quantile}
    return evaluation


def summarize(rows):
    pooled = [row for row in rows if row["successful_2pt"]]
    probe_names = sorted({name for row in pooled for name in row["probes"]})
    fields = ["auroc"] + [
        f"{q}:{kind}" for q in HEADLINE_QUANTILES for kind in ("tpr", "realized_fpr")
    ]

    def metric(evaluation, field):
        if field == "auroc":
            return evaluation["auroc"]
        quantile, kind = field.split(":")
        return evaluation["at_fpr"][quantile][kind]

    summary = {
        "pooled_folders": [row["folder"] for row in pooled],
        "unpooled_folders": [
            row["folder"] for row in rows if not row["successful_2pt"]
        ],
        "references": {},
        "probes": {},
    }
    for reference in ("tm_alone", "union_tm_middle", "union_tm_late"):
        summary["references"][reference] = {
            field: mean([metric(row["references"][reference], field) for row in pooled])
            for field in fields
        }

    for name in probe_names:
        covered = [row for row in pooled if name in row["probes"]]
        summary["probes"][name] = {
            "n": len(covered),
            "rules": sorted({row["probes"][name]["rule"] for row in covered}),
            "rates": {row["folder"]: row["probes"][name]["rate"] for row in covered},
        }
        for reading in ("standalone", "union_tm", "union_tm_middle"):
            block = {}
            for field in fields:
                values = [
                    metric(row["probes"][name][reading], field) for row in covered
                ]
                block[field] = {
                    "vs_tm_alone": paired_summary(
                        values,
                        [
                            metric(row["references"]["tm_alone"], field)
                            for row in covered
                        ],
                    ),
                    "vs_union_tm_middle": paired_summary(
                        values,
                        [
                            metric(row["references"]["union_tm_middle"], field)
                            for row in covered
                        ],
                    ),
                }
            block["beats_both_at_every_fpr"] = all(
                block[f"{q}:tpr"][reference]["mean_difference"] > 0
                for q in HEADLINE_QUANTILES
                for reference in ("vs_tm_alone", "vs_union_tm_middle")
            )
            summary["probes"][name][reading] = block
    return summary


def summarize_benign(rows):
    probe_names = sorted({name for row in rows for name in row["probes"]})
    summary = {"references": {}, "probes": {}}
    for reference in ("tm_alone", "union_tm_middle"):
        aurocs = [row["references"][reference]["auroc"] for row in rows]
        summary["references"][reference] = benign_block(aurocs)
    for name in probe_names:
        summary["probes"][name] = {
            reading: benign_block(
                [row["probes"][name][reading]["auroc"] for row in rows]
            )
            for reading in ("standalone", "union_tm")
        }
    return summary


def benign_block(aurocs):
    finite = [value for value in aurocs if not math.isnan(value)]
    block = {
        "n": len(finite),
        "mean": mean(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "within_tolerance": all(
            abs(value - 0.5) <= BENIGN_TOLERANCE for value in finite
        ),
    }
    return block


def mean(values):
    average = sum(values) / len(values) if values else None
    return average


def plot_backdoored(summary, model_set, json_path, suffix):
    names = sorted(summary["probes"])
    references = summary["references"]
    figure, axes = plt.subplots(
        1, len(HEADLINE_QUANTILES), figsize=(18, 6), sharey=True
    )
    plotted = {}
    width = 0.4
    for axis, quantile in zip(axes, HEADLINE_QUANTILES):
        field = f"{quantile}:tpr"
        standalone = [
            summary["probes"][n]["standalone"][field]["vs_tm_alone"]["mean"]
            for n in names
        ]
        with_tm = [
            summary["probes"][n]["union_tm"][field]["vs_tm_alone"]["mean"]
            for n in names
        ]
        positions = list(range(len(names)))
        axis.bar(
            [p - width / 2 for p in positions],
            standalone,
            width,
            color=OKABE_ITO[0],
            label="probe alone",
        )
        axis.bar(
            [p + width / 2 for p in positions],
            with_tm,
            width,
            color=OKABE_ITO[2],
            label="min rank with PSBD-TM",
        )
        axis.axhline(
            references["tm_alone"][field], color="black", label="reference 1, PSBD-TM"
        )
        axis.axhline(
            references["union_tm_middle"][field],
            color=OKABE_ITO[1],
            linestyle="--",
            label="reference 2, PSBD-TM with blocks 5 to 8",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        axis.set_title(f"mean TPR at {float(quantile[1:]):.0%} nominal FPR")
        axis.set_ylim(0, 1)
        plotted[field] = {
            "probes": names,
            "standalone": standalone,
            "union_tm": with_tm,
            "tm_alone": references["tm_alone"][field],
            "union_tm_middle": references["union_tm_middle"][field],
        }
    axes[0].set_ylabel(f"mean TPR over {len(summary['pooled_folders'])} models")
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Novel probes against the 2 references, {model_set} set")

    figure_path = os.path.join(FIGURES_DIR, f"tpr_{model_set}{suffix}.png")
    save_figure(figure, figure_path)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/novel_probes/evaluate.py",
        [json_path],
        plotted,
    )


def plot_benign(summary, rows, json_path, suffix):
    names = sorted(summary["probes"])
    figure, axis = plt.subplots(figsize=(12, 5))
    plotted = {}
    for index, name in enumerate(names):
        values = [row["probes"][name]["standalone"]["auroc"] for row in rows]
        axis.scatter([index] * len(values), values, color=OKABE_ITO[0], s=14)
        plotted[name] = {
            f"{row['folder']}:{row['probe_attack']}": value
            for row, value in zip(rows, values)
        }
    tm = [row["references"]["tm_alone"]["auroc"] for row in rows]
    axis.scatter([len(names)] * len(tm), tm, color="black", s=14, label="PSBD-TM")
    plotted["psbd_tm"] = tm
    axis.axhspan(0.5 - BENIGN_TOLERANCE, 0.5 + BENIGN_TOLERANCE, color="0.9")
    axis.set_xticks(range(len(names) + 1))
    axis.set_xticklabels(names + ["psbd_tm"], rotation=60, ha="right", fontsize=8)
    axis.set_ylabel("AUROC with a trigger the model never learned")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.suptitle("Benign control, 1 dot per benign model and stamped trigger")

    figure_path = os.path.join(FIGURES_DIR, f"benign{suffix}.png")
    save_figure(figure, figure_path)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/novel_probes/evaluate.py",
        [json_path],
        plotted,
    )


if __name__ == "__main__":
    main()

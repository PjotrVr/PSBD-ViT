"""Aggregate the why_psbd_works records into summary.json and the README tables.

CPU only, seconds. Reads every per-model JSON measure.py wrote, the seed
ensemble records and cached_reads.json, reduces each hypothesis to a few numbers
per model, averages them per architecture and attack category and applies the
verdict rules stated in README.md. The notebook reads summary.json only.

    PYTHONPATH=. python experiments/why_psbd_works/summarize.py
"""

import argparse
import collections
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from experiments._paths import experiment_results_dir  # noqa: E402
from scripts.coverage_ledger import SUCCESS_BARS, success_verdicts  # noqa: E402
import experiments.why_psbd_works.measure as measure  # noqa: E402
from experiments.why_psbd_works.measure import (  # noqa: E402
    OPERATOR_SPECS,
    SLUG,
    auroc_low_is_positive,
)

CATEGORY_ORDER = ("patch", "blend", "frequency", "warp", "quantization", "benign")
HEADLINE_OPERATORS = (
    "token_mask",
    "dropout",
    "channel_mask",
    "gaussian",
    "residual_dropout",
)
TACT_CONDITIONAL_ACCURACY = 0.5
QUARANTINED_ATTACKS = ("sig",)
# The site and step the flatness verdict reads: the stream entering the last
# probed block, where every operator's disturbance has reached, at the smaller
# of the 2 steps.
FLAT_EPSILON = "0.05"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    directory = experiment_results_dir(SLUG, args.output_root)
    records, seeds, cached = read_records(directory)

    per_model = [model_row(record) for record in records]
    success = success_flags([row["folder"] for row in per_model])
    for row in per_model:
        row.update(success[row["folder"]])
    # The panel is the models successful at the 2 point clean-accuracy bar (the
    # user's decision of 2026-09-29), the benign probes kept as the control. The
    # 5 point bar is read beside it where it adds models.
    panel = [r for r in per_model if r["successful_2pt"] or r["category"] == "benign"]
    wider = [r for r in per_model if r["successful_5pt"] or r["category"] == "benign"]
    summary = {
        "models": len(per_model),
        "panel_models": len(panel),
        "per_model": per_model,
        "groups": group_means(panel),
        "groups_5pt": group_means(wider),
        "seed_ensemble": [seed_row(record) for record in seeds],
        "cached_reads": cached_summary(cached),
    }
    summary["wanet_probe"] = wanet_probe_rows(directory)
    summary["critical_rate"] = critical_rate_summary(directory, per_model)
    summary["sufficiency"] = sufficiency_summary(directory)
    summary["thresholds"] = THRESHOLDS
    summary["parameters"] = measurement_parameters()
    summary["verdicts"] = verdicts(summary["groups"])
    summary["verdicts_5pt"] = verdicts(summary["groups_5pt"])
    summary["panel_verdicts"] = panel_verdicts(cached, summary["seed_ensemble"])
    with open(os.path.join(directory, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(markdown(summary))


# successful_2pt and successful_5pt from the coverage ledger for ViT. Swin is not
# in the ledger, so its verdicts are rebuilt the ledger's way: ASR from args.json
# against the declared bar, clean accuracy against the same dataset's benign Swin
# model, both through scripts.coverage_ledger.success_verdicts.
def success_flags(folders, results_dir="results", checkpoints_dir="checkpoints"):
    with open(os.path.join(results_dir, "coverage", "coverage.json")) as handle:
        coverage = json.load(handle)
    cells = {cell["folder_name"]: cell for cell in coverage["cells"]}
    declaration = {
        key: coverage[key]
        for key in (
            "asr_bar",
            "clean_accuracy_drop_bar",
            "clean_accuracy_drop_bar_headline",
        )
    }
    flags = {}
    for folder in folders:
        if folder in cells:
            cell = cells[folder]
            flags[folder] = {key: bool(cell.get(key)) for key in SUCCESS_BARS}
            continue
        metadata = read_args(checkpoints_dir, folder)
        if metadata.get("attack") == "benign":
            flags[folder] = {key: False for key in SUCCESS_BARS}
            continue
        benign = read_args(
            checkpoints_dir, f"{metadata['architecture']}_{metadata['dataset']}_benign"
        )
        cell = {
            "asr_class": "clears"
            if (metadata.get("asr") or 0) >= declaration["asr_bar"]
            else "below_bar",
            "clean_accuracy_drop": metadata["clean_accuracy"]
            - benign["clean_accuracy"],
        }
        flags[folder] = success_verdicts(cell, declaration)
    return flags


def read_args(checkpoints_dir, folder):
    with open(os.path.join(checkpoints_dir, folder, "args.json")) as handle:
        metadata = json.load(handle)
    return metadata


# The measurement settings, read from measure.py so every document that quotes
# 1 of them quotes the value the records were made with.
def measurement_parameters():
    parameters = {
        name: getattr(measure, name)
        for name in (
            "PAIR_COUNT",
            "FORWARD_PASSES",
            "KEEP_FRACTIONS",
            "SUBSET_DRAWS",
            "STREAM_BLOCKS",
            "JACOBIAN_PAIRS",
            "TOP_TOKENS",
            "ATTRIBUTION_PAIRS",
            "ATTRIBUTION_SAMPLES",
            "ATTRIBUTION_TOP_SHARE",
            "FLAT_PAIRS",
            "FLAT_DIRECTIONS",
            "FLAT_EPSILONS",
            "HEADLINE_QUANTILE",
            "KNN_NEIGHBOURS",
            "FOREIGN_DATASET",
            "NEURON_SHARE",
            "NEURON_DRAWS",
            "LATE_BLOCK_COUNT",
            "LATE_UNITS",
            "SUFFICIENCY_FRACTIONS",
            "SPREAD_PAIRS",
            "SPREAD_BLOCKS",
            "SMALL_NOISE_RATES",
            "SMALL_NOISE_PASSES",
            "FALLBACK_IMAGES",
            "FALLBACK_PASSES",
            "ADAPTIVE_TARGET",
            "BATCH_SIZE",
            "GPU_MEMORY_GB",
            "CPU_THREADS",
        )
    }
    parameters["OPERATORS"] = {
        name: {"placement": spec[0], "positions": list(spec[1])}
        for name, spec in measure.OPERATOR_SPECS.items()
    }
    return parameters


# The measure.py operators and the cached placements they correspond to.
OPERATOR_PLACEMENTS = {
    name: spec[0]
    for name, spec in measure.OPERATOR_SPECS.items()
    if name in HEADLINE_OPERATORS
}


# The critical-rate account (critical_rate.py): its predictions, A* per attack and
# placement, and which mechanism reading of this experiment tracks the p* gap
# across (model, operator). The link is a correlation over a few models, so it
# is reported as the hypothesis-level reading it is.
def critical_rate_summary(directory, per_model):
    path = os.path.join(directory, "critical_rate.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    rows = payload["rows"]
    by_cell = collections.defaultdict(list)
    for row in rows:
        by_cell[(row["attack"], row["placement"])].append(row)
    table = {
        f"{attack}|{placement}": {
            "models": len(group),
            "a_star": mean([r["a_star"] for r in group]),
            "auroc_adaptive": mean([r["auroc_adaptive"] for r in group]),
            "clean_median": mean([r["clean_median"] for r in group]),
            "triggered_median": mean([r["triggered_median"] for r in group]),
        }
        for (attack, placement), group in by_cell.items()
    }
    lookup = {(r["folder"], r["placement"]): r for r in rows}
    joined = []
    for model in per_model:
        if model["architecture"] != "vit" or model["category"] == "benign":
            continue
        excess = (model["redundancy"].get("0.3") or {}).get("excess_retention")
        for operator, placement in OPERATOR_PLACEMENTS.items():
            reading = lookup.get((model["folder"], placement))
            block = model["operators"].get(operator)
            if reading is None or block is None:
                continue
            joined.append(
                {
                    "folder": model["folder"],
                    "operator": operator,
                    "a_star": reading["a_star"],
                    "margin_gap": delta(
                        block["triggered_margin_retention"],
                        block["clean_margin_retention"],
                    ),
                    "spr_ratio": ratio(
                        block["triggered_backdoor_spr"], block["clean_class_spr"]
                    ),
                    "l8": block["spearman_psu_backdoor_drop_above_floor"],
                    "redundancy_excess": excess,
                }
            )
    link = {
        key: spearman([j["a_star"] for j in joined], [j[key] for j in joined])
        for key in ("margin_gap", "spr_ratio", "l8", "redundancy_excess")
    }
    summary = {
        "predictions": payload["predictions"],
        "by_attack_and_placement": table,
        "joined": joined,
        "link_spearman": link,
        "link_pairs": len(joined),
    }
    return summary


# The over-determination account (sufficiency.py): per model, how much of the
# attack label the trigger carries on content-free carriers (blank excess) and
# with the image content hidden (content excess), against PSBD-TM's AUROC and
# TPR at the 0.01 quantile at the adaptive rate, and against A* where
# critical_rate.py read the model.
def sufficiency_summary(directory, results_dir="results"):
    # The test runs on the evidence-surplus set of 10 ViT models, 2 ResNet-18
    # reproductions and the benign controls (the user's scope of 2026-09-30).
    folder = os.path.join(
        results_dir, "_experiments", "evidence_surplus", "sufficiency"
    )
    if not os.path.isdir(folder):
        return None
    critical_path = os.path.join(directory, "critical_rate.json")
    a_star = {}
    if os.path.exists(critical_path):
        with open(critical_path) as handle:
            for row in json.load(handle)["rows"]:
                if row["placement"] == "before_attention_norm_token_mask":
                    a_star[row["folder"]] = row["a_star"]
    rows = []
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name)) as handle:
            record = json.load(handle)
        tm = psbd_tm_reading(results_dir, record["folder"], record["architecture"])
        rows.append(
            {
                "name": name[: -len(".json")],
                "folder": record["folder"],
                "architecture": record["architecture"],
                "attack": "benign" if record["probe_attack"] else record["attack"],
                "probe": record["probe_attack"],
                "blank_excess": record["blank"]["excess"],
                "content_excess": (record["content"] or {}).get("excess"),
                "content_collapsed": (record["content"] or {}).get("collapsed", True),
                "any_class": record["classes"]["non_source_stamped_on_target"],
                "source_class": record["classes"]["source_stamped_on_target"],
                "auroc": tm["auroc"] if tm and not record["probe_attack"] else None,
                "tpr_q01": tm["tpr"] if tm and not record["probe_attack"] else None,
                "a_star": a_star.get(record["folder"])
                if not record["probe_attack"]
                else None,
            }
        )
    backdoored = [r for r in rows if r["attack"] != "benign" and r["auroc"] is not None]
    benign = [r for r in rows if r["attack"] == "benign"]
    by_attack = collections.defaultdict(list)
    for r in backdoored:
        by_attack[(r["architecture"], r["attack"])].append(r)
    with_a_star = [r for r in backdoored if r["a_star"] is not None]
    summary = {
        "rows": rows,
        "S1_models": len(backdoored),
        "S1_spearman_auroc": spearman(
            [r["blank_excess"] for r in backdoored], [r["auroc"] for r in backdoored]
        ),
        "S1_spearman_tpr": spearman(
            [r["blank_excess"] for r in backdoored], [r["tpr_q01"] for r in backdoored]
        ),
        "S1_content_spearman_auroc": spearman(
            [r["content_excess"] for r in backdoored if not r["content_collapsed"]],
            [r["auroc"] for r in backdoored if not r["content_collapsed"]],
        ),
        "by_attack": {
            f"{a}|{attack}": {
                "models": len(group),
                "blank_excess": mean([r["blank_excess"] for r in group]),
                "content_excess": mean([r["content_excess"] for r in group]),
                "any_class": mean([r["any_class"] for r in group]),
                "auroc": mean([r["auroc"] for r in group]),
                "tpr_q01": mean([r["tpr_q01"] for r in group]),
            }
            for (a, attack), group in sorted(by_attack.items())
        },
        "S3_benign_max_blank_excess": max(
            (abs(r["blank_excess"]) for r in benign), default=None
        ),
        "S3_benign_max_content_excess": max(
            (abs(r["content_excess"]) for r in benign), default=None
        ),
        # Hypothesis-level: does sufficiency account for what A* leaves over?
        "a_star_residual_spearman": spearman(
            [r["blank_excess"] for r in with_a_star],
            [r["auroc"] - r["a_star"] for r in with_a_star],
        ),
        "a_star_models": len(with_a_star),
    }
    return summary


# PSBD-TM on the transformers, PSBD-RD on the ResNet-18 reproduction, whose only
# site is after the residual add.
def psbd_tm_reading(results_dir, folder, architecture):
    path = os.path.join(results_dir, folder, "psbd_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        placement = (
            "post_residual"
            if architecture == "resnet18"
            else "before_attention_norm_token_mask"
        )
        block = json.load(handle)["placements"].get(placement)
    if not block or block.get("adaptive_rate") is None:
        return None
    row = next(r for r in block["rates"] if r["rate"] == block["adaptive_rate"])
    reading = {
        "auroc": row["detection_psu_ratio"]["q0.25"]["auroc"],
        "tpr": row["detection_psu_ratio"]["q0.01"]["tpr"],
    }
    return reading


# Memo L26: probe accuracy for the exact warp against a random warp, on single
# tokens and on the class token, per block.
def wanet_probe_rows(directory):
    rows = []
    for name in sorted(os.listdir(directory)):
        if not name.startswith("wanet_probe__"):
            continue
        with open(os.path.join(directory, name)) as handle:
            record = json.load(handle)
        success = success_flags([record["folder"]])[record["folder"]]
        rows.append({**record, **success})
    return rows


def read_records(directory):
    records, seeds, cached = [], [], None
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name in ("summary.json",):
            continue
        with open(os.path.join(directory, name)) as handle:
            payload = json.load(handle)
        if name.startswith("seed_ensemble__"):
            seeds.append(payload)
        elif name == "cached_reads.json":
            cached = payload
        elif name in ("critical_rate.json", "shift_curves.json") or not name.endswith(
            ".json"
        ):
            continue
        elif name.startswith(("sanity_", "wanet_probe__")):
            continue
        else:
            records.append(payload)
    return records, seeds, cached


# Everything a verdict or a figure reads, 1 flat row per model. Medians over
# images, since several per-image ratios have heavy tails where a margin or a
# component sits near 0.
def model_row(record):
    hit = torch.tensor(record["baseline"]["hit"], dtype=torch.bool)
    row = {
        "name": record_name(record),
        "folder": record["folder"],
        "architecture": record["architecture"],
        "attack": record["attack"],
        "category": record["category"],
        "group": analysis_group(record),
        "dataset": record["dataset"],
        "pairs": record["pairs"],
        "hit_pairs": record["hit_pairs"],
        "clean_accuracy": record["baseline"]["clean_accuracy"],
        "triggered_asr": record["baseline"]["triggered_asr"],
        "auroc_margin_unperturbed": record["baseline"]["auroc_triggered_larger_margin"],
        "auroc_confidence_unperturbed": record["baseline"][
            "auroc_triggered_larger_confidence"
        ],
        "clean_margin": median(record["baseline"]["clean_margin"]),
        "triggered_margin": median(select(record["baseline"]["triggered_margin"], hit)),
        "cosine_backdoor_target_readout": record["direction_geometry"][
            "cosine_backdoor_target_readout"
        ],
        "cache_agreement": record.get("cache_agreement"),
        "sanity": record.get("sanity"),
        "rates": record["rates"],
        "operators": {
            name: operator_row(block, hit, record)
            for name, block in record["operators"].items()
        },
        "ood": {
            name: {
                "auroc_triggered_vs_clean": block["auroc_triggered_vs_clean"],
                "auroc_foreign_vs_clean": block["auroc_foreign_vs_clean"],
            }
            for name, block in record["ood"].items()
        },
        "redundancy": redundancy_row(record["redundancy"]),
        "low_dim": low_dim_row(record["low_dim"], record),
        "sufficiency": sufficiency_row(record["sufficiency"]),
        "flatness": flatness_row(record["flatness"]),
        "neurons": neuron_row(record["neurons"]),
        "small_noise": small_noise_row(record.get("small_noise"), record),
        "token_spread": {
            block: {
                "cv": median(values["coefficient_of_variation"]),
                "top_token_share": median(values["top_token_share"]),
            }
            for block, values in record["token_spread"].items()
        },
    }
    return row


def operator_row(block, hit, record):
    clean, triggered = block["clean"], block["triggered"]
    psu_clean = torch.tensor(clean["psu_ratio"])
    psu_triggered = torch.tensor(triggered["psu_ratio"])[hit]
    margin_clean = torch.tensor(record["baseline"]["clean_margin"])
    margin_triggered = torch.tensor(record["baseline"]["triggered_margin"])[hit]
    retention_triggered = select(triggered["backdoor_retention"], hit)

    row = {
        "rate": block["rate"],
        "auroc_paired": block["auroc_paired_pipeline_form"],
        "auroc_hit_only": block["auroc_hit_only"],
        "auroc_without_target": block.get("auroc_hit_only_without_target"),
        "auroc_paired_without_target": block.get("auroc_paired_without_target"),
        "margin_matched_auroc": margin_matched_auroc(
            psu_triggered, margin_triggered, psu_clean, margin_clean
        ),
        "clean_kept": mean(clean["kept"]),
        "triggered_kept": mean(select(triggered["kept"], hit)),
        "clean_margin_retention": median(clean["margin_retention"]),
        "triggered_margin_retention": median(
            select(triggered["margin_retention"], hit)
        ),
        "clean_margin_drop": median(clean["margin_drop"]),
        "triggered_margin_drop": median(select(triggered["margin_drop"], hit)),
        "auroc_margin_retention": auroc_low_is_positive(
            -torch.tensor(select(triggered["margin_retention"], hit)),
            -torch.tensor(finite(clean["margin_retention"])),
        ),
        "triggered_backdoor_share": median(select(triggered["backdoor_share"], hit)),
        "clean_relative_change": median(clean["relative_change"]),
        "triggered_relative_change": median(select(triggered["relative_change"], hit)),
        "triggered_backdoor_retention": median(retention_triggered),
        "clean_class_retention": median(clean["class_retention"]),
        "triggered_backdoor_spr": median(
            select(triggered["backdoor_signal_to_perturbation"], hit)
        ),
        "clean_class_spr": median(clean["class_signal_to_perturbation"]),
        "spearman_psu_backdoor_drop": spearman(
            select(triggered["psu_ratio"], hit),
            [None if r is None else 1.0 - r for r in retention_triggered],
        ),
        # docs/why-psbd-works-theory.md, L8: most triggered images never flip, so
        # their statistic sits at the bfloat16 floor. The 2 forms it asks for are
        # triggered images with phi above 0.01, and clean and triggered pooled.
        "spearman_psu_backdoor_drop_above_floor": spearman_above(
            select(triggered["psu_ratio"], hit),
            [None if r is None else 1.0 - r for r in retention_triggered],
            0.01,
        ),
        "triggered_share_above_floor": share_above(
            select(triggered["psu_ratio"], hit), 0.01
        ),
        "spearman_psu_backdoor_drop_pooled": spearman(
            select(triggered["psu_ratio"], hit) + list(clean["psu_ratio"]),
            [None if r is None else 1.0 - r for r in retention_triggered]
            + [None if r is None else 1.0 - r for r in clean["backdoor_retention"]],
        ),
        "clean_shift_to_target": block["clean_shift_to_target"],
        "uniform_share": 1.0 / record["num_classes"],
        "flagged": block["flagged"],
    }
    for quantity in ("predictive_entropy", "mutual_information", "own_prob_std"):
        row[f"auroc_{quantity}"] = block[f"auroc_{quantity}"]
    return row


# Each triggered image is paired with the clean image whose unperturbed margin is
# nearest, with replacement. An AUROC that stays high on these pairs is not
# explained by the triggered margin being larger to begin with.
def spearman_above(first, second, floor):
    kept = [
        (a, b)
        for a, b in zip(first, second)
        if a is not None and b is not None and a > floor
    ]
    value = spearman([k[0] for k in kept], [k[1] for k in kept])
    return value


def share_above(values, floor):
    kept = finite(values)
    share = sum(v > floor for v in kept) / len(kept) if kept else None
    return share


def margin_matched_auroc(psu_triggered, margin_triggered, psu_clean, margin_clean):
    if len(psu_triggered) == 0:
        return None
    distance = (margin_triggered[:, None] - margin_clean[None, :]).abs()  # (t, c)
    nearest = distance.argmin(dim=1)  # (t,)
    matched_clean = psu_clean[nearest]  # (t,)
    auroc = auroc_low_is_positive(psu_triggered, matched_clean)
    return auroc


def redundancy_row(redundancy):
    reference = redundancy["1.0"]["mean"]
    reference_excess = reference["triggered_on_target"] - reference["clean_on_target"]
    row = {}
    for fraction, block in redundancy.items():
        mean_reading = block["mean"]
        excess = mean_reading["triggered_on_target"] - mean_reading["clean_on_target"]
        entry = {
            "clean_kept": mean_reading["clean_kept"],
            "triggered_kept": mean_reading["triggered_kept"],
            "clean_on_target": mean_reading["clean_on_target"],
            "excess_retention": excess / reference_excess
            if reference_excess > 0
            else None,
            "draws": len(block["draws"]),
        }
        forced = block.get("trigger_forced_visible")
        if forced:
            entry["trigger_forced_triggered_kept"] = forced["triggered_kept"]
            hidden = [d for d in block["draws"] if d.get("trigger_visible") == 0]
            entry["trigger_hidden_triggered_kept"] = (
                sum(d["triggered_kept"] for d in hidden) / len(hidden)
                if hidden
                else None
            )
            entry["trigger_hidden_draws"] = len(hidden)
        row[fraction] = entry
    return row


def low_dim_row(low_dim, record):
    spectrum = low_dim["difference_spectrum"]
    row = {
        "trigger_top1_share": spectrum["trigger_change"]["top1_share"],
        "trigger_participation": spectrum["trigger_change"]["participation_ratio"],
        "clean_pair_top1_share": spectrum["clean_to_clean_change"]["top1_share"],
        "clean_pair_participation": spectrum["clean_to_clean_change"][
            "participation_ratio"
        ],
        "jacobian": {},
        "attribution": {},
    }
    for block, splits in low_dim["jacobian"].items():
        row["jacobian"][block] = {
            f"{split}_{key}": median(values[key])
            for split, values in splits.items()
            for key in ("top_token_share", "effective_tokens", "effective_rank")
        }
        if "trigger_token_share" in splits["triggered"]:
            row["jacobian"][block]["triggered_trigger_token_share"] = median(
                splits["triggered"]["trigger_token_share"]
            )
    for split, values in low_dim["attribution"].items():
        row["attribution"][f"{split}_gini"] = median(values["gini"])
        row["attribution"][f"{split}_top_pixel_share"] = median(
            values["top_pixel_share"]
        )
        if "trigger_pixel_share" in values:
            row["attribution"][f"{split}_trigger_pixel_share"] = median(
                values["trigger_pixel_share"]
            )
    return row


def sufficiency_row(sufficiency):
    row = {}
    for order in ("random", "ranked"):
        clean = sufficiency["clean"][order]["smallest_fraction"]
        triggered = sufficiency["triggered"][order]["smallest_fraction"]
        row[order] = {
            "clean_median": median(clean),
            "triggered_median": median(triggered),
            "auroc": auroc_low_is_positive(
                torch.tensor(triggered), torch.tensor(clean)
            ),
        }
    return row


def flatness_row(flatness):
    row = {}
    for site, splits in flatness.items():
        entry = {}
        for quantity in ("curvature_margin", "curvature_log_prob", "slope_margin"):
            clean = splits["clean"][FLAT_EPSILON][quantity]
            triggered = splits["triggered"][FLAT_EPSILON][quantity]
            entry[f"clean_{quantity}"] = median(clean)
            entry[f"triggered_{quantity}"] = median(triggered)
            # Flatter means a smaller absolute value, so the triggered image is
            # the positive and a low absolute value flags it.
            entry[f"auroc_flatter_{quantity}"] = auroc_low_is_positive(
                torch.tensor(finite(triggered)).abs(), torch.tensor(finite(clean)).abs()
            )
        clean_margin = torch.tensor(splits["clean"]["margin"])
        triggered_margin = torch.tensor(splits["triggered"]["margin"])
        relative_clean = torch.tensor(splits["clean"][FLAT_EPSILON]["curvature_margin"])
        relative_triggered = torch.tensor(
            splits["triggered"][FLAT_EPSILON]["curvature_margin"]
        )
        entry["clean_curvature_over_margin"] = float(
            (relative_clean / clean_margin).median()
        )
        entry["triggered_curvature_over_margin"] = float(
            (relative_triggered / triggered_margin).median()
        )
        row[site] = entry
    return row


# docs/why-psbd-works-theory.md, L2 and P4: a second-order account predicts
# phi(2r) / phi(r) near 4 per image, a rank order that holds across r (Spearman
# at least 0.9) and an AUROC at small r within 0.02 of the operating one.
def small_noise_row(small_noise, record):
    if not small_noise:
        return None
    rates = sorted(small_noise, key=float)
    # Keyed by rate rather than listed, so the group means average them.
    row = {
        "auroc_hit_only": {r: small_noise[r]["auroc_hit_only"] for r in rates},
        "auroc_at_smallest_rate": small_noise[rates[0]]["auroc_hit_only"],
        "operating_gaussian_auroc": record["operators"]["gaussian"]["auroc_hit_only"],
        "operating_gaussian_rate": record["rates"]["gaussian"]["rate"],
    }
    low, high = rates[0], rates[-1]
    for split in ("clean", "triggered"):
        first = small_noise[low][f"{split}_psu_ratio"]
        second = small_noise[high][f"{split}_psu_ratio"]
        kept = [
            (a, b)
            for a, b in zip(first, second)
            if a is not None and b is not None and a > 1e-4
        ]
        row[f"{split}_ratio_median"] = median([b / a for a, b in kept])
        row[f"{split}_spearman_across_rates"] = spearman(
            [k[0] for k in kept], [k[1] for k in kept]
        )
    return row


def neuron_row(neurons):
    row = {
        "late_top30_triggered_kept": neurons["ablate_late_top30"]["triggered_kept"],
        "late_random30_triggered_kept": neurons["ablate_late_random30"][
            "triggered_kept"
        ],
        "late_top30_triggered_on_target": neurons["ablate_late_top30"][
            "triggered_on_target"
        ],
        "late_random30_triggered_on_target": neurons["ablate_late_random30"][
            "triggered_on_target"
        ],
        "late_top30_clean_kept": neurons["ablate_late_top30"]["clean_kept"],
        "all_top_triggered_kept": neurons["ablate_top_tac"]["triggered_kept"],
        "all_random_triggered_kept": neurons["ablate_random"]["triggered_kept"],
        "all_top_clean_kept": neurons["ablate_top_tac"]["clean_kept"],
        "retention": {
            name: {
                "triggered_top_tac": median(block["triggered_top_tac"]),
                "clean_own_top": median(block["clean_own_top"]),
            }
            for name, block in neurons["retention"].items()
        },
    }
    return row


def seed_row(record):
    row = {
        "cell": record["cell"],
        "psbd_tm_cached_auroc": record["psbd_tm_cached_auroc"],
        "member_triggered_asr": record["member_triggered_asr"],
        "clean_spearman_with_psbd_tm": record.get("clean_spearman_with_psbd_tm"),
        "triggered_spearman_with_psbd_tm": record.get(
            "triggered_spearman_with_psbd_tm"
        ),
    }
    for name in ("ensemble_psu_ratio", "disagreement", "mutual_information"):
        row[name] = record[name]["auroc_paired_pipeline_form"]
        row[f"{name}_hit_only"] = record[name]["auroc_hit_only"]
    return row


def cached_summary(cached):
    if cached is None:
        return None
    # The panel is the successful_2pt models, the benign references kept apart.
    cached = dict(
        cached, models=[r for r in cached["models"] if r.get("successful_2pt")]
    )
    summary = {}
    for group in ("models", "benign"):
        summary[group] = {}
        for name in ("token_mask", "residual_dropout"):
            rows = [r for r in cached[group] if r.get(name)]
            easy = [
                r[name]["target_class_easy"]
                for r in rows
                if r[name]["target_class_easy"]
            ]
            destination = [
                r[name]["shift_destination"]
                for r in rows
                if r[name]["shift_destination"]
            ]
            summary[group][name] = {
                "models": len(easy),
                "target_flagged_share": mean([e["target_flagged_share"] for e in easy]),
                "other_flagged_share": mean([e["other_flagged_share"] for e in easy]),
                "target_shift_share": mean([d["target_share"] for d in destination]),
                "largest_class_share": mean(
                    [d["largest_class_share"] for d in destination]
                ),
                "uniform_share": mean([d["uniform_share"] for d in destination]),
            }
    by_attack = collections.defaultdict(list)
    for row in cached["models"]:
        by_attack[row["attack"]].append(row)
    summary["by_attack"] = {
        attack: {
            name: {
                "target_flagged_share": mean(
                    [
                        r[name]["target_class_easy"]["target_flagged_share"]
                        for r in rows
                        if r.get(name) and r[name]["target_class_easy"]
                    ]
                ),
                "target_shift_share": mean(
                    [
                        r[name]["shift_destination"]["target_share"]
                        for r in rows
                        if r.get(name) and r[name]["shift_destination"]
                    ]
                ),
            }
            for name in ("token_mask", "residual_dropout")
        }
        for attack, rows in by_attack.items()
    }
    return summary


# Means per architecture and analysis group over every numeric leaf of the rows,
# so each table and figure reads 1 number per group.
def group_means(per_model):
    grouped = collections.defaultdict(list)
    for row in per_model:
        grouped[(row["architecture"], row["group"])].append(row)
    means = {}
    for (architecture, group), rows in grouped.items():
        means.setdefault(architecture, {})[group] = {
            "models": [r["name"] for r in rows],
            "mean": nested_mean(rows),
        }
    return means


def nested_mean(rows):
    # Keys are gathered from every row, since a statistic can be None on the
    # first model of a group (no triggered image above the floor) and numeric on
    # the next.
    keys = []
    for row in rows:
        keys += [key for key in row if key not in keys]
    result = {}
    for key in keys:
        present = [r[key] for r in rows if r.get(key) is not None]
        if any(isinstance(v, dict) for v in present):
            children = [v for v in present if isinstance(v, dict)]
            result[key] = nested_mean(children)
        elif any(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in present
        ):
            values = [
                v
                for v in present
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            result[key] = sum(values) / len(values)
        elif key in rows[0] and not isinstance(rows[0][key], (str, list, bool)):
            result[key] = None
    return result


# Every threshold a verdict reads, named once. The ones marked with a memo id are
# the supported and refuted values of docs/why-psbd-works-theory.md, table
# "Numeric predictions for the open hypotheses". The rest are this experiment's
# own and README.md gives their reasons.
THRESHOLDS = {
    "separating_auroc": 0.7,
    "separating_operators": 4,
    "margin_gap_supported": 0.3,
    "margin_gap_refuted": 0.1,
    "margin_matching_cost": 0.05,
    "l8_supported": 0.7,
    "l8_refuted": 0.3,
    "l8_floor": 0.01,
    "backdoor_share": 0.5,
    "redundancy_fraction": "0.3",
    "redundancy_supported": 0.5,
    "redundancy_above_clean": 0.2,
    "redundancy_refuted_above_clean": 0.05,
    "redundancy_luck": 0.1,
    "low_dim_top1": 0.5,
    "low_dim_top1_refuted": 0.3,
    "low_dim_sufficiency_ratio": 0.5,
    "low_dim_sufficiency_auroc": 0.8,
    "flat_ratio_low": 3.0,
    "flat_ratio_high": 5.0,
    "flat_rank": 0.9,
    "flat_supported_gap": 0.02,
    "flat_refuted_gap": 0.1,
    "l10_supported": 0.10,
    "l10_refuted": 0.02,
    "uncertainty_supported_gap": 0.02,
    "uncertainty_refuted_gap": 0.1,
    "ood_knn_supported": 0.8,
    "ood_knn_refuted": 0.6,
    "ood_foreign_supported": 0.25,
    "ood_foreign_refuted": 0.1,
    "l7_supported": 0.1,
    "l7_refuted": 0.9,
    "l7_random_gap": 0.05,
    "l22_supported": 0.02,
    "l22_refuted": 0.05,
    "p1_margin": 0.02,
    "p10_supported": 0.75,
    "p10_benign": 0.35,
    "p10_refuted": 0.40,
    "p2_supported_auroc": 0.9,
    "p2_supported_spearman": 0.5,
    "p2_refuted_spearman": 0.2,
}
T = THRESHOLDS


# The 3 accounts read on the whole panel from the caches (P1, P10) or on the
# seed-replicated cells (P2), per attack.
def panel_verdicts(cached, seeds):
    if cached is None:
        return None
    ceiling = {
        r["folder"]: r
        for r in cached["refutations"]["confidence_ceiling"]
        if r.get("successful_2pt")
    }
    per_model = {r["folder"]: r for r in cached["refutations"]["per_model"]}
    by_attack = collections.defaultdict(list)
    # SIG is quarantined for the amplitude drift of the 2026-09-29 audit, so no
    # verdict reads its model even where the cache is internally consistent.
    for folder, row in ceiling.items():
        if row["attack"] in QUARANTINED_ATTACKS:
            continue
        by_attack[row["attack"]].append((row, per_model[folder]))
    confidence = {}
    for attack, rows in by_attack.items():
        margin = mean([m["psbd_tm"] - c["a_star"] for c, m in rows])
        beats = sum(m["psbd_tm"] > c["a_star"] for c, m in rows)
        if margin >= T["p1_margin"] and beats > len(rows) / 2:
            status = "refuted"
        elif margin <= 0:
            status = "supported"
        else:
            status = "partial"
        confidence[attack] = verdict(
            status,
            "PSBD-TM minus A*(P_c)",
            margin,
            f"P1: confidence carries the signal if PSBD-TM does not beat A*(P_c), "
            f"refuted when it beats it by {T['p1_margin']} or more on most models",
        )
        confidence[attack]["models"] = len(rows)
        confidence[attack]["beats"] = beats
        confidence[attack]["rd_margin"] = mean(
            [m["psbd_rd"] - c["a_star"] for c, m in rows]
        )

    easy = {}
    benign = mean(
        [
            r["token_mask"]["target_class_easy"]["target_flagged_share"]
            for r in cached["benign"]
            if r.get("token_mask") and r["token_mask"]["target_class_easy"]
        ]
    )
    for attack, block in cached_summary(cached)["by_attack"].items():
        if attack in QUARANTINED_ATTACKS:
            continue
        share = block["token_mask"]["target_flagged_share"]
        if share is None:
            continue
        if share >= T["p10_supported"] and (benign or 1) <= T["p10_benign"]:
            status = "supported"
        elif share <= T["p10_refuted"]:
            status = "refuted"
        else:
            status = "partial"
        easy[attack] = verdict(
            status,
            "clean target-class images flagged, PSBD-TM",
            share,
            f"P10: supported at {T['p10_supported']} or more with benign at "
            f"{T['p10_benign']} or less, refuted at {T['p10_refuted']} or less",
        )
        easy[attack]["benign"] = benign

    ensemble = {}
    for row in seeds:
        attack = row["cell"].split("_", 2)[2].rsplit("_", 2)[0]
        ensemble.setdefault(attack, []).append(row)
    epistemic = {}
    for attack, rows in ensemble.items():
        auroc = mean([r["ensemble_psu_ratio"] for r in rows])
        rho = mean([r["clean_spearman_with_psbd_tm"] for r in rows])
        a_star = mean(
            [ceiling[r["cell"]]["a_star"] for r in rows if r["cell"] in ceiling]
        )
        if (
            auroc is not None
            and auroc >= T["p2_supported_auroc"]
            and (rho or 0) >= T["p2_supported_spearman"]
        ):
            status = "supported"
        elif (a_star is not None and auroc is not None and auroc <= a_star) or (
            rho is not None and rho <= T["p2_refuted_spearman"]
        ):
            status = "refuted"
        else:
            status = "partial"
        epistemic[attack] = verdict(
            status,
            "3-seed ensemble statistic AUROC",
            auroc,
            f"P2: supported at {T['p2_supported_auroc']} or more with a clean Spearman "
            f"of {T['p2_supported_spearman']} or more against PSBD-TM, refuted at or "
            f"below A*(P_c) or at a Spearman of {T['p2_refuted_spearman']} or less",
        )
        epistemic[attack]["clean_spearman"] = rho
        epistemic[attack]["a_star"] = a_star
        epistemic[attack]["cells"] = len(rows)
    readings = {"confidence": confidence, "target_easy": easy, "epistemic": epistemic}
    return readings


def verdict(status, key, value, rule):
    reading = {"status": status, "key": key, "value": value, "rule": rule}
    return reading


# The verdict rules, applied per architecture and group. Each rule returns a
# status, the key number it read and its text built from THRESHOLDS, so the
# notebook's grid and the README print the same thing.
def verdicts(grouped):
    table = {}
    for architecture, groups in grouped.items():
        for group, block in groups.items():
            m = block["mean"]
            table.setdefault(architecture, {})[group] = {
                "phenomenon": phenomenon_verdict(m),
                "margin": margin_verdict(m),
                "direction": direction_verdict(m),
                "redundancy": redundancy_verdict(m),
                "low_dim": low_dim_verdict(m),
                "flatness": flatness_verdict(m),
                "neuron_bias": neuron_bias_verdict(m),
                "pass_uncertainty": pass_uncertainty_verdict(m),
                "ood": ood_verdict(m),
                "trigger_neurons": trigger_neuron_verdict(m),
                "missingness": missingness_verdict(m),
            }
    return table


def phenomenon_verdict(m):
    values = [m["operators"][o]["auroc_hit_only"] for o in HEADLINE_OPERATORS]
    separating = sum(v is not None and v >= T["separating_auroc"] for v in values)
    if separating >= T["separating_operators"]:
        status = "supported"
    elif separating >= 1:
        status = "partial"
    else:
        status = "refuted"
    reading = verdict(
        status,
        f"headline operators of {len(HEADLINE_OPERATORS)} with AUROC at least "
        f"{T['separating_auroc']}",
        separating,
        f"the statistic separates with at least {T['separating_operators']} of the "
        f"{len(HEADLINE_OPERATORS)} headline operators",
    )
    reading["token_mask_auroc"] = m["operators"]["token_mask"]["auroc_hit_only"]
    return reading


def margin_verdict(m):
    operators = m["operators"]
    gaps, drops = [], []
    for name in HEADLINE_OPERATORS:
        block = operators[name]
        if (block["auroc_hit_only"] or 0) < T["separating_auroc"]:
            continue
        gaps.append(
            (block["triggered_margin_retention"] or 0)
            - (block["clean_margin_retention"] or 0)
        )
        drops.append(
            (block["auroc_hit_only"] or 0) - (block["margin_matched_auroc"] or 0)
        )
    if not gaps:
        return verdict("no data", "retention gap", None, "no separating operator")
    gap, drop = mean(gaps), mean(drops)
    if gap >= T["margin_gap_supported"] and drop <= T["margin_matching_cost"]:
        status = "supported"
    elif gap < T["margin_gap_refuted"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        "triggered minus clean median margin retention",
        gap,
        f"supported if the retention gap is at least {T['margin_gap_supported']} on "
        f"the separating operators and margin matching costs at most "
        f"{T['margin_matching_cost']} AUROC, refuted if the gap is below "
        f"{T['margin_gap_refuted']}",
    )
    reading["auroc_lost_to_margin_matching"] = drop
    return reading


def direction_verdict(m):
    tm = m["operators"]["token_mask"]
    l8 = tm.get("spearman_psu_backdoor_drop_above_floor")
    share = tm.get("triggered_backdoor_share")
    if l8 is None:
        return verdict("no data", "L8 Spearman", None, "")
    if l8 >= T["l8_supported"] and (share or 0) >= T["backdoor_share"]:
        status = "supported"
    elif l8 <= T["l8_refuted"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        f"L8 Spearman, triggered phi above {T['l8_floor']}, PSBD-TM",
        l8,
        f"L8: supported at {T['l8_supported']} or more with a backdoor share of at "
        f"least {T['backdoor_share']}, refuted at {T['l8_refuted']} or less",
    )
    reading["backdoor_share"] = share
    reading["pooled_spearman"] = tm.get("spearman_psu_backdoor_drop_pooled")
    return reading


def redundancy_verdict(m):
    entry = m["redundancy"].get(T["redundancy_fraction"])
    key = f"triggered excess retention at {T['redundancy_fraction']} visible"
    if not entry or entry.get("excess_retention") is None:
        return verdict("no data", key, None, "")
    excess, clean = entry["excess_retention"], entry["clean_kept"]
    forced = entry.get("trigger_forced_triggered_kept")
    luck = (
        forced is not None and forced - entry["triggered_kept"] > T["redundancy_luck"]
    )
    if (
        excess >= T["redundancy_supported"]
        and excess >= clean + T["redundancy_above_clean"]
        and not luck
    ):
        status = "supported"
    elif excess <= clean + T["redundancy_refuted_above_clean"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        key,
        excess,
        f"supported at {T['redundancy_supported']} or more and "
        f"{T['redundancy_above_clean']} above clean retention (L20, L23), with a patch "
        f"trigger's forced-visible run within {T['redundancy_luck']}, refuted at "
        f"{T['redundancy_refuted_above_clean']} or less above clean",
    )
    reading["clean_kept"] = clean
    reading["trigger_forced_triggered_kept"] = forced
    return reading


def low_dim_verdict(m):
    ranked = m["sufficiency"]["ranked"]
    triggered, clean = ranked["triggered_median"], ranked["clean_median"]
    top1 = m["low_dim"]["trigger_top1_share"]
    key = "top eigenvalue share of the triggered minus clean feature change"
    if triggered is None or clean is None or top1 is None:
        return verdict("no data", key, None, "")
    few_directions = top1 >= T["low_dim_top1"]
    small_input = (
        triggered <= clean * T["low_dim_sufficiency_ratio"]
        and (ranked["auroc"] or 0) >= T["low_dim_sufficiency_auroc"]
    )
    if few_directions and small_input:
        status = "supported"
    elif triggered >= clean and top1 < T["low_dim_top1_refuted"]:
        status = "refuted"
    else:
        status = "partial"
    last_block = str(max(int(b) for b in m["low_dim"]["jacobian"]))
    jacobian = m["low_dim"]["jacobian"][last_block]
    reading = verdict(
        status,
        key,
        top1,
        f"supported if 1 direction holds at least {T['low_dim_top1']} of the paired "
        f"change and the ranked sufficient token share is at most "
        f"{T['low_dim_sufficiency_ratio']} of the clean one with an AUROC of "
        f"{T['low_dim_sufficiency_auroc']} (L6), refuted if the triggered share is "
        f"not below the clean one and the top direction holds under "
        f"{T['low_dim_top1_refuted']}",
    )
    reading["ranked_sufficient_triggered"] = triggered
    reading["ranked_sufficient_clean"] = clean
    reading["jacobian_effective_tokens_triggered"] = jacobian[
        "triggered_effective_tokens"
    ]
    reading["jacobian_effective_tokens_clean"] = jacobian["clean_effective_tokens"]
    return reading


def flatness_verdict(m):
    small = m.get("small_noise")
    key = "Gaussian AUROC at the smallest rate, float32"
    if not small or small.get("auroc_at_smallest_rate") is None:
        return verdict("no data", key, None, "")
    at_small = small["auroc_at_smallest_rate"]
    operating = small["operating_gaussian_auroc"]
    ratio_value = small.get("clean_ratio_median")
    rank = small.get("clean_spearman_across_rates")
    if operating is not None and at_small <= operating - T["flat_refuted_gap"]:
        status = "refuted"
    elif (
        operating is not None
        and abs(at_small - operating) <= T["flat_supported_gap"]
        and ratio_value is not None
        and T["flat_ratio_low"] <= ratio_value <= T["flat_ratio_high"]
        and (rank or 0) >= T["flat_rank"]
    ):
        status = "supported"
    else:
        status = "partial"
    reading = verdict(
        status,
        key,
        at_small,
        f"L2, P4: supported if phi(2r)/phi(r) lies between {T['flat_ratio_low']} and "
        f"{T['flat_ratio_high']}, ranks hold across r at {T['flat_rank']} and the "
        f"small-r AUROC is within {T['flat_supported_gap']} of the operating one, "
        f"refuted if it is {T['flat_refuted_gap']} or more below",
    )
    reading["operating_auroc"] = operating
    reading["clean_ratio_median"] = ratio_value
    reading["clean_rank_spearman"] = rank
    reading["finite_difference_auroc"] = {
        site: entry["auroc_flatter_curvature_log_prob"]
        for site, entry in m["flatness"].items()
    }
    return reading


def neuron_bias_verdict(m):
    rd, tm = m["operators"]["residual_dropout"], m["operators"]["token_mask"]
    change = delta(rd.get("auroc_hit_only"), rd.get("auroc_without_target"))
    key = "PSBD-RD AUROC lost when the target's gain is redistributed"
    if change is None:
        return verdict("no data", key, None, "")
    if change >= T["l10_supported"]:
        status = "supported"
    elif abs(change) < T["l10_refuted"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        key,
        change,
        f"L10: supported if the AUROC falls by {T['l10_supported']} or more, "
        f"refuted if it moves by less than {T['l10_refuted']}",
    )
    reading["token_mask_change"] = delta(
        tm.get("auroc_hit_only"), tm.get("auroc_without_target")
    )
    reading["shift_to_target_rd"] = rd.get("clean_shift_to_target")
    reading["shift_to_target_tm"] = tm.get("clean_shift_to_target")
    reading["uniform"] = tm.get("uniform_share")
    return reading


def pass_uncertainty_verdict(m):
    tm = m["operators"]["token_mask"]
    psu, bald = tm["auroc_hit_only"], tm["auroc_mutual_information"]
    if psu is None or bald is None:
        return verdict("no data", "BALD AUROC", None, "")
    if bald >= psu - T["uncertainty_supported_gap"]:
        status = "supported"
    elif bald <= psu - T["uncertainty_refuted_gap"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        "BALD AUROC over the PSBD-TM passes",
        bald,
        f"supported if the class-blind uncertainty separates within "
        f"{T['uncertainty_supported_gap']} of the statistic, refuted if it trails by "
        f"{T['uncertainty_refuted_gap']} or more",
    )
    reading["statistic_auroc"] = psu
    return reading


def ood_verdict(m):
    knn = m["ood"]["knn_distance"]["auroc_triggered_vs_clean"]
    flagged = m["operators"]["token_mask"]["flagged"]
    key = "foreign images flagged minus clean flagged, PSBD-TM"
    if knn is None:
        return verdict("no data", key, None, "")
    foreign_excess = flagged["foreign"] - flagged["clean"]
    if knn >= T["ood_knn_supported"] and foreign_excess >= T["ood_foreign_supported"]:
        status = "supported"
    elif knn <= T["ood_knn_refuted"] or foreign_excess <= T["ood_foreign_refuted"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        key,
        foreign_excess,
        f"supported if the kNN distance separates at {T['ood_knn_supported']} and "
        f"PSBD flags foreign images {T['ood_foreign_supported']} above clean ones, "
        f"refuted if the kNN AUROC is at most {T['ood_knn_refuted']} or the foreign "
        f"excess at most {T['ood_foreign_refuted']}",
    )
    reading["knn_auroc"] = knn
    return reading


def trigger_neuron_verdict(m):
    top = m["neurons"]["late_top30_triggered_kept"]
    random_units = m["neurons"]["late_random30_triggered_kept"]
    key = "triggered answers kept with the top units of the last 4 blocks zeroed"
    if top is None:
        return verdict("no data", key, None, "")
    if top <= T["l7_supported"]:
        status = "supported"
    elif top > T["l7_refuted"] and abs(top - random_units) <= T["l7_random_gap"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        key,
        top,
        f"L7: supported at {T['l7_supported']} or less, refuted above "
        f"{T['l7_refuted']} and within {T['l7_random_gap']} of as many random units",
    )
    reading["random_units"] = random_units
    return reading


def missingness_verdict(m):
    operators = m["operators"]
    substitute = operators.get("token_substitute", {}).get("auroc_hit_only")
    change = delta(operators["token_mask"]["auroc_hit_only"], substitute)
    key = "token mask minus token substitute AUROC"
    if change is None:
        return verdict("no data", key, None, "")
    if abs(change) <= T["l22_supported"]:
        status = "supported"
    elif change >= T["l22_refuted"]:
        status = "refuted"
    else:
        status = "partial"
    reading = verdict(
        status,
        key,
        change,
        f"L22: supported within {T['l22_supported']}, refuted when substitution reads "
        f"{T['l22_refuted']} or more lower (read at the adaptive rule, the theory note "
        f"asks for the matched rule)",
    )
    return reading


def analysis_group(record):
    if record["attack"] == "benign" or record.get("probe_attack"):
        group = f"benign_probe_{record['probe_attack']}"
        return group
    if record["attack"] != "tact":
        return record["category"]
    conditional = record["baseline"]["clean_accuracy"] >= TACT_CONDITIONAL_ACCURACY
    group = "patch_tact" if conditional else "patch_tact_source_mapped"
    return group


def record_name(record):
    name = record["folder"]
    if record.get("probe_attack"):
        name = f"{name}:{record['probe_attack']}"
    return name


def select(values, mask):
    chosen = [v for v, keep in zip(values, mask.tolist()) if keep and v is not None]
    return chosen


def finite(values):
    kept = [v for v in values if v is not None]
    return kept


def median(values):
    kept = finite(values)
    value = statistics.median(kept) if kept else None
    return value


def mean(values):
    kept = finite(values)
    value = sum(kept) / len(kept) if kept else None
    return value


def ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    value = numerator / denominator
    return value


def delta(after, before):
    if after is None or before is None:
        return None
    value = after - before
    return value


def spearman(first, second):
    pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    a = torch.tensor([p[0] for p in pairs]).argsort().argsort().double()
    b = torch.tensor([p[1] for p in pairs]).argsort().argsort().double()
    value = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    return value


def markdown(summary):
    lines = []
    for architecture, groups in summary["verdicts"].items():
        lines.append(f"\n{architecture}")
        header = "| group | " + " | ".join(OPERATOR_SPECS) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(OPERATOR_SPECS))
        for group in sorted(groups):
            operators = summary["groups"][architecture][group]["mean"]["operators"]
            cells = " | ".join(
                fmt(operators.get(name, {}).get("auroc_hit_only"))
                for name in OPERATOR_SPECS
            )
            lines.append(f"| {group} | {cells} |")
    text = "\n".join(lines)
    return text


def fmt(value):
    text = "n/a" if value is None else f"{value:.3f}"
    return text


if __name__ == "__main__":
    main()

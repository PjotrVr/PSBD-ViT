"""The adaptive attackers on disk, read from their caches and sidecars.

2 kinds of evasive checkpoint exist. The `_evade_l1` models (ViT and Swin,
CIFAR-100 and Tiny ImageNet, 10 attacks at 3 rates) were each trained with a
hinge against 1 probe. The 14 `vit_cifar100_*_evade_union` models were trained
against 3 probes at once. For each this script reads attack success, clean
accuracy against its non-evasive twin and against the benign reference, the
ledger's success verdicts at both clean-accuracy bars, and the detection of every
cached placement at the adaptive rate, one-sided with the two-sided value as a
diagnostic. The same placements are read on each twin, so every drop is paired.

    .venv/bin/python -m experiments.final_method.adaptive_attackers
"""

import json
import os
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.splits import SPLITS  # noqa: E402
from defenses.cache import read_split_manifest  # noqa: E402
from defenses.decision import (  # noqa: E402
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    attack_success_mask,
)
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.cache_readouts.shared import (  # noqa: E402
    REPO_ROOT,
    choose_rate,
    evaluate_scores,
    load_baselines,
    load_passes,
    validation_shift_by_rate,
    write_json,
)
from scripts.coverage_ledger import (  # noqa: E402
    benign_reference_accuracy,
    classify_cell,
    clean_accuracy_of,
    load_declaration,
    retarget_declaration,
)
from scripts.paper._common import OKABE_ITO, figure_sidecar, save_figure  # noqa: E402

SLUG = "final_method"
CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"
DECLARATION = "configs/psbd_basis.json"
FIGURES_DIR = os.path.join(REPO_ROOT, "experiments", SLUG, "figures")
# The 2 canonical evasion families. Smoke runs, the CIFAR-10 lambda sweep and
# the paper-attack variants are other experiments and stay out.
EVADER = re.compile(r"^(vit|swin)_(cifar100|tiny)_(.+)_(0_\d+)_evade_(l1|union)$")
QUANTILES = (0.01, 0.05, 0.10)
HEADLINE = ("q0.01", "q0.05", "q0.10")
NAMED = {RECOMMENDED_PLACEMENT: "PSBD-TM", PUBLISHED_PLACEMENT: "PSBD-RD"}


def main():
    started = time.perf_counter()
    declaration = load_declaration(DECLARATION)
    references = {
        architecture: benign_reference_accuracy(
            CHECKPOINTS_DIR,
            RESULTS_DIR,
            retarget_declaration(declaration, architecture)["benign_reference"],
        )
        for architecture in ("vit", "swin")
    }

    folders = sorted(f for f in os.listdir(CHECKPOINTS_DIR) if EVADER.match(f))
    rows = [measure_evader(folder, declaration, references) for folder in folders]
    summary = summarize(rows)

    payload = {
        "experiment": "adaptive attackers on disk",
        "quantiles": list(QUANTILES),
        "rate_rule": "adaptive at the canonical shift target, nearest rate when unreached",
        "summary": summary,
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    path = experiment_result_path(SLUG, "adaptive_attackers.json", RESULTS_DIR)
    write_json(payload, path)
    plot(summary, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def measure_evader(folder, declaration, references):
    architecture, dataset, attack, rate_tag, family = EVADER.match(folder).groups()
    twin = folder[: folder.rindex("_evade_")]
    with open(os.path.join(CHECKPOINTS_DIR, folder, "args.json")) as handle:
        metadata = json.load(handle)

    retargeted = retarget_declaration(declaration, architecture)
    reference = references[architecture].get(dataset)
    evader_cell = ledger_cell(folder, metadata, reference, retargeted)
    twin_cell = ledger_cell(twin, read_args(twin), reference, retargeted)

    probes = probed_placements(metadata["evasion"])
    evader_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    twin_dir = os.path.join(RESULTS_DIR, twin, "psbd")
    placements = cached_placements(evader_dir)

    row = {
        "folder": folder,
        "twin": twin,
        "architecture": architecture,
        "dataset": dataset,
        "attack": attack,
        "family": family,
        "poison_rate": metadata.get("poison_rate"),
        "probes": probes,
        "evader": evader_cell,
        "twin_cell": twin_cell,
        "clean_accuracy_drop_vs_twin": difference(
            evader_cell["clean_accuracy"], twin_cell["clean_accuracy"]
        ),
        "placements_cached": placements,
        "detection": {
            placement: {
                "probed": placement in probes,
                "evader": detection_at_adaptive(evader_dir, placement),
                "twin": detection_at_adaptive(twin_dir, placement)
                if placement in cached_placements(twin_dir)
                else None,
            }
            for placement in placements
        },
    }
    return row


def read_args(folder):
    path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        metadata = json.load(handle)
    return metadata


def ledger_cell(folder, metadata, reference, declaration):
    # Attack success is read from the PSBD baseline cache, the source the
    # ledger uses, so evader and twin are measured on the same triggered split.
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    asr = asr_from_cache(psbd_dir) if os.path.isdir(psbd_dir) else None
    cell = {
        "folder_name": folder,
        "dataset": metadata.get("dataset"),
        "attack": metadata.get("attack"),
        "asr": asr if asr is not None else metadata.get("asr"),
        "asr_sidecar": metadata.get("asr"),
        "clean_accuracy": clean_accuracy_of(CHECKPOINTS_DIR, RESULTS_DIR, folder),
    }
    classify_cell(cell, reference, declaration, CHECKPOINTS_DIR, RESULTS_DIR)
    return cell


def asr_from_cache(psbd_dir):
    baselines = load_baselines(psbd_dir)
    _, labels, loader_labels = baselines["backdoor"]
    captured = attack_success_mask(labels, loader_labels)  # (n_backdoor,) bool
    if captured is None:
        return None
    asr = float(captured.float().mean())
    return asr


def probed_placements(evasion):
    probes = evasion.get("probes") or [evasion]
    names = [
        probe["position"]
        if probe["operator"] == "dropout"
        else f"{probe['position']}_{probe['operator']}"
        for probe in probes
    ]
    return names


def cached_placements(psbd_dir):
    if not os.path.isdir(psbd_dir):
        return []
    names = sorted(
        entry
        for entry in os.listdir(psbd_dir)
        if os.path.isdir(os.path.join(psbd_dir, entry))
    )
    return names


def detection_at_adaptive(psbd_dir, placement):
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)
    shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
    if not shift_by_rate:
        return None
    rate = choose_rate(shift_by_rate, "adaptive")
    rule = "adaptive"
    if rate is None:
        rate = choose_rate(shift_by_rate, "nearest")
        rule = "nearest"

    passes = load_passes(psbd_dir, placement, rate, baselines)
    scores = {
        split: psu_ratio_from_cache(
            passes[split]["baseline_probs"],
            passes[split]["baseline_labels"],
            passes[split]["per_pass_probs"],
        )  # (n_split,)
        for split in SPLITS
    }
    evaluation = evaluate_scores(scores, manifest, QUANTILES)
    detection = {
        "rate": rate,
        "rate_rule": rule,
        "validation_shift": shift_by_rate[rate],
        "auroc": evaluation["auroc"],
        # A diagnostic only. The decision rule is one-sided, so an evader that
        # inverts the signal is a detector failure, whatever this reads.
        "auroc_two_sided": max(evaluation["auroc"], 1 - evaluation["auroc"]),
        "inverted": evaluation["auroc"] < 0.5,
        "at_fpr": evaluation["at_fpr"],
    }
    return detection


def difference(value, reference):
    if value is None or reference is None:
        return None
    result = value - reference
    return result


def summarize(rows):
    groups = {
        "vit_l1": [
            r for r in rows if r["architecture"] == "vit" and r["family"] == "l1"
        ],
        "swin_l1": [
            r for r in rows if r["architecture"] == "swin" and r["family"] == "l1"
        ],
        "vit_union": [r for r in rows if r["family"] == "union"],
    }
    summary = {name: summarize_group(group) for name, group in groups.items()}
    return summary


def summarize_group(rows):
    def mean(values):
        values = [v for v in values if v is not None]
        result = sum(values) / len(values) if values else None
        return result

    evaders = [r["evader"] for r in rows]
    group = {
        "n": len(rows),
        "n_asr_clears": sum(c["asr_class"] == "clears" for c in evaders),
        "n_successful_2pt": sum(c["successful_2pt"] for c in evaders),
        "n_successful_5pt": sum(c["successful_5pt"] for c in evaders),
        "n_twin_successful_2pt": sum(r["twin_cell"]["successful_2pt"] for r in rows),
        "mean_asr": mean([c["asr"] for c in evaders]),
        "mean_twin_asr": mean([r["twin_cell"]["asr"] for r in rows]),
        "mean_clean_accuracy_drop_vs_twin": mean(
            [r["clean_accuracy_drop_vs_twin"] for r in rows]
        ),
        "mean_clean_accuracy_drop_vs_benign": mean(
            [c["clean_accuracy_drop"] for c in evaders]
        ),
        "placements_cached_counts": placement_counts(rows),
        "placements": {},
    }
    placements = sorted({p for r in rows for p in r["placements_cached"]})
    for placement in placements:
        for subset_name, subset in (
            ("all", rows),
            ("asr_clears", [r for r in rows if r["evader"]["asr_class"] == "clears"]),
        ):
            readings = [
                r["detection"][placement]
                for r in subset
                if placement in r["detection"] and r["detection"][placement]["evader"]
            ]
            paired = [d for d in readings if d["twin"]]
            block = {
                "n": len(readings),
                "n_probed": sum(d["probed"] for d in readings),
                "n_inverted": sum(d["evader"]["inverted"] for d in readings),
                "n_nearest_rate": sum(
                    d["evader"]["rate_rule"] == "nearest" for d in readings
                ),
                "evader_auroc": mean([d["evader"]["auroc"] for d in readings]),
                "evader_auroc_two_sided": mean(
                    [d["evader"]["auroc_two_sided"] for d in readings]
                ),
                "n_paired": len(paired),
                "twin_auroc": mean([d["twin"]["auroc"] for d in paired]),
                "evader_auroc_paired": mean([d["evader"]["auroc"] for d in paired]),
            }
            for quantile in HEADLINE:
                block[f"evader_tpr_{quantile}"] = mean(
                    [d["evader"]["at_fpr"][quantile]["tpr"] for d in readings]
                )
                block[f"evader_fpr_{quantile}"] = mean(
                    [d["evader"]["at_fpr"][quantile]["realized_fpr"] for d in readings]
                )
                block[f"twin_tpr_{quantile}"] = mean(
                    [d["twin"]["at_fpr"][quantile]["tpr"] for d in paired]
                )
                block[f"evader_tpr_{quantile}_paired"] = mean(
                    [d["evader"]["at_fpr"][quantile]["tpr"] for d in paired]
                )
            group["placements"].setdefault(placement, {})[subset_name] = block
    return group


def placement_counts(rows):
    counts = {}
    for row in rows:
        for placement in row["placements_cached"]:
            counts[placement] = counts.get(placement, 0) + 1
    return counts


def plot(summary, json_path):
    groups = list(summary)
    figure, axes = plt.subplots(
        len(groups), len(HEADLINE), figsize=(15, 4 * len(groups)), squeeze=False
    )
    plotted = {}
    for row_index, group in enumerate(groups):
        placements = summary[group]["placements"]
        names = list(placements)
        for column_index, quantile in enumerate(HEADLINE):
            axis = axes[row_index][column_index]
            twin = [placements[p]["all"][f"twin_tpr_{quantile}"] or 0 for p in names]
            evader = [
                placements[p]["all"][f"evader_tpr_{quantile}_paired"] or 0
                for p in names
            ]
            positions = range(len(names))
            axis.bar(
                [i - 0.2 for i in positions],
                twin,
                0.4,
                label="twin",
                color=OKABE_ITO[0],
            )
            axis.bar(
                [i + 0.2 for i in positions],
                evader,
                0.4,
                label="evader",
                color=OKABE_ITO[1],
            )
            plotted[f"{group}/{quantile}"] = {
                p: {"twin": t, "evader": e} for p, t, e in zip(names, twin, evader)
            }
            axis.set_xticks(list(positions))
            axis.set_xticklabels(
                [NAMED.get(p, p).replace("_", "\n", 2) for p in names], fontsize=7
            )
            axis.set_ylim(0, 1)
            axis.set_title(f"{group}, TPR at {float(quantile[1:]):.0%} FPR", fontsize=9)
        axes[row_index][0].set_ylabel("mean TPR, paired models")
    axes[0][0].legend(fontsize=8)
    figure.suptitle("Adaptive attackers against their non-evasive twins, adaptive rate")

    figure_path = os.path.join(FIGURES_DIR, "adaptive_attackers.png")
    save_figure(figure, figure_path)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/final_method/adaptive_attackers.py",
        [json_path],
        plotted,
    )


if __name__ == "__main__":
    main()

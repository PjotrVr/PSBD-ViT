"""The 7 figure types beyond the histograms and ROC, each with a JSON sidecar of its numbers.

Every figure reads the stage-1 caches or an earlier record, never a GPU. A probe is
read at the adaptive rate its model's numbers.json records, so these figures and
the scores always describe the same passes. Called from make.py.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import (  # noqa: E402
    complete_rates,
    detection_report,
    pair_clean_to_backdoor,
    threshold_at_quantile,
)
from defenses.scores import (  # noqa: E402
    psu_from_cache,
    psu_ratio_from_cache,
    shift_ratio,
    shift_target_histogram,
    to_rank,
)
from experiments.cache_readouts.shared import (  # noqa: E402
    load_baselines,
    load_passes,
    ordered_attacks,
)
from experiments.score_figures import config  # noqa: E402
from scripts.paper._common import OKABE_ITO  # noqa: E402

RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
ACROSS_SETS = "_across_sets"
CLEAN_COLOR, TRIGGERED_COLOR = OKABE_ITO[0], OKABE_ITO[1]
SET_COLORS = {
    "vit_panel": OKABE_ITO[0],
    "swin_panel": OKABE_ITO[1],
    "backdoorbench": OKABE_ITO[2],
}
# Scatter plots draw at most this many points per split, chosen with a fixed
# seed, so a 10000-image triggered split does not hide the clean one.
MAX_POINTS = 2500
POINT_SEED = 0
PICTURE_NAMES = (
    "scatter",
    "confidence",
    "shift_ladder",
    "flip_targets",
    "evaders",
    "negative_psu",
    "threshold_transfer",
)


def run_pictures(names, set_names, root):
    settings = {
        "scatter": config.SCATTER,
        "confidence": config.CONFIDENCE,
        "shift_ladder": config.SHIFT_LADDER,
        "flip_targets": config.FLIP_TARGETS,
        "evaders": config.EVADERS,
        "negative_psu": config.NEGATIVE_PSU,
        "threshold_transfer": config.THRESHOLD_TRANSFER,
    }
    per_model = {
        "scatter": draw_scatter,
        "confidence": draw_confidence,
        "flip_targets": draw_flip_targets,
    }
    for name in names:
        options = settings[name]
        if not options["enabled"]:
            continue
        if name in per_model:
            for set_name, folders in options["models"].items():
                if set_name not in set_names:
                    continue
                for folder in folders:
                    model_dir = os.path.join(root, set_name, folder)
                    per_model[name](model_dir, read_numbers(model_dir), options)
        elif name == "shift_ladder":
            for set_name in options["sets"]:
                if set_name in set_names:
                    draw_shift_ladder(root, set_name, options)
        elif name == "evaders":
            draw_evaders(os.path.join(root, ACROSS_SETS), options)
        elif name == "negative_psu":
            draw_negative_psu(root, options)
        elif name == "threshold_transfer":
            draw_threshold_transfer(root, options)
        print(f"picture {name} done", flush=True)


def draw_scatter(model_dir, numbers, options):
    anchor, partner = options["anchor"], options["partner"]
    psbd_dir = psbd_dir_of(numbers["folder"])
    baselines = load_baselines(psbd_dir)
    anchor_psu = probe_psu(psbd_dir, numbers, anchor, baselines)
    partner_psu = probe_psu(psbd_dir, numbers, partner, baselines)

    ranks = {
        split: torch.stack(
            [
                to_rank(psu[split], psu["validation"])
                for psu in (anchor_psu, partner_psu)
            ]
        ).numpy()  # (2, n_split)
        for split in ("validation", "backdoor")
    }
    n_validation = ranks["validation"].shape[1]

    rules = {}
    for rule, shares in (("min", (1.0, 1.0)), ("weighted", weighted_pair_shares())):
        fused_validation = np.min(
            ranks["validation"] / np.array(shares)[:, None], axis=0
        )  # (n_validation,)
        threshold = threshold_at_quantile(
            torch.from_numpy(fused_validation), options["budget"]
        )
        cutoffs = [share * threshold for share in shares]
        rules[rule] = {
            "shares": list(shares),
            "threshold": threshold,
            "cutoffs": cutoffs,
        }
        for split, split_ranks in ranks.items():
            by_anchor = split_ranks[0] < cutoffs[0]  # (n_split,)
            by_partner = split_ranks[1] < cutoffs[1]  # (n_split,)
            rules[rule][split] = {
                "flagged": float((by_anchor | by_partner).mean()),
                "anchor_only": float((by_anchor & ~by_partner).mean()),
                "partner_only": float((by_partner & ~by_anchor).mean()),
                "both": float((by_anchor & by_partner).mean()),
            }

    # A rank of 0 sits below every clean-validation image and has no place on a
    # log axis, so those points are spread just below 1 / (2 n) and the floor is
    # drawn as a line.
    floor = 0.5 / n_validation
    shown = {
        split: spread_floor(split_ranks, floor) for split, split_ranks in ranks.items()
    }
    sidecar = {
        "folder": numbers["folder"],
        "anchor": {"name": anchor, "rate": numbers["probes"][anchor]["rate"]},
        "partner": {"name": partner, "rate": numbers["probes"][partner]["rate"]},
        "budget": options["budget"],
        "rank_floor": floor,
        "share_at_rank_0": {
            split: [float((split_ranks[i] == 0).mean()) for i in range(2)]
            for split, split_ranks in ranks.items()
        },
        "rules": rules,
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for axis, (rule, reading) in zip(axes, rules.items()):
        for split, color, label in (
            ("validation", CLEAN_COLOR, "clean validation"),
            ("backdoor", TRIGGERED_COLOR, "triggered"),
        ):
            points = subsample(shown[split])
            axis.scatter(
                points[0],
                points[1],
                s=4,
                alpha=0.35,
                color=color,
                label=label,
                rasterized=True,
            )
        low = 0.2 / n_validation
        x_cut, y_cut = reading["cutoffs"]
        axis.axvspan(low, max(x_cut, low), color="0.5", alpha=0.2, lw=0)
        axis.axhspan(
            low,
            max(y_cut, low),
            color="0.5",
            alpha=0.2,
            lw=0,
            label=f"flagged at {options['budget'] * 100:g}% FPR",
        )
        axis.axvline(max(x_cut, low), color="k", lw=1)
        axis.axhline(max(y_cut, low), color="k", lw=1)
        axis.axvline(floor, color="0.4", lw=0.6, ls=":")
        axis.axhline(floor, color="0.4", lw=0.6, ls=":")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(low, 1)
        axis.set_ylim(low, 1)
        rule_name = (
            "plain min"
            if rule == "min"
            else f"weighted {reading['shares'][0]:g}/{reading['shares'][1]:g}"
        )
        axis.set_title(
            f"{rule_name}: TPR {reading['backdoor']['flagged']:.3f}, "
            f"validation flagged {reading['validation']['flagged']:.3f}",
            fontsize=9,
        )
        axis.set_xlabel(f"{anchor} percentile in clean validation (log)")
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=7, markerscale=3)
    axes[0].set_ylabel(f"{partner} percentile in clean validation (log)")
    figure.suptitle(
        f"{numbers['folder']}: where each fusion rule draws its line "
        "(dotted, below every clean image)",
        fontsize=10,
    )
    write_figure(
        figure, sidecar, os.path.join(model_dir, f"scatter_{anchor}-{partner}")
    )


def weighted_pair_shares():
    shares = (config.WEIGHTED_FIRST_SHARE, 1.0 - config.WEIGHTED_FIRST_SHARE)
    return shares


def spread_floor(split_ranks, floor):
    generator = np.random.default_rng(POINT_SEED)
    jitter = 10 ** generator.uniform(
        np.log10(0.25 * floor), np.log10(floor), size=split_ranks.shape
    )  # (2, n_split)
    spread = np.where(split_ranks == 0, jitter, split_ranks)  # (2, n_split)
    return spread


def subsample(points):
    if points.shape[1] <= MAX_POINTS:
        return points
    generator = np.random.default_rng(POINT_SEED)
    keep = generator.choice(points.shape[1], MAX_POINTS, replace=False)
    sampled = points[:, keep]  # (2, MAX_POINTS)
    return sampled


def draw_confidence(model_dir, numbers, options):
    probe = options["probe"]
    psbd_dir = psbd_dir_of(numbers["folder"])
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)
    placement = numbers["probes"][probe]["placement"]
    passes = load_passes(
        psbd_dir, placement, numbers["probes"][probe]["rate"], baselines
    )

    forms = {"absolute": psu_from_cache, "fractional": psu_ratio_from_cache}
    psu = {
        form: {
            split: function(
                passes[split]["baseline_probs"],
                passes[split]["baseline_labels"],
                passes[split]["per_pass_probs"],
            )  # (n_split,)
            for split in passes
        }
        for form, function in forms.items()
    }
    confidence = {
        split: passes[split]["baseline_probs"].max(dim=1).values.float()  # (n_split,)
        for split in passes
    }

    sidecar = {
        "folder": numbers["folder"],
        "probe": probe,
        "rate": numbers["probes"][probe]["rate"],
        "budget": options["budget"],
        "median_confidence": {
            split: float(values.median()) for split, values in confidence.items()
        },
        "forms": {},
    }
    for form, by_split in psu.items():
        report = detection_report(
            by_split["validation"],
            pair_clean_to_backdoor(by_split["clean"], manifest),
            by_split["backdoor"],
            options["budget"],
        )
        flagged = by_split["validation"] < report["threshold"]  # (n_validation,)
        sidecar["forms"][form] = {
            "auroc": report["auroc"],
            "tpr": report["tpr"],
            "realized_fpr": report["fpr"],
            "threshold": report["threshold"],
            "median_confidence_of_flagged_validation": float(
                confidence["validation"][flagged].median()
            )
            if flagged.any()
            else None,
            "spearman_with_confidence_validation": float(
                spearmanr(confidence["validation"], by_split["validation"]).statistic
            ),
        }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, form in zip(axes, forms):
        for split, color, label in (
            ("validation", CLEAN_COLOR, "clean validation"),
            ("backdoor", TRIGGERED_COLOR, "triggered"),
        ):
            points = subsample(
                np.stack([confidence[split].numpy(), psu[form][split].numpy()])
            )
            axis.scatter(
                points[0],
                points[1],
                s=4,
                alpha=0.3,
                color=color,
                label=label,
                rasterized=True,
            )
        reading = sidecar["forms"][form]
        axis.axhline(
            reading["threshold"],
            color="k",
            lw=1,
            ls="--",
            label=f"threshold at {options['budget'] * 100:g}% FPR",
        )
        if form == "absolute":
            axis.plot([0, 1], [0, 1], color="0.3", lw=1, label="PSU = $P_c$")
        axis.set_xlabel("starting confidence $P_c$")
        axis.set_ylabel(f"{form} PSU")
        axis.set_title(
            f"{form}: AUROC {reading['auroc']:.3f}, TPR at "
            f"{options['budget'] * 100:g}% FPR {reading['tpr']:.3f}",
            fontsize=9,
        )
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=7, markerscale=3)
    figure.suptitle(
        f"{numbers['folder']}: {probe} PSU against starting confidence", fontsize=10
    )
    write_figure(figure, sidecar, os.path.join(model_dir, f"confidence_{probe}"))


def draw_shift_ladder(root, set_name, options):
    set_dir = os.path.join(root, set_name)
    all_numbers = read_all_numbers(set_dir)
    attacks = ordered_attacks({numbers["attack"] for numbers in all_numbers})

    # rows[probe][attack][rate] collects 1 tuple per model of (clean shift,
    # triggered shift, clean surviving probability, triggered surviving).
    rows = {probe: {attack: {} for attack in attacks} for probe in options["probes"]}
    adaptive = {
        probe: {attack: [] for attack in attacks} for probe in options["probes"]
    }
    for numbers in all_numbers:
        psbd_dir = psbd_dir_of(numbers["folder"])
        baselines = {
            split: load_baseline(baseline_path(psbd_dir, split))
            for split in ("validation", "backdoor")
        }
        for probe in options["probes"]:
            reading = numbers["probes"][probe]
            if reading["placement"] is None:
                continue
            if reading["rate"] is not None:
                adaptive[probe][numbers["attack"]].append(reading["rate"])
            for rate in complete_rates(psbd_dir, reading["placement"]):
                values = ladder_point(psbd_dir, reading["placement"], rate, baselines)
                rows[probe][numbers["attack"]].setdefault(rate, []).append(values)

    sidecar = {
        "set": set_name,
        "probes": {},
        "fields": [
            "clean_shift",
            "triggered_shift",
            "clean_surviving",
            "triggered_surviving",
        ],
    }
    for probe in options["probes"]:
        sidecar["probes"][probe] = {}
        for attack in attacks:
            ladder = rows[probe][attack]
            sidecar["probes"][probe][attack] = {
                "adaptive_rates": sorted(adaptive[probe][attack]),
                "median_adaptive_rate": float(np.median(adaptive[probe][attack]))
                if adaptive[probe][attack]
                else None,
                "by_rate": {
                    f"{rate:g}": {
                        "n_models": len(values),
                        **dict(
                            zip(
                                sidecar["fields"],
                                np.mean(values, axis=0).round(5).tolist(),
                            )
                        ),
                    }
                    for rate, values in sorted(ladder.items())
                },
            }

    figure, axes = plt.subplots(
        len(options["probes"]),
        len(attacks),
        figsize=(2.6 * len(attacks), 2.6 * len(options["probes"]) + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, probe in enumerate(options["probes"]):
        for column, attack in enumerate(attacks):
            axis = axes[row, column]
            reading = sidecar["probes"][probe][attack]
            rates = [float(rate) for rate in reading["by_rate"]]
            series = {
                field: [point[field] for point in reading["by_rate"].values()]
                for field in sidecar["fields"]
            }
            axis.plot(
                rates,
                series["clean_shift"],
                color=CLEAN_COLOR,
                lw=1.6,
                marker=".",
                label="clean, share changed",
            )
            axis.plot(
                rates,
                series["triggered_shift"],
                color=TRIGGERED_COLOR,
                lw=1.6,
                marker=".",
                label="triggered, share changed",
            )
            axis.plot(
                rates,
                series["clean_surviving"],
                color=CLEAN_COLOR,
                lw=1.2,
                ls="--",
                label="clean, surviving probability",
            )
            axis.plot(
                rates,
                series["triggered_surviving"],
                color=TRIGGERED_COLOR,
                lw=1.2,
                ls="--",
                label="triggered, surviving probability",
            )
            if reading["median_adaptive_rate"] is not None:
                axis.axvline(
                    reading["median_adaptive_rate"],
                    color="k",
                    lw=0.8,
                    ls=":",
                    label="median adaptive rate",
                )
            n_models = max(
                (p["n_models"] for p in reading["by_rate"].values()), default=0
            )
            axis.set_title(f"{probe}, {attack} ({n_models})", fontsize=8)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.3)
            if row == len(options["probes"]) - 1:
                axis.set_xlabel("perturbation rate", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(
        f"{set_name}: prediction shift and surviving probability along "
        "the rate ladder, mean per attack (model count)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    write_figure(figure, sidecar, os.path.join(set_dir, "shift_vs_rate"), tight=False)


def ladder_point(psbd_dir, placement, rate, baselines):
    values = []
    for split in ("validation", "backdoor"):
        _, labels, _ = baselines[split]
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )  # (passes, n_split)
        values.append(shift_ratio(labels, argmax))
    for split in ("validation", "backdoor"):
        per_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )  # (passes, n_split)
        values.append(float(per_pass_probs.float().mean()))
    return values


def draw_flip_targets(model_dir, numbers, options):
    probe = options["probe"]
    psbd_dir = psbd_dir_of(numbers["folder"])
    baseline_probs, baseline_labels, _ = load_baseline(
        baseline_path(psbd_dir, "validation")
    )  # (n_validation, num_classes), (n_validation,)
    reading = numbers["probes"][probe]
    _, argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, reading["placement"], reading["rate"], "validation")
    )  # (passes, n_validation)
    num_classes = baseline_probs.shape[1]
    counts = np.array(shift_target_histogram(baseline_labels, argmax, num_classes))
    predicted = np.bincount(baseline_labels.long().numpy(), minlength=num_classes)

    metadata = checkpoint_metadata(numbers["folder"])
    target = metadata.get("target_label")
    total = int(counts.sum())
    sidecar = {
        "folder": numbers["folder"],
        "probe": probe,
        "rate": reading["rate"],
        "num_classes": num_classes,
        "target_label": target,
        "source_classes": metadata.get("source_classes"),
        "changed_predictions": total,
        "changed_share": total / argmax.numel(),
        "target_share_of_changed": float(counts[target] / total)
        if target is not None and total
        else None,
        "top_class": int(counts.argmax()),
        "top_share_of_changed": float(counts.max() / total) if total else None,
        "uniform_share": 1.0 / (num_classes - 1),
        "counts": counts.tolist(),
        "validation_predicted_counts": predicted.tolist(),
    }

    figure, axis = plt.subplots(figsize=(8, 4))
    classes = np.arange(num_classes)
    axis.bar(classes, counts / max(total, 1), color=CLEAN_COLOR, width=0.85)
    axis.axhline(
        sidecar["uniform_share"],
        color="0.3",
        lw=0.8,
        ls="--",
        label="uniform over the other classes",
    )
    # The target bar can be empty, so the target is marked by a line that shows
    # either way.
    if target is not None:
        axis.axvline(
            target,
            color=TRIGGERED_COLOR,
            lw=1.5,
            ls=":",
            label=f"target class {target}, "
            f"{sidecar['target_share_of_changed']:.3f} of changed",
        )
    axis.annotate(
        f"class {sidecar['top_class']}: {sidecar['top_share_of_changed']:.3f}",
        (sidecar["top_class"], counts.max() / max(total, 1)),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=8,
    )
    axis.set_xlabel("class the changed prediction moves to")
    axis.set_ylabel("share of changed clean predictions")
    axis.set_ylim(0, 1.1)
    axis.set_title(
        f"{numbers['folder']}, {probe} at rate {reading['rate']:g}: "
        f"{total} changed predictions",
        fontsize=9,
    )
    axis.grid(alpha=0.3, axis="y")
    axis.legend(loc="best", fontsize=8)
    write_figure(figure, sidecar, os.path.join(model_dir, f"flip_targets_{probe}"))


def draw_evaders(out_dir, options):
    with open(options["record"]) as handle:
        record = json.load(handle)

    points = []
    for model in record["models"]:
        reading = model["detection"].get("before_attention_norm_token_mask") or {}
        evader = reading.get("evader")
        if evader is None:
            continue
        # clean_accuracy_drop is accuracy minus the benign reference, so a loss
        # is negative there and positive here.
        points.append(
            {
                "folder": model["folder"],
                "architecture": model["architecture"],
                "family": model["family"],
                "probed": model["probes"],
                "tm_probed": "before_attention_norm_token_mask" in model["probes"],
                "clean_accuracy_loss_points": -100
                * model["evader"]["clean_accuracy_drop"],
                "asr": model["evader"]["asr"],
                "clears_asr_bar": model["evader"]["asr_class"] == "clears",
                "tm_auroc": evader["auroc"],
                "tm_rate_rule": evader["rate_rule"],
            }
        )

    groups = {
        ("vit", "l1"): (
            "o",
            OKABE_ITO[0],
            "ViT, single-probe attacker (probes PSBD-TM)",
        ),
        ("swin", "l1"): (
            "s",
            OKABE_ITO[2],
            "Swin-S, single-probe attacker (probes attention-input dropout)",
        ),
        ("vit", "union"): ("^", OKABE_ITO[1], "ViT, all-probe attacker"),
    }
    bar = options["success_bar_points"]
    summary = {}
    figure, axis = plt.subplots(figsize=(8, 7))
    for (architecture, family), (marker, color, label) in groups.items():
        members = [
            p
            for p in points
            if p["architecture"] == architecture and p["family"] == family
        ]
        evading = [
            p
            for p in members
            if p["clears_asr_bar"]
            and p["clean_accuracy_loss_points"] <= bar
            and p["tm_auroc"] < 0.5
        ]
        summary[f"{architecture}_{family}"] = {
            "n": len(members),
            "n_clears_asr_bar": sum(p["clears_asr_bar"] for p in members),
            "n_within_bar": sum(
                p["clean_accuracy_loss_points"] <= bar for p in members
            ),
            "n_successful_and_tm_below_0.5": len(evading),
            "mean_tm_auroc": float(np.mean([p["tm_auroc"] for p in members]))
            if members
            else None,
        }
        for clears in (True, False):
            subset = [p for p in members if p["clears_asr_bar"] == clears]
            axis.scatter(
                [p["clean_accuracy_loss_points"] for p in subset],
                [p["tm_auroc"] for p in subset],
                marker=marker,
                s=36,
                edgecolors=color,
                facecolors=color if clears else "none",
                label=f"{label}, ASR {'clears' if clears else 'below'} the bar",
            )
    axis.axvline(bar, color="k", lw=1, ls="--", label=f"{bar}-point clean-accuracy bar")
    axis.axhline(0.5, color="0.4", lw=1, ls=":", label="AUROC 0.5")
    axis.set_xlabel("clean-accuracy loss against the benign reference (points)")
    axis.set_ylabel("PSBD-TM AUROC")
    axis.set_title("Adaptive attackers: what evading PSBD-TM costs", fontsize=10)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7)
    sidecar = {
        "record": options["record"],
        "success_bar_points": bar,
        "summary": summary,
        "points": points,
    }
    write_figure(figure, sidecar, os.path.join(out_dir, "evader_frontier"))


def draw_negative_psu(root, options):
    probe, budget = options["probe"], options["budget"]
    key = f"q{budget:.2f}"
    points = []
    for set_name in options["sets"]:
        for numbers in read_all_numbers(os.path.join(root, set_name)):
            reading = numbers["probes"][probe]
            if reading["rate"] is None:
                continue
            psbd_dir = psbd_dir_of(numbers["folder"])
            probs, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
            per_pass_probs, _ = load_dropout_pass_probs(
                dropout_pass_path(
                    psbd_dir, reading["placement"], reading["rate"], "validation"
                )
            )  # (passes, n_validation)
            psu = psu_ratio_from_cache(probs, labels, per_pass_probs)  # (n_validation,)
            evaluation = numbers["combinations"][probe]["psu"]
            points.append(
                {
                    "set": set_name,
                    "folder": numbers["folder"],
                    "attack": numbers["attack"],
                    "negative_share": float((psu < 0).float().mean()),
                    "tpr": evaluation["at_fpr"][key]["tpr"],
                    "threshold": evaluation["at_fpr"][key]["threshold"],
                }
            )

    summary = {}
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for set_name in options["sets"]:
        members = [p for p in points if p["set"] == set_name]
        shares = [p["negative_share"] for p in members]
        tprs = [p["tpr"] for p in members]
        summary[set_name] = {
            "n": len(members),
            "spearman": float(spearmanr(shares, tprs).statistic)
            if len(members) > 2
            else None,
            "n_threshold_below_0": sum(p["threshold"] < 0 for p in members),
            "n_negative_share_above_budget": sum(s > budget for s in shares),
        }
        axis.scatter(
            shares,
            tprs,
            s=28,
            color=SET_COLORS.get(set_name, "0.3"),
            alpha=0.8,
            label=f"{set_name} ({len(members)})",
        )
    axis.axvline(
        budget, color="k", lw=1, ls="--", label=f"{budget * 100:g}% of clean validation"
    )
    axis.set_xscale("symlog", linthresh=1e-3)
    axis.set_xlim(0, max(0.05, 1.2 * max(p["negative_share"] for p in points)))
    axis.set_ylim(-0.02, 1.02)
    # BackdoorBench has few models and the ones the report discusses are named.
    for point in points:
        if point["set"] == "backdoorbench":
            axis.annotate(
                point["folder"][3:],
                (point["negative_share"], point["tpr"]),
                textcoords="offset points",
                xytext=(4, -3),
                fontsize=6,
            )
    axis.set_xlabel(
        f"share of clean validation with {probe} fractional PSU below 0 (symlog)"
    )
    axis.set_ylabel(f"TPR at {budget * 100:g}% FPR")
    axis.set_title(
        f"Share of clean images whose confidence rises under {probe}, against TPR",
        fontsize=10,
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    sidecar = {"probe": probe, "budget": budget, "summary": summary, "points": points}
    write_figure(figure, sidecar, os.path.join(root, ACROSS_SETS, "negative_psu"))


def draw_threshold_transfer(root, options):
    sidecar = {"curves": options["curves"], "sets": {}}
    figure, axes = plt.subplots(
        1, len(options["sets"]), figsize=(11, 5), sharey=True, squeeze=False
    )
    for axis, set_name in zip(axes[0], options["sets"]):
        all_numbers = read_all_numbers(os.path.join(root, set_name))
        sidecar["sets"][set_name] = {}
        for index, (key, rule) in enumerate(options["curves"]):
            nominal, realized = [], []
            for numbers in all_numbers:
                evaluation = numbers["combinations"].get(key, {}).get(rule)
                if evaluation is None:
                    continue
                for budget in config.BUDGETS:
                    nominal.append(budget)
                    realized.append(
                        evaluation["at_fpr"][f"q{budget:.2f}"]["realized_fpr"]
                    )
            nominal, realized = np.array(nominal), np.array(realized)
            sidecar["sets"][set_name][f"{key} {rule}"] = {
                f"{budget:g}": {
                    "n": int((nominal == budget).sum()),
                    "mean": float(realized[nominal == budget].mean()),
                    "median": float(np.median(realized[nominal == budget])),
                    "max": float(realized[nominal == budget].max()),
                    "share_above_1.5x": float(
                        (realized[nominal == budget] > 1.5 * budget).mean()
                    ),
                }
                for budget in config.BUDGETS
            }
            # A small sideways offset keeps the 2 curves' points apart.
            offset = 1.0 + 0.08 * (index - 0.5)
            label = key if rule == "psu" else f"{key}, {rule}"
            axis.scatter(
                nominal * offset,
                realized,
                s=14,
                alpha=0.6,
                color=OKABE_ITO[index],
                label=label,
            )
        axis.plot(
            [5e-3, 0.3], [5e-3, 0.3], color="0.3", lw=1, label="realized = nominal"
        )
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=3e-3, linscale=0.3)
        axis.set_xlim(6e-3, 0.3)
        axis.set_ylim(0, 0.5)
        axis.set_xlabel("nominal FPR (clean-validation quantile)")
        axis.set_title(f"{set_name}: 1 point per model and budget", fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    axes[0, 0].set_ylabel("FPR realized on the paired clean test split (symlog)")
    write_figure(figure, sidecar, os.path.join(root, ACROSS_SETS, "threshold_transfer"))


def probe_psu(psbd_dir, numbers, probe, baselines):
    reading = numbers["probes"][probe]
    assert reading["rate"] is not None, (numbers["folder"], probe, reading["skipped"])
    passes = load_passes(psbd_dir, reading["placement"], reading["rate"], baselines)
    psu = {
        split: psu_ratio_from_cache(
            passes[split]["baseline_probs"],
            passes[split]["baseline_labels"],
            passes[split]["per_pass_probs"],
        )  # (n_split,)
        for split in passes
    }
    return psu


def psbd_dir_of(folder):
    path = os.path.join(RESULTS_DIR, folder, "psbd")
    return path


def checkpoint_metadata(folder):
    path = os.path.join(CHECKPOINTS_DIR, folder, "args.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        metadata = json.load(handle)
    return metadata


def read_numbers(model_dir):
    with open(os.path.join(model_dir, "numbers.json")) as handle:
        numbers = json.load(handle)
    return numbers


def read_all_numbers(set_dir):
    all_numbers = []
    for name in sorted(os.listdir(set_dir)):
        path = os.path.join(set_dir, name, "numbers.json")
        if os.path.exists(path):
            all_numbers.append(read_numbers(os.path.join(set_dir, name)))
    return all_numbers


def write_figure(figure, sidecar, stem, tight=True):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(f"{stem}.png", dpi=config.DPI)
    plt.close(figure)
    with open(f"{stem}.json", "w") as handle:
        json.dump(sidecar, handle, indent=1)

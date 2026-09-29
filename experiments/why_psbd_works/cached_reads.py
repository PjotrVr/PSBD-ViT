"""CPU reads of the cached sweeps for the why_psbd_works hypotheses, no GPU.

2 tests need only the per-pass tensors cli.sweep already wrote under
results/<folder>/psbd/.

    P10  Is the target class simply easy? If it were, any clean image of the target
         class would keep its prediction under the probe and score low, triggered
         or not. Fractional PSU of clean images whose true class is the target,
         against clean images of every other class, at the adaptive rate of
         PSBD-TM and PSBD-RD, on every clearing ViT model and on the benign
         reference of the same dataset read at the same class.
    P6   Where do shifted clean predictions go? The share of the clean split's
         shifted (pass, image) predictions that land on the target, against the
         share the most popular class takes and against 1/classes, read off the
         histogram cli.analyze stored per rate.

The flag for P10 is the pipeline's own: an image is flagged when its fractional
PSU falls below the 0.25 quantile of the clean held-out split's.

The same script gathers, per panel model, the cached numbers behind the memo's
refutations, so the notebook can redraw each one from its source file.

    P1   the confidence-only detector (detectors/confidence_metrics.json) against
         PSBD-TM and PSBD-RD, fractional AUROC at the adaptive rate
    P5   the benign references' own PSBD-TM and PSBD-RD AUROC
    P9   final rank ratio, CKA and separation against the best deployable AUROC,
         from experiments/latent_geometry_predicts_detection
    P3   ASR after removing the backdoor direction, the top 20 TAC coordinates or
         a random direction, from experiments/backdoor_neurons
    L14  the target share and clean shift ratio of gain_scale at mlp_norm_out at
         the largest amplification of its ladder, IBD-PSC's own operator
    L13  input_pixels_scale_up by attack and poison rate, SCALE-UP's theorem
    L18  after_embedding_token_mask against PSBD-TM, Doan et al.'s patch drop

2 reads follow docs/why-psbd-works-theory.md.

    P1   A*(P_c), the AUROC of the cross-fitted likelihood ratio of the
         unperturbed confidence, the best any function of confidence can do and
         the bar every account claiming to carry the signal has to clear
    clean fragility  the clean keep curve of PSBD-TM's ladder per dataset against
         a per-image critical rate (logistic plus a default-class floor) and
         against the homogeneous AND power law, the flip count histogram at the
         adaptive rate against a binomial, the intra-image flip correlation, the
         share of clean images that never flip and their share of the false
         positives

    PYTHONPATH=. python experiments/why_psbd_works/cached_reads.py
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import collections  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import binom  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import (  # noqa: E402
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    pair_clean_to_backdoor,
    threshold_at_quantile,
)
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.why_psbd_works.measure import SLUG  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    clearing_cells,
    excluded_folders,
    load_coverage,
    load_psbd_metrics,
)

PLACEMENTS = {
    "token_mask": RECOMMENDED_PLACEMENT,
    "residual_dropout": PUBLISHED_PLACEMENT,
}
HEADLINE_QUANTILE = 0.25
# Every panel model targets class 0 and the benign references were swept with a
# BadNets probe at class 0, so class 0 is the class read on both.
PANEL_TARGET = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    excluded = excluded_folders(args.results_dir)
    cells = [
        cell
        for cell in clearing_cells(load_coverage(args.results_dir))
        if cell["folder_name"] not in excluded
        and cell["folder_name"].startswith("vit_")
        # The panel is the models successful at the 2 point clean-accuracy bar
        # (54 on ViT). The 5 point bar adds a few, which every row marks, so a
        # reader can take either population.
        and cell.get("successful_5pt")
        # The 2 clearing cells still queued on the GPU have no cache yet.
        and os.path.exists(
            os.path.join(
                args.results_dir, cell["folder_name"], "psbd", "split_manifest.json"
            )
        )
        and load_psbd_metrics(args.results_dir, cell["folder_name"]) is not None
    ]
    datasets = sorted({cell["dataset"] for cell in cells})

    rows = []
    for cell in cells:
        rows.append(read_model(args.results_dir, cell["folder_name"], cell))
    benign_rows = [
        read_model(
            args.results_dir,
            f"vit_{dataset}_benign",
            {"attack": "benign", "dataset": dataset, "target_label": PANEL_TARGET},
        )
        for dataset in datasets
    ]

    payload = {
        "models": rows,
        "benign": benign_rows,
        "refutations": {
            "per_model": [refutation_row(args.results_dir, cell) for cell in cells],
            "benign": [
                refutation_row(
                    args.results_dir,
                    {
                        "folder_name": f"vit_{d}_benign",
                        "attack": "benign",
                        "dataset": d,
                        "poison_rate": None,
                    },
                )
                for d in datasets
            ],
            "latent_geometry": latent_geometry_rows(args.results_dir, cells),
            "direction_ablation": direction_ablation_rows(args.results_dir),
            "confidence_ceiling": [
                confidence_ceiling_row(args.results_dir, cell) for cell in cells
            ],
        },
        "clean_fragility": clean_fragility(
            args.results_dir, [cell for cell in cells if cell.get("successful_2pt")]
        ),
    }
    out_path = experiment_result_path(SLUG, "cached_reads.json", args.output_root)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print_summary(payload)


def read_model(results_dir, folder, cell):
    report = load_psbd_metrics(results_dir, folder)
    target = cell.get("target_label")
    target = PANEL_TARGET if target is None else target
    row = {
        "folder": folder,
        "attack": cell["attack"],
        "dataset": cell["dataset"],
        "target_label": target,
        "successful_2pt": bool(cell.get("successful_2pt")),
    }
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    for name, placement in PLACEMENTS.items():
        block = report["placements"].get(placement) if report else None
        rate = psbd_rate(block, "adaptive") if block else None
        if rate is None:
            row[name] = None
            continue
        row[name] = {
            "rate": rate,
            "target_class_easy": target_class_reading(
                psbd_dir, placement, rate, target
            ),
            "shift_destination": shift_destination(block, rate, target),
        }
    return row


# Clean images of the analysis split and of the held-out split together, split by
# whether their true class is the target. The held-out split also sets the
# threshold, so its own flagged share is 0.25 by construction over all classes.
def target_class_reading(psbd_dir, placement, rate, target):
    scores, classes = {}, {}
    for split in ("validation", "clean"):
        probs, labels, loader_labels = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        scores[split] = psu_ratio_from_cache(probs, labels, per_pass)  # (n,)
        classes[split] = loader_labels  # (n,)
    if classes["clean"].numel() == 0:
        return None

    threshold = threshold_at_quantile(scores["validation"], HEADLINE_QUANTILE)
    pooled_scores = torch.cat([scores["validation"], scores["clean"]])  # (n_all,)
    pooled_classes = torch.cat([classes["validation"], classes["clean"]])
    is_target = pooled_classes == target  # (n_all,)
    flagged = pooled_scores < threshold  # (n_all,)

    reading = {
        "threshold": float(threshold),
        "target_images": int(is_target.sum()),
        "target_flagged_share": float(flagged[is_target].float().mean()),
        "other_flagged_share": float(flagged[~is_target].float().mean()),
        "target_median_psu_ratio": float(pooled_scores[is_target].median()),
        "other_median_psu_ratio": float(pooled_scores[~is_target].median()),
    }
    return reading


def shift_destination(block, rate, target):
    row = next(r for r in block["rates"] if r["rate"] == rate)
    histogram = row["shift_target_histogram"]["clean"]
    total = sum(histogram)
    if not total:
        return None
    destination = {
        "target_share": histogram[target] / total,
        "largest_class_share": max(histogram) / total,
        "largest_class": histogram.index(max(histogram)),
        "uniform_share": 1.0 / len(histogram),
        "clean_shift_ratio": row["shift_ratio"]["clean"],
    }
    return destination


def refutation_row(results_dir, cell):
    folder = cell["folder_name"]
    report = load_psbd_metrics(results_dir, folder) or {"placements": {}}
    placements = report["placements"]
    row = {
        "folder": folder,
        "attack": cell["attack"],
        "dataset": cell["dataset"],
        "poison_rate": cell.get("poison_rate"),
        "successful_2pt": bool(cell.get("successful_2pt")),
        "psbd_tm": fractional_auroc(placements.get(RECOMMENDED_PLACEMENT)),
        "psbd_rd": fractional_auroc(placements.get(PUBLISHED_PLACEMENT)),
        "after_embedding_token_mask": fractional_auroc(
            placements.get("after_embedding_token_mask")
        ),
        "scale_up": fractional_auroc(placements.get("input_pixels_scale_up")),
        "confidence_only": detector_auroc(results_dir, folder, "confidence"),
        "gain_scale_top": gain_scale_top(placements.get("mlp_norm_out_gain_scale")),
    }
    return row


def fractional_auroc(block):
    if not block:
        return None
    rate = psbd_rate(block, "adaptive")
    if rate is None:
        return None
    row = next(r for r in block["rates"] if r["rate"] == rate)
    auroc = row["detection_psu_ratio"][f"q{HEADLINE_QUANTILE:.2f}"]["auroc"]
    return auroc


def detector_auroc(results_dir, folder, name):
    path = os.path.join(results_dir, folder, "detectors", f"{name}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        record = json.load(handle)
    detection = record.get("detection") or {}
    block = detection.get(f"q{HEADLINE_QUANTILE:.2f}")
    auroc = block["auroc"] if block else None
    return auroc


# IBD-PSC's theorem predicts that past some amplification every clean prediction
# goes to the target. The largest factor on the ladder is where it should show.
def gain_scale_top(block):
    if not block or not block.get("rates"):
        return None
    row = max(block["rates"], key=lambda r: r["rate"])
    histogram = row["shift_target_histogram"]["clean"]
    total = sum(histogram)
    reading = {
        "factor": 1.0 + row["rate"],
        "clean_shift_ratio": row["shift_ratio"]["clean"],
        "target_share": histogram[PANEL_TARGET] / total if total else None,
        "largest_class_share": max(histogram) / total if total else None,
    }
    return reading


def latent_geometry_rows(results_dir, cells):
    path = os.path.join(
        results_dir,
        "_experiments",
        "latent_geometry_predicts_detection",
        "geometry_vs_detection.csv",
    )
    if not os.path.exists(path):
        return None
    wanted = {cell["folder_name"] for cell in cells}
    with open(path) as handle:
        header = handle.readline().strip().split(",")
        rows = [dict(zip(header, line.strip().split(","))) for line in handle]
    kept = []
    for row in rows:
        if row["folder"] not in wanted or row.get("error"):
            continue
        kept.append(
            {
                "folder": row["folder"],
                "attack": row["attack"],
                "best_deployable_auroc": float(row["best_deployable_auroc"]),
                "final_rank_ratio": float(row["final_rank_ratio"]),
                "final_cka": float(row["final_cka"]),
                "final_separation_auroc": float(row["final_separation_auroc"]),
            }
        )
    return kept


def direction_ablation_rows(results_dir):
    path = os.path.join(
        results_dir, "_experiments", "backdoor_neurons", "backdoor_neuron_ablation.json"
    )
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        records = json.load(handle)
    rows = [
        {
            "folder": record["folder_name"],
            "attack": record["attack"],
            "baseline_asr": record["baseline"]["asr"],
            "direction_removed_asr": record["ablated"]["direction"]["asr"],
            "random_direction_asr": record["ablated"]["random_dir_0"]["asr"],
            "top_20_coordinates_asr": record["ablated"]["top_20"]["asr"],
            "random_20_coordinates_asr": record["ablated"]["random_20"]["asr"],
        }
        for record in records
        if "sam_rho" not in record["folder_name"]
    ]
    return rows


# The P1 bound of docs/why-psbd-works-theory.md, rebuilt as
# experiments/theory_checks/lrconf2.py builds it: the likelihood ratio of
# triggered against clean on 40 quantile bins of -log(1 - P_c), fitted on 1 fold
# of pairs and applied to the other, both ways. Folds split by pair so a clean
# image and its triggered twin never sit on opposite sides. The fold draw is
# seeded per model here, where lrconf2.py draws them from 1 generator in panel
# order, so single models differ from its numbers by fold noise.
def confidence_ceiling_row(results_dir, cell):
    folder = cell["folder_name"]
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    confidence = {}
    for split in ("clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        confidence[split] = probs.gather(1, labels.view(-1, 1).long())[:, 0].double()
    clean = pair_clean_to_backdoor(confidence["clean"], manifest).numpy()
    triggered = confidence["backdoor"].numpy()
    pairs = len(triggered)
    generator = np.random.default_rng(0)
    fold = generator.integers(0, 2, pairs)
    values = np.r_[clean, triggered]
    labels = np.r_[np.zeros(pairs), np.ones(pairs)]
    folds = np.r_[fold, fold]
    transformed = -np.log1p(-np.minimum(values, 1 - 1e-12))
    ratio = np.zeros_like(transformed)
    for held_out in (0, 1):
        fit, apply = folds != held_out, folds == held_out
        edges = np.unique(np.quantile(transformed[fit], np.linspace(0, 1, 41)))
        bins = len(edges) - 1
        fit_bins = np.clip(
            np.searchsorted(edges, transformed[fit], side="right") - 1, 0, bins - 1
        )
        apply_bins = np.clip(
            np.searchsorted(edges, transformed[apply], side="right") - 1, 0, bins - 1
        )
        likelihood = (np.bincount(fit_bins[labels[fit] == 1], minlength=bins) + 0.5) / (
            np.bincount(fit_bins[labels[fit] == 0], minlength=bins) + 0.5
        )
        ratio[apply] = likelihood[apply_bins]
    row = {
        "folder": folder,
        "attack": cell["attack"],
        "dataset": cell["dataset"],
        "successful_2pt": bool(cell.get("successful_2pt")),
        "a_star": float(roc_auc_score(labels, ratio)),
        "raw_confidence": float(roc_auc_score(labels, transformed)),
    }
    return row


def clean_fragility(results_dir, cells):
    by_dataset = collections.defaultdict(list)
    per_model = []
    for cell in cells:
        folder = cell["folder_name"]
        report = load_psbd_metrics(results_dir, folder)
        block = report["placements"][RECOMMENDED_PLACEMENT]
        keep = {r["rate"]: 1.0 - r["shift_ratio"]["validation"] for r in block["rates"]}
        by_dataset[cell["dataset"]].append(keep)
        per_model.append(flip_counts(results_dir, folder, block))

    curves = {}
    for dataset, ladders in by_dataset.items():
        rates = sorted(set.intersection(*[set(k) for k in ladders]))
        mean_keep = np.array([np.mean([k[r] for k in ladders]) for r in rates])
        curves[dataset] = fit_keep_curve(np.array(rates), mean_keep)
        curves[dataset]["models"] = len(ladders)
    summary = {
        "curves": curves,
        "per_model": per_model,
        "median_intra_image_correlation": float(
            np.median([m["intra_image_correlation"] for m in per_model])
        ),
        "mean_never_flip_share": float(
            np.mean([m["validation_hist"][0] for m in per_model])
        ),
        "mean_false_positive_never_flip_share": float(
            np.mean([m["false_positive_never_flip_share"] for m in per_model])
        ),
    }
    return summary


# keep(p) = floor + (1 - floor) G(p), with G the survival function of a logistic
# per-image critical rate, against the homogeneous AND law (1 - p)^n on the same
# floor-removed curve.
def fit_keep_curve(rates, keep):
    floor = float(keep[np.argmax(rates)])
    scaled = (keep - floor) / (1.0 - floor)

    def logistic(p, median, scale):
        return 1.0 / (1.0 + np.exp((p - median) / scale))

    def and_law(p, units):
        return (1.0 - p) ** units

    (median, scale), _ = curve_fit(logistic, rates, scaled, p0=(0.3, 0.1))
    (units,), _ = curve_fit(and_law, rates, scaled, p0=(3.0,))
    curve = {
        "rates": rates.tolist(),
        "keep": keep.tolist(),
        "floor": floor,
        "logistic_median": float(median),
        "logistic_scale": float(scale),
        "critical_rate_sd": float(scale * np.pi / np.sqrt(3.0)),
        "logistic_rmse": float(
            np.sqrt(np.mean((logistic(rates, median, scale) - scaled) ** 2))
        ),
        "and_units": float(units),
        "and_rmse": float(np.sqrt(np.mean((and_law(rates, units) - scaled) ** 2))),
    }
    return curve


# At PSBD-TM's adaptive rate with the sweep's k = 3: how many of the 3 passes
# change each held-out image's answer, against a binomial with the same mean, and
# the intra-image correlation that makes the difference,
# Var(N) = k q (1 - q) (1 + (k - 1) rho).
def flip_counts(results_dir, folder, block):
    rate = psbd_rate(block, "adaptive")
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    counts, scores = {}, {}
    for split in ("validation", "clean"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, RECOMMENDED_PLACEMENT, rate, split)
        )
        counts[split] = (argmax.long() != labels.view(1, -1).long()).sum(dim=0)  # (n,)
        scores[split] = psu_ratio_from_cache(probs, labels, per_pass)  # (n,)
    passes = int(argmax.shape[0])
    validation = counts["validation"].double()
    q = float(validation.mean()) / passes
    variance = float(validation.var(unbiased=False))
    rho = (variance / (passes * q * (1 - q)) - 1.0) / (passes - 1)
    histogram = torch.bincount(counts["validation"], minlength=passes + 1).double()
    threshold = threshold_at_quantile(scores["validation"], HEADLINE_QUANTILE)
    manifest = read_split_manifest(psbd_dir)
    paired_scores = pair_clean_to_backdoor(scores["clean"], manifest)
    paired_counts = pair_clean_to_backdoor(counts["clean"], manifest)
    false_positive = paired_scores < threshold
    row = {
        "folder": folder,
        "rate": rate,
        "passes": passes,
        "validation_hist": (histogram / histogram.sum()).tolist(),
        "binomial_hist": [float(binom.pmf(n, passes, q)) for n in range(passes + 1)],
        "intra_image_correlation": rho,
        "false_positive_never_flip_share": float(
            (paired_counts[false_positive] == 0).double().mean()
        ),
        "never_flip_flagged_share": float(
            false_positive[paired_counts == 0].double().mean()
        ),
    }
    return row


def print_summary(payload):
    for group in ("models", "benign"):
        for name in PLACEMENTS:
            readings = [
                r[name]["target_class_easy"]
                for r in payload[group]
                if r[name] and r[name]["target_class_easy"]
            ]
            target = sum(x["target_flagged_share"] for x in readings) / len(readings)
            other = sum(x["other_flagged_share"] for x in readings) / len(readings)
            print(
                f"P10 {group} {name}: target-class clean flagged {target:.3f}, "
                f"other classes {other:.3f}, over {len(readings)} models"
            )


if __name__ == "__main__":
    main()

"""Threshold gallery: the clean images that set PSBD-TM's 1% FPR threshold and the triggered images that escape it.

CPU only. PSBD-TM's fractional PSU is read from the stage-1 cache at the adaptive
rate (the labeled nearest rate where no cached rate reaches the target). The 20
clean validation images with the lowest PSU are the ones whose scores fix the 1%
FPR threshold, since a low score is read as poisoned. The 20 triggered images
with the highest PSU are the ones that escape any threshold first. The images are
rebuilt through cli.sweep's own loader, and the model is run on them on the CPU
to hold each image to its cached unperturbed prediction.

    .venv/bin/python -m experiments.internal_maps.gallery
"""

import argparse
import collections
import os
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from cli.sweep import load_model_and_loaders  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from defenses.decision import RECOMMENDED_PLACEMENT  # noqa: E402
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.cache_readouts.shared import (  # noqa: E402
    choose_rate,
    load_baselines,
    load_passes,
    validation_shift_by_rate,
)
from experiments.internal_maps import config  # noqa: E402
from experiments.internal_maps.measure import write_json  # noqa: E402

SLUG = "internal_maps"
RESULTS_DIR = "results"
CPU_THREADS = 4
CONFIDENT = 0.9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="", help="comma separated, default all")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(CPU_THREADS)
    output_root = experiment_results_dir(SLUG, RESULTS_DIR)
    selected = [name for name in args.models.split(",") if name] or list(
        config.GALLERY_MODELS
    )
    summary = {}
    for folder in selected:
        record, images = gallery_record(folder)
        model_dir = os.path.join(output_root, record["architecture"], folder)
        write_json(os.path.join(model_dir, "gallery_numbers.json"), record)
        draw_gallery(
            os.path.join(model_dir, "gallery_threshold_clean.png"),
            record,
            "lowest_clean",
            images["lowest_clean"],
        )
        draw_gallery(
            os.path.join(model_dir, "gallery_escaping_triggered.png"),
            record,
            "highest_triggered",
            images["highest_triggered"],
        )
        summary[folder] = {
            key: record[key]["statistics"]
            for key in ("lowest_clean", "highest_triggered")
        }
        summary[folder]["validation_base_rates"] = record["validation_base_rates"]
        for key in ("threshold", "tpr_at_budget", "tpr_at_budget_hits", "hit_share"):
            summary[folder][key] = record[key]
        summary[folder]["non_hit_share_of_highest_overall"] = record[
            "non_hit_share_of_highest_overall"
        ]
        print(f"[ok] {folder}", flush=True)
    write_json(os.path.join(output_root, "gallery_summary.json"), summary)


def gallery_record(folder):
    psu, baselines, rate, rule, shift = read_tm_psu(folder)
    loader_args = types.SimpleNamespace(
        checkpoints_dir="checkpoints",
        raw_data_dir="raw_data",
        batch_size=64,
        num_workers=0,
        max_samples=None,
        probe_attack=None,
        probe_target_label=None,
    )
    model, architecture, loaders, manifest, metadata = load_model_and_loaders(
        loader_args, folder, torch.device("cpu")
    )
    target = int(metadata["target_label"])
    validation_set = loaders["validation"].dataset
    clean_set = loaders["clean"].dataset
    backdoor_set = loaders["backdoor"].dataset
    clean_row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}

    _, validation_predictions, _ = baselines["validation"]
    validation_probabilities = baselines["validation"][0]  # (n, classes)
    validation_labels = torch.tensor(
        [int(validation_set[row][1]) for row in range(len(validation_set))]
    )  # (n,)
    validation_confidence = validation_probabilities.max(dim=1).values  # (n,)
    threshold = float(torch.quantile(psu["validation"], config.GALLERY_BUDGET))

    lowest_rows = torch.argsort(psu["validation"])[: config.GALLERY_COUNT]
    # A triggered image the unperturbed model does not send to the target carries
    # no working backdoor, and a high PSU there is the right answer. The escapes
    # are drawn among the hits only, and the share of non-hits among the highest
    # PSU of the whole split is recorded beside them.
    _, backdoor_hits_predictions, _ = baselines["backdoor"]
    hits = backdoor_hits_predictions == target  # (n_backdoor,)
    overall_highest = torch.argsort(psu["backdoor"], descending=True)[
        : config.GALLERY_COUNT
    ]
    non_hit_share_overall = float((~hits[overall_highest]).float().mean())
    hit_rows = hits.nonzero(as_tuple=True)[0]  # (hits,)
    highest_rows = hit_rows[
        torch.argsort(psu["backdoor"][hit_rows], descending=True)[
            : config.GALLERY_COUNT
        ]
    ]

    lowest = []
    for row in lowest_rows.tolist():
        image, label = validation_set[row]
        lowest.append(
            {
                "row": row,
                "test_index": int(manifest["heldout_indices"][row]),
                "true_class": int(label),
                "predicted_class": int(validation_predictions[row]),
                "confidence": float(validation_confidence[row]),
                "psu": float(psu["validation"][row]),
                "is_target_class": int(label) == target,
                "image": image,
            }
        )

    backdoor_probabilities, backdoor_predictions, _ = baselines["backdoor"]
    highest = []
    for row in highest_rows.tolist():
        image, _ = backdoor_set[row]
        original = manifest["analysis_backdoor_indices"][row]
        _, true_class = clean_set[clean_row_of[original]]
        predicted = int(backdoor_predictions[row])
        highest.append(
            {
                "row": row,
                "test_index": int(original),
                "true_class": int(true_class),
                "predicted_class": predicted,
                "confidence": float(backdoor_probabilities[row, predicted]),
                "psu": float(psu["backdoor"][row]),
                "is_target_class": predicted == target,
                "image": image,
            }
        )

    check_against_cache(
        model,
        lowest + highest,
        {"validation": validation_set, "backdoor": backdoor_set},
        baselines,
    )
    mean, std = normalization(manifest, metadata["dataset"])
    images = {
        "lowest_clean": [to_pixels(entry.pop("image"), mean, std) for entry in lowest],
        "highest_triggered": [
            to_pixels(entry.pop("image"), mean, std) for entry in highest
        ],
    }

    record = {
        "folder": folder,
        "architecture": architecture,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "target_label": target,
        "placement": RECOMMENDED_PLACEMENT,
        "rate": rate,
        "rate_rule": rule,
        "validation_shift": shift,
        "budget": config.GALLERY_BUDGET,
        "threshold": threshold,
        "validation_size": int(len(psu["validation"])),
        "triggered_size": int(len(psu["backdoor"])),
        "tpr_at_budget": float((psu["backdoor"] <= threshold).float().mean()),
        "tpr_at_budget_hits": float(
            (psu["backdoor"][hits] <= threshold).float().mean()
        ),
        "hit_share": float(hits.float().mean()),
        "non_hit_share_of_highest_overall": non_hit_share_overall,
        "validation_base_rates": base_rates(
            validation_labels,
            validation_predictions,
            validation_confidence,
            psu["validation"],
            target,
        ),
        "lowest_clean": {
            "entries": lowest,
            "statistics": statistics(lowest, target, triggered=False),
        },
        "highest_triggered": {
            "entries": highest,
            "statistics": statistics(highest, target, triggered=True),
        },
    }
    return record, images


def read_tm_psu(folder):
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    shift_by_rate = validation_shift_by_rate(psbd_dir, RECOMMENDED_PLACEMENT, baselines)
    rate = choose_rate(shift_by_rate, "adaptive")
    rule = "adaptive"
    if rate is None:
        rate = choose_rate(shift_by_rate, "nearest")
        rule = "nearest"
    passes = load_passes(psbd_dir, RECOMMENDED_PLACEMENT, rate, baselines)
    psu = {
        split: psu_ratio_from_cache(
            passes[split]["baseline_probs"],
            passes[split]["baseline_labels"],
            passes[split]["per_pass_probs"],
        )  # (n_split,)
        for split in passes
    }
    reading = (psu, baselines, float(rate), rule, float(shift_by_rate[rate]))
    return reading


# The cache was written by cli.sweep in loader order. If the rebuilt rows do not
# reproduce the cached predictions, the gallery shows the wrong images, so the
# run stops. The check runs on the first rows of each split and on the gallery
# entries themselves, whose CPU prediction is recorded: the cache holds bfloat16
# passes and this rerun is float32, which can flip an image whose top 2 classes
# are close.
ALIGNMENT_ROWS = 64
ALIGNMENT_FLOOR = 0.95


@torch.inference_mode()
def check_against_cache(model, entries, datasets, baselines):
    for split, dataset in datasets.items():
        rows = range(min(ALIGNMENT_ROWS, len(dataset)))
        images = torch.stack([dataset[row][0] for row in rows])  # (n, 3, s, s)
        predictions = model(images).argmax(dim=1)  # (n,)
        agreement = float(
            (predictions == baselines[split][1][: len(rows)]).float().mean()
        )
        assert agreement >= ALIGNMENT_FLOOR, (split, agreement)

    images = torch.stack([entry["image"] for entry in entries])  # (n, 3, s, s)
    probabilities = model(images).softmax(dim=1)  # (n, classes)
    for entry, row in zip(entries, probabilities):
        entry["cpu_prediction"] = int(row.argmax())
        entry["cpu_confidence"] = float(row.max())
    agreement = sum(e["cpu_prediction"] == e["predicted_class"] for e in entries)
    assert agreement >= ALIGNMENT_FLOOR * len(entries), agreement


def base_rates(labels, predictions, confidence, psu, target):
    rates = {
        "target_class_share": float((labels == target).float().mean()),
        "predicted_target_share": float((predictions == target).float().mean()),
        "negative_psu_share": float((psu < 0).float().mean()),
        "below_confident_share": float((confidence < CONFIDENT).float().mean()),
        "misclassified_share": float((predictions != labels).float().mean()),
    }
    return rates


def statistics(entries, target, triggered):
    true_counts = collections.Counter(entry["true_class"] for entry in entries)
    predicted_counts = collections.Counter(
        entry["predicted_class"] for entry in entries
    )
    count = len(entries)
    summary = {
        "count": count,
        "target_class_share": sum(e["true_class"] == target for e in entries) / count,
        "predicted_target_share": sum(e["predicted_class"] == target for e in entries)
        / count,
        "negative_psu_share": sum(e["psu"] < 0 for e in entries) / count,
        "below_confident_share": sum(e["confidence"] < CONFIDENT for e in entries)
        / count,
        "misclassified_share": sum(
            e["predicted_class"] != e["true_class"] for e in entries
        )
        / count,
        "top_true_classes": true_counts.most_common(3),
        "top_predicted_classes": predicted_counts.most_common(3),
        "largest_true_class_share": true_counts.most_common(1)[0][1] / count,
        "distinct_true_classes": len(true_counts),
        "psu_range": [min(e["psu"] for e in entries), max(e["psu"] for e in entries)],
    }
    if triggered:
        summary["escaping_share_of_target_predictions"] = summary[
            "predicted_target_share"
        ]
    return summary


def normalization(manifest, dataset):
    stored = manifest.get("normalization")
    if stored:
        statistics_pair = (stored["mean"], stored["std"])
        return statistics_pair
    spec = DATASET_REGISTRY[dataset]
    return spec.mean, spec.std


def to_pixels(image, mean, std):
    restored = (
        image * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    )
    pixels = (restored.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
    return pixels.numpy()


def draw_gallery(path, record, key, images):
    entries = record[key]["entries"]
    columns = 5
    rows = (len(entries) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.9 * rows))
    for axis, entry, pixels in zip(axes.flat, entries, images):
        axis.imshow(pixels, interpolation="nearest")
        target_mark = " TARGET" if entry["is_target_class"] else ""
        axis.set_title(
            f"true {entry['true_class']}, pred {entry['predicted_class']}{target_mark}\n"
            f"P_c {entry['confidence']:.3f}, PSU {entry['psu']:+.3f}",
            fontsize=8,
            color="#D55E00" if entry["is_target_class"] else "black",
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in list(axes.flat)[len(entries) :]:
        axis.axis("off")
    stats = record[key]["statistics"]
    if key == "lowest_clean":
        heading = (
            f"{record['folder']}: the {len(entries)} clean validation images with the lowest"
            f" PSBD-TM PSU (threshold at {record['budget']:.0%} FPR {record['threshold']:+.3f})"
        )
        target_line = f"true class is the target {stats['target_class_share']:.2f}"
    else:
        heading = (
            f"{record['folder']}: the {len(entries)} triggered images sent to the target"
            f" with the highest PSU (TPR at {record['budget']:.0%} FPR"
            f" {record['tpr_at_budget']:.3f})"
        )
        target_line = f"predicted target {stats['predicted_target_share']:.2f}"
    footer = (
        f"{target_line}, negative PSU {stats['negative_psu_share']:.2f},"
        f" P_c < {CONFIDENT} {stats['below_confident_share']:.2f},"
        f" misclassified {stats['misclassified_share']:.2f},"
        f" largest true class share {stats['largest_true_class_share']:.2f}"
        f" ({stats['distinct_true_classes']} classes). PSBD-TM rate {record['rate']}"
        f" ({record['rate_rule']}), target class {record['target_label']}"
    )
    figure.suptitle(heading, fontsize=10)
    figure.text(0.5, 0.01, footer, ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=config.DPI)
    plt.close(figure)


if __name__ == "__main__":
    main()

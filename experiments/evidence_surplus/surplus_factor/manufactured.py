"""Experiment A2 of evidence surplus: give clean images surplus with no trigger and ask whether PSBD flags them.

A clean image x predicted as class c gets its own class's direction added at a
block's output,

    h_l(x) <- h_l(x) + (S - 1) * ||o_c|| * o_c / ||o_c||,   o_c = mu_c - mu

mu_c the mean pooled feature (class token) of class c and mu the mean over all
classes, both fitted on the fit images. The steered image then carries S times a
typical image's own-class evidence along o_c. PSBD-TM and PSBD-RD score it with the
fractional PSU at the model's adaptive rate p* and 3 passes, and the flag rate is
the share at or below a quantile of the unsteered clean-validation scores.

Controls: a random unit direction of the same added length on every token, and the
own-class direction added to the class token only. Steering is applied at block 8
and at block 12. After block 12 only the class token reaches the head.

    prepare    CPU, the clean validation split and the clean analysis images
    probe      GPU under the shared lock, 1 model per call
    summarize  CPU, the table and the figure
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning import seed_everything  # noqa: E402

from analysis.features import captured_layers, transformer_blocks  # noqa: E402
from analysis.samples import collect_split  # noqa: E402
from cli.sweep import PSBD_MASK_SEED, bound_operator  # noqa: E402
from data.splits import build_psbd_loaders_from_checkpoint  # noqa: E402
from defenses.inference import forward_probs  # noqa: E402
from defenses.operators import effective_forward_passes  # noqa: E402
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.backdoor_manifestation.measure import checkpoint_path  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout  # noqa: E402

SLUG = "evidence_surplus"
CACHE = os.path.join("scratch", SLUG, "manufactured")
MODELS = {
    "vit_cifar10_benign": "cifar10",
    "vit_gtsrb_benign": "gtsrb",
    "vit_cifar10_badnet_a2o_0_01": "cifar10",
    "vit_cifar10_blend_0_1": "cifar10",
}
PLACEMENTS = {
    "PSBD-TM": (
        "before_attention_norm",
        "token_mask",
        "before_attention_norm_token_mask",
    ),
    "PSBD-RD": ("post_residual", "dropout", "post_residual"),
}
STEER_BLOCKS = (8, 12)
SURPLUS_GRID = (1.0, 2.0, 4.0, 8.0)
QUANTILES = (0.01, 0.05, 0.25)
FORWARD_PASSES = 3
FIT_IMAGES = 5000
STEER_IMAGES = 1000
BATCH_SIZE = 200
GPU_MEMORY_FRACTION = 0.35
SEED = 0


def main():
    args = parse_args()
    if args.stage == "summarize":
        summarize(args.results_dir)
        return
    if args.stage == "prepare":
        prepare(args.model, args.raw_data_dir)
        return
    probe(args.model, args.results_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probe", "summarize"))
    parser.add_argument("--model", default=None, choices=sorted(MODELS))
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def prepare(folder, raw_data_dir):
    benign = folder.endswith("_benign")
    # Only the clean splits are read. A benign checkpoint has no trigger of its
    # own and the loader refuses it without one, so a probe trigger is named.
    loaders, _ = build_psbd_loaders_from_checkpoint(
        checkpoint_path(folder),
        raw_data_dir=raw_data_dir,
        batch_size=256,
        num_workers=8,
        probe_attack="badnet_a2o" if benign else None,
        probe_target_label=0 if benign else None,
    )
    validation_images, validation_labels = collect_split(loaders["validation"])
    analysis_images, analysis_labels = collect_split(loaders["clean"])
    cached = {
        "validation": validation_images,  # (2000, C, H, W)
        "validation_labels": validation_labels,
        "fit": analysis_images[:FIT_IMAGES],  # (FIT_IMAGES, C, H, W)
        "fit_labels": analysis_labels[:FIT_IMAGES],
        "steer": analysis_images[FIT_IMAGES : FIT_IMAGES + STEER_IMAGES],
        "steer_labels": analysis_labels[FIT_IMAGES : FIT_IMAGES + STEER_IMAGES],
    }
    os.makedirs(CACHE, exist_ok=True)
    torch.save(cached, os.path.join(CACHE, f"{folder}.pt"))
    print(
        f"{folder}: {len(validation_images)} validation, {len(cached['fit'])} fit, {len(cached['steer'])} steer"
    )


def probe(folder, results_dir):
    seed_everything(SEED)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    data = torch.load(os.path.join(CACHE, f"{folder}.pt"))
    model = load_checkpoint("vit", checkpoint_path(folder), device)
    blocks = transformer_blocks(network_core(model), "vit")
    rates = adaptive_rates(results_dir, folder)
    surplus_values = surplus_from_experiment_a(results_dir, folder)

    class_directions = {
        block: class_offsets(model, data["fit"], data["fit_labels"], block, device)
        for block in STEER_BLOCKS
    }
    validation_scores = {
        name: psu_scores(
            model, blocks, spec, rates[name], data["validation"], None, device
        )
        for name, spec in PLACEMENTS.items()
    }
    thresholds = {
        name: {str(q): float(torch.quantile(scores["psu_ratio"], q)) for q in QUANTILES}
        for name, scores in validation_scores.items()
    }

    unsteered = {
        name: psu_scores(model, blocks, spec, rates[name], data["steer"], None, device)
        for name, spec in PLACEMENTS.items()
    }
    steer_predictions = unsteered["PSBD-TM"]["baseline_labels"]  # (n,)
    conditions = [{"block": None, "surplus": None, "variant": "unsteered"}]
    results = [
        condition_record(conditions[0], unsteered, thresholds, steer_predictions)
    ]
    generator = torch.Generator().manual_seed(SEED)
    grid = sorted(
        set(SURPLUS_GRID) | {round(value, 3) for value in surplus_values.values()}
    )
    for block in STEER_BLOCKS:
        offsets, norms = class_directions[block]  # (classes, dim), (classes,)
        for surplus in grid:
            lengths = (surplus - 1.0) * norms[steer_predictions]  # (n,)
            own = (
                offsets[steer_predictions] / norms[steer_predictions, None]
            )  # (n, dim)
            random = torch.randn(own.shape, generator=generator)
            random = random / random.norm(dim=1, keepdim=True)  # (n, dim)
            for variant, units, class_token_only in (
                ("own_all_tokens", own, False),
                ("own_class_token", own, True),
                ("random_all_tokens", random, False),
            ):
                steer = {
                    "block": blocks[block - 1],
                    "shifts": units * lengths[:, None],  # (n, dim)
                    "class_token_only": class_token_only,
                }
                scores = {
                    name: psu_scores(
                        model, blocks, spec, rates[name], data["steer"], steer, device
                    )
                    for name, spec in PLACEMENTS.items()
                }
                condition = {"block": block, "surplus": surplus, "variant": variant}
                results.append(
                    condition_record(condition, scores, thresholds, steer_predictions)
                )

    record = {
        "model": folder,
        "adaptive_rates": rates,
        "surplus_from_a": surplus_values,
        "thresholds": thresholds,
        "cached_threshold_q0_25": cached_thresholds(results_dir, folder),
        "n_validation": int(data["validation"].shape[0]),
        "n_steer": int(data["steer"].shape[0]),
        "forward_passes": FORWARD_PASSES,
        "conditions": results,
    }
    directory = os.path.join(experiment_results_dir(SLUG, results_dir), "manufactured")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{folder}.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    for row in results:
        print(
            row["block"],
            row["surplus"],
            row["variant"],
            row["kept_prediction"],
            row["flag_rate"],
        )


def adaptive_rates(results_dir, folder):
    with open(os.path.join(results_dir, folder, "psbd_metrics.json")) as handle:
        placements = json.load(handle)["placements"]
    rates = {
        name: float(placements[spec[2]]["adaptive_rate"])
        for name, spec in PLACEMENTS.items()
    }
    return rates


def cached_thresholds(results_dir, folder):
    """The adaptive-rate q 0.25 threshold psbd_metrics.json holds, to check this script's scoring against."""
    with open(os.path.join(results_dir, folder, "psbd_metrics.json")) as handle:
        placements = json.load(handle)["placements"]
    thresholds = {
        name: placements[spec[2]]["adaptive"]["threshold"]
        for name, spec in PLACEMENTS.items()
    }
    return thresholds


def surplus_from_experiment_a(results_dir, folder):
    """The last-block surplus factor A measured, the model's own or the median of its dataset's backdoored models."""
    runs = os.path.join(experiment_results_dir(SLUG, results_dir), "runs")
    if not folder.endswith("_benign"):
        with open(os.path.join(runs, f"{folder}.json")) as handle:
            value = json.load(handle)["layers"][-1]["backdoor"]["median_surplus"]
        return {folder: value}
    dataset = MODELS[folder]
    values = {}
    for path in glob.glob(os.path.join(runs, f"vit_{dataset}_*.json")):
        record = json.load(open(path))
        if record["run"]["benign"]:
            continue
        values[record["run"]["name"]] = record["layers"][-1]["backdoor"][
            "median_surplus"
        ]
    median = float(np.median([value for value in values.values() if value is not None]))
    return {f"median of {len(values)} {dataset} models": median}


@torch.inference_mode()
def class_offsets(model, images, labels, block, device):
    """o_c = mu_c - mu of the class token at block's output, (classes, dim), with its norms."""
    features = []
    with captured_layers(model, (block,), "vit") as captured:
        for batch in images.split(BATCH_SIZE):
            forward_probs(model, batch, device, True)
            features.append(captured[block][:, 0, :].float().cpu())
    stacked = torch.cat(features)  # (n, dim)
    classes = int(labels.max()) + 1

    overall = stacked.mean(dim=0)  # (dim,)
    means = torch.stack(
        [
            stacked[labels == c].mean(dim=0) if (labels == c).any() else overall
            for c in range(classes)
        ]
    )  # (classes, dim)
    offsets = means - overall  # (classes, dim)
    norms = offsets.norm(dim=1).clamp_min(1e-8)  # (classes,)
    return offsets, norms


def steer_hook(shifts, class_token_only, rows):
    """Add each image's own shift to its tokens, or to its class token only."""

    def hook(_module, _inputs, output):
        shift = shifts[rows["start"] : rows["start"] + output.shape[0]].to(
            output.device, output.dtype
        )
        if class_token_only:
            steered = output.clone()
            steered[:, 0, :] = steered[:, 0, :] + shift
            return steered
        steered = output + shift[:, None, :]
        return steered

    return hook


@torch.inference_mode()
def psu_scores(model, blocks, spec, rate, images, steer, device):
    """Fractional PSU of every image under 1 placement at rate, with an optional steer applied throughout."""
    position, operator, _ = spec
    rows = {"start": 0}
    handle = (
        steer["block"].register_forward_hook(
            steer_hook(steer["shifts"], steer["class_token_only"], rows)
        )
        if steer is not None
        else None
    )
    try:
        baseline = batched_probs(model, images, rows, device)  # (n, classes)
        labels = baseline.argmax(dim=1)  # (n,)
        factory = {
            name: bound_operator(operator, None)
            for name in DROPOUT_CONFIGS.get(position, (position,))
        }
        probe_handles = plug_dropout(model, "vit", tuple(factory), factory, rate)
        try:
            seed_everything(PSBD_MASK_SEED)
            passes = []
            for _ in range(effective_forward_passes(operator, FORWARD_PASSES)):
                probs = batched_probs(model, images, rows, device)  # (n, classes)
                passes.append(probs.gather(1, labels[:, None]).squeeze(1))
        finally:
            unplug_dropout(probe_handles)
    finally:
        if handle is not None:
            handle.remove()

    per_pass = torch.stack(passes)  # (passes, n)
    scores = {
        "psu_ratio": psu_ratio_from_cache(baseline, labels, per_pass),  # (n,)
        "baseline_labels": labels,
        "baseline_confidence": baseline.gather(1, labels[:, None]).squeeze(1),
    }
    return scores


def batched_probs(model, images, rows, device):
    chunks = []
    for start in range(0, images.shape[0], BATCH_SIZE):
        rows["start"] = start
        chunks.append(
            forward_probs(model, images[start : start + BATCH_SIZE], device, True).cpu()
        )
    probs = torch.cat(chunks)  # (n, classes)
    return probs


def condition_record(condition, scores, thresholds, unsteered_predictions):
    record = {
        **condition,
        "kept_prediction": float(
            (scores["PSBD-TM"]["baseline_labels"] == unsteered_predictions)
            .float()
            .mean()
        ),
        "median_confidence": float(scores["PSBD-TM"]["baseline_confidence"].median()),
        "median_psu_ratio": {
            name: float(value["psu_ratio"].median()) for name, value in scores.items()
        },
        "flag_rate": {
            name: {
                q: float((value["psu_ratio"] <= threshold).float().mean())
                for q, threshold in thresholds[name].items()
            }
            for name, value in scores.items()
        },
    }
    return record


def summarize(results_dir):
    directory = os.path.join(experiment_results_dir(SLUG, results_dir), "manufactured")
    records = [
        json.load(open(path))
        for path in sorted(glob.glob(os.path.join(directory, "*.json")))
    ]
    figure, axes = plt.subplots(
        len(records), 4, figsize=(7.0, 1.7 * len(records)), sharey=True, squeeze=False
    )
    for row, record in enumerate(records):
        for column, (placement, block) in enumerate(
            [(p, b) for p in PLACEMENTS for b in STEER_BLOCKS]
        ):
            axis = axes[row, column]
            for variant, style in (
                ("own_all_tokens", "-"),
                ("own_class_token", "--"),
                ("random_all_tokens", ":"),
            ):
                points = sorted(
                    (c["surplus"], c["flag_rate"][placement]["0.05"])
                    for c in record["conditions"]
                    if c["block"] == block and c["variant"] == variant
                )
                axis.plot(
                    *zip(*points),
                    linestyle=style,
                    marker="o",
                    markersize=2,
                    label=variant.replace("_", " "),
                )
            axis.axhline(0.05, color="grey", linewidth=0.6)
            axis.set_xscale("log", base=2)
            axis.set_title(f"{record['model']}\n{placement}, block {block}", fontsize=6)
            if row == len(records) - 1:
                axis.set_xlabel("S")
            if column == 0:
                axis.set_ylabel("flag rate at q 0.05")
    axes[0, 0].legend(fontsize=5)
    figure.tight_layout()
    figures = os.path.join(experiment_results_dir(SLUG, results_dir), "figures")
    os.makedirs(figures, exist_ok=True)
    figure.savefig(os.path.join(figures, "manufactured_surplus.png"), dpi=150)
    figure.savefig(os.path.join(figures, "manufactured_surplus.pdf"))
    plt.close(figure)
    print(f"{len(records)} models summarized")


if __name__ == "__main__":
    main()

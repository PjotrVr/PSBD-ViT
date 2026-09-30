"""Experiment A3 of evidence surplus: remove surplus from PSBD-TM's false positives and ask whether they stop being flagged.

False positives are clean validation images whose cached PSBD-TM fractional PSU
at the adaptive rate is at or below the cached 0.01 or 0.05 quantile. 2 edits
remove evidence while keeping the prediction.

Stream edit, at the last block's output on every token. With o_c the class-mean
offset of the image's predicted class c (fitted on clean analysis images), e its
own-class evidence and beta* the removal that flips the prediction,

    e = (h - mu) . o_c / ||o_c||,   b = e - beta*,   s = e / b
    h <- h - beta * o_c / ||o_c||,  beta = max(0, e - s_med * b)

so the edited surplus equals s_med, the median s of non-flagged clean images.

Input edit, at the first block's input. Patch tokens are ranked by how much
replacing each alone with the per-position mean clean token lowers the predicted
logit. The smallest m whose top-m tokens kept (the rest replaced) keep the
prediction is found by bisection, and exactly those m are kept.

Controls: a random direction with the same removed length, a random token subset
of the same size, and both edits on non-flagged clean images. Readout: PSBD-TM at
the adaptive rate with 3 fresh passes, flagged at the cached thresholds.

    prepare    CPU, reuses manufactured.prepare (validation, fit and steer images)
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
from cli.sweep import PSBD_MASK_SEED, bound_operator  # noqa: E402
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defenses.decision import RECOMMENDED_PLACEMENT, threshold_at_quantile  # noqa: E402
from defenses.inference import forward_logits  # noqa: E402
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.backdoor_manifestation.measure import checkpoint_path  # noqa: E402
from experiments.evidence_surplus.surplus_factor.manufactured import (  # noqa: E402
    CACHE,
    class_offsets,
)
from experiments.evidence_surplus.surplus_factor.manufactured import (  # noqa: E402
    prepare as prepare_images,
)
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402

SLUG = "evidence_surplus"
MODELS = [
    "vit_cifar10_badnet_a2o_0_01",
    "vit_tiny_badnet_a2o_0_05",
    "vit_cifar100_blend_0_1",
    "vit_cifar10_blend_0_1",
    "vit_cifar10_bpp_0_05",
    "vit_gtsrb_lf_0_01",
    "vit_cifar10_wanet_0_1",
    "vit_tiny_wanet_0_05",
    "vit_gtsrb_tact_0_05",
    "vit_cifar10_tact_0_01",
    "vit_cifar10_benign",
    "vit_gtsrb_benign",
]
QUANTILES = (0.01, 0.05)
NON_FLAGGED_SAMPLE = 200
FORWARD_PASSES = 3
EDIT_BLOCK = 12
SEARCH_FACTOR = 4.0
BISECTION_STEPS = 12
BATCH_SIZE = 196
GPU_MEMORY_FRACTION = 0.35
SEED = 0


def main():
    args = parse_args()
    if args.stage == "summarize":
        summarize(args.results_dir)
        return
    if args.stage == "prepare":
        if not os.path.exists(os.path.join(CACHE, f"{args.model}.pt")):
            prepare_images(args.model, args.raw_data_dir)
        return
    probe(args.model, args.results_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probe", "summarize"))
    parser.add_argument("--model", default=None, choices=MODELS)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def probe(folder, results_dir):
    seed_everything(SEED)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    data = torch.load(os.path.join(CACHE, f"{folder}.pt"))
    model = load_checkpoint("vit", checkpoint_path(folder), device)
    blocks = transformer_blocks(network_core(model), "vit")

    rate, cached_psu, cached_labels = cached_validation(results_dir, folder)
    thresholds = {str(q): threshold_at_quantile(cached_psu, q) for q in QUANTILES}
    flagged = torch.nonzero(cached_psu <= thresholds["0.05"]).squeeze(1)  # (n_fp,)
    flagged_at_1 = cached_psu[flagged] <= thresholds["0.01"]  # (n_fp,)
    generator = torch.Generator().manual_seed(SEED)
    unflagged_pool = torch.nonzero(cached_psu > thresholds["0.05"]).squeeze(1)
    unflagged = unflagged_pool[
        torch.randperm(len(unflagged_pool), generator=generator)[:NON_FLAGGED_SAMPLE]
    ]

    offsets, norms = class_offsets(
        model, data["fit"], data["fit_labels"], EDIT_BLOCK, device
    )
    overall = mean_class_token(model, data["fit"], EDIT_BLOCK, device)  # (dim,)
    mean_tokens = mean_first_block_input(model, data["fit"], device)  # (197, dim)

    sets = {"false_positive": flagged, "non_flagged": unflagged}
    readings = {
        name: read_evidence(
            model, blocks, data["validation"][rows], offsets, norms, overall, device
        )
        for name, rows in sets.items()
    }
    reachable = ~torch.isnan(readings["non_flagged"]["surplus"])
    median_surplus = float(readings["non_flagged"]["surplus"][reachable].median())

    results = {}
    for name, rows in sets.items():
        images = data["validation"][rows]  # (n, C, H, W)
        reading = readings[name]
        results[name] = edit_and_score(
            model,
            blocks,
            images,
            reading,
            median_surplus,
            mean_tokens,
            rate,
            thresholds,
            generator,
            device,
        )
        results[name]["cached_label_agreement"] = float(
            (reading["predictions"] == cached_labels[rows]).float().mean()
        )
    results["false_positive"]["flagged_at_0.01"] = flagged_at_1.tolist()

    record = {
        "model": folder,
        "adaptive_rate": rate,
        "cached_thresholds": thresholds,
        "n_validation": int(cached_psu.numel()),
        "n_false_positive": int(flagged.numel()),
        "n_false_positive_at_0.01": int(flagged_at_1.sum()),
        "n_non_flagged": int(unflagged.numel()),
        "median_non_flagged_surplus": median_surplus,
        "false_positive_median_surplus": float(
            np.nanmedian(readings["false_positive"]["surplus"].numpy())
        ),
        "false_positive_reachable_share": float(
            (~torch.isnan(readings["false_positive"]["surplus"])).float().mean()
        ),
        "non_flagged_reachable_share": float(reachable.float().mean()),
        "sets": results,
    }
    directory = os.path.join(experiment_results_dir(SLUG, results_dir), "removal")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{folder}.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    for name, entry in results.items():
        print(
            folder,
            name,
            {edit: value["flag_rate"] for edit, value in entry["edits"].items()},
        )


def cached_validation(results_dir, folder):
    """PSBD-TM's adaptive rate and the cached fractional PSU of the clean validation split."""
    with open(os.path.join(results_dir, folder, "psbd_metrics.json")) as handle:
        rate = float(
            json.load(handle)["placements"][RECOMMENDED_PLACEMENT]["adaptive_rate"]
        )
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
    per_pass, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, RECOMMENDED_PLACEMENT, rate, "validation")
    )
    psu = psu_ratio_from_cache(probs, labels, per_pass)  # (n,)
    return rate, psu, labels


@torch.inference_mode()
def mean_class_token(model, images, block, device):
    features = []
    with captured_layers(model, (block,), "vit") as captured:
        for batch in images.split(BATCH_SIZE):
            forward_logits(model, batch, device, True)
            features.append(captured[block][:, 0, :].float().cpu())
    overall = torch.cat(features).mean(dim=0)  # (dim,)
    return overall


@torch.inference_mode()
def mean_first_block_input(model, images, device):
    total, count = 0.0, 0
    with captured_layers(model, (0,), "vit") as captured:
        for batch in images.split(BATCH_SIZE):
            forward_logits(model, batch, device, True)
            total = total + captured[0].float().sum(dim=0).cpu()  # (197, dim)
            count += batch.shape[0]
    mean_tokens = total / count  # (197, dim)
    return mean_tokens


def stream_hook(units, amounts, rows):
    """Subtract amounts[i] * units[i] from every token of image i at the hooked block's output."""

    def hook(_module, _inputs, output):
        start = rows["start"]
        shift = (
            units[start : start + output.shape[0]]
            * amounts[start : start + output.shape[0], None]
        ).to(output.device, output.dtype)
        edited = output - shift[:, None, :]
        return edited

    return hook


def token_hook(keep, mean_tokens, rows):
    """Replace every patch token not kept with the per-position mean clean token, at the first block's input."""

    def pre_hook(_module, inputs):
        start = rows["start"]
        tokens = inputs[0]
        kept = keep[start : start + tokens.shape[0]].to(tokens.device)  # (batch, 197)
        replacement = mean_tokens.to(tokens.device, tokens.dtype)[None]  # (1, 197, dim)
        edited = torch.where(kept[:, :, None], tokens, replacement)
        return (edited,)

    return pre_hook


def forward_installer(module, make_hook):
    """A function that registers make_hook(rows) as a forward hook on module and returns the handles."""

    def install(rows):
        handles = [module.register_forward_hook(make_hook(rows))]
        return handles

    return install


def pre_installer(first_block, keep, mean_tokens):
    """A function that registers the token replacement as a pre-hook on the first block."""

    def install(rows):
        handles = [
            first_block.register_forward_pre_hook(token_hook(keep, mean_tokens, rows))
        ]
        return handles

    return install


@torch.inference_mode()
def predictions_with(model, images, install, device):
    rows = {"start": 0}
    handles = install(rows)
    chunks = []
    try:
        for start in range(0, images.shape[0], BATCH_SIZE):
            rows["start"] = start
            chunks.append(
                forward_logits(
                    model, images[start : start + BATCH_SIZE], device, True
                ).cpu()
            )
    finally:
        for handle in handles:
            handle.remove()
    logits = torch.cat(chunks)  # (n, classes)
    return logits


@torch.inference_mode()
def read_evidence(model, blocks, images, offsets, norms, overall, device):
    """Predicted class, own-class evidence e, flip removal beta* and surplus e / (e - beta*) per image."""
    features = []
    with captured_layers(model, (EDIT_BLOCK,), "vit") as captured:
        logits = []
        for batch in images.split(BATCH_SIZE):
            logits.append(forward_logits(model, batch, device, True).cpu())
            features.append(captured[EDIT_BLOCK][:, 0, :].float().cpu())
    predictions = torch.cat(logits).argmax(dim=1)  # (n,)
    units = offsets[predictions] / norms[predictions, None]  # (n, dim)
    evidence = ((torch.cat(features) - overall) * units).sum(dim=1)  # (n,)

    limit = SEARCH_FACTOR * evidence.abs().clamp_min(1e-3)  # (n,)
    block = blocks[EDIT_BLOCK - 1]

    def flips(amounts):
        install = forward_installer(
            block, lambda rows: stream_hook(units, amounts, rows)
        )
        changed = (
            predictions_with(model, images, install, device).argmax(dim=1)
            != predictions
        )
        return changed

    reachable = flips(limit)
    low, high = torch.zeros_like(limit), limit.clone()
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        changed = flips(middle)
        high = torch.where(changed, middle, high)
        low = torch.where(changed, low, middle)
    needed = torch.where(reachable, high, torch.full_like(high, float("nan")))  # (n,)
    boundary = evidence - needed  # (n,)
    surplus = torch.where(
        boundary > 0, evidence / boundary, torch.full_like(boundary, float("nan"))
    )
    reading = {
        "predictions": predictions,
        "units": units,
        "evidence": evidence,
        "boundary": boundary,
        "surplus": surplus,
    }
    return reading


@torch.inference_mode()
def token_ranking(model, first_block, images, predictions, mean_tokens, device):
    """Per image, patch tokens ordered by the logit drop of replacing each alone with the mean token."""
    rankings = []
    positions = torch.arange(1, 197)
    for index in range(images.shape[0]):
        keep = torch.ones(196, 197, dtype=torch.bool)
        keep[torch.arange(196), positions] = False  # row j drops patch token j
        copies = images[index : index + 1].expand(196, -1, -1, -1)
        install = pre_installer(first_block, keep, mean_tokens)
        logits = predictions_with(model, copies, install, device)  # (196, classes)
        drop = -logits[:, predictions[index]]  # (196,)
        rankings.append(
            torch.argsort(drop, descending=True) + 1
        )  # most important first, token indices
    ranking = torch.stack(rankings)  # (n, 196)
    return ranking


def keep_mask(ranking, counts):
    """(n, 197) mask keeping the class token and the first counts[i] tokens of ranking[i]."""
    n = ranking.shape[0]
    keep = torch.zeros(n, 197, dtype=torch.bool)
    keep[:, 0] = True
    order = torch.arange(196)[None, :] < counts[:, None]  # (n, 196)
    keep.scatter_(1, ranking, order)
    return keep


def smallest_sufficient(
    model, first_block, images, predictions, ranking, mean_tokens, device
):
    low, high = (
        torch.zeros(images.shape[0], dtype=torch.long),
        torch.full((images.shape[0],), 196),
    )
    for _ in range(8):
        middle = (low + high) // 2
        keep = keep_mask(ranking, middle)
        install = pre_installer(first_block, keep, mean_tokens)
        same = (
            predictions_with(model, images, install, device).argmax(dim=1)
            == predictions
        )
        high = torch.where(same, middle, high)
        low = torch.where(same, low, middle)
    counts = high.clamp_min(1)  # (n,)
    return counts


def edit_and_score(
    model,
    blocks,
    images,
    reading,
    median_surplus,
    mean_tokens,
    rate,
    thresholds,
    generator,
    device,
):
    first_block = blocks[0]
    block = blocks[EDIT_BLOCK - 1]
    predictions = reading["predictions"]
    removal = (reading["evidence"] - median_surplus * reading["boundary"]).clamp_min(
        0
    )  # (n,)
    removal = torch.nan_to_num(removal, nan=0.0)
    random_units = torch.randn(reading["units"].shape, generator=generator)
    random_units = random_units / random_units.norm(dim=1, keepdim=True)  # (n, dim)

    ranking = token_ranking(
        model, first_block, images, predictions, mean_tokens, device
    )  # (n, 196)
    counts = smallest_sufficient(
        model, first_block, images, predictions, ranking, mean_tokens, device
    )  # (n,)
    random_ranking = torch.stack(
        [torch.randperm(196, generator=generator) + 1 for _ in range(images.shape[0])]
    )

    edits = {
        "unedited": lambda rows: [],
        "stream_own_class": forward_installer(
            block, lambda rows: stream_hook(reading["units"], removal, rows)
        ),
        "stream_random_direction": forward_installer(
            block, lambda rows: stream_hook(random_units, removal, rows)
        ),
        "input_sufficient_tokens": pre_installer(
            first_block, keep_mask(ranking, counts), mean_tokens
        ),
        "input_random_tokens": pre_installer(
            first_block, keep_mask(random_ranking, counts), mean_tokens
        ),
    }
    scored = {}
    for name, install in edits.items():
        baseline = torch.softmax(
            predictions_with(model, images, install, device), dim=1
        )  # (n, classes)
        labels = baseline.argmax(dim=1)  # (n,)
        per_pass = []
        handles = plug_dropout(
            model,
            "vit",
            ("before_attention_norm",),
            {"before_attention_norm": bound_operator("token_mask", None)},
            rate,
        )
        try:
            seed_everything(PSBD_MASK_SEED)
            for _ in range(FORWARD_PASSES):
                probs = torch.softmax(
                    predictions_with(model, images, install, device), dim=1
                )
                per_pass.append(probs.gather(1, labels[:, None]).squeeze(1))
        finally:
            unplug_dropout(handles)
        psu = psu_ratio_from_cache(baseline, labels, torch.stack(per_pass))  # (n,)
        scored[name] = {
            "kept_prediction": float((labels == predictions).mean(dtype=torch.float)),
            "median_psu_ratio": float(psu.median()),
            "flag_rate": {
                q: float((psu <= threshold).float().mean())
                for q, threshold in thresholds.items()
            },
            "psu_ratio": [round(float(value), 4) for value in psu],
        }
    entry = {
        "n": int(images.shape[0]),
        "edited_by_stream": float((removal > 0).float().mean()),
        "median_removal_over_evidence": float(
            np.nanmedian((removal / reading["evidence"].abs().clamp_min(1e-6)).numpy())
        ),
        "median_sufficient_tokens": float(counts.float().median()),
        "edits": scored,
    }
    return entry


def summarize(results_dir):
    directory = os.path.join(experiment_results_dir(SLUG, results_dir), "removal")
    records = [
        json.load(open(path))
        for path in sorted(glob.glob(os.path.join(directory, "*.json")))
    ]
    edits = [
        "unedited",
        "stream_own_class",
        "stream_random_direction",
        "input_sufficient_tokens",
        "input_random_tokens",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for axis, set_name in zip(axes, ("false_positive", "non_flagged")):
        positions = np.arange(len(edits))
        for record in records:
            values = [
                record["sets"][set_name]["edits"][edit]["flag_rate"]["0.05"]
                for edit in edits
            ]
            style = "--" if record["model"].endswith("benign") else "-"
            axis.plot(
                positions,
                values,
                style,
                marker="o",
                markersize=2,
                linewidth=0.7,
                label=record["model"],
            )
        axis.axhline(0.05, color="grey", linewidth=0.6)
        axis.set_xticks(
            positions,
            [edit.replace("_", " ") for edit in edits],
            rotation=60,
            fontsize=6,
        )
        axis.set_title(
            f"{set_name.replace('_', ' ')} images, flag rate at q 0.05", fontsize=7
        )
    axes[0].set_ylabel("PSBD-TM flag rate")
    axes[1].legend(fontsize=4)
    figure.tight_layout()
    figures = os.path.join(experiment_results_dir(SLUG, results_dir), "figures")
    os.makedirs(figures, exist_ok=True)
    figure.savefig(os.path.join(figures, "false_positive_removal.png"), dpi=150)
    figure.savefig(os.path.join(figures, "false_positive_removal.pdf"))
    plt.close(figure)
    for record in records:
        fp = record["sets"]["false_positive"]["edits"]
        print(
            record["model"],
            record["n_false_positive"],
            {
                edit: (
                    round(fp[edit]["flag_rate"]["0.05"], 3),
                    round(fp[edit]["kept_prediction"], 3),
                )
                for edit in edits
            },
        )


if __name__ == "__main__":
    main()

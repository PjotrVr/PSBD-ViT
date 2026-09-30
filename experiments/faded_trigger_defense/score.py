"""Per-input PSBD-TM scores on faded triggers, the GPU half of the faded trigger defenses.

The trigger dose experiment (experiments/evidence_surplus/trigger_dose/) stores
only aggregates per dose. This pass writes 1 score per input so the CPU readout
(analyze.py) can pool queries, combine rates and test a preprocessing step
without another GPU run. Everything the detector reads is canonical: PSBD-TM
(token mask at before_attention_norm), the fractional PSU, k = 3 passes under
bfloat16 with mask seed 0, the adaptive rate and the clean-validation thresholds
of the canonical sweep (experiments/evidence_surplus/common.operating_point).

Part "plain" re-creates the first PAIRS pairs of the PSBD eval ASR split with
the dose code's pair loader and trigger fade at the doses in DOSES, and scores
the faded triggered images and their clean twins at every rate of the placement's
cached ladder. Clean validation and the clean test split are read from the
canonical cache in the same per-input format, since the thresholds are read from
exactly those tensors.

Part "features" stores the last block's class token of every image the other
parts score, for the matched filter of analyze.py.

Part "amplified" is exploratory. Every input is sharpened before scoring,
x' = clip(x + lambda (x - blur(x)), 0, 1) in pixel space at the dataset's own
resolution. lambda is the largest value of LAMBDAS whose clean validation
accuracy stays within 1 point of the unsharpened accuracy. The adaptive rate and
the thresholds are then recomputed on sharpened clean validation by the canonical
rules, and the twins and faded triggered images are scored at that rate.

    PYTHONPATH=. .venv/bin/python experiments/faded_trigger_defense/score.py --list
    PYTHONPATH=. .venv/bin/python experiments/faded_trigger_defense/score.py \\
        --models vit_cifar10_blend_0_1 --parts plain amplified
"""

import argparse
import os
import subprocess
import sys
import time
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lightning import seed_everything  # noqa: E402
from torchvision.transforms.v2.functional import gaussian_blur  # noqa: E402

from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    complete_rates,
    select_rate_adaptively,
    threshold_at_quantile,
)
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.evidence_surplus.common import (  # noqa: E402
    MASK_SEED,
    PASSES,
    QUANTILES,
    logits_of,
    normalized,
    operating_point,
    pixel_space,
)
from experiments.evidence_surplus.trigger_dose import (  # noqa: E402
    measure as trigger_dose,
)
from experiments.why_psbd_works.measure import load_pairs  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402

SLUG = "faded_trigger_defense"
DOSES = (1.0, 0.8, 0.6, 0.5, 0.4)
PAIRS = 1024
BACKDOORED = (
    "vit_cifar10_blend_0_1",
    "vit_cifar100_blend_0_1",
    "vit_cifar10_badnet_a2o_0_01",
    "vit_gtsrb_lf_0_01",
    "vit_cifar10_bpp_0_05",
    "vit_tiny_badnet_a2o_0_05",
)
# Each benign model is probed with the triggers its dataset's backdoored models
# carry, so every attack has a control that never learned it where 1 exists.
BENIGN_PROBES = (
    ("vit_cifar10_benign", "badnet_a2o"),
    ("vit_cifar10_benign", "blend"),
    ("vit_cifar10_benign", "bpp"),
    ("vit_gtsrb_benign", "lf"),
)
PARTS = ("plain", "amplified", "features")
LAMBDAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
CLEAN_ACCURACY_TOLERANCE = 0.01
BLUR_KERNEL = 5
BLUR_SIGMA = 1.0
GPU_MEMORY_FRACTION = 0.15
CPU_THREADS = 4
PAIR_BATCH = 128

# The dose code fades inside its measure() and exposes no function for it, so the
# formula is restated in faded_images below. The doses must stay a subset of its
# series so the first rows reproduce its readings.
assert set(DOSES) <= set(trigger_dose.DOSES), "doses outside the dose series"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    # Smoke-test truncations. A real run leaves all 4 at their defaults.
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--max-clean", type=int, default=None)
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument(
        "--clean-accuracy-tolerance", type=float, default=CLEAN_ACCURACY_TOLERANCE
    )
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    entries = (
        [tuple(m.partition(":")[::2]) for m in args.models]
        if args.models
        else [(f, "") for f in BACKDOORED] + list(BENIGN_PROBES)
    )
    if args.list:
        print("\n".join(f"{f}:{p}" if p else f for f, p in entries))
        return

    torch.set_num_threads(CPU_THREADS)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device.index)
    output_dir = args.output_dir or experiment_results_dir(SLUG, args.results_dir)

    for folder, probe in entries:
        name = entry_name(folder, probe)
        for part in args.parts:
            path = os.path.join(output_dir, part, f"{name}.pt")
            if os.path.exists(path):
                continue
            started = time.time()
            scorer = {
                "plain": score_plain,
                "amplified": score_amplified,
                "features": score_features,
            }[part]
            record = scorer(folder, probe or None, args, device)
            record["meta"]["seconds"] = round(time.time() - started, 1)
            save_record(record, path)
            print(f"[ok] {part} {name} {record['meta']['seconds']}s", flush=True)


def entry_name(folder, probe):
    name = f"{folder}__probe_{probe}" if probe else folder
    return name


def score_plain(folder, probe, args, device):
    context = load_context(folder, probe, args, device)
    point = context["point"]
    rates = context["rates"]

    triggered = {}
    for dose in DOSES:
        faded = faded_images(context, dose)  # (n, C, H, W), pixels
        triggered[dose] = readings(
            context, normalized(faded, context["dataset"]), rates, device
        )
    twins = readings(context, context["pairs"]["clean"], rates, device)
    twins["label"] = context["pairs"]["clean_labels"]
    for dose in DOSES:
        triggered[dose]["label"] = context["pairs"]["clean_labels"]

    record = {
        "meta": meta(context, args),
        "rates": rates,
        "adaptive_rate": point["rate"],
        "thresholds": point["thresholds"],
        "validation": cached_readings(context, "validation", args.max_validation),
        "clean_test": cached_readings(context, "clean", args.max_clean),
        "twins": twins,
        "triggered": triggered,
        # Row of each pair in the cached clean and backdoor splits, for the check
        # that fresh passes and the cache agree on the same images.
        "cache_rows": {
            "twins": context["twin_rows"],
            "triggered": torch.arange(len(context["twin_rows"])),
        },
        "cached_pairs": cached_pair_readings(context),
    }
    return record


def score_amplified(folder, probe, args, device):
    context = load_context(folder, probe, args, device)
    dataset = context["dataset"]
    rates = context["rates"]
    validation_pixels = pixel_space(context["validation_images"], dataset)
    validation_labels = context["validation_labels"]

    # lambda is fixed on clean validation accuracy alone, before any triggered
    # image is sharpened.
    accuracy = {
        0.0: clean_accuracy(context, validation_pixels, validation_labels, 0.0, device)
    }
    for amount in LAMBDAS:
        accuracy[amount] = clean_accuracy(
            context, validation_pixels, validation_labels, amount, device
        )
    chosen = choose_lambda(accuracy, args.clean_accuracy_tolerance)

    record = {
        "meta": meta(context, args),
        "lambdas": list(LAMBDAS),
        "validation_accuracy": accuracy,
        "chosen_lambda": chosen,
        "rates": rates,
    }
    if chosen is None:
        return record

    validation = readings(
        context,
        normalized(sharpened(validation_pixels, chosen), dataset),
        rates,
        device,
    )
    validation["label"] = validation_labels
    shift_by_rate = {
        rate: float(1.0 - validation["kept"][i].mean()) for i, rate in enumerate(rates)
    }
    adaptive_rate = select_rate_adaptively(shift_by_rate, ADAPTIVE_SHIFT_TARGET)
    record.update(
        validation=validation, shift_by_rate=shift_by_rate, adaptive_rate=adaptive_rate
    )
    if adaptive_rate is None:
        return record

    rate_index = rates.index(adaptive_rate)
    validation_scores = validation["psu"][rate_index]  # (n_validation,)
    record["thresholds"] = {
        str(q): threshold_at_quantile(validation_scores, q) for q in QUANTILES
    }
    twin_pixels = pixel_space(context["pairs"]["clean"], dataset)
    record["twins"] = readings(
        context,
        normalized(sharpened(twin_pixels, chosen), dataset),
        [adaptive_rate],
        device,
    )
    record["twins"]["label"] = context["pairs"]["clean_labels"]
    record["triggered"] = {}
    for dose in DOSES:
        faded = faded_images(context, dose)  # (n, C, H, W), pixels
        record["triggered"][dose] = readings(
            context,
            normalized(sharpened(faded, chosen), dataset),
            [adaptive_rate],
            device,
        )
        record["triggered"][dose]["label"] = context["pairs"]["clean_labels"]
    return record


# The last block's class token as the head reads it, for the matched filter, on
# every split the other parts score.
def score_features(folder, probe, args, device):
    context = load_context(folder, probe, args, device)
    dataset = context["dataset"]
    clean_set = context["loaders"]["clean"].dataset
    clean_count = min(args.max_clean or len(clean_set), len(clean_set))
    clean_images = torch.stack([clean_set[row][0] for row in range(clean_count)]).to(
        device
    )  # (n_clean, C, H, W)

    record = {
        "meta": meta(context, args),
        "validation": features_of(context, context["validation_images"], device),
        "clean_test": features_of(context, clean_images, device),
        "twins": features_of(context, context["pairs"]["clean"], device),
        "triggered": {
            dose: features_of(
                context, normalized(faded_images(context, dose), dataset), device
            )
            for dose in DOSES
        },
    }
    return record


@torch.inference_mode()
def features_of(context, images, device):
    captured = []
    head = network_core(context["model"]).heads
    handle = head.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].float().cpu())
    )
    try:
        logits = logits_of(context["model"], images, device)  # (n, classes)
    finally:
        handle.remove()
    feature = torch.cat(captured)  # (n, hidden)
    assert feature.shape[0] == logits.shape[0], "1 class token per image"
    result = {"feature": feature.half(), "prediction": logits.argmax(dim=1)}
    return result


def load_context(folder, probe, args, device):
    checkpoint = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, checkpoint, device).eval()
    point = operating_point(args.results_dir, folder, architecture)
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    rates = args.rates or complete_rates(psbd_dir, point["placement"])
    assert point["rate"] in rates, "the adaptive rate must be on the scored ladder"

    loader_args = types.SimpleNamespace(
        raw_data_dir=args.raw_data_dir, pairs=args.pairs, batch_size=PAIR_BATCH
    )
    pairs, _ = load_pairs(checkpoint, probe, loader_args, device)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=PAIR_BATCH,
        num_workers=0,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    twin_rows = cache_rows_of_twins(psbd_dir, manifest, len(pairs["targets"]))

    validation_set = loaders["validation"].dataset
    validation_count = min(
        args.max_validation or len(validation_set), len(validation_set)
    )
    validation_items = [validation_set[row] for row in range(validation_count)]
    validation_images = torch.stack([image for image, _ in validation_items]).to(
        device
    )  # (n_validation, C, H, W)
    validation_labels = torch.tensor([int(label) for _, label in validation_items])

    context = {
        "folder": folder,
        "probe": probe,
        "metadata": metadata,
        "architecture": architecture,
        "dataset": metadata["dataset"],
        "attack": probe or metadata["attack"],
        "model": model,
        "point": point,
        "rates": [float(r) for r in rates],
        "psbd_dir": psbd_dir,
        "pairs": pairs,
        "twin_rows": twin_rows,
        "validation_images": validation_images,
        "validation_labels": validation_labels,
        "loaders": loaders,
    }
    return context


# The clean twin of pair i is the clean-split row holding the same test image as
# backdoor-split row i, which is how load_pairs picks it too.
def cache_rows_of_twins(psbd_dir, manifest, count):
    cached = read_split_manifest(psbd_dir)
    assert cached["analysis_clean_indices"] == manifest["analysis_clean_indices"], (
        "the fresh split must match the cached one row for row"
    )
    row_of = {
        original: row for row, original in enumerate(manifest["analysis_clean_indices"])
    }
    rows = torch.tensor(
        [row_of[original] for original in manifest["analysis_backdoor_indices"][:count]]
    )  # (count,)
    return rows


# The dose code's fade, restated because it lives inline in its measure():
# linear in pixel space for every attack in this set, x(a) = x_clean + a
# (x_triggered - x_clean), and WaNet by its warp strength through its own warped().
def faded_images(context, dose):
    dataset = context["dataset"]
    pairs = context["pairs"]
    clean_pixels = pixel_space(pairs["clean"], dataset)  # (n, C, H, W)
    if context["attack"] == "wanet":
        faded = trigger_dose.warped(
            clean_pixels,
            context["metadata"],
            context["attack"],
            int(pairs["targets"][0]),
            dose,
            pairs,
        )
        return faded
    triggered_pixels = pixel_space(pairs["triggered"], dataset)  # (n, C, H, W)
    faded = clean_pixels + dose * (triggered_pixels - clean_pixels)  # (n, C, H, W)
    return faded


def sharpened(pixels, amount):
    if amount == 0.0:
        return pixels
    blurred = gaussian_blur(
        pixels, kernel_size=[BLUR_KERNEL, BLUR_KERNEL], sigma=[BLUR_SIGMA, BLUR_SIGMA]
    )  # (n, C, H, W)
    residual = pixels - blurred  # (n, C, H, W), the high-frequency part
    # Clipped because a sharpened pixel outside [0, 1] is an image no camera or
    # attacker could hand the model.
    result = (pixels + amount * residual).clamp(0.0, 1.0)  # (n, C, H, W)
    return result


def clean_accuracy(context, pixels, labels, amount, device):
    images = normalized(sharpened(pixels, amount), context["dataset"])
    logits = logits_of(context["model"], images, device)  # (n, classes)
    accuracy = float((logits.argmax(dim=1) == labels).float().mean())
    return accuracy


# The largest lambda whose clean accuracy stays within tolerance of lambda = 0,
# or None when even the smallest breaks it.
def choose_lambda(accuracy_by_lambda, tolerance):
    reference = accuracy_by_lambda[0.0]
    passing = [
        amount
        for amount, accuracy in accuracy_by_lambda.items()
        if amount > 0.0 and accuracy >= reference - tolerance
    ]
    chosen = max(passing) if passing else None
    return chosen


# Unperturbed prediction and confidence of every image, and at every rate its
# fractional PSU and the share of the k passes that kept the prediction. The mask
# stream is reseeded per rate, the way the dose code seeds each call.
@torch.inference_mode()
def readings(context, images, rates, device):
    model = context["model"]
    point = context["point"]
    base = F.softmax(logits_of(model, images, device), dim=1)  # (n, classes)
    confidence, prediction = base.max(dim=1)  # (n,), (n,)

    psu_by_rate = []
    kept_by_rate = []
    for rate in rates:
        handles = plug_dropout(
            model,
            context["architecture"],
            point["positions"],
            {p: point["factory"] for p in point["positions"]},
            rate,
        )
        try:
            seed_everything(MASK_SEED, verbose=False)
            pass_probs = [
                F.softmax(logits_of(model, images, device), dim=1)
                for _ in range(PASSES)
            ]  # k of (n, classes)
        finally:
            unplug_dropout(handles)
        tracked = torch.stack(
            [probs.gather(1, prediction[:, None])[:, 0] for probs in pass_probs]
        )  # (k, n)
        kept = torch.stack(
            [probs.argmax(dim=1) == prediction for probs in pass_probs]
        )  # (k, n)
        psu_by_rate.append(psu_ratio_from_cache(base, prediction, tracked))
        kept_by_rate.append(kept.float().mean(dim=0))

    result = {
        "prediction": prediction,
        "confidence": confidence,
        "psu": torch.stack(psu_by_rate),  # (rates, n)
        "kept": torch.stack(kept_by_rate),  # (rates, n)
    }
    return result


# The same per-input fields read from the canonical cache of a clean split.
def cached_readings(context, split, limit):
    psbd_dir = context["psbd_dir"]
    placement = context["point"]["placement"]
    probs, prediction, loader_labels = load_baseline(baseline_path(psbd_dir, split))
    count = min(limit or len(prediction), len(prediction))
    rows = torch.arange(count)
    result = rows_of_cache(psbd_dir, placement, split, context["rates"], rows)
    result["label"] = loader_labels[:count]
    return result


def cached_pair_readings(context):
    psbd_dir = context["psbd_dir"]
    placement = context["point"]["placement"]
    rates = [context["point"]["rate"]]
    twins = rows_of_cache(psbd_dir, placement, "clean", rates, context["twin_rows"])
    # The cached backdoor split holds the checkpoint's own trigger, or BadNets on
    # a benign checkpoint, so a blend or bpp probe has no cached triggered rows.
    probe_matches_cache = context["probe"] in (None, "badnet_a2o")
    triggered = (
        rows_of_cache(
            psbd_dir,
            placement,
            "backdoor",
            rates,
            torch.arange(len(context["twin_rows"])),
        )
        if probe_matches_cache
        else None
    )
    result = {"rate": rates[0], "twins": twins, "triggered": triggered}
    return result


def rows_of_cache(psbd_dir, placement, split, rates, rows):
    probs, prediction, _ = load_baseline(baseline_path(psbd_dir, split))
    probs = probs[rows]  # (n, classes)
    prediction = prediction[rows].long()  # (n,)
    psu_by_rate = []
    kept_by_rate = []
    for rate in rates:
        per_pass, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )  # (k, n_split), (k, n_split)
        psu_by_rate.append(psu_ratio_from_cache(probs, prediction, per_pass[:, rows]))
        kept_by_rate.append((argmax[:, rows].long() == prediction).float().mean(dim=0))
    result = {
        "prediction": prediction,
        "confidence": probs.max(dim=1).values.float(),
        "psu": torch.stack(psu_by_rate),  # (rates, n)
        "kept": torch.stack(kept_by_rate),  # (rates, n)
    }
    return result


def meta(context, args):
    pairs = context["pairs"]
    record = {
        "folder": context["folder"],
        "probe_attack": context["probe"],
        "attack": context["attack"],
        "dataset": context["dataset"],
        "architecture": context["architecture"],
        "target": int(pairs["targets"][0]),
        "backdoored": context["metadata"]["attack"] != "benign",
        "placement": context["point"]["placement"],
        "passes": PASSES,
        "mask_seed": MASK_SEED,
        "doses": list(DOSES),
        "pairs": len(pairs["targets"]),
        "smoke": bool(
            args.max_validation
            or args.max_clean
            or args.rates
            or args.clean_accuracy_tolerance != CLEAN_ACCURACY_TOLERANCE
        ),
        "git_commit": git_commit(),
        "blur_kernel": BLUR_KERNEL,
        "blur_sigma": BLUR_SIGMA,
        "clean_accuracy_tolerance": args.clean_accuracy_tolerance,
    }
    return record


def git_commit():
    completed = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    commit = completed.stdout.strip() or None
    return commit


def save_record(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = f"{path}.partial"
    torch.save(record, partial)
    os.replace(partial, path)


if __name__ == "__main__":
    main()

"""Sanity gates for measure.py, run on 1 model per architecture before any full GPU run.

    1  clean accuracy and attack success measured through measure.py's forward path
       match the checkpoint's recorded metrics within 0.01
    2  the shift ratio measure.py's own perturbation path produces on the clean
       validation split matches the stage-1 cache at the same rates, for PSBD-RD
       and PSBD-TM, and the ResNet-18 probe sits after each residual add
    3  the triggered set is the PSBD split's AttackSuccessSet and every clean and
       triggered image is the same test image
    4  the model is in eval mode, owns no active dropout, the hooks come off and
       the perturbation is resampled per pass
    5  gate 2 repeated in float32 on ViT, since every cache was written in bfloat16

    flock scratch/gpu.lock env PYTHONPATH=. python \\
        experiments/prediction_shift_phenomenon/sanity.py --folder F [--float32]
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from attacks.poisoning import AttackSuccessSet  # noqa: E402
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
)
from defenses.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from defenses.scores import shift_ratio  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.prediction_shift_phenomenon.measure import (  # noqa: E402
    forward_features,
    limit_gpu_memory,
    plug_condition,
)
from analysis.features import transformer_blocks  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import unplug_dropout  # noqa: E402

SLUG = "prediction_shift_phenomenon"
METRIC_TOLERANCE = 0.01
# 3 passes over 2000 images is 6000 draws, whose binomial standard error is at
# most 0.0065. Masks are drawn in a different batch order from the sweep's, so
# only the distribution can agree, and 0.03 is about 4.5 standard errors.
SHIFT_TOLERANCE = 0.03
GATE_PASSES = 3
RESIDUAL_GATE_RATES = {"resnet18": (0.3, 0.5, 0.7), "vit": (0.03, 0.07, 0.09)}
RESIDUAL_GATE_RATES["swin"] = RESIDUAL_GATE_RATES["vit"]
TOKEN_GATE_RATES = (0.3, 0.5, 0.7)
CHUNK = 256
MAP_IMAGES = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--only-shift", action="store_true")
    parser.add_argument("--only-state", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    use_bfloat16 = not args.float32

    checkpoint_path = os.path.join(
        args.checkpoints_dir, args.folder, "attack_result.pt"
    )
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device).eval()
    last_layer = len(transformer_blocks(network_core(model), architecture))
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    splits = {name: stack_split(loader.dataset) for name, loader in loaders.items()}
    context = (model, architecture, last_layer, args.batch_size, device, use_bfloat16)

    record = {
        "folder": args.folder,
        "architecture": architecture,
        "precision": "bfloat16" if use_bfloat16 else "float32",
    }
    if args.only_state:
        record["gate_4_state"] = gate_model_state(context, splits["validation"])
    else:
        record["gate_2_shift_ratio"] = gate_shift_ratio(
            context, splits["validation"], args.results_dir, args.folder
        )
    if not args.only_shift and not args.only_state:
        record["gate_1_metrics"] = gate_metrics(
            context, splits, args.checkpoints_dir, args.folder
        )
        record["gate_3_pairs"] = gate_pairs(loaders, manifest, splits)
        record["gate_4_state"] = gate_model_state(context, splits["validation"])

    suffix = "_float32" if args.float32 else ""
    suffix = "_state" if args.only_state else suffix
    output_dir = os.path.join(experiment_results_dir(SLUG, args.output_root), "sanity")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{args.folder}{suffix}.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps(record, indent=1))


def stack_split(dataset):
    images, labels = [], []
    for index in range(len(dataset)):
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
    split = {
        "images": torch.stack(images),  # (n, 3, height, width)
        "labels": torch.tensor(labels),  # (n,)
    }
    return split


def predict(context, images, condition, seed):
    model, architecture, last_layer, batch_size, device, use_bfloat16 = context
    handles = plug_condition(model, architecture, condition)
    try:
        parts = [
            forward_features(
                model,
                architecture,
                last_layer,
                images[start : start + CHUNK],
                seed * 7919 + start,
                batch_size,
                device,
                use_bfloat16,
            )["predictions"]
            for start in range(0, len(images), CHUNK)
        ]
    finally:
        unplug_dropout(handles)
    predictions = torch.cat(parts)  # (n,)
    return predictions


def perturbed_maps(context, images, condition, seed):
    model, architecture, last_layer, batch_size, device, use_bfloat16 = context
    handles = plug_condition(model, architecture, condition)
    try:
        maps = forward_features(
            model,
            architecture,
            last_layer,
            images,
            seed,
            batch_size,
            device,
            use_bfloat16,
        )["maps"]  # (n, channels, height, width)
    finally:
        unplug_dropout(handles)
    return maps


def gate_metrics(context, splits, checkpoints_dir, folder):
    unperturbed = {"placement": None, "rate": 0.0}
    correct = []
    for name in ("validation", "clean"):
        predictions = predict(context, splits[name]["images"], unperturbed, 0)
        correct.append(predictions == splits[name]["labels"])
    clean_accuracy = float(torch.cat(correct).float().mean())

    triggered = predict(context, splits["backdoor"]["images"], unperturbed, 0)
    attack_success = float((triggered == splits["backdoor"]["labels"]).float().mean())

    recorded = recorded_metrics(checkpoints_dir, folder)
    result = {
        "clean_accuracy": clean_accuracy,
        "attack_success_rate": attack_success,
        "recorded": recorded,
        "clean_accuracy_gap": abs(clean_accuracy - recorded["clean_accuracy"]),
        "attack_success_gap": abs(attack_success - recorded["asr"]),
    }
    result["passes"] = (
        result["clean_accuracy_gap"] <= METRIC_TOLERANCE
        and result["attack_success_gap"] <= METRIC_TOLERANCE
    )
    return result


def recorded_metrics(checkpoints_dir, folder):
    # ResNet-18 checkpoints carry no metrics.json. Their args.json holds the
    # numbers training measured on the same test set.
    for name in ("metrics.json", "args.json"):
        path = os.path.join(checkpoints_dir, folder, name)
        if os.path.exists(path):
            with open(path) as handle:
                payload = json.load(handle)
            if payload.get("asr") is not None:
                recorded = {
                    "source": name,
                    "clean_accuracy": payload["clean_accuracy"],
                    "asr": payload["asr"],
                }
                return recorded
    raise FileNotFoundError(f"no recorded metrics for {folder}")


def gate_shift_ratio(context, validation, results_dir, folder):
    architecture = context[1]
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    _, cached_baseline_labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
    unperturbed = predict(
        context, validation["images"], {"placement": None, "rate": 0.0}, 0
    )
    baseline_agreement = float((unperturbed == cached_baseline_labels).float().mean())

    placements = [(PUBLISHED_PLACEMENT, RESIDUAL_GATE_RATES[architecture])]
    if architecture != "resnet18":
        placements.append((RECOMMENDED_PLACEMENT, TOKEN_GATE_RATES))

    rows = []
    for placement, rates in placements:
        for rate in rates:
            condition = {"placement": placement, "rate": rate}
            passes = torch.stack(
                [
                    predict(context, validation["images"], condition, draw + 1)
                    for draw in range(GATE_PASSES)
                ]
            )  # (passes, n)
            ours = shift_ratio(unperturbed, passes)
            _, cached_argmax = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, "validation")
            )
            cached = shift_ratio(cached_baseline_labels, cached_argmax)
            rows.append(
                {
                    "placement": placement,
                    "rate": rate,
                    "ours": ours,
                    "cached": cached,
                    "gap": abs(ours - cached),
                }
            )

    result = {
        "baseline_prediction_agreement": baseline_agreement,
        "rows": rows,
        "tolerance": SHIFT_TOLERANCE,
        "passes": all(row["gap"] <= SHIFT_TOLERANCE for row in rows),
    }
    if architecture == "resnet18":
        result["probe_sites"] = resnet_probe_sites(context)
    return result


def resnet_probe_sites(context):
    # Li et al. place dropout "after each residual connection in the residual
    # basic block, before the activation function". 1 probe per BasicBlock, and
    # a block's output is a ReLU of the probe's output, so it is never negative.
    model, architecture = context[0], context[1]
    condition = {"placement": PUBLISHED_PLACEMENT, "rate": 0.5}
    handles = plug_condition(model, architecture, condition)
    blocks = transformer_blocks(network_core(model), architecture)
    wrapped = sum("forward" in vars(block) for block in blocks)
    unplug_dropout(handles)
    sites = {"basic_blocks": len(blocks), "probes": len(handles), "wrapped": wrapped}
    return sites


def gate_pairs(loaders, manifest, splits):
    backdoor_set = loaders["backdoor"].dataset
    clean_indices = manifest["analysis_clean_indices"]
    backdoor_indices = manifest["analysis_backdoor_indices"]
    row_of = {original: row for row, original in enumerate(clean_indices)}
    clean_rows = torch.tensor([row_of[original] for original in backdoor_indices])

    clean_images = splits["clean"]["images"][clean_rows]  # (n_backdoor, 3, h, w)
    triggered_images = splits["backdoor"]["images"]
    changed = (clean_images - triggered_images).abs().amax(dim=1) > 1e-6  # (n, h, w)
    changed_share = changed.flatten(1).float().mean(dim=1)  # (n,)
    correlation = batch_correlation(clean_images, triggered_images)  # (n,)

    result = {
        "backdoor_set_class": type(backdoor_set).__name__,
        "is_attack_success_set": isinstance(backdoor_set, AttackSuccessSet),
        "pairs": int(len(backdoor_indices)),
        "backdoor_labels": sorted(set(splits["backdoor"]["labels"].tolist())),
        "clean_labels_of_pairs_include_target": bool(
            (
                splits["clean"]["labels"][clean_rows] == splits["backdoor"]["labels"]
            ).any()
        ),
        "changed_pixel_share_mean": float(changed_share.mean()),
        "changed_pixel_share_max": float(changed_share.max()),
        "pixel_correlation_min": float(correlation.min()),
        "pixel_correlation_mean": float(correlation.mean()),
    }
    result["passes"] = (
        result["is_attack_success_set"]
        and not result["clean_labels_of_pairs_include_target"]
    )
    return result


def batch_correlation(first, second):
    first_flat = first.flatten(1) - first.flatten(1).mean(dim=1, keepdim=True)
    second_flat = second.flatten(1) - second.flatten(1).mean(dim=1, keepdim=True)
    numerator = (first_flat * second_flat).sum(dim=1)  # (n,)
    denominator = first_flat.norm(dim=1) * second_flat.norm(dim=1)
    correlation = numerator / denominator.clamp_min(1e-12)
    return correlation


def gate_model_state(context, validation):
    model, architecture = context[0], context[1]
    images = validation["images"][:CHUNK]
    training_modules = [
        name for name, module in model.named_modules() if module.training
    ]
    owned_dropouts = {
        name: module.p
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout)
    }
    hooks_before = hook_count(model)

    unperturbed = {"placement": None, "rate": 0.0}
    logits_before = predict(context, images, unperturbed, 0)
    readings = {}
    for placement in placements_of(architecture):
        condition = {"placement": placement, "rate": 0.5}
        # Features rather than predictions, because at a saturating rate every
        # pass sends every image to the same class whatever the mask was.
        first = perturbed_maps(context, images[:MAP_IMAGES], condition, 1)
        repeat = perturbed_maps(context, images[:MAP_IMAGES], condition, 1)
        second = perturbed_maps(context, images[:MAP_IMAGES], condition, 2)
        readings[placement] = {
            "same_seed_identical": bool((first == repeat).all()),
            "new_seed_differs_share": float((first != second).float().mean()),
        }
    logits_after = predict(context, images, unperturbed, 0)
    hooks_after = hook_count(model)
    wrapped_after = sum("forward" in vars(module) for module in model.modules())

    result = {
        "modules_in_training_mode": training_modules,
        "model_owned_dropouts": owned_dropouts,
        "hooks_before": hooks_before,
        "hooks_after": hooks_after,
        "forward_wrappers_left": wrapped_after,
        "unperturbed_predictions_unchanged": bool(
            (logits_before == logits_after).all()
        ),
        "resampling": readings,
    }
    result["passes"] = (
        not training_modules
        and hooks_before == hooks_after
        and wrapped_after == 0
        and result["unperturbed_predictions_unchanged"]
        and all(
            reading["same_seed_identical"] and reading["new_seed_differs_share"] > 0
            for reading in readings.values()
        )
    )
    return result


def placements_of(architecture):
    if architecture == "resnet18":
        return (PUBLISHED_PLACEMENT,)
    return (PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT)


def hook_count(model):
    count = sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )
    return count


if __name__ == "__main__":
    main()

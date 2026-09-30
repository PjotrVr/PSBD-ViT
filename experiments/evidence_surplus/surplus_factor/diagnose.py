"""The 2 diagnostic measurements the failure protocol asked for, each targeting 1 failed prediction.

head       Predictions 1 to 3 of A failed at the last block's output, which sits
           before the final LayerNorm. LayerNorm rescales every token to unit
           variance, so a shift measured before it need not be the shift the head
           reads. This repeats A's surplus factor at the head input, after the
           final LayerNorm, where the logits are linear in the feature.

survival   Prediction 9 of A2 failed: at block 8, PSBD-TM did not respond more to
           all-token steering than to class-token-only steering. This measures how
           much of the added own-class evidence reaches the class token at block
           12, with no probe and averaged over PSBD-TM's 3 masked passes, for the 2
           steering variants.

    flock scratch/gpu.lock env PYTHONPATH=. python experiments/evidence_surplus/surplus_factor/diagnose.py head --model vit_cifar10_blend_0_1
    flock scratch/gpu.lock env PYTHONPATH=. python experiments/evidence_surplus/surplus_factor/diagnose.py survival --model vit_cifar10_benign
"""

import argparse
import json
import os

import torch
from lightning import seed_everything

from analysis.features import captured_layers, transformer_blocks
from cli.sweep import PSBD_MASK_SEED, bound_operator
from defenses.inference import forward_logits
from experiments._paths import experiment_results_dir
from experiments.backdoor_manifestation.measure import checkpoint_path
from experiments.evidence_surplus.surplus_factor.manufactured import (
    CACHE,
    class_offsets,
    steer_hook,
)
from experiments.evidence_surplus.surplus_factor.measure import (
    EVAL_IMAGES,
    SEARCH_FACTOR,
    pair_path,
    panel_runs,
    surplus_record,
)
from models.backbones import load_checkpoint, network_core
from models.positions import plug_dropout, unplug_dropout

SLUG = "evidence_surplus"
BISECTION_STEPS = 12
BATCH_SIZE = 200
GPU_MEMORY_FRACTION = 0.35
SEED = 0
SURVIVAL_BLOCK = 8
READ_BLOCK = 12
SURVIVAL_SURPLUS = (2.0, 4.0)
FORWARD_PASSES = 3


def main():
    args = parse_args()
    seed_everything(SEED)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    device = torch.device("cuda")
    record = (
        head_surplus(args.model, device)
        if args.part == "head"
        else survival(args.model, device)
    )
    directory = os.path.join(
        experiment_results_dir(SLUG, args.results_dir), "diagnostics", args.part
    )
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{args.model}.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps(record)[:600])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("head", "survival"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


@torch.inference_mode()
def head_inputs(model, head, images, device):
    store, features, predictions = {}, [], []
    handle = head.register_forward_pre_hook(
        lambda _module, inputs: store.__setitem__("x", inputs[0])
    )
    try:
        for batch in images.split(BATCH_SIZE):
            predictions.append(
                forward_logits(model, batch, device, True).argmax(dim=1).cpu()
            )
            features.append(store["x"].float().cpu())
    finally:
        handle.remove()
    stacked = torch.cat(features)  # (N, dim)
    all_predictions = torch.cat(predictions)  # (N,)
    return stacked, all_predictions


@torch.inference_mode()
def lands_at_head(model, head, unit, images, scales, target, device):
    def pre_hook(_module, inputs):
        shifted = inputs[0] + scales.to(inputs[0].device, inputs[0].dtype)[
            :, None
        ] * unit.to(inputs[0].device, inputs[0].dtype)
        return (shifted,)

    handle = head.register_forward_pre_hook(pre_hook)
    try:
        logits = forward_logits(model, images, device, True)  # (batch, classes)
    finally:
        handle.remove()
    landed = (logits.argmax(dim=1) == target).cpu()  # (batch,)
    return landed


def head_surplus(folder_or_run, device):
    """A's surplus factor with the direction fitted and added at the head input, after the final LayerNorm."""
    run = next(run for run in panel_runs() if run["name"] == folder_or_run)
    pairs = torch.load(pair_path(run["name"]))
    model = load_checkpoint(run["architecture"], checkpoint_path(run["folder"]), device)
    head = network_core(model).heads.head
    clean, clean_predictions = head_inputs(model, head, pairs["clean"], device)
    triggered, _ = head_inputs(model, head, pairs["triggered"], device)

    half = clean.shape[0] // 2
    eval_rows = torch.arange(half, min(clean.shape[0], half + EVAL_IMAGES))
    direction = (triggered[:half] - clean[:half]).mean(dim=0)  # (dim,)
    unit = direction / direction.norm().clamp_min(1e-8)
    actual = (triggered[eval_rows] - clean[eval_rows]) @ unit  # (n,)
    off_target = clean_predictions[eval_rows] != run["target"]
    limit = SEARCH_FACTOR * float(actual.median().clamp_min(1e-6))

    images = pairs["clean"][eval_rows][off_target]
    needed_chunks = []
    for batch in images.split(BATCH_SIZE):
        count = batch.shape[0]
        high = torch.full((count,), limit)
        reachable = lands_at_head(model, head, unit, batch, high, run["target"], device)
        low = torch.zeros(count)
        for _ in range(BISECTION_STEPS):
            middle = (low + high) / 2
            landed = lands_at_head(
                model, head, unit, batch, middle, run["target"], device
            )
            high = torch.where(landed, middle, high)
            low = torch.where(landed, low, middle)
        needed_chunks.append(
            torch.where(reachable, high, torch.full((count,), float("nan")))
        )
    needed = torch.cat(needed_chunks)  # (n_off,)

    record = {
        "run": run,
        "site": "head input, after the final LayerNorm",
        **surplus_record(actual[off_target], needed),
    }
    return record


@torch.inference_mode()
def survival(folder, device):
    """Own-class evidence reaching the class token at block 12 after steering at block 8, with and without PSBD-TM."""
    data = torch.load(os.path.join(CACHE, f"{folder}.pt"))
    model = load_checkpoint("vit", checkpoint_path(folder), device)
    blocks = transformer_blocks(network_core(model), "vit")
    with open(os.path.join("results", folder, "psbd_metrics.json")) as handle:
        rate = float(
            json.load(handle)["placements"]["before_attention_norm_token_mask"][
                "adaptive_rate"
            ]
        )

    steer_offsets, steer_norms = class_offsets(
        model, data["fit"], data["fit_labels"], SURVIVAL_BLOCK, device
    )
    read_offsets, read_norms = class_offsets(
        model, data["fit"], data["fit_labels"], READ_BLOCK, device
    )
    images = data["steer"]
    predictions = torch.cat(
        [
            forward_logits(model, batch, device, True).argmax(dim=1).cpu()
            for batch in images.split(BATCH_SIZE)
        ]
    )  # (n,)
    read_units = read_offsets[predictions] / read_norms[predictions, None]  # (n, dim)

    def evidence(steer, masked):
        """Mean own-class projection of the block-12 class token, over passes when masked."""
        rows = {"start": 0}
        handles = []
        if steer is not None:
            handles.append(
                blocks[SURVIVAL_BLOCK - 1].register_forward_hook(
                    steer_hook(steer[0], steer[1], rows)
                )
            )
        probes = (
            plug_dropout(
                model,
                "vit",
                ("before_attention_norm",),
                {"before_attention_norm": bound_operator("token_mask", None)},
                rate,
            )
            if masked
            else []
        )
        passes = FORWARD_PASSES if masked else 1
        try:
            seed_everything(PSBD_MASK_SEED)
            totals = torch.zeros(images.shape[0])
            for _ in range(passes):
                with captured_layers(model, (READ_BLOCK,), "vit") as captured:
                    for start in range(0, images.shape[0], BATCH_SIZE):
                        rows["start"] = start
                        forward_logits(
                            model, images[start : start + BATCH_SIZE], device, True
                        )
                        token = (
                            captured[READ_BLOCK][:, 0, :].float().cpu()
                        )  # (batch, dim)
                        units = read_units[start : start + BATCH_SIZE]
                        totals[start : start + BATCH_SIZE] += (token * units).sum(dim=1)
        finally:
            if probes:
                unplug_dropout(probes)
            for handle in handles:
                handle.remove()
        mean_evidence = totals / passes  # (n,)
        return mean_evidence

    base = {masked: evidence(None, masked) for masked in (False, True)}
    rows_out = []
    for surplus in SURVIVAL_SURPLUS:
        lengths = (surplus - 1.0) * steer_norms[predictions]  # (n,)
        shifts = (
            steer_offsets[predictions]
            / steer_norms[predictions, None]
            * lengths[:, None]
        )  # (n, dim)
        for variant, class_token_only in (
            ("own_all_tokens", False),
            ("own_class_token", True),
        ):
            added = {
                masked: evidence((shifts, class_token_only), masked) - base[masked]
                for masked in (False, True)
            }
            rows_out.append(
                {
                    "surplus": surplus,
                    "variant": variant,
                    "median_added_evidence_unmasked": float(added[False].median()),
                    "median_added_evidence_masked": float(added[True].median()),
                    "median_surviving_share": float(
                        (added[True] / added[False].clamp_min(1e-6)).median()
                    ),
                }
            )
    record = {
        "model": folder,
        "steer_block": SURVIVAL_BLOCK,
        "read_block": READ_BLOCK,
        "adaptive_rate": rate,
        "median_base_evidence_unmasked": float(base[False].median()),
        "median_base_evidence_masked": float(base[True].median()),
        "rows": rows_out,
    }
    return record


if __name__ == "__main__":
    main()

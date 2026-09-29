"""Sanity gates the probe stage must pass on 1 model before the panel runs.

1. Clean accuracy and ASR read through measure.batch_readout on the whole PSBD
   analysis split match checkpoints/<folder>/metrics.json within 0.01.
2. The tensor measure.py reads as layer l is block l's output. It is compared
   against a forward that walks the blocks by hand, and the head input it keeps
   is compared against the final norm applied to the last block's output.
3. The cached pairs are the PSBD split's eligible images, paired: every pair's
   clean label is eligible under is_eval_poisonable, the manifest maps each
   backdoor row to the clean row of the same test image, and for a patch trigger
   the 2 images differ only inside the trigger.

Run under the GPU lock, 1 model per call:
    flock scratch/gpu.lock env PYTHONPATH=. python experiments/backdoor_manifestation/sanity.py --run vit_cifar10_badnet_a2o_0_05
"""

import argparse
import json
import os

import torch
from lightning import seed_everything

from analysis.features import captured_layers, transformer_blocks
from attacks.poisoning import is_eval_poisonable
from data.splits import build_psbd_loaders_from_checkpoint
from experiments._paths import experiment_results_dir
from experiments.backdoor_manifestation.measure import (
    GPU_MEMORY_FRACTION,
    PAIR_CACHE,
    SEED,
    SLUG,
    batch_readout,
    checkpoint_path,
    panel_runs,
)
from models.backbones import load_checkpoint, network_core

TOLERANCE = 0.01
CHECK_IMAGES = 8


def main():
    args = parse_args()
    seed_everything(SEED)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    run = next(run for run in panel_runs() if run["name"] == args.run)

    model = load_checkpoint(run["architecture"], checkpoint_path(run["folder"]), device)
    core = network_core(model)
    blocks = transformer_blocks(core, run["architecture"])
    head = core.heads.head if run["architecture"] == "vit" else core.head
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path(run["folder"]),
        batch_size=64,
        num_workers=8,
        probe_attack=run["probe_attack"],
        probe_target_label=run["probe_target"],
    )

    record = {
        "run": run["name"],
        "gate_1_behavior": behavior_gate(model, run, blocks, head, loaders, device),
        "gate_2_layers": layer_gate(model, core, run, blocks, head, loaders, device),
        "gate_3_pairs": pair_gate(run, manifest),
    }
    directory = os.path.join(experiment_results_dir(SLUG), "sanity")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{run['name']}.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps(record, indent=1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    return parser.parse_args()


def behavior_gate(model, run, blocks, head, loaders, device):
    clean_correct, clean_total = 0, 0
    for images, labels in loaders["clean"]:
        predictions = batch_readout(model, run, blocks, head, images, device)[
            "predictions"
        ].cpu()  # (batch,)
        clean_correct += int((predictions == labels).sum())
        clean_total += labels.shape[0]

    hits, triggered_total = 0, 0
    for images, intended in loaders["backdoor"]:
        predictions = batch_readout(model, run, blocks, head, images, device)[
            "predictions"
        ].cpu()  # (batch,)
        hits += int((predictions == intended).sum())
        triggered_total += intended.shape[0]

    with open(os.path.join("checkpoints", run["folder"], "metrics.json")) as handle:
        reference = json.load(handle)
    measured_accuracy = clean_correct / clean_total
    measured_asr = hits / triggered_total
    record = {
        "clean_accuracy": measured_accuracy,
        "asr": measured_asr,
        "n_clean": clean_total,
        "n_triggered": triggered_total,
        "metrics_json_clean_accuracy": reference["clean_accuracy"],
        "metrics_json_asr": reference["asr"],
        "passes": abs(measured_accuracy - reference["clean_accuracy"]) <= TOLERANCE
        and abs(measured_asr - reference["asr"]) <= TOLERANCE,
    }
    return record


@torch.inference_mode()
def layer_gate(model, core, run, blocks, head, loaders, device):
    images, _ = next(iter(loaders["clean"]))
    images = images[:CHECK_IMAGES]  # (k, C, H, W)
    layers = tuple(range(len(blocks) + 1))

    # fp32 on both sides, so the comparison is exact up to kernel order.
    with captured_layers(model, layers, run["architecture"]) as captured:
        model(images.to(device))
        library = {layer: captured[layer].float().clone() for layer in layers}
    readout = batch_readout(model, run, blocks, head, images, device, False)
    manual = manual_block_outputs(model, core, run["architecture"], images, device)
    manual_head_input = manual_head(core, run["architecture"], manual[len(blocks)])

    record = {
        "shapes": {str(layer): list(library[layer].shape) for layer in layers},
        "max_abs_difference_to_manual": max(
            float((library[layer] - manual[layer]).abs().max()) for layer in layers
        ),
        "max_abs_difference_head_input": float(
            (readout["head"] - manual_head_input.float()).abs().max()
        ),
        "pooled_is_class_token_or_mean": all(
            torch.allclose(
                readout["pooled"][layer].cpu(),
                library[layer][:, 0].cpu()
                if run["architecture"] == "vit"
                else library[layer].flatten(1, 2).mean(dim=1).cpu(),
            )
            for layer in layers
        ),
    }
    record["passes"] = (
        record["max_abs_difference_to_manual"] < 1e-3
        and record["max_abs_difference_head_input"] < 1e-3
        and record["pooled_is_class_token_or_mean"]
    )
    return record


def manual_block_outputs(model, core, architecture, images, device):
    """Layer 0 is the first block's input, layer l the output of block l, walked by hand."""
    resized = model[0](images.to(device))  # (k, C, 224, 224)
    outputs = {}
    if architecture == "vit":
        tokens = core._process_input(resized)  # (k, 196, 768)
        class_token = core.class_token.expand(tokens.shape[0], -1, -1)  # (k, 1, 768)
        x = torch.cat([class_token, tokens], dim=1) + core.encoder.pos_embedding
        x = core.encoder.dropout(x)  # (k, 197, 768)
        outputs[0] = x.float().clone()
        for index, block in enumerate(core.encoder.layers, start=1):
            x = block(x)
            outputs[index] = x.float().clone()
        return outputs

    x = resized
    index = 0
    for stage in core.features:
        members = list(stage) if isinstance(stage, torch.nn.Sequential) else [stage]
        for member in members:
            if type(member).__name__.startswith("SwinTransformerBlock"):
                if index == 0:
                    outputs[0] = x.float().clone()
                x = member(x)
                index += 1
                outputs[index] = x.float().clone()
            else:
                x = member(x)
    return outputs


def manual_head(core, architecture, last_block_output):
    if architecture == "vit":
        head_input = core.encoder.ln(last_block_output)[:, 0]  # (k, 768)
        return head_input
    normalized = core.norm(last_block_output)  # (k, 7, 7, 768)
    head_input = normalized.mean(dim=(1, 2))  # (k, 768)
    return head_input


def pair_gate(run, manifest):
    pairs = torch.load(os.path.join(PAIR_CACHE, f"{run['name']}.pt"))
    label_mode = manifest["label_mode"]
    target = manifest["probe_target_label"]
    eligible = all(
        is_eval_poisonable(label_mode, int(label), target) for label in pairs["labels"]
    )
    backdoor_indices = set(manifest["analysis_backdoor_indices"])
    in_split = all(int(index) in backdoor_indices for index in pairs["test_indices"])

    difference = (pairs["triggered"] - pairs["clean"]).abs().amax(dim=1)  # (N, H, W)
    changed = (difference > 1e-6).float().mean(dim=0)  # (H, W)
    record = {
        "n_pairs": int(pairs["clean"].shape[0]),
        "all_labels_eval_poisonable": eligible,
        "all_test_indices_in_backdoor_split": in_split,
        "share_of_pixels_ever_changed": float((changed > 0).float().mean()),
        "share_of_pixels_changed_in_every_pair": float((changed == 1).float().mean()),
        "label_mode": label_mode,
        "target": target,
    }
    record["passes"] = eligible and in_split
    return record


if __name__ == "__main__":
    main()

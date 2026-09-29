"""How a backdoor manifests inside a ViT-B/16 and a Swin-S, and how to find it.

1 run is 1 checkpoint probed with 1 trigger: a backdoored model with its own
trigger, or a benign model of the same dataset with that trigger as the control.
3 stages, so the GPU lock is held only while a model is on the GPU.

    prepare    CPU. Builds the paired eligible images through the PSBD split and
               caches them under scratch/backdoor_manifestation/pairs/.
    probe      GPU. Reads every block of the residual stream for the pairs, the
               class token's attention on ViT, and runs the ablations. Writes
               results/_experiments/backdoor_manifestation/runs/<run>.json and
               caches the head features for the embeddings.
    summarize  CPU. PCA and UMAP of the head features, cross-attack overlap of the
               top TAC dimensions, and 1 row per run in summary.json.

The pairs are split in 2 halves after a seeded shuffle. Every direction, TAC
ranking and neuron choice is fitted on the first half and every separability or
ablation number is read on the second, so a direction cannot score well by
having memorised its own samples.

The forward runs in bfloat16 by the GPU rule of this run. The benign control is
measured with the same precision, so any bfloat16 floor in TAC or in the direction
norm is present in the control too and a backdoored model is read against it.

Example
    PYTHONPATH=. python experiments/backdoor_manifestation/measure.py prepare --run vit_cifar10_badnet_a2o_0_05
    flock scratch/gpu.lock env PYTHONPATH=. python experiments/backdoor_manifestation/measure.py probe --run vit_cifar10_badnet_a2o_0_05
    PYTHONPATH=. python experiments/backdoor_manifestation/measure.py summarize
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from analysis.cka import debiased_linear_cka
from analysis.direction import outlier_dimensions
from analysis.features import as_token_sequence, captured_layers, transformer_blocks
from analysis.samples import collect_split
from attacks import build_attack, default_config
from data.registry import DATASET_REGISTRY
from data.splits import build_psbd_loaders_from_checkpoint, read_checkpoint_metadata
from defenses.decision import pair_clean_to_backdoor
from defenses.inference import forward_logits
from experiments._paths import experiment_results_dir
from models.backbones import load_checkpoint, network_core

SLUG = "backdoor_manifestation"
PAIR_CACHE = os.path.join("scratch", SLUG, "pairs")
FEATURE_CACHE = os.path.join("scratch", SLUG, "features")

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
ARCHITECTURES = ("vit", "swin")
CATEGORY = {
    "badnet_a2o": "patch",
    "tact": "patch",
    "blend": "blend",
    "lf": "blend",
    "sig": "frequency",
    "wanet": "warp",
    "bpp": "quantization",
}

# 5% is the rate every attack reaches on every dataset for 4 attacks. Every cell
# must be successful at the 2-point bar (ASR at least 0.85 and clean accuracy
# within 2 points of the benign model), which backdoored_run enforces. No ViT
# WaNet model is successful on CIFAR-100 or GTSRB and the CIFAR-10 one is at
# 10%, and TaCT is trigger-conditional on 3 ViT models only, so those cells are
# named individually.
# SIG is left out. attacks/sig.py builds amplitude 0.157 since 2026-09-09 and the
# ViT SIG checkpoints learned 0.1 with no override in their args.json, so a
# rebuilt SIG trigger is not the trigger those models were trained on
# (docs/audits/2026-09-29-experiment-audit.md).
BACKDOORED = [
    *[
        f"{architecture}_{dataset}_{attack}_0_05"
        for attack in ("badnet_a2o", "blend", "lf", "bpp")
        for architecture in ARCHITECTURES
        for dataset in DATASETS
    ],
    "vit_cifar10_tact_0_01",
    "vit_cifar10_tact_0_05",
    "vit_gtsrb_tact_0_05",
    "swin_cifar10_tact_0_05",
    "vit_cifar10_wanet_0_1",
    "vit_tiny_wanet_0_05",
    "swin_cifar10_wanet_0_05",
    "swin_cifar100_wanet_0_1",
    "swin_gtsrb_wanet_0_1",
    "swin_tiny_wanet_0_05",
]

MAX_PAIRS = 800
TARGET_IMAGES = 200
EXAMPLE_PAIRS = 3
BATCH_SIZE = 128
# 14 GB of the shared 40 GB A100, the per-process share of the GPU rule, so 2
# jobs from the queue still leave about 12 GB for anyone else.
GPU_MEMORY_FRACTION = 0.35
TOP_DIMENSIONS = 20
NEURON_COUNTS = (20, 100, 300)
RANDOM_DIRECTIONS = 3
HISTOGRAM_BINS = 40
TOKEN_MAP_DIMENSIONS = 4
# The trigger whose direction serves as the "another attack" control: a global
# trigger for the patch and warp attacks and a patch trigger for the global ones,
# so the control direction is never the backdoor's own kind.
OTHER_TRIGGER = {
    "badnet_a2o": "blend",
    "tact": "blend",
    "blend": "badnet_a2o",
    "lf": "badnet_a2o",
    "wanet": "blend",
    "bpp": "badnet_a2o",
}
# A token counts as a trigger token when its mean input change is at least this
# share of the largest one. Patch triggers light 1 to 4 tokens, global triggers
# light most of the grid, which is what the attention reading needs to separate.
TRIGGER_TOKEN_SHARE = 0.25
# ViT-B/16 reads a 224 input as a 14 by 14 grid of 16 pixel patches.
PATCH_GRID = 14
EMBEDDING_POINTS = 300
SEED = 0
# The coverage ledger scores ViT cells only. A Swin TaCT run is screened by
# measurement instead, with the rule experiments/why_token_masking_works uses.
LEDGER = {
    cell["folder_name"]: cell
    for cell in json.load(open(os.path.join("results", "coverage", "coverage.json")))[
        "cells"
    ]
}
ASR_BAR = 0.85
MATCHED_PC_MAX_COSINE = 0.2
CLEAN_ACCURACY_BAR = 0.02
SOURCE_MAPPED_CLEAN_ACCURACY = 0.5


def main():
    args = parse_args()
    if args.stage == "list":
        print("\n".join(run["name"] for run in panel_runs()))
        return
    if args.stage == "summarize":
        summarize(args.results_dir)
        return

    run = next(run for run in panel_runs() if run["name"] == args.run)
    if args.stage == "prepare":
        prepare(run, args.raw_data_dir)
        return
    probe(run, args.results_dir, torch.device(args.device), args.max_pairs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "prepare", "probe", "summarize"))
    parser.add_argument("--run", default=None)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    # A CPU smoke run of the whole probe stage on a few pairs, before the GPU window.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    return parser.parse_args()


def panel_runs():
    """1 model family at a time, benign controls before the backdoored runs of the family.

    A backdoored run reads its benign control's direction, so the control has to
    run first. Grouping by family puts the first backdoored run early in the
    queue rather than after every control. A benign model is probed only with
    the triggers of the family's backdoored runs, since a control with no
    backdoored twin reads nothing.
    """
    runs = []
    for architecture in ARCHITECTURES:
        for dataset in DATASETS:
            family = f"{architecture}_{dataset}_"
            backdoored = [
                backdoored_run(folder)
                for folder in BACKDOORED
                if folder.startswith(family)
            ]
            attacks = sorted(
                {run["attack"] for run in backdoored}, key=list(CATEGORY).index
            )
            runs += [benign_run(architecture, dataset, attack) for attack in attacks]
            runs += backdoored
    return runs


def backdoored_run(folder):
    metadata = read_checkpoint_metadata(checkpoint_path(folder))
    ledger_class = LEDGER.get(folder, {}).get("asr_class")
    if ledger_class in ("diverged", "source_mapped"):
        raise ValueError(f"{folder} is {ledger_class} in the coverage ledger")
    if not successful_at_2_points(folder, metadata):
        raise ValueError(f"{folder} is not successful at the 2-point bar")
    run = {
        "name": folder,
        "folder": folder,
        "architecture": metadata["architecture"],
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "category": CATEGORY[metadata["attack"]],
        "benign": False,
        "probe_attack": None,
        "probe_target": None,
        "target": int(metadata["target_label"]),
    }
    return run


def successful_at_2_points(folder, metadata):
    """The ledger's successful_2pt for a ViT cell, the same rule from args.json for a Swin cell.

    Successful means ASR at least ASR_BAR and clean accuracy no more than 2
    points below the benign model of the same architecture and dataset. Swin
    cells are not in the ledger, so their args.json values are read against the
    Swin benign model's. args.json holds the ASR measured on the PSBD split,
    which for TaCT is the source-restricted one, where the July metrics.json
    files predate the restriction.
    """
    if folder in LEDGER:
        successful = bool(LEDGER[folder]["successful_2pt"])
        return successful
    benign = read_checkpoint_metadata(
        checkpoint_path(f"{metadata['architecture']}_{metadata['dataset']}_benign")
    )
    successful = (
        metadata["asr"] >= ASR_BAR
        and metadata["clean_accuracy"] >= benign["clean_accuracy"] - CLEAN_ACCURACY_BAR
    )
    return successful


def benign_run(architecture, dataset, attack):
    # The GTSRB SIG model trains on target class 1 because class 0 caps a
    # clean-label rate at 0.56%, so its control is probed at the same target.
    target = 1 if (dataset == "gtsrb" and attack == "sig") else 0
    folder = f"{architecture}_{dataset}_benign"
    run = {
        "name": f"{folder}__{attack}",
        "folder": folder,
        "architecture": architecture,
        "dataset": dataset,
        "attack": attack,
        "category": CATEGORY[attack],
        "benign": True,
        "probe_attack": attack,
        "probe_target": target,
        "target": target,
    }
    return run


def checkpoint_path(folder):
    path = os.path.join("checkpoints", folder, "attack_result.pt")
    return path


def prepare(run, raw_data_dir):
    path = os.path.join(PAIR_CACHE, f"{run['name']}.pt")
    pairs = (
        torch.load(path) if os.path.exists(path) else paired_images(run, raw_data_dir)
    )
    if "other_triggered" not in pairs:
        pairs["other_triggered"] = other_triggered_images(run, pairs)
        pairs["other_trigger"] = OTHER_TRIGGER[run["attack"]]
    write_pairs(run["name"], pairs)
    print(
        f"{run['name']}: {pairs['clean'].shape[0]} pairs of {pairs['n_eligible']} eligible"
    )


def paired_images(run, raw_data_dir):
    seed_everything(SEED)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path(run["folder"]),
        raw_data_dir=raw_data_dir,
        batch_size=128,
        num_workers=8,
        probe_attack=run["probe_attack"],
        probe_target_label=run["probe_target"],
    )
    clean_images, clean_labels = collect_split(
        loaders["clean"]
    )  # (n_analysis, C, H, W)
    triggered_images, _ = collect_split(loaders["backdoor"])  # (n_eligible, C, H, W)
    clean_rows = pair_clean_to_backdoor(
        torch.arange(clean_images.shape[0]), manifest
    )  # (n_eligible,)
    assert clean_rows.shape[0] == triggered_images.shape[0]

    generator = torch.Generator().manual_seed(SEED)
    chosen = torch.randperm(clean_rows.shape[0], generator=generator)[:MAX_PAIRS]
    target_rows = torch.nonzero(clean_labels == run["target"]).squeeze(1)
    target_rows = target_rows[:TARGET_IMAGES]

    pairs = {
        "clean": clean_images[clean_rows[chosen]],  # (pairs, C, H, W)
        "triggered": triggered_images[chosen],  # (pairs, C, H, W)
        "labels": clean_labels[clean_rows[chosen]],  # (pairs,)
        "test_indices": torch.tensor(manifest["analysis_backdoor_indices"])[chosen],
        "target_images": clean_images[target_rows],  # (targets, C, H, W)
        "n_eligible": int(clean_rows.shape[0]),
    }
    return pairs


def other_triggered_images(run, pairs):
    """The same clean images stamped with a trigger the model was not trained on.

    The trigger is planted in pixel space and normalized after, as the PSBD
    loaders do. The test index stands in for the loader's position argument,
    which only an index-seeded trigger reads.
    """
    spec = DATASET_REGISTRY[run["dataset"]]
    name = OTHER_TRIGGER[run["attack"]]
    attack = build_attack(name, default_config(name), spec.image_size, run["target"])
    plant = attack.apply_trigger_eval or attack.apply_trigger
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    pixels = pixel_images(pairs["clean"], spec)  # (pairs, C, H, W)

    stamped = torch.stack(
        [
            normalize(plant(image, int(index)))
            for image, index in zip(pixels, pairs["test_indices"])
        ]
    )  # (pairs, C, H, W)
    return stamped


def write_pairs(name, pairs):
    os.makedirs(PAIR_CACHE, exist_ok=True)
    torch.save(pairs, os.path.join(PAIR_CACHE, f"{name}.pt"))


def probe(run, results_dir, device, max_pairs):
    seed_everything(SEED)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)

    pairs = limited(
        torch.load(os.path.join(PAIR_CACHE, f"{run['name']}.pt")), max_pairs
    )
    model = load_checkpoint(run["architecture"], checkpoint_path(run["folder"]), device)
    core = network_core(model)
    blocks = transformer_blocks(core, run["architecture"])
    num_blocks = len(blocks)
    head = core.heads.head if run["architecture"] == "vit" else core.head
    spec = DATASET_REGISTRY[run["dataset"]]

    num_pairs = pairs["clean"].shape[0]
    fit_rows = torch.arange(num_pairs // 2)
    eval_rows = torch.arange(num_pairs // 2, num_pairs)

    example_row = int(eval_rows[0])
    clean_pass, triggered_pass, token_maps, example_tokens = paired_forward(
        model,
        run,
        blocks,
        head,
        pairs["clean"],
        pairs["triggered"],
        example_row,
        device,
    )
    target_pass = single_forward(
        model, run, blocks, head, pairs["target_images"], device
    )
    other_pass = single_forward(
        model, run, blocks, head, pairs["other_triggered"], device
    )

    input_map = input_change_map(pairs, spec)  # (height, width)
    input_grid = input_change_grid(input_map)  # (14, 14)
    trigger_tokens = torch.nonzero(
        input_grid.flatten() >= TRIGGER_TOKEN_SHARE * input_grid.max()
    ).squeeze(1)  # (n_trigger_tokens,)

    layers = layer_table(clean_pass, triggered_pass, fit_rows)
    final_layer = num_blocks
    middle_layer = (2 * num_blocks) // 3
    neurons = neuron_readout(
        clean_pass["pooled"][final_layer],
        triggered_pass["pooled"][final_layer],
        fit_rows,
        eval_rows,
    )
    top_dimension_maps = {
        str(layer): top_dimension_token_maps(
            run,
            clean_pass,
            triggered_pass,
            example_tokens,
            pairs,
            fit_rows,
            example_row,
            layer,
        )
        for layer in (middle_layer, final_layer)
    }
    attention = (
        attention_readout(clean_pass, triggered_pass, trigger_tokens)
        if run["architecture"] == "vit"
        else None
    )
    readout = readout_alignment(
        clean_pass["head"], triggered_pass["head"], fit_rows, head, run["target"]
    )
    passes = {
        "clean": clean_pass,
        "triggered": triggered_pass,
        "target": target_pass,
        "other": other_pass,
    }
    benign_directions = read_benign_directions(run)
    ablation = {
        str(layer): ablation_readout(
            model,
            run,
            blocks[layer - 1],
            passes,
            pairs,
            fit_rows,
            eval_rows,
            layer,
            benign_directions,
            device,
        )
        for layer in (middle_layer, final_layer)
    }
    if run["benign"]:
        write_benign_directions(run, passes, fit_rows, (middle_layer, final_layer))

    record = {
        "run": run,
        "ledger_class": LEDGER.get(run["folder"], {}).get("asr_class"),
        "below_asr_bar": None,
        "other_trigger": pairs["other_trigger"],
        "benign_direction_available": benign_directions is not None,
        "num_blocks": num_blocks,
        "n_pairs": num_pairs,
        "n_eligible": pairs["n_eligible"],
        "n_target_images": int(pairs["target_images"].shape[0]),
        "middle_layer": middle_layer,
        "behavior": behavior(
            clean_pass, triggered_pass, target_pass, pairs, run["target"]
        ),
        "examples": examples(pairs, spec),
        "input_change_map": rounded(input_map),
        "input_change_grid": rounded(input_grid),
        "trigger_tokens": trigger_tokens.tolist(),
        "layers": layers,
        "peak_layer": max(layers, key=lambda row: row["rel_direction_norm"])["layer"],
        "onset_layer": onset_layer(layers),
        "neurons": neurons,
        "token_maps": token_maps,
        "top_dimension_maps": top_dimension_maps,
        "attention": attention,
        "readout": readout,
        "ablation": ablation,
    }
    record["below_asr_bar"] = (not run["benign"]) and record["behavior"][
        "asr"
    ] < ASR_BAR
    write_run(results_dir, run["name"], record)
    write_head_features(run["name"], clean_pass, triggered_pass, target_pass, pairs)
    print(summary_line(record))


def limited(pairs, max_pairs):
    kept = {
        key: value[:max_pairs]
        if key
        in (
            "clean",
            "triggered",
            "other_triggered",
            "labels",
            "test_indices",
            "target_images",
        )
        else value
        for key, value in pairs.items()
    }
    return kept


def paired_forward(
    model, run, blocks, head, clean_images, triggered_images, example_row, device
):
    """Every block's pooled feature, the per-token change and the head input, for index-aligned pairs.

    Pooling follows the head: the class token on ViT, the token mean on Swin.
    Token sequences are reduced batch by batch to per-token change sums and are
    kept whole only for the example row, because a Swin sequence at the first
    stage is 3136 tokens and 25 layers of it for 800 images do not fit.
    """
    layers = tuple(range(len(blocks) + 1))
    change_sums = {layer: 0.0 for layer in layers}
    clean_norm_sum = {layer: 0.0 for layer in layers}
    clean_parts, triggered_parts = [], []
    example_tokens = {}

    for start in range(0, clean_images.shape[0], BATCH_SIZE):
        clean_batch = batch_readout(
            model, run, blocks, head, clean_images[start : start + BATCH_SIZE], device
        )
        triggered_batch = batch_readout(
            model,
            run,
            blocks,
            head,
            triggered_images[start : start + BATCH_SIZE],
            device,
        )
        for layer in layers:
            clean_tokens = clean_batch["tokens"][layer]  # (batch, tokens, dim)
            triggered_tokens = triggered_batch["tokens"][layer]  # (batch, tokens, dim)
            change_sums[layer] += (
                (triggered_tokens - clean_tokens).norm(dim=2).sum(dim=0).cpu()
            )  # (tokens,)
            clean_norm_sum[layer] += float(clean_tokens.norm(dim=2).mean(dim=1).sum())
        if start <= example_row < start + BATCH_SIZE:
            offset = example_row - start
            example_tokens = {
                "clean": {
                    layer: clean_batch["tokens"][layer][offset].cpu()
                    for layer in layers
                },
                "triggered": {
                    layer: triggered_batch["tokens"][layer][offset].cpu()
                    for layer in layers
                },
            }
        clean_parts.append(detached_summary(clean_batch))
        triggered_parts.append(detached_summary(triggered_batch))

    num_pairs = clean_images.shape[0]
    grids = clean_batch["grids"]
    token_maps = token_change_maps(change_sums, clean_norm_sum, num_pairs, grids)
    clean_pass = joined(clean_parts, grids)
    triggered_pass = joined(triggered_parts, grids)
    return clean_pass, triggered_pass, token_maps, example_tokens


def single_forward(model, run, blocks, head, images, device):
    parts = [
        detached_summary(batch_readout(model, run, blocks, head, batch, device))
        for batch in images.split(BATCH_SIZE)
    ]
    single_pass = (
        joined(parts, {})
        if parts
        else {"head": torch.zeros(0), "predictions": torch.zeros(0)}
    )
    return single_pass


def batch_readout(model, run, blocks, head, batch, device, use_bfloat16=True):
    """1 forward of 1 batch with every block's tokens, the head input and ViT's class-token attention."""
    layers = tuple(range(len(blocks) + 1))
    head_store, attention_store = {}, {}
    handles = [
        head.register_forward_pre_hook(
            lambda _module, inputs: head_store.__setitem__("x", inputs[0])
        )
    ]
    if run["architecture"] == "vit":
        handles += [
            block.ln_1.register_forward_hook(
                class_token_attention_hook(block, attention_store, index)
            )
            for index, block in enumerate(blocks)
        ]

    try:
        with (
            torch.inference_mode(),
            captured_layers(model, layers, run["architecture"]) as captured,
        ):
            logits = forward_logits(
                model, batch, device, use_bfloat16
            )  # (batch, classes)
            tokens = {
                layer: as_token_sequence(captured[layer]).float() for layer in layers
            }
            grids = {
                layer: tuple(captured[layer].shape[1:3])
                for layer in layers
                if captured[layer].dim() == 4
            }
    finally:
        for handle in handles:
            handle.remove()

    attention = (
        torch.stack([attention_store[index] for index in range(len(blocks))], dim=1)
        if attention_store
        else None
    )  # (batch, blocks, heads, tokens)
    readout = {
        "tokens": tokens,
        "grids": grids,
        "pooled": {
            layer: pool_tokens(tokens[layer], run["architecture"]) for layer in layers
        },
        "head": head_store["x"].float(),  # (batch, dim)
        "predictions": logits.argmax(dim=1),  # (batch,)
        "attention": attention,
    }
    return readout


def detached_summary(readout):
    summary = {
        "pooled": {layer: values.cpu() for layer, values in readout["pooled"].items()},
        "head": readout["head"].cpu(),
        "predictions": readout["predictions"].cpu(),
        "attention": readout["attention"].cpu()
        if readout["attention"] is not None
        else None,
    }
    return summary


def joined(parts, grids):
    layers = parts[0]["pooled"].keys()
    single_pass = {
        "pooled": {
            layer: torch.cat([part["pooled"][layer] for part in parts])
            for layer in layers
        },
        "head": torch.cat([part["head"] for part in parts]),  # (N, dim)
        "predictions": torch.cat([part["predictions"] for part in parts]),  # (N,)
        "attention": torch.cat([part["attention"] for part in parts])
        if parts[0]["attention"] is not None
        else None,  # (N, blocks, heads, tokens)
        "grids": grids,
    }
    return single_pass


def pool_tokens(tokens, architecture):
    pooled = (
        tokens[:, 0, :] if architecture == "vit" else tokens.mean(dim=1)
    )  # (batch, dim)
    return pooled


def class_token_attention_hook(block, store, index):
    """Recompute the block's attention from its normalized input and keep the class token's row.

    torchvision calls the attention with need_weights=False, so the weights never
    leave the module. Calling it again on the same normalized input returns the
    identical weights without changing the forward.
    """

    def hook(_module, _inputs, output):
        _, weights = block.self_attention(
            output, output, output, need_weights=True, average_attn_weights=False
        )  # (batch, heads, tokens, tokens)
        store[index] = weights[:, :, 0, :].float()  # (batch, heads, tokens)

    return hook


def behavior(clean_pass, triggered_pass, target_pass, pairs, target):
    labels = pairs["labels"]
    record = {
        "asr": float((triggered_pass["predictions"] == target).float().mean()),
        "clean_accuracy": float((clean_pass["predictions"] == labels).float().mean()),
        "clean_to_target": float((clean_pass["predictions"] == target).float().mean()),
        "target_accuracy": float((target_pass["predictions"] == target).float().mean())
        if target_pass["predictions"].numel()
        else None,
    }
    return record


def pixel_images(images, spec):
    mean = torch.tensor(spec.mean).view(1, -1, 1, 1)
    std = torch.tensor(spec.std).view(1, -1, 1, 1)
    pixels = (images * std + mean).clamp(0, 1)  # (N, C, H, W)
    return pixels


def input_change_map(pairs, spec):
    """Mean absolute pixel change the trigger makes, (height, width) at native resolution."""
    clean = pixel_images(pairs["clean"], spec)  # (N, C, H, W)
    triggered = pixel_images(pairs["triggered"], spec)  # (N, C, H, W)

    change = (triggered - clean).abs().mean(dim=(0, 1))  # (H, W)
    return change


def input_change_grid(change_map):
    """The pixel change averaged over each 16 pixel patch of the 224 input, (14, 14).

    The model's own Resize wrapper upsamples the native image to 224, so the
    change is upsampled the same way before pooling it onto the patch grid.
    """
    upsampled = F.interpolate(
        change_map[None, None], size=(PATCH_GRID * 16, PATCH_GRID * 16), mode="bilinear"
    )  # (1, 1, 224, 224)

    grid = F.avg_pool2d(upsampled, 16)[0, 0]  # (14, 14)
    return grid


def examples(pairs, spec):
    clean = pixel_images(pairs["clean"][:EXAMPLE_PAIRS], spec)  # (k, C, H, W)
    triggered = pixel_images(pairs["triggered"][:EXAMPLE_PAIRS], spec)  # (k, C, H, W)

    record = {
        "clean": image_bytes(clean),
        "triggered": image_bytes(triggered),
        "labels": pairs["labels"][:EXAMPLE_PAIRS].tolist(),
        "test_indices": pairs["test_indices"][:EXAMPLE_PAIRS].tolist(),
    }
    return record


def image_bytes(images):
    listed = (images.permute(0, 2, 3, 1) * 255).round().byte().tolist()  # (k, H, W, C)
    return listed


def layer_table(clean_pass, triggered_pass, fit_rows):
    final_layer = max(clean_pass["pooled"])
    final_direction = paired_direction(
        clean_pass["pooled"][final_layer], triggered_pass["pooled"][final_layer]
    )  # (dim,)
    final_top = torch.argsort(
        tac(clean_pass["pooled"][final_layer], triggered_pass["pooled"][final_layer]),
        descending=True,
    )[:TOP_DIMENSIONS]

    rows = [
        layer_row(
            layer,
            clean_pass["pooled"][layer],
            triggered_pass["pooled"][layer],
            final_direction,
            final_top,
            fit_rows,
        )
        for layer in sorted(clean_pass["pooled"])
    ]
    return rows


def paired_direction(clean, triggered):
    direction = (triggered - clean).mean(dim=0)  # (dim,)
    return direction


def tac(clean, triggered):
    values = (triggered - clean).abs().mean(dim=0)  # (dim,)
    return values


def unit(vector):
    normalized = vector / vector.norm().clamp_min(1e-8)
    return normalized


def layer_row(layer, clean, triggered, final_direction, final_top, fit_rows):
    difference = triggered - clean  # (N, dim)
    direction = difference.mean(dim=0)  # (dim,)
    clean_scale = clean.norm(dim=1).mean()
    values = tac(clean, triggered)  # (dim,)
    top = torch.argsort(values, descending=True)[:TOP_DIMENSIONS]
    direction_unit = unit(direction)  # (dim,)
    same_space = direction.shape == final_direction.shape

    row = {
        "layer": layer,
        "dim": int(direction.shape[0]),
        "rel_direction_norm": float(direction.norm() / clean_scale),
        "direction_share": direction_share(difference),
        "cka": float(debiased_linear_cka(clean, triggered)),
        "max_tac": float(values.max()),
        "mean_tac": float(values.mean()),
        "mean_abs_clean": float(clean.abs().mean()),
        "tac_outliers": int(outlier_dimensions(values).numel()),
        "top_tac_dimensions": top.tolist(),
        "top_tac_values": rounded(values[top]),
        "tac_of_final_top": rounded(values[final_top]) if same_space else None,
        "participation": participation(direction_unit),
        "top_tac_direction_energy": float((direction_unit[top] ** 2).sum()),
        "cosine_to_final": float(direction_unit @ unit(final_direction))
        if same_space
        else None,
        "split_half_top_jaccard": split_half_jaccard(clean, triggered, fit_rows),
    }
    return row


def direction_share(difference):
    """Share of the paired change's energy that lies along its own mean.

        share = ||mean_i d_i||^2 / mean_i ||d_i||^2

    1 when every image moves by the same vector, near 0 when the moves point
    in unrelated directions and cancel in the mean.
    """
    mean_energy = difference.mean(dim=0).pow(2).sum()
    total_energy = difference.pow(2).sum(dim=1).mean().clamp_min(1e-12)

    share = float(mean_energy / total_energy)
    return share


def participation(direction_unit):
    """How many coordinates a unit direction is spread across, 1 for a single axis.

        PR = (sum_k u_k^2)^2 / sum_k u_k^4 = 1 / sum_k u_k^4 for a unit u

    A Gaussian direction in D dimensions reads about D / 3.
    """
    ratio = float(1.0 / direction_unit.pow(4).sum())
    return ratio


def split_half_jaccard(clean, triggered, fit_rows):
    half = fit_rows.shape[0]
    first = set(
        torch.argsort(tac(clean[:half], triggered[:half]), descending=True)[
            :TOP_DIMENSIONS
        ].tolist()
    )
    second = set(
        torch.argsort(tac(clean[half:], triggered[half:]), descending=True)[
            :TOP_DIMENSIONS
        ].tolist()
    )

    jaccard = len(first & second) / len(first | second)
    return jaccard


def onset_layer(rows):
    """The first layer whose relative direction norm reaches half the largest one."""
    peak = max(row["rel_direction_norm"] for row in rows)
    onset = next(row["layer"] for row in rows if row["rel_direction_norm"] >= peak / 2)
    return onset


def neuron_readout(clean, triggered, fit_rows, eval_rows):
    """Clean against triggered separability of single neurons and of the direction.

    Every choice (which neuron, its sign, the direction) is made on the fit half,
    every AUROC is read on the eval half.
    """
    fit_difference = triggered[fit_rows] - clean[fit_rows]  # (half, dim)
    fit_direction = fit_difference.mean(dim=0)  # (dim,)
    fit_tac = fit_difference.abs().mean(dim=0)  # (dim,)
    top = torch.argsort(fit_tac, descending=True)[:TOP_DIMENSIONS]
    fit_auroc = oriented_auroc_per_dimension(
        clean[fit_rows], triggered[fit_rows]
    )  # (dim,)
    best_fit_dimension = int(torch.argmax(fit_auroc))

    eval_clean, eval_triggered = (
        clean[eval_rows],
        triggered[eval_rows],
    )  # (half, dim) each
    signs = torch.sign(fit_direction)  # (dim,)
    projection_clean = eval_clean @ unit(fit_direction)  # (half,)
    projection_triggered = eval_triggered @ unit(fit_direction)  # (half,)

    record = {
        "top_tac_dimensions": top.tolist(),
        "top_tac_eval_auroc": [
            auroc(signs[k] * eval_clean[:, k], signs[k] * eval_triggered[:, k])
            for k in top[:5].tolist()
        ],
        "best_fit_dimension": best_fit_dimension,
        "best_fit_dimension_eval_auroc": auroc(
            signs[best_fit_dimension] * eval_clean[:, best_fit_dimension],
            signs[best_fit_dimension] * eval_triggered[:, best_fit_dimension],
        ),
        "direction_eval_auroc": auroc(projection_clean, projection_triggered),
        "histograms": {
            **{
                f"dimension_{k}": histogram(eval_clean[:, k], eval_triggered[:, k])
                for k in top[:3].tolist()
            },
            "direction": histogram(projection_clean, projection_triggered),
        },
    }
    return record


def oriented_auroc_per_dimension(clean, triggered):
    """AUROC of every coordinate for triggered above clean, folded so 0.5 is the floor."""
    labels = np.concatenate([np.zeros(clean.shape[0]), np.ones(triggered.shape[0])])
    values = torch.cat([clean, triggered]).numpy()  # (2 * half, dim)

    raw = torch.tensor(
        [roc_auc_score(labels, values[:, k]) for k in range(values.shape[1])]
    )
    folded = torch.maximum(raw, 1 - raw)  # (dim,)
    return folded


def auroc(clean_scores, triggered_scores):
    labels = np.concatenate(
        [np.zeros(len(clean_scores)), np.ones(len(triggered_scores))]
    )
    scores = torch.cat([clean_scores, triggered_scores]).float().numpy()

    value = float(roc_auc_score(labels, scores))
    return value


def histogram(clean_values, triggered_values):
    combined = torch.cat([clean_values, triggered_values])
    edges = torch.linspace(
        float(combined.min()), float(combined.max()), HISTOGRAM_BINS + 1
    )

    record = {
        "edges": rounded(edges),
        "clean": torch.histogram(clean_values.float(), edges)[0].int().tolist(),
        "triggered": torch.histogram(triggered_values.float(), edges)[0].int().tolist(),
    }
    return record


def token_change_maps(change_sums, clean_norm_sum, num_pairs, grids):
    """Per-token size of the trigger's change at every layer, over the patch grid.

        m_l(t) = mean_i || h_l(x_i + delta)_t - h_l(x_i)_t || / mean_{i,t} || h_l(x_i)_t ||

    ViT's class token is reported apart from the grid. A Swin grid shrinks by 2
    at every stage, so each layer carries its own grid shape.
    """
    maps = {}
    for layer, change_sum in change_sums.items():
        scale = clean_norm_sum[layer] / num_pairs
        relative = change_sum / num_pairs / scale  # (tokens,)
        if layer in grids:
            maps[str(layer)] = {
                "grid": list(grids[layer]),
                "class_token": None,
                "map": rounded(relative),
            }
        else:
            maps[str(layer)] = {
                "grid": [PATCH_GRID, PATCH_GRID],
                "class_token": float(relative[0]),
                "map": rounded(relative[1:]),
            }
    return maps


def top_dimension_token_maps(
    run, clean_pass, triggered_pass, example_tokens, pairs, fit_rows, row, layer
):
    """The token grid of the top TAC dimensions for 1 image, clean and triggered.

    The statistic of visualization.cheap_tools.run_token_maps, with the
    dimensions ranked on the fit half and the image taken from the eval half.
    """
    fit_tac = tac(
        clean_pass["pooled"][layer][fit_rows], triggered_pass["pooled"][layer][fit_rows]
    )
    dimensions = torch.argsort(fit_tac, descending=True)[:TOKEN_MAP_DIMENSIONS]

    clean_tokens = example_tokens["clean"][layer]  # (tokens, dim)
    triggered_tokens = example_tokens["triggered"][layer]  # (tokens, dim)
    first_patch = 1 if run["architecture"] == "vit" else 0
    grid = clean_pass["grids"].get(layer, (PATCH_GRID, PATCH_GRID))

    record = {
        "dimensions": dimensions.tolist(),
        "test_index": int(pairs["test_indices"][row]),
        "row": row,
        "clean": rounded(dimension_grids(clean_tokens, first_patch, dimensions, grid)),
        "triggered": rounded(
            dimension_grids(triggered_tokens, first_patch, dimensions, grid)
        ),
    }
    return record


def dimension_grids(tokens, first_patch, dimensions, grid):
    patches = tokens[first_patch:, dimensions]  # (patches, k)
    grids = patches.T.reshape(len(dimensions), *grid)  # (k, grid_h, grid_w)
    return grids


def attention_readout(clean_pass, triggered_pass, trigger_tokens):
    """The class token's attention per block, and its mass on the trigger's tokens.

    Attention rows are (N, blocks, heads, 197). Trigger tokens index the 196
    patch tokens, so they shift by 1 past the class token.
    """
    clean = clean_pass["attention"]  # (N, blocks, heads, tokens)
    triggered = triggered_pass["attention"]  # (N, blocks, heads, tokens)
    positions = trigger_tokens + 1

    record = {
        "n_trigger_tokens": int(trigger_tokens.numel()),
        "uniform_mass": float(trigger_tokens.numel() / clean.shape[-1]),
        "clean_mass": rounded(
            clean[..., positions].sum(dim=-1).mean(dim=0)
        ),  # (blocks, heads)
        "triggered_mass": rounded(triggered[..., positions].sum(dim=-1).mean(dim=0)),
        "clean_row": rounded(clean.mean(dim=(0, 2))[:, 1:]),  # (blocks, 196)
        "triggered_row": rounded(triggered.mean(dim=(0, 2))[:, 1:]),
        "clean_self": rounded(clean.mean(dim=(0, 2))[:, 0]),  # (blocks,)
        "triggered_self": rounded(triggered.mean(dim=(0, 2))[:, 0]),
    }
    return record


def readout_alignment(clean_head, triggered_head, fit_rows, head, target):
    """The backdoor direction at the head input against the classifier's rows.

    Rows are centered over classes first, because adding the same vector to every
    row shifts every logit equally and cannot change a prediction.
    """
    direction = paired_direction(
        clean_head[fit_rows], triggered_head[fit_rows]
    )  # (dim,)
    weight = head.weight.detach().float().cpu()  # (classes, dim)
    centered = weight - weight.mean(dim=0, keepdim=True)  # (classes, dim)
    logit_shift = centered @ direction  # (classes,)
    cosines = (centered / centered.norm(dim=1, keepdim=True)) @ unit(
        direction
    )  # (classes,)
    others = torch.cat([cosines[:target], cosines[target + 1 :]])

    record = {
        "cosine_target_row": float(cosines[target]),
        "max_cosine_other_row": float(others.max()),
        "target_logit_rank": int((logit_shift > logit_shift[target]).sum()) + 1,
        "target_logit_shift": float(logit_shift[target]),
        "max_other_logit_shift": float(
            torch.cat([logit_shift[:target], logit_shift[target + 1 :]]).max()
        ),
        "random_cosine_scale": float(1 / np.sqrt(direction.shape[0])),
    }
    return record


def ablation_readout(
    model,
    run,
    block,
    passes,
    pairs,
    fit_rows,
    eval_rows,
    layer,
    benign_directions,
    device,
):
    """ASR, clean accuracy and per-class recall on the eval half with 1 thing removed at 1 block's output.

    Removing k coordinates and removing 1 direction are the same operation in 2
    bases, zeroing the residual stream's component along chosen unit vectors, so
    the comparison asks which basis the backdoor lives in. Applied to every token,
    since a later block can re-read a token the edit skipped.

    Every removal has a control of its own kind. A direction is read against
    isotropic random directions, the top clean principal direction, the clean
    principal direction whose variance matches the backdoor direction's (the
    energy-matched control, taken among principal directions at cosine below
    MATCHED_PC_MAX_COSINE to the backdoor direction), the target class's mean
    offset, the backdoor direction with that offset projected out, another trigger's
    direction on the same model and the direction the same trigger induces in
    the benign model. k coordinates are read against k random coordinates and the
    k coordinates of largest clean magnitude, since a ViT carries a few massive
    dimensions that any "top neuron" rule could be picking.
    """
    clean_fit = passes["clean"]["pooled"][layer][fit_rows]  # (half, dim)
    triggered_fit = passes["triggered"]["pooled"][layer][fit_rows]  # (half, dim)
    other_fit = passes["other"]["pooled"][layer][fit_rows]  # (half, dim)
    target_all = passes["target"]["pooled"][layer]  # (targets, dim)
    target_fit = target_all[: target_all.shape[0] // 2]  # (targets / 2, dim)

    direction = paired_direction(clean_fit, triggered_fit)  # (dim,)
    width = direction.shape[0]
    generator = torch.Generator().manual_seed(SEED)
    components, variances = clean_principal_directions(clean_fit)  # (r, dim), (r,)
    backdoor_variance = float((clean_fit @ unit(direction)).var())
    # The energy-matched control must be a different direction, so principal
    # directions within MATCHED_PC_MAX_COSINE of the backdoor direction are
    # excluded before matching the variance. At the last block the backdoor
    # direction lies close to the target class's axis, which is itself a clean
    # principal direction, and matching without the exclusion returns it.
    cosines = (components @ unit(direction)).abs()  # (r,)
    distance = (variances - backdoor_variance).abs()  # (r,)
    distance[cosines >= MATCHED_PC_MAX_COSINE] = float("inf")
    matched = int(torch.argmin(distance))

    directions = {
        "direction": direction,
        **{
            f"random_direction_{index}": torch.randn(width, generator=generator)
            for index in range(RANDOM_DIRECTIONS)
        },
        "clean_pc1": components[0],
        "variance_matched_pc": components[matched],
        "target_class_direction": target_fit.mean(dim=0) - clean_fit.mean(dim=0),
        "other_trigger_direction": paired_direction(clean_fit, other_fit),
    }
    # The backdoor direction with its target-class component projected out. If
    # the backdoor were only "look like the target class", removing this rest
    # would leave ASR intact.
    target_unit = unit(directions["target_class_direction"])  # (dim,)
    directions["direction_orthogonal_to_target"] = (
        direction - (direction @ target_unit) * target_unit
    )
    if benign_directions is not None and layer in benign_directions:
        directions["benign_direction"] = benign_directions[layer]

    tac_order = torch.argsort(tac(clean_fit, triggered_fit), descending=True)
    direction_order = torch.argsort(direction.abs(), descending=True)
    magnitude_order = torch.argsort(clean_fit.abs().mean(dim=0), descending=True)
    random_order = torch.randperm(width, generator=generator)
    coordinate_rules = {
        "top_tac": tac_order,
        "top_direction_coordinates": direction_order,
        "top_magnitude": magnitude_order,
        "random_coordinates": random_order,
    }

    hooks = {"none": None}
    for name, vector in directions.items():
        hooks[name] = remove_direction_hook(unit(vector).to(device))
    for rule, order in coordinate_rules.items():
        for k in NEURON_COUNTS:
            hooks[f"{rule}_{k}"] = zero_coordinates_hook(order[:k].to(device))

    evaluation = evaluation_images(pairs, eval_rows)
    results = {
        name: hooked_behavior(model, block, hook, evaluation, run, device)
        for name, hook in hooks.items()
    }

    # The sufficiency half: add a direction to clean images and count how many
    # the model now sends to the target. Every control is rescaled to the
    # backdoor direction's norm, so only the orientation differs.
    steering = {}
    for name, vector in directions.items():
        if name.startswith("random_direction_") and name != "random_direction_0":
            continue
        scaled = unit(vector) * direction.norm()  # (dim,)
        steered = hooked_predictions(
            model,
            block,
            add_direction_hook(scaled.to(device)),
            evaluation["clean"],
            device,
        )
        steering[name] = float((steered == run["target"]).float().mean())

    record = {
        "variants": results,
        "steer_clean_to_target": steering,
        "direction_norm": float(direction.norm()),
        "clean_variance_along": {
            name: float((clean_fit @ unit(vector)).var())
            for name, vector in directions.items()
        },
        "variance_matched_pc_index": matched,
        "top_tac_magnitude_overlap": {
            str(k): len(set(tac_order[:k].tolist()) & set(magnitude_order[:k].tolist()))
            / k
            for k in NEURON_COUNTS
        },
        "cosine_to_direction": {
            name: float(unit(vector) @ unit(direction))
            for name, vector in directions.items()
        },
    }
    return record


def clean_principal_directions(clean_fit):
    """Principal directions of the centered clean features and the clean variance along each."""
    centered = clean_fit - clean_fit.mean(dim=0, keepdim=True)  # (half, dim)
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)

    variances = singular_values**2 / (clean_fit.shape[0] - 1)  # (r,)
    return right, variances


def evaluation_images(pairs, eval_rows):
    target_images = pairs["target_images"]
    target_eval = target_images[target_images.shape[0] // 2 :]  # (targets / 2, C, H, W)
    evaluation = {
        "clean": pairs["clean"][eval_rows],  # (half, C, H, W)
        "triggered": pairs["triggered"][eval_rows],  # (half, C, H, W)
        "labels": pairs["labels"][eval_rows],  # (half,)
        "target_images": target_eval,
    }
    return evaluation


def hooked_behavior(model, block, hook, evaluation, run, device):
    """ASR, paired clean accuracy, target-class recall and per-class recall under 1 hook.

    Per-class recall pools the eval-half clean pairs with the eval-half clean
    target images, so the target class, which the eligible pairs exclude, is
    covered. A class absent from the pool reads None.
    """
    target = run["target"]
    clean_predictions = hooked_predictions(
        model, block, hook, evaluation["clean"], device
    )
    triggered_predictions = hooked_predictions(
        model, block, hook, evaluation["triggered"], device
    )
    target_predictions = (
        hooked_predictions(model, block, hook, evaluation["target_images"], device)
        if evaluation["target_images"].shape[0]
        else torch.zeros(0, dtype=torch.long)
    )

    pooled_labels = torch.cat(
        [evaluation["labels"], torch.full((target_predictions.shape[0],), target)]
    )  # (half + targets / 2,)
    pooled_predictions = torch.cat([clean_predictions, target_predictions])
    num_classes = DATASET_REGISTRY[run["dataset"]].num_classes
    per_class = [
        float((pooled_predictions[pooled_labels == c] == c).float().mean())
        if bool((pooled_labels == c).any())
        else None
        for c in range(num_classes)
    ]
    behavior_record = {
        "asr": float((triggered_predictions == target).float().mean()),
        "clean_accuracy": float(
            (clean_predictions == evaluation["labels"]).float().mean()
        ),
        "clean_to_target": float((clean_predictions == target).float().mean()),
        "target_recall": float((target_predictions == target).float().mean())
        if target_predictions.numel()
        else None,
        "per_class_recall": [
            None if value is None else round(value, 4) for value in per_class
        ],
    }
    return behavior_record


def read_benign_directions(run):
    """The fit-half directions the benign model's own run saved for the same trigger, if it ran."""
    if run["benign"]:
        return None
    control = f"{run['architecture']}_{run['dataset']}_benign__{run['attack']}"
    path = os.path.join(FEATURE_CACHE, f"{control}_directions.pt")
    if not os.path.exists(path):
        return None
    directions = torch.load(path)
    return directions


def write_benign_directions(run, passes, fit_rows, layers):
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    directions = {
        layer: paired_direction(
            passes["clean"]["pooled"][layer][fit_rows],
            passes["triggered"]["pooled"][layer][fit_rows],
        )
        for layer in layers
    }
    torch.save(directions, os.path.join(FEATURE_CACHE, f"{run['name']}_directions.pt"))


def hooked_predictions(model, block, hook, images, device):
    handle = block.register_forward_hook(hook) if hook is not None else None
    chunks = []
    try:
        with torch.inference_mode():
            for batch in images.split(BATCH_SIZE):
                chunks.append(
                    forward_logits(model, batch, device, True).argmax(dim=1).cpu()
                )
    finally:
        if handle is not None:
            handle.remove()

    predictions = torch.cat(chunks)  # (N,)
    return predictions


def remove_direction_hook(direction_unit):
    """Sets x to x - (x . u) u on every token, the zero ablation of 1 rotated coordinate."""

    def hook(_module, _inputs, output):
        u = direction_unit.to(output.dtype)  # (dim,)
        component = (output * u).sum(dim=-1, keepdim=True)  # (..., 1)
        ablated = output - component * u
        return ablated

    return hook


def zero_coordinates_hook(dimensions):
    def hook(_module, _inputs, output):
        ablated = output.clone()
        ablated[..., dimensions] = 0.0
        return ablated

    return hook


def add_direction_hook(direction):
    def hook(_module, _inputs, output):
        steered = output + direction.to(output.dtype)
        return steered

    return hook


def rounded(values, digits=5):
    array = (
        values.detach().float().cpu().numpy()
        if torch.is_tensor(values)
        else np.asarray(values)
    )
    listed = np.round(array, digits).tolist()
    return listed


def write_run(results_dir, name, record):
    directory = os.path.join(experiment_results_dir(SLUG, results_dir), "runs")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{name}.json"), "w") as handle:
        json.dump(record, handle)


def write_head_features(name, clean_pass, triggered_pass, target_pass, pairs):
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    features = {
        "clean": clean_pass["head"],
        "triggered": triggered_pass["head"],
        "target": target_pass["head"],
        "labels": pairs["labels"],
    }
    torch.save(features, os.path.join(FEATURE_CACHE, f"{name}.pt"))


def summary_line(record):
    ablated = record["ablation"][str(record["num_blocks"])]["variants"]
    line = (
        f"{record['run']['name']:40} asr {record['behavior']['asr']:.3f} "
        f"ca {record['behavior']['clean_accuracy']:.3f} peak {record['peak_layer']} "
        f"dir-ablated asr {ablated['direction']['asr']:.3f} "
        f"top300 asr {ablated['top_tac_300']['asr']:.3f} "
        f"auroc dir {record['neurons']['direction_eval_auroc']:.3f} "
        f"best neuron {record['neurons']['best_fit_dimension_eval_auroc']:.3f}"
    )
    return line


def summarize(results_dir):
    directory = experiment_results_dir(SLUG, results_dir)
    records = [
        json.load(open(path))
        for path in sorted(glob.glob(os.path.join(directory, "runs", "*.json")))
    ]
    embeddings_directory = os.path.join(directory, "embeddings")
    os.makedirs(embeddings_directory, exist_ok=True)
    for record in records:
        name = record["run"]["name"]
        path = os.path.join(embeddings_directory, f"{name}.json")
        if os.path.exists(path):
            continue
        with open(path, "w") as handle:
            json.dump(embedding_record(name), handle)

    summary = {
        "rows": [summary_row(record) for record in records],
        "top_dimension_overlap": top_dimension_overlap(records),
        "chance_jaccard": chance_jaccard(768, TOP_DIMENSIONS),
    }
    with open(os.path.join(directory, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(f"{len(records)} runs summarized into {directory}/summary.json")


def embedding_record(name):
    """PCA and UMAP of the head input, fitted on the fit half and applied to the eval half.

    3 populations: clean and triggered pairs and clean target-class images, each
    split in 2 halves the same way. Fitting on 1 half and drawing the other
    keeps the picture from showing structure the projection found in its own
    points. Capped at EMBEDDING_POINTS per population.
    """
    # Imported here because umap's numba start-up costs 10 seconds, which the
    # probe stage would otherwise spend holding the shared GPU lock.
    import umap

    features = torch.load(os.path.join(FEATURE_CACHE, f"{name}.pt"))
    fit, shown = {}, {}
    for population in ("clean", "triggered", "target"):
        values = features[population]  # (n, dim)
        half = values.shape[0] // 2
        fit[population] = values[:half][:EMBEDDING_POINTS]
        shown[population] = values[half:][:EMBEDDING_POINTS]
    fit_stack = torch.cat(list(fit.values())).numpy()  # (n_fit, dim)
    shown_stack = torch.cat(list(shown.values())).numpy()  # (n_shown, dim)

    pca = PCA(n_components=2).fit(fit_stack)
    projector = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit(
        fit_stack
    )
    half = features["clean"].shape[0] // 2
    record = {
        "populations": ["clean"] * len(shown["clean"])
        + ["triggered"] * len(shown["triggered"])
        + ["target_class"] * len(shown["target"]),
        "labels": features["labels"][half:][:EMBEDDING_POINTS].tolist(),
        "pca": rounded(pca.transform(shown_stack), 4),
        "pca_explained": rounded(pca.explained_variance_ratio_, 4),
        "umap": rounded(projector.transform(shown_stack), 4),
        "umap_settings": {"num_neighbors": 15, "min_distance": 0.1},
    }
    return record


def summary_row(record):
    final = str(record["num_blocks"])
    middle = str(record["middle_layer"])
    last = record["layers"][-1]
    row = {
        "name": record["run"]["name"],
        "architecture": record["run"]["architecture"],
        "dataset": record["run"]["dataset"],
        "attack": record["run"]["attack"],
        "category": record["run"]["category"],
        "benign": record["run"]["benign"],
        "n_pairs": record["n_pairs"],
        "excluded": exclusion(record),
        "below_asr_bar": (not record["run"]["benign"])
        and record["behavior"]["asr"] < ASR_BAR,
        **record["behavior"],
        "peak_layer": record["peak_layer"],
        "onset_layer": record["onset_layer"],
        "final_rel_direction_norm": last["rel_direction_norm"],
        "max_rel_direction_norm": max(
            row["rel_direction_norm"] for row in record["layers"]
        ),
        "final_cka": last["cka"],
        "final_direction_share": last["direction_share"],
        "final_participation": last["participation"],
        "final_top_tac_direction_energy": last["top_tac_direction_energy"],
        "final_tac_outliers": last["tac_outliers"],
        "final_split_half_jaccard": last["split_half_top_jaccard"],
        "direction_eval_auroc": record["neurons"]["direction_eval_auroc"],
        "best_neuron_eval_auroc": record["neurons"]["best_fit_dimension_eval_auroc"],
        "top_tac_neuron_eval_auroc": record["neurons"]["top_tac_eval_auroc"][0],
        "cosine_target_row": record["readout"]["cosine_target_row"],
        "max_cosine_other_row": record["readout"]["max_cosine_other_row"],
        "target_logit_rank": record["readout"]["target_logit_rank"],
        "ablation_final": lean_ablation(record["ablation"][final]),
        "ablation_middle": lean_ablation(record["ablation"][middle]),
        "attention_final_mass_ratio": attention_mass_ratio(record),
    }
    return row


def lean_ablation(ablation):
    """The ablation record without per-class recall, which the run JSON keeps."""
    lean = {
        **ablation,
        "variants": {
            name: {
                key: value for key, value in entry.items() if key != "per_class_recall"
            }
            for name, entry in ablation["variants"].items()
        },
    }
    return lean


def exclusion(record):
    """source_mapped when a TaCT model misclassifies its clean source images without a trigger.

    TaCT's pairs are its source class only, so the paired clean accuracy is the
    source-class accuracy. Below 0.5 the model maps the class to the target with
    no trigger, and every trigger reading would be averaged with that mapping.
    """
    run = record["run"]
    mapped = (
        run["attack"] == "tact"
        and not run["benign"]
        and record["behavior"]["clean_accuracy"] < SOURCE_MAPPED_CLEAN_ACCURACY
    )
    reason = "source_mapped" if mapped else None
    return reason


def attention_mass_ratio(record):
    """Triggered over clean class-token attention on the trigger's tokens, last 4 blocks, head mean."""
    attention = record["attention"]
    if attention is None:
        return None
    clean = np.asarray(attention["clean_mass"])[-4:].mean()
    triggered = np.asarray(attention["triggered_mass"])[-4:].mean()

    ratio = float(triggered / max(clean, 1e-8))
    return ratio


def top_dimension_overlap(records):
    """Jaccard of the final layer's top TAC dimensions between every 2 runs on 1 model family.

    A family is 1 architecture on 1 dataset. Every model there is fine-tuned from
    the same ImageNet weights, so a coordinate index names the same unit before
    fine-tuning, which is what makes a shared index meaningful at all.
    """
    families = {}
    for record in records:
        run = record["run"]
        key = f"{run['architecture']}_{run['dataset']}"
        families.setdefault(key, []).append(
            (run["name"], set(record["layers"][-1]["top_tac_dimensions"]))
        )

    overlap = {}
    for key, members in families.items():
        names = [name for name, _ in members]
        matrix = [[len(a & b) / len(a | b) for _, b in members] for _, a in members]
        overlap[key] = {"names": names, "jaccard": np.round(matrix, 4).tolist()}
    return overlap


def chance_jaccard(width, k):
    """Expected Jaccard of 2 independent uniform k-subsets of width coordinates, by exact sum."""
    from math import comb

    expected = sum(
        comb(k, shared)
        * comb(width - k, k - shared)
        / comb(width, k)
        * shared
        / (2 * k - shared)
        for shared in range(k + 1)
    )
    return expected


if __name__ == "__main__":
    main()

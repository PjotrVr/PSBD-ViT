"""GPU stage of the internal maps: every number the figures draw, 1 numbers.json per model.

The figures explain why PSBD-TM (token_mask at before_attention_norm) separates
triggered from clean inputs. Each measurement is inference only on paired clean and
triggered images of the PSBD analysis split (data.splits, seed PSBD_SPLIT_SEED),
built exactly as cli.sweep builds them. make.py draws the figures from the JSON on
the CPU. README.md states what each figure shows.

    flock -E 75 scratch/gpu.lock .venv/bin/python -m experiments.internal_maps.measure \
        --models vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import math
import os
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything

from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.decision import RECOMMENDED_PLACEMENT
from defenses.inference import forward_logits
from defenses.operators import TokenMask
from experiments._paths import experiment_results_dir
from experiments.cache_readouts.shared import (
    choose_rate,
    load_baselines,
    validation_shift_by_rate,
)
from experiments.internal_maps import config
from experiments.why_token_masking_works.measure import RecordingTokenMask
from experiments.why_token_masking_works.tokens import (
    BLOCK_GRIDS,
    block_trigger_positions,
    cached_baseline_agreement,
    example_images,
    model_blocks,
    model_seed,
    touched_tokens,
    trigger_pixel_map,
)
from models.backbones import load_checkpoint, network_core
from models.positions import plug_dropout, unplug_dropout

SLUG = "internal_maps"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
RAW_DATA_DIR = "raw_data"
TM_POSITION = "before_attention_norm"
PATCH_ATTACKS = ("badnet_a2o", "tact", "lc")
# The login node is shared, so torch gets a few CPU threads and a capped share of
# the A100 (the user's rule for this run, 0.3 or less).
CPU_THREADS = 4
GPU_MEMORY_FRACTION = 0.3
BATCH_SIZE = 192
VIT_HEADS = 12
RANDOM_CONTROL_DRAWS = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="", help="comma separated, default all")
    parser.add_argument("--figures", default=",".join(config.FIGURES))
    parser.add_argument("--force", action="store_true")
    # Recomputes only --figures and keeps every other entry of an existing record.
    parser.add_argument("--update", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    output_root = experiment_results_dir(SLUG, RESULTS_DIR)

    selected = [name for name in args.models.split(",") if name]
    figures = [name for name in args.figures.split(",") if name]
    for architecture, models in config.MODELS.items():
        for folder, probe in models.items():
            if selected and folder not in selected:
                continue
            out_path = os.path.join(output_root, architecture, folder, "numbers.json")
            exists = os.path.exists(out_path)
            if exists and not (args.force or args.update):
                print(f"[skip] {folder}", flush=True)
                continue
            started = time.time()
            record = measure_model(folder, probe, figures, device)
            if exists and args.update:
                record = dict(read_json(out_path), **record)
            record["seconds"] = round(time.time() - started, 1)
            write_json(out_path, record)
            print(f"[ok] {folder} {record['seconds']}s", flush=True)


def measure_model(folder, probe, figures, device):
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device).eval()
    trigger_metadata = dict(metadata, attack=probe or metadata["attack"])
    if probe:
        trigger_metadata["target_label"] = 0

    pool, validation = load_pairs(checkpoint_path, probe, device)
    pool["clean_base"] = predict(model, pool["clean"], device)  # (pool,)
    pool["triggered_base"] = predict(model, pool["triggered"], device)  # (pool,)
    agreement = cached_baseline_agreement(RESULTS_DIR, folder, pool)
    assert min(agreement.values()) >= 0.98, f"{folder} disagrees with cache {agreement}"
    pairs, probe_pairs = eligible_pairs(pool, backdoored=probe is None)

    pixel_map = trigger_pixel_map(trigger_metadata)  # (224, 224)
    grid = config.MAP_GRID[architecture]
    trigger_cells = touched_tokens(pixel_map, grid)  # (count,) on the map grid
    tm_rate, validation_shift, rate_rule = adaptive_rate(folder)

    record = {
        "folder": folder,
        "architecture": architecture,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "probe_attack": probe,
        "trigger_attack": trigger_metadata["attack"],
        "target_label": int(trigger_metadata["target_label"]),
        "poison_rate": metadata["poison_rate"],
        "git_commit": git_commit(),
        "map_grid": grid,
        "trigger_cells": trigger_cells.tolist(),
        "trigger_pixel_map_56": pooled_pixel_map(pixel_map),
        "is_patch": trigger_metadata["attack"] in PATCH_ATTACKS,
        "tm_rate": tm_rate,
        "tm_validation_shift": validation_shift,
        "tm_rate_rule": rate_rule,
        "baseline_agreement_with_cache": agreement,
        "pool_pairs": int(len(pool["clean"])),
        "eligible_pairs": int(len(pairs["clean"]) + len(probe_pairs["clean"])),
        "examples": example_images(pairs, metadata["dataset"]),
        "example_classes": {
            "clean_label": int(pairs["clean_labels"][0]),
            "clean_prediction": int(pairs["clean_base"][0]),
            "triggered_prediction": int(pairs["triggered_base"][0]),
        },
    }

    if "token_removal" in figures:
        record["token_removal"] = measure_token_removal(
            model, architecture, pairs, trigger_cells, folder, device
        )
    if "trigger_reading" in figures:
        if architecture == "vit":
            record["trigger_reading"] = measure_class_token_attention(
                model, pairs, pixel_map, device
            )
        else:
            record["trigger_reading"] = measure_readout_attribution(
                model, pairs, pixel_map, device
            )
    if "depth_survival" in figures:
        record["depth_survival"] = measure_depth_survival(
            model, architecture, pairs, probe_pairs, validation, tm_rate, folder, device
        )
    if "direction" in figures:
        record["direction"] = measure_direction(
            model, architecture, pairs, pixel_map, device
        )
    if "masks" in figures:
        record["masks"] = measure_example_masks(
            model, architecture, pairs, pixel_map, tm_rate, folder, device
        )
    return record


# The analysis pairs of the PSBD split, the first POOL_PAIRS triggered rows and
# the clean twin of each, plus clean validation images for fitting the probes.
def load_pairs(checkpoint_path, probe, device):
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=RAW_DATA_DIR,
        batch_size=64,
        num_workers=0,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    clean_set = loaders["clean"].dataset
    backdoor_set = loaders["backdoor"].dataset
    validation_set = loaders["validation"].dataset
    row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    count = min(config.POOL_PAIRS, len(backdoor_set))
    clean_rows = [row_of[o] for o in manifest["analysis_backdoor_indices"][:count]]

    clean_items = [clean_set[row] for row in clean_rows]
    triggered_items = [backdoor_set[row] for row in range(count)]
    pool = {
        "clean": torch.stack([image for image, _ in clean_items]).to(device),
        "clean_labels": torch.tensor([int(label) for _, label in clean_items]),
        "triggered": torch.stack([image for image, _ in triggered_items]).to(device),
        "targets": torch.tensor([int(target) for _, target in triggered_items]),
    }
    assert pool["clean"].shape == pool["triggered"].shape, "pairs must align"
    validation_count = min(config.PROBE_VALIDATION_IMAGES, len(validation_set))
    validation = torch.stack(
        [validation_set[row][0] for row in range(validation_count)]
    ).to(device)  # (m, 3, s, s)
    return pool, validation


# A pair is used when the unperturbed model sends the triggered image to the
# target and classifies the clean twin correctly, so both have a decision to keep.
# A benign model has no target, so its triggered decision is whatever it predicts.
def eligible_pairs(pool, backdoored):
    clean_base = pool["clean_base"].cpu()
    triggered_base = pool["triggered_base"].cpu()
    hit = (
        triggered_base == pool["targets"]
        if backdoored
        else torch.ones_like(triggered_base, dtype=torch.bool)
    )
    correct = clean_base == pool["clean_labels"]
    rows = (hit & correct).nonzero(as_tuple=True)[0]  # (eligible,)

    evaluation_rows = rows[: config.EVAL_PAIRS]
    probe_rows = rows[
        config.EVAL_PAIRS : config.EVAL_PAIRS + config.PROBE_TRIGGERED_PAIRS
    ]
    pairs = select_rows(pool, evaluation_rows)
    probe_pairs = select_rows(pool, probe_rows)
    return pairs, probe_pairs


def select_rows(pool, rows):
    selected = {}
    for key, value in pool.items():
        selected[key] = value[rows.to(value.device)]
    # The class whose probability every map follows. The target on a backdoored
    # model, the benign model's own prediction on the triggered image otherwise.
    selected["triggered_class"] = selected["triggered_base"]
    selected["clean_class"] = selected["clean_base"]
    return selected


# Figure 1. Every map unit is removed from attention in every block, alone (the
# drop map) or together with everything outside a kept window (the kept map).
def measure_token_removal(model, architecture, pairs, trigger_cells, folder, device):
    grid = config.MAP_GRID[architecture]
    units = grid * grid
    keeper = GridKeep()
    handles = plug_grid_keep(model, architecture, keeper)
    seed = model_seed(folder)
    result = {}
    try:
        for split in ("triggered", "clean"):
            images = pairs[split][: config.REMOVAL_PAIRS]  # (n, 3, s, s)
            classes = pairs[f"{split}_class"][: config.REMOVAL_PAIRS].to(device)
            nothing = torch.ones(1, grid, grid, device=device)
            base_probability, _ = masked_readout(
                model, keeper, images, nothing, classes
            )

            # Each mask hides 1 unit and leaves the rest visible.
            single = torch.ones(units, grid, grid, device=device)  # (units, g, g)
            single.view(units, units)[torch.arange(units), torch.arange(units)] = 0.0
            single_probability, _ = masked_readout(
                model, keeper, images, single, classes
            )  # (units, n)
            drop = base_probability - single_probability  # (units, n)

            split_result = {
                "base_probability": float(base_probability.mean()),
                "drop_map": drop.mean(dim=1).reshape(grid, grid).tolist(),
                "kept_maps": {},
            }
            for window in config.KEPT_WINDOWS[architecture]:
                kept_masks = window_masks(grid, window, device)  # (units, g, g)
                probability, kept = masked_readout(
                    model, keeper, images, kept_masks, classes
                )
                split_result["kept_maps"][str(window)] = {
                    "probability": probability.mean(dim=1).reshape(grid, grid).tolist(),
                    "kept": kept.float().mean(dim=1).reshape(grid, grid).tolist(),
                    "visible_units": window * window,
                }
                # A model that sees almost nothing can fall onto 1 default class,
                # and on some models that class is the target. The clean twin's
                # target probability under the same window is that floor.
                if split == "clean":
                    target_classes = pairs["triggered_class"][: config.REMOVAL_PAIRS]
                    target_probability, _ = masked_readout(
                        model, keeper, images, kept_masks, target_classes.to(device)
                    )
                    split_result["kept_maps"][str(window)]["target_probability"] = (
                        target_probability.mean(dim=1).reshape(grid, grid).tolist()
                    )
            split_result["controls"] = removal_controls(
                model, keeper, images, classes, trigger_cells, grid, seed, device
            )
            # The same visible shares on the clean twin, read as how often it
            # lands on the target, the floor a collapsing model gives the
            # triggered survival.
            if split == "clean":
                target_classes = pairs["triggered_class"][: config.REMOVAL_PAIRS]
                floor = removal_controls(
                    model,
                    keeper,
                    images,
                    target_classes.to(device),
                    trigger_cells,
                    grid,
                    seed,
                    device,
                )
                split_result["controls_on_target"] = floor
            result[split] = split_result
    finally:
        unplug_dropout(handles)
    result["pairs"] = int(min(config.REMOVAL_PAIRS, len(pairs["clean"])))
    return result


# The trigger's own units hidden in every block against as many random units, the
# reading why_token_masking_works part A recorded, and random visible shares, its
# part D, so this hook can be held to those records.
def removal_controls(model, keeper, images, classes, trigger_cells, grid, seed, device):
    units = grid * grid
    controls = {}
    if 0 < len(trigger_cells) < units // 2:
        trigger_mask = torch.ones(1, units, device=device)
        trigger_mask[0, trigger_cells.to(device)] = 0.0
        _, kept = masked_readout(
            model, keeper, images, trigger_mask.reshape(1, grid, grid), classes
        )
        controls["trigger_hidden_kept"] = float(kept.float().mean())

        others = torch.tensor(
            [u for u in range(units) if u not in set(trigger_cells.tolist())]
        )
        draws = []
        for draw in range(RANDOM_CONTROL_DRAWS):
            generator = torch.Generator().manual_seed(seed + draw)
            chosen = others[torch.randperm(len(others), generator=generator)]
            random_mask = torch.ones(1, units, device=device)
            random_mask[0, chosen[: len(trigger_cells)].to(device)] = 0.0
            _, kept = masked_readout(
                model, keeper, images, random_mask.reshape(1, grid, grid), classes
            )
            draws.append(float(kept.float().mean()))
        controls["random_hidden_kept"] = sum(draws) / len(draws)

    for fraction in config.VISIBLE_FRACTIONS:
        visible = round(fraction * units)
        draws = []
        for draw in range(config.VISIBLE_DRAWS):
            generator = torch.Generator().manual_seed(seed + 100 + draw)
            order = torch.randperm(units, generator=generator)
            subset_mask = torch.zeros(1, units, device=device)
            subset_mask[0, order[:visible].to(device)] = 1.0
            _, kept = masked_readout(
                model, keeper, images, subset_mask.reshape(1, grid, grid), classes
            )
            draws.append(float(kept.float().mean()))
        controls[f"visible_{fraction}_kept"] = sum(draws) / len(draws)
    return controls


def window_masks(grid, window, device):
    half = window // 2
    rows = torch.arange(grid, device=device)
    centers = torch.cartesian_prod(rows, rows)  # (units, 2)
    inside_row = (rows[None, :] - centers[:, :1]).abs() <= half  # (units, g)
    inside_column = (rows[None, :] - centers[:, 1:]).abs() <= half  # (units, g)
    masks = (
        inside_row[:, :, None] & inside_column[:, None, :]
    ).float()  # (units, g, g)
    return masks


# Every (mask, image) combination in batches. Returns the probability of each
# image's class and whether its argmax stays that class, both (masks, n).
@torch.inference_mode()
def masked_readout(model, keeper, images, masks, classes):
    count = len(images)
    masks_per_batch = max(1, BATCH_SIZE // count)
    probabilities, kept = [], []
    for start in range(0, len(masks), masks_per_batch):
        chunk = masks[start : start + masks_per_batch]  # (m, g, g)
        batch_images = images.repeat(len(chunk), 1, 1, 1)  # (m * n, 3, s, s)
        keeper.keep = chunk.repeat_interleave(count, dim=0)  # (m * n, g, g)
        logits = forward_logits(model, batch_images, images.device, use_bfloat16=True)
        keeper.keep = None
        probability = logits.softmax(dim=1).reshape(len(chunk), count, -1)  # (m, n, c)
        chosen = probability.gather(2, classes[None, :, None].expand(len(chunk), -1, 1))
        probabilities.append(chosen[:, :, 0])  # (m, n)
        kept.append(probability.argmax(dim=2) == classes[None])  # (m, n)
    readout = (torch.cat(probabilities).cpu(), torch.cat(kept).cpu())
    return readout


# Multiplies the attention input of every block by a per-image keep map on the
# map grid. ViT's class token is never hidden, as TokenMask never masks it. A Swin
# keep map is repeated up to each stage's grid, so 1 cell hides all its tokens.
# The library operator rescales survivors by 1 / (1 - p), which the LayerNorm right
# after this position cancels, so a plain 0 or 1 keep is the same operation.
class GridKeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = None

    def forward(self, x):
        if self.keep is None:
            return x
        batch = x.shape[0]
        assert self.keep.shape[0] == batch, "1 keep map per image"
        if x.dim() == 3:
            class_token = torch.ones(batch, 1, device=x.device)
            keep = torch.cat([class_token, self.keep.reshape(batch, -1)], dim=1)
            assert keep.shape[1] == x.shape[1], "ViT keeps 1 + 196 tokens"
            masked = x * keep[:, :, None].to(x.dtype)  # (batch, 197, channels)
            return masked
        height = x.shape[1]
        factor = height // self.keep.shape[1]
        keep = self.keep.repeat_interleave(factor, dim=1).repeat_interleave(
            factor, dim=2
        )  # (batch, height, width)
        masked = x * keep[..., None].to(x.dtype)  # (batch, height, width, channels)
        return masked


def plug_grid_keep(model, architecture, keeper):
    handles = plug_dropout(
        model, architecture, (TM_POSITION,), {TM_POSITION: lambda _rate: keeper}, 0.0
    )
    return handles


# Figure 2 on ViT. Attention of the class token to the trigger's tokens, per block
# and head, recomputed from the attention module's input (its query and key
# projections), since torchvision calls it with need_weights off. Also the head-mean
# attention of every block for attention rollout (Abnar and Zuidema 2020).
def measure_class_token_attention(model, pairs, pixel_map, device):
    trigger = touched_tokens(pixel_map, 14) + 1  # (count,), class token at 0
    result = {"trigger_tokens": int(len(trigger))}
    for split in ("triggered", "clean"):
        images = pairs[split][: config.EVAL_PAIRS]
        attention = vit_attention(model, images, device)  # 12 x (n, 12, 197, 197)
        mass = torch.stack(
            [a[:, :, 0, trigger.to(device)].sum(dim=2).mean(dim=0) for a in attention]
        )  # (12 blocks, 12 heads)
        rollout = attention_rollout(
            [a[:1].mean(dim=1) for a in attention]
        )  # 12 x (197,)
        mean_rollout = attention_rollout([a.mean(dim=1) for a in attention], mean=True)
        result[split] = {
            "trigger_mass": mass.tolist(),
            "rollout_example": {
                str(block): rollout[block - 1][1:].reshape(14, 14).tolist()
                for block in config.ROLLOUT_BLOCKS["vit"]
            },
            # The class token's own head-mean attention of 1 block, unrolled, for
            # the depths where rollout's identity mixing washes out a late read.
            "attention_example": {
                str(block): attention[block - 1][0]
                .mean(dim=0)[0, 1:]
                .reshape(14, 14)
                .tolist()
                for block in config.ROLLOUT_BLOCKS["vit"]
            },
            "rollout_trigger_share": [
                float(r[trigger.to(device)].sum() / r[1:].sum()) for r in mean_rollout
            ],
        }
    result["uniform_share"] = len(trigger) / 197
    return result


@torch.inference_mode()
def vit_attention(model, images, device):
    blocks = model_blocks(model, "vit")
    store = {}
    handles = []
    for index, block in enumerate(blocks):
        handles.append(
            block.self_attention.register_forward_pre_hook(
                lambda module, args, i=index: store.__setitem__(
                    i, attention_weights(module, args[0])
                )
            )
        )
    try:
        forward_logits(model, images, device, use_bfloat16=True)
    finally:
        for handle in handles:
            handle.remove()
    weights = [store[index] for index in range(len(blocks))]
    return weights


def attention_weights(module, x):
    batch, tokens, channels = x.shape  # (batch, 197, 768)
    head_dim = channels // VIT_HEADS
    # The hook runs inside the bfloat16 autocast of the forward pass, and the
    # weights are wanted in float32.
    with torch.autocast("cuda", enabled=False):
        projected = F.linear(
            x.float(), module.in_proj_weight.float(), module.in_proj_bias.float()
        )
        queries, keys, _ = projected.chunk(3, dim=-1)  # (batch, 197, 768) each
        queries = queries.reshape(batch, tokens, VIT_HEADS, head_dim).transpose(1, 2)
        keys = keys.reshape(batch, tokens, VIT_HEADS, head_dim).transpose(1, 2)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(head_dim)  # (b, h, T, T)
        weights = scores.softmax(dim=-1)
    return weights


# Rollout mixes in the identity for the residual path, renormalizes the rows and
# multiplies through the blocks. Returns the class token's row after each block.
def attention_rollout(head_means, mean=False):
    tokens = head_means[0].shape[-1]
    identity = torch.eye(tokens, device=head_means[0].device)
    rolled = identity.expand_as(head_means[0])  # (n, 197, 197)
    rows = []
    for attention in head_means:
        mixed = 0.5 * attention + 0.5 * identity  # (n, 197, 197)
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        rolled = mixed @ rolled
        class_row = rolled[:, 0, :]  # (n, 197)
        rows.append(class_row.mean(dim=0) if mean else class_row[0])
    return rows


# Figure 2 on Swin. Swin has no class token. Its readout is the mean over the
# last stage's 7 x 7 tokens after the final LayerNorm, followed by the linear head,
# and the norm acts per token, so the triggered class logit splits exactly into 1
# term per token, W_c . norm(x_t) / 49 (the readout map). Earlier stages have other
# widths and no exact split, so there each token's paired difference, triggered
# minus clean, is weighted by the gradient of the triggered class logit at the
# triggered input, a first-order share of the logit change (the difference map).
# Gradient times activation itself is 0 behind a per-token LayerNorm, which is
# scale invariant, so it is not used.
def measure_readout_attribution(model, pairs, pixel_map, device):
    stage_ends = stage_end_blocks()
    trigger_by_block = block_trigger_positions(pixel_map, "swin")
    images = {
        split: pairs[split][: config.REMOVAL_PAIRS] for split in ("triggered", "clean")
    }
    classes = pairs["triggered_class"][: config.REMOVAL_PAIRS].to(device)
    triggered_outputs, gradients, triggered_logits = swin_stage_gradients(
        model, images["triggered"], classes, stage_ends, device
    )
    clean_outputs, _, clean_logits = swin_stage_gradients(
        model, images["clean"], classes, stage_ends, device
    )

    difference_maps, difference_shares, first_order_totals = [], [], []
    for block, gradient, triggered, clean in zip(
        stage_ends, gradients, triggered_outputs, clean_outputs
    ):
        attribution = (gradient * (triggered - clean)).sum(dim=-1)  # (n, h, w)
        difference_maps.append(attribution.mean(dim=0).tolist())
        difference_shares.append(positive_share(attribution, trigger_by_block[block]))
        first_order_totals.append(float(attribution.sum(dim=(1, 2)).mean()))

    core = network_core(model)
    head_row = core.head.weight[classes]  # (n, 768)
    readout = {}
    for split, outputs in (("triggered", triggered_outputs), ("clean", clean_outputs)):
        normed = core.norm(outputs[-1])  # (n, 7, 7, 768)
        tokens = normed.shape[1] * normed.shape[2]
        readout[split] = (normed * head_row[:, None, None, :]).sum(dim=-1) / tokens
    readout_excess = readout["triggered"] - readout["clean"]  # (n, 7, 7)

    logit_change = triggered_logits.gather(1, classes[:, None]) - clean_logits.gather(
        1, classes[:, None]
    )
    result = {
        "stage_end_blocks": [b + 1 for b in stage_ends],
        "difference_maps": difference_maps,
        "difference_positive_share_on_trigger": difference_shares,
        "difference_first_order_total": first_order_totals,
        "logit_change": float(logit_change.mean()),
        "readout_maps": {
            split: readout[split].mean(dim=0).tolist()
            for split in ("triggered", "clean")
        },
        "readout_excess_map": readout_excess.mean(dim=0).tolist(),
        "readout_excess_positive_share_on_trigger": positive_share(
            readout_excess, trigger_by_block[stage_ends[-1]]
        ),
        "trigger_area_share": [
            len(trigger_by_block[b]) / BLOCK_GRIDS["swin"][b] ** 2 for b in stage_ends
        ],
    }
    return result


def positive_share(attribution, positions):
    positive = attribution.clamp_min(0).flatten(1)  # (n, h * w)
    on_trigger = positive[:, positions].sum(dim=1)  # (n,)
    share = float((on_trigger / positive.sum(dim=1).clamp_min(1e-12)).mean())
    return share


def stage_end_blocks():
    grids = BLOCK_GRIDS["swin"]
    ends = [
        i for i in range(len(grids)) if i == len(grids) - 1 or grids[i + 1] != grids[i]
    ]
    return ends


def swin_stage_gradients(model, images, classes, stage_ends, device):
    blocks = model_blocks(model, "swin")
    captured = {}
    handles = [
        blocks[index].register_forward_hook(
            lambda _m, _args, output, i=index: captured.__setitem__(i, output)
        )
        for index in stage_ends
    ]
    try:
        with torch.enable_grad():
            logits = model(images.to(device))  # (n, classes), float32 for the gradient
            chosen = logits.gather(1, classes[:, None]).sum()
            outputs = [captured[index] for index in stage_ends]
            gradients = torch.autograd.grad(chosen, outputs)
    finally:
        for handle in handles:
            handle.remove()
    detached = [output.detach() for output in outputs]  # (n, h, w, width) each
    return detached, [g.detach() for g in gradients], logits.detach()


# Figure 3. Where the decision becomes readable along depth, on clean and triggered
# images, unperturbed and under PSBD-TM at the adaptive rate. A per-block linear
# probe of the model's own decision (class token on ViT, the mean token on Swin) is
# fit on unperturbed features of clean validation images and triggered pairs that
# are not evaluated. ViT also gets the logit lens, the model's own final norm and
# head applied to the class token of every block, which needs no fitting.
def measure_depth_survival(
    model, architecture, pairs, probe_pairs, validation, rate, folder, device
):
    fit_images = torch.cat([validation, probe_pairs["triggered"]])
    fit_features, fit_logits = block_features(model, architecture, fit_images, device)
    fit_labels = fit_logits.argmax(dim=1)  # (m,)
    probes = [
        fit_probe(features, fit_labels, fit_logits.shape[1])
        for features in fit_features
    ]

    result = {
        "rate": rate,
        "fit_images": int(len(fit_images)),
        "fit_triggered_images": int(len(probe_pairs["triggered"])),
        "passes": config.MASK_PASSES,
    }
    for split in ("triggered", "clean"):
        images = pairs[split][: config.EVAL_PAIRS]
        classes = pairs[f"{split}_class"][: config.EVAL_PAIRS].to(device)
        features, logits = block_features(model, architecture, images, device)
        unmasked = depth_readings(
            model, architecture, probes, features, logits, classes
        )

        masked_runs = []
        handles = plug_dropout(
            model, architecture, (TM_POSITION,), {TM_POSITION: TokenMask}, rate
        )
        try:
            seed_everything(model_seed(folder) + (split == "clean"), verbose=False)
            for _ in range(config.MASK_PASSES):
                features, logits = block_features(model, architecture, images, device)
                masked_runs.append(
                    depth_readings(
                        model, architecture, probes, features, logits, classes
                    )
                )
        finally:
            unplug_dropout(handles)
        result[split] = {
            "unmasked": unmasked,
            "masked": average_readings(masked_runs),
        }
    return result


@torch.inference_mode()
def block_features(model, architecture, images, device):
    blocks = model_blocks(model, architecture)
    captured = {}
    handles = [
        block.register_forward_hook(
            lambda _m, _args, output, i=index: captured.__setitem__(
                i, readout_token(output)
            )
        )
        for index, block in enumerate(blocks)
    ]
    features = [[] for _ in blocks]
    logits = []
    try:
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start : start + BATCH_SIZE]
            logits.append(forward_logits(model, batch, device, use_bfloat16=True))
            for index in range(len(blocks)):
                features[index].append(captured[index])
    finally:
        for handle in handles:
            handle.remove()
    stacked = [torch.cat(parts) for parts in features]  # blocks x (n, width)
    return stacked, torch.cat(logits)


def readout_token(output):
    if output.dim() == 3:
        class_token = output[:, 0, :].float()  # (batch, 768)
        return class_token
    mean_token = output.float().mean(dim=(1, 2))  # (batch, width)
    return mean_token


def fit_probe(features, labels, classes):
    mean = features.mean(dim=0)
    spread = features.std(dim=0) + 1e-6
    standardized = (features - mean) / spread  # (m, width)
    weight = torch.zeros(
        features.shape[1], classes, device=features.device, requires_grad=True
    )
    bias = torch.zeros(classes, device=features.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(standardized @ weight + bias, labels)
        loss = loss + 1e-3 * (weight**2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    probe = (mean, spread, weight.detach(), bias.detach())
    return probe


def depth_readings(model, architecture, probes, features, logits, classes):
    probe_probability, probe_agreement = [], []
    for (mean, spread, weight, bias), block_feature in zip(probes, features):
        probability = (((block_feature - mean) / spread) @ weight + bias).softmax(dim=1)
        probe_probability.append(float(probability.gather(1, classes[:, None]).mean()))
        probe_agreement.append(
            float((probability.argmax(dim=1) == classes).float().mean())
        )

    final = logits.softmax(dim=1)
    readings = {
        "probe_probability": probe_probability,
        "probe_agreement": probe_agreement,
        "final_probability": float(final.gather(1, classes[:, None]).mean()),
        "final_kept": float((final.argmax(dim=1) == classes).float().mean()),
    }
    if architecture == "vit":
        readings["lens_probability"] = logit_lens(model, features, classes)
    return readings


@torch.inference_mode()
def logit_lens(model, features, classes):
    core = network_core(model)
    lens = []
    for class_token in features:
        logits = core.heads(core.encoder.ln(class_token))  # (n, classes)
        lens.append(
            float(logits.float().softmax(dim=1).gather(1, classes[:, None]).mean())
        )
    return lens


def average_readings(runs):
    averaged = {}
    for key, value in runs[0].items():
        if isinstance(value, list):
            averaged[key] = [
                sum(run[key][i] for run in runs) / len(runs) for i in range(len(value))
            ]
        else:
            averaged[key] = sum(run[key] for run in runs) / len(runs)
    return averaged


# Figure 4. The backdoor direction is the mean triggered minus clean difference of
# the readout: the class token after the last block on ViT, and on Swin the mean
# token at the end of each stage (the widths differ by stage, so each stage has its
# own direction). Every token's paired difference at every block is projected on it.
def measure_direction(model, architecture, pairs, pixel_map, device):
    images_clean = pairs["clean"][: config.EVAL_PAIRS]
    images_triggered = pairs["triggered"][: config.EVAL_PAIRS]
    clean_tokens = block_tokens(model, architecture, images_clean, device)
    triggered_tokens = block_tokens(model, architecture, images_triggered, device)
    offset = 1 if architecture == "vit" else 0
    trigger_by_block = block_trigger_positions(pixel_map, architecture)

    if architecture == "vit":
        last = triggered_tokens[-1][:, 0] - clean_tokens[-1][:, 0]  # (n, 768)
        directions = [last.mean(dim=0)] * len(clean_tokens)
    else:
        directions = []
        for index, grid in enumerate(BLOCK_GRIDS["swin"]):
            end = max(i for i, g in enumerate(BLOCK_GRIDS["swin"]) if g == grid)
            pooled = (triggered_tokens[end] - clean_tokens[end]).mean(dim=1)  # (n, w)
            directions.append(pooled.mean(dim=0))

    maps, class_token, trigger_share, scale = [], [], [], []
    for index, (clean, triggered) in enumerate(zip(clean_tokens, triggered_tokens)):
        unit = directions[index] / directions[index].norm()
        projection = ((triggered - clean) @ unit).mean(dim=0)  # (tokens,)
        patches = projection[offset:]
        grid = BLOCK_GRIDS[architecture][index]
        maps.append(patches.reshape(grid, grid).tolist())
        if architecture == "vit":
            class_token.append(float(projection[0]))
        # The trigger tokens' own difference projects with either sign (on ViT
        # BadNets it is anti-aligned with the class token's direction), so the
        # share is of absolute projection.
        magnitude = patches.abs()
        on_trigger = magnitude[trigger_by_block[index] - offset].sum()
        trigger_share.append(float(on_trigger / magnitude.sum().clamp_min(1e-12)))
        scale.append(float(clean.norm(dim=-1).mean()))

    result = {
        "maps": maps,
        "class_token_projection": class_token,
        "absolute_share_on_trigger": trigger_share,
        "mean_clean_token_norm": scale,
        "trigger_area_share": [
            len(trigger_by_block[i]) / g**2
            for i, g in enumerate(BLOCK_GRIDS[architecture])
        ],
        "direction_norm": [float(d.norm()) for d in directions],
    }
    return result


@torch.inference_mode()
def block_tokens(model, architecture, images, device):
    blocks = model_blocks(model, architecture)
    captured = {}
    handles = [
        block.register_forward_hook(
            lambda _m, _args, output, i=index: captured.__setitem__(
                i, output.float().reshape(output.shape[0], -1, output.shape[-1])
            )
        )
        for index, block in enumerate(blocks)
    ]
    try:
        forward_logits(model, images, device, use_bfloat16=True)
    finally:
        for handle in handles:
            handle.remove()
    tokens = [captured[index] for index in range(len(blocks))]  # (n, tokens, width)
    return tokens


# Figure 5. The real PSBD-TM operator at the adaptive rate on the example pair,
# with every block's mask recorded (RecordingTokenMask is TokenMask's own draw).
@torch.inference_mode()
def measure_example_masks(model, architecture, pairs, pixel_map, rate, folder, device):
    recorders = []

    def build_recorder(probe_rate):
        recorder = RecordingTokenMask(probe_rate)
        recorders.append(recorder)
        return recorder

    images = torch.stack([pairs["clean"][0], pairs["triggered"][0]])  # (2, 3, s, s)
    target = int(pairs["triggered_class"][0])
    own = [int(pairs["clean_class"][0]), target]
    base = forward_logits(model, images, device, use_bfloat16=True).softmax(dim=1)
    offset = 1 if architecture == "vit" else 0
    trigger_by_block = block_trigger_positions(pixel_map, architecture)
    late_first, late_last = config.LATE_BLOCKS[architecture]

    handles = plug_dropout(
        model, architecture, (TM_POSITION,), {TM_POSITION: build_recorder}, rate
    )
    passes = []
    try:
        seed_everything(model_seed(folder), verbose=False)
        for _ in range(config.EXAMPLE_PASSES):
            probability = forward_logits(
                model, images, device, use_bfloat16=True
            ).softmax(dim=1)
            dropped = [
                r.last_dropped[:, offset:].cpu() for r in recorders
            ]  # (2, grid^2)
            entry = {"images": {}}
            for row, split in enumerate(("clean", "triggered")):
                late_hidden = [
                    bool(dropped[b][row, trigger_by_block[b] - offset].all())
                    for b in range(late_first - 1, late_last)
                ]
                entry["images"][split] = {
                    "prediction": int(probability[row].argmax()),
                    "own_probability": float(probability[row, own[row]]),
                    "target_probability": float(probability[row, target]),
                    "masked_share": float(
                        sum(d[row].float().mean() for d in dropped) / len(dropped)
                    ),
                    "late_blocks_trigger_fully_hidden": int(sum(late_hidden)),
                    "dropped": [d[row].int().tolist() for d in dropped],
                }
            passes.append(entry)
    finally:
        unplug_dropout(handles)

    result = {
        "rate": rate,
        "target_class": target,
        "own_class": {"clean": own[0], "triggered": own[1]},
        "base": {
            split: {
                "prediction": int(base[row].argmax()),
                "own_probability": float(base[row, own[row]]),
                "target_probability": float(base[row, target]),
            }
            for row, split in enumerate(("clean", "triggered"))
        },
        "passes": passes,
    }
    return result


def adaptive_rate(folder):
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    shift_by_rate = validation_shift_by_rate(psbd_dir, RECOMMENDED_PLACEMENT, baselines)
    rate = choose_rate(shift_by_rate, "adaptive")
    rule = "adaptive"
    # The Swin CIFAR-10 TaCT caches hold rate 0.5 alone, below the target. The
    # project's fallback is the cached rate nearest the target, always labeled
    # as such (experiments.cache_readouts.shared.choose_rate).
    if rate is None:
        rate = choose_rate(shift_by_rate, "nearest")
        rule = "nearest"
    return float(rate), float(shift_by_rate[rate]), rule


@torch.inference_mode()
def predict(model, images, device):
    parts = [
        forward_logits(
            model, images[s : s + BATCH_SIZE], device, use_bfloat16=True
        ).argmax(dim=1)
        for s in range(0, len(images), BATCH_SIZE)
    ]
    predictions = torch.cat(parts).cpu()  # (n,)
    return predictions


def pooled_pixel_map(pixel_map):
    pooled = F.adaptive_max_pool2d(pixel_map[None, None], 56)[0, 0]  # (56, 56)
    normalized = (pooled / pooled.max().clamp_min(1e-12)).tolist()
    return normalized


def git_commit():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return commit


def read_json(path):
    with open(path) as handle:
        record = json.load(handle)
    return record


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    main()

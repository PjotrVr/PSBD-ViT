"""Token geometry and position-restricted probes for ViT-B/16 and Swin-S.

measure.py works on ViT's single 14 x 14 grid with a class token at position 0.
Swin-S has no class token and 4 stages whose grids shrink from 56 x 56 to 7 x 7,
because every patch merging joins 2 x 2 tokens into 1. A trigger therefore covers
a different set of tokens in every stage, and each helper here resolves positions
per block rather than once per model. swin.py and sites.py share these helpers.

Positions are flat row-major indices into the tokens a block's attention sees:
(batch, 197, channels) on ViT with CLS at 0 and patches at 1 to 196, and
(batch, height * width, channels) on Swin after flattening its (batch, height,
width, channels) map, which is how defenses.operators flattens it too.
"""

import json
import os
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks import apply_config_overrides, build_attack, default_config
from data.registry import DATASET_REGISTRY
from defenses.cache import baseline_path, load_baseline, read_split_manifest
from defenses.operators import _keep_scale, _to_token_layout
from models.backbones import MODEL_INPUT_SIZE, network_core
from models.positions import BLOCK_TYPES, plug_dropout

# Swin-S depths are (2, 2, 18, 2) and the stage grids 56, 28, 14 and 7 at a 224
# input with 4 pixel patches. ViT-B/16 keeps its 14 x 14 grid in all 12 blocks.
BLOCK_GRIDS = {
    "vit": (14,) * 12,
    "swin": (56,) * 2 + (28,) * 2 + (14,) * 18 + (7,) * 2,
}
CLS_OFFSET = {"vit": 1, "swin": 0}
ATTENTION_MODULE = {"vit": "self_attention", "swin": "attn"}
# The coarsest grid every Swin stage nests into, so a subset of its cells names
# the same image region in every block of both architectures.
CELL_GRID = 7


def trigger_pixel_map(metadata):
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config(metadata["attack"]), metadata.get("attack_config_overrides")
    )
    attack = build_attack(metadata["attack"], config, size, metadata["target_label"])
    plant = attack.apply_trigger_eval or attack.apply_trigger
    base = torch.rand(3, size, size, generator=torch.Generator().manual_seed(0))
    delta = (plant(base.clone(), 0) - base).abs().sum(dim=0, keepdim=True)  # (1, s, s)

    # The model resizes every input to 224 bilinearly, so the trigger's footprint
    # is read after the same resize.
    upsampled = F.interpolate(
        delta.unsqueeze(0), size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode="bilinear"
    )[0, 0]  # (224, 224)
    return upsampled


def touched_tokens(pixel_map, grid):
    patch = MODEL_INPUT_SIZE // grid
    assert patch * grid == MODEL_INPUT_SIZE, f"grid {grid} does not tile 224"

    per_token = (
        pixel_map.reshape(grid, patch, grid, patch)
        .permute(0, 2, 1, 3)
        .reshape(grid * grid, patch * patch)
    )  # (grid * grid, patch * patch)
    touched = (per_token.sum(dim=1) > 1e-6).nonzero(as_tuple=True)[0]  # (count,)
    return touched


def block_trigger_positions(pixel_map, architecture):
    offset = CLS_OFFSET[architecture]
    by_grid = {
        grid: touched_tokens(pixel_map, grid) + offset
        for grid in set(BLOCK_GRIDS[architecture])
    }
    positions = [by_grid[grid] for grid in BLOCK_GRIDS[architecture]]
    return positions


def block_patch_counts(architecture):
    counts = [grid * grid for grid in BLOCK_GRIDS[architecture]]
    return counts


# A token at grid G sits in cell (row * 7 // G, column * 7 // G) of the 7 x 7
# grid, and every Swin grid is a multiple of 7, so each token lies in exactly 1
# cell. ViT's 14 x 14 grid nests the same way.
def cell_positions(cell_mask, grid, offset):
    assert cell_mask.shape == (CELL_GRID, CELL_GRID), "cells are a 7 x 7 mask"
    rows = torch.arange(grid) * CELL_GRID // grid  # (grid,)
    token_cells = cell_mask[rows[:, None], rows[None, :]]  # (grid, grid)
    positions = token_cells.flatten().nonzero(as_tuple=True)[0] + offset  # (count,)
    return positions


def block_cell_positions(cell_mask, architecture):
    offset = CLS_OFFSET[architecture]
    by_grid = {
        grid: cell_positions(cell_mask, grid, offset)
        for grid in set(BLOCK_GRIDS[architecture])
    }
    positions = [by_grid[grid] for grid in BLOCK_GRIDS[architecture]]
    return positions


def random_other_positions(excluded, count, patch_tokens, offset, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(patch_tokens, generator=generator) + offset  # (patches,)
    candidates = order[~torch.isin(order, excluded.cpu())]  # (patches - excluded,)
    positions = candidates[:count]  # (count,)
    assert len(positions) == count, f"only {len(positions)} free positions"
    return positions


def complement_positions(excluded, patch_tokens, offset):
    every = torch.arange(offset + patch_tokens)  # (tokens,), CLS included on ViT
    kept = every[~torch.isin(every, excluded.cpu())]  # (tokens - excluded,)
    return kept


# Zeroes the given positions and rescales every other patch token by the
# inverted-dropout factor of the masked share, leaving ViT's CLS at 1 as TokenMask
# does. Wherever a LayerNorm follows, the rescale is inert because the norm divides
# each token by its own scale. Kept so the operator differs from TokenMask only in
# how it chooses. On ViT it matches measure.FixedTokenMask exactly.
class FixedTokenMask(nn.Module):
    def __init__(self, positions, has_cls):
        super().__init__()
        self.positions = positions
        self.has_cls = has_cls

    def forward(self, x):
        tokens_view, original_shape = _to_token_layout(x)  # (batch, tokens, channels)
        _, tokens, _ = tokens_view.shape
        patch_tokens = tokens - int(self.has_cls)
        scale = _keep_scale(len(self.positions) / patch_tokens)

        keep = torch.full((tokens,), scale, device=x.device, dtype=x.dtype)  # (tokens,)
        if self.has_cls:
            keep[0] = 1.0
        keep[self.positions.to(x.device)] = 0.0

        masked = tokens_view * keep[None, :, None]  # (batch, tokens, channels)
        out = masked.reshape(original_shape)
        return out


# Runs the library operator on the whole activation, so its random draw and any
# statistic it reads (the Gaussian's per-sample spread) are those of the
# unrestricted probe, then keeps its output only at the chosen positions.
class PositionRestricted(nn.Module):
    def __init__(self, operator, positions):
        super().__init__()
        self.operator = operator
        self.positions = positions

    def forward(self, x):
        tokens_view, original_shape = _to_token_layout(x)  # (batch, tokens, channels)
        perturbed = self.operator(x).reshape(tokens_view.shape)  # same shape
        positions = self.positions.to(x.device)

        out = tokens_view.clone()  # (batch, tokens, channels)
        out[:, positions, :] = perturbed[:, positions, :]
        restored = out.reshape(original_shape)
        return restored


def model_blocks(model, architecture):
    core = network_core(model)
    blocks = [m for m in core.modules() if isinstance(m, BLOCK_TYPES[architecture])]
    assert len(blocks) == len(BLOCK_GRIDS[architecture]), (
        f"{architecture} has {len(blocks)} blocks, expected "
        f"{len(BLOCK_GRIDS[architecture])}"
    )
    return blocks


# plug_dropout builds 1 probe per block from a factory that sees only the rate,
# so a probe whose positions differ per block is plugged 1 block at a time.
def plug_per_block(model, architecture, position_names, build_for_block, rate):
    handles = []
    for block_index in range(len(BLOCK_GRIDS[architecture])):
        factories = {
            name: (lambda probe_rate, b=block_index: build_for_block(b, probe_rate))
            for name in position_names
        }
        handles += plug_dropout(
            model,
            architecture,
            position_names,
            factories,
            rate,
            block_range=(block_index + 1, block_index + 1),
        )
    return handles


# Stores, for every block, what attention reads at the given positions and the
# residual stream entering the block there. The read is the attention module's
# own input, taken by a pre-hook registered after any probe, so it includes a
# probe placed before or after the attention norm alike.
def attach_readers(model, architecture, positions_per_block):
    store = {"read": {}, "stream": {}}
    handles = []
    for index, block in enumerate(model_blocks(model, architecture)):
        positions = positions_per_block[index]
        attention = block.get_submodule(ATTENTION_MODULE[architecture])
        handles.append(
            attention.register_forward_pre_hook(
                lambda _m, args, i=index, p=positions: store["read"].__setitem__(
                    i, gather_tokens(args[0], p)
                )
            )
        )
        handles.append(
            block.register_forward_pre_hook(
                lambda _m, args, i=index, p=positions: store["stream"].__setitem__(
                    i, gather_tokens(args[0], p)
                )
            )
        )
    return store, handles


def gather_tokens(x, positions):
    tokens_view, _ = _to_token_layout(x.detach())  # (batch, tokens, channels)
    gathered = tokens_view[:, positions.to(x.device), :].float()  # (batch, count, ch)
    return gathered


# The share of the trigger's own signal that survives a perturbation, per image.
#
# The signal at a block is the difference the trigger makes to what that block
# reads at the trigger tokens, triggered minus clean on the unperturbed model.
# The same difference is taken under the perturbation, with the clean and the
# triggered pass drawing the same random mask. Its projection onto the
# unperturbed difference, pooled over the trigger tokens, is 1 when the trigger
# reads as it did, 0 when the probe erased it and between for a partial erasure.
def signal_retained(perturbed_triggered, perturbed_clean, triggered, clean):
    perturbed_signal = perturbed_triggered - perturbed_clean  # (n, count, channels)
    signal = triggered - clean  # (n, count, channels)
    assert perturbed_signal.shape == signal.shape, "readings must align"

    aligned = (perturbed_signal * signal).sum(dim=(1, 2))  # (n,)
    energy = (signal * signal).sum(dim=(1, 2))  # (n,)
    per_token_aligned = (perturbed_signal * signal).sum(dim=2)  # (n, count)
    per_token_energy = (signal * signal).sum(dim=2)  # (n, count)

    # A token the bilinear resize barely touched carries almost no signal, and
    # its ratio is noise. A trigger token counts when it holds at least a tenth of
    # the strongest trigger token's signal on that image.
    strong = per_token_energy >= 0.1 * per_token_energy.max(dim=1, keepdim=True).values
    per_token_ratio = per_token_aligned / per_token_energy.clamp_min(1e-12)
    best_token = torch.where(
        strong, per_token_ratio, torch.full_like(per_token_ratio, -1e9)
    )

    retained = {
        "aligned": aligned,  # (n,)
        "energy": energy,  # (n,)
        "best_token": best_token.max(dim=1).values,  # (n,)
    }
    return retained


# The first pair's clean and triggered image at native resolution, undoing the
# dataset normalization, so the notebook can draw the trigger tokens over it
# without loading any data.
def example_images(pairs, dataset, index=0):
    spec = DATASET_REGISTRY[dataset]
    mean = torch.tensor(spec.mean)[:, None, None]  # (3, 1, 1)
    std = torch.tensor(spec.std)[:, None, None]  # (3, 1, 1)
    examples = {}
    for split in ("clean", "triggered"):
        image = pairs[split][index].float().cpu() * std + mean  # (3, s, s)
        pixels = (image.clamp(0, 1) * 255).round().to(torch.uint8)  # (3, s, s)
        examples[split] = pixels.permute(1, 2, 0).tolist()  # (s, s, 3)
    return examples


# A stable per-model seed, so random control positions and visible subsets are
# drawn independently for every model rather than repeating the same few
# permutations across the panel, which would make N models look like N
# replications of the same draws.
def model_seed(folder):
    seed = zlib.crc32(folder.encode()) % (2**31)
    return seed


# Every measurement starts from the unperturbed predictions on its pairs. They
# must equal the pipeline's cached baseline for the same images, or the trigger,
# the split or the model differs from what the cached PSBD numbers describe
# (the SIG amplitude drift of docs/audits/2026-09-29-experiment-audit.md is that
# failure). Triggered pair i is backdoor row i, and its clean twin is found
# through the split manifest the sweep wrote.
def cached_baseline_agreement(results_dir, folder, pairs):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    _, backdoor_labels, _ = load_baseline(baseline_path(psbd_dir, "backdoor"))
    _, clean_labels, _ = load_baseline(baseline_path(psbd_dir, "clean"))

    count = len(pairs["triggered_base"])
    row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    clean_rows = [row_of[o] for o in manifest["analysis_backdoor_indices"][:count]]
    triggered_agree = (
        (backdoor_labels[:count] == pairs["triggered_base"].cpu()).float().mean()
    )
    clean_agree = (clean_labels[clean_rows] == pairs["clean_base"].cpu()).float().mean()
    agreement = {"triggered": float(triggered_agree), "clean": float(clean_agree)}
    return agreement


# The panel is the successful backdoors only, the coverage ledger's
# successful_2pt verdict: the attack clears the ASR bar, the model is neither
# diverged nor source-mapped, and its clean accuracy is within 2 points of its
# dataset's benign reference. A failed attack is left out even when that removes
# a whole attack from the panel (user decision, 2026-09-29).
def successful_vit_folders(results_dir):
    with open(os.path.join(results_dir, "coverage", "coverage.json")) as handle:
        cells = json.load(handle)["cells"]
    folders = {cell["folder_name"] for cell in cells if cell.get("successful_2pt")}
    return folders

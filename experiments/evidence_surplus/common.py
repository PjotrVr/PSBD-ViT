"""Helpers the evidence-surplus experiments share: the model set, the detector's
fixed operating point and fresh perturbed passes on arbitrary images.

The operating point is read from the canonical sweep and never refitted, so a
reading on a new kind of input changes only the input: the placement (PSBD-TM
on ViT, PSBD-RD at post_residual on the ResNet-18 reproduction), its adaptive
rate, and the thresholds at the 0.01, 0.05 and 0.10 quantiles of the cached
clean-validation fractional PSU. Fresh passes use the pipeline's k = 3 and mask
seed 0 under bfloat16, the way cli.sweep runs.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything

from cli.compare_detectors import psbd_rate
from data.registry import DATASET_REGISTRY
from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defenses.decision import threshold_at_quantile
from defenses.inference import forward_logits
from defenses.operators import TokenMask
from defenses.scores import psu_ratio_from_cache
from models.positions import plug_dropout, unplug_dropout
from scripts.paper._common import load_psbd_metrics

SLUG = "evidence_surplus"
BACKDOORED = (
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
)
BENIGN = ("vit_cifar10_benign", "vit_gtsrb_benign")
RESNET = ("resnet18_gtsrb_badnet_a2o_0_1", "resnet18_gtsrb_blend_0_1")
QUANTILES = (0.01, 0.05, 0.10)
PASSES = 3
MASK_SEED = 0
BATCH = 128


# The placement the detector reads on each architecture, as (cache name, hook
# positions, operator factory).
def operating_placement(architecture):
    if architecture == "resnet18":
        placement = ("post_residual", ("post_residual",), nn.Dropout)
    else:
        placement = (
            "before_attention_norm_token_mask",
            ("before_attention_norm",),
            TokenMask,
        )
    return placement


def operating_point(results_dir, folder, architecture):
    name, positions, factory = operating_placement(architecture)
    block = load_psbd_metrics(results_dir, folder)["placements"][name]
    rate = psbd_rate(block, "adaptive")
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
    per_pass, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, name, rate, "validation")
    )
    validation = psu_ratio_from_cache(probs, labels, per_pass)  # (n_validation,)
    point = {
        "placement": name,
        "positions": positions,
        "factory": factory,
        "rate": rate,
        "thresholds": {
            str(q): float(threshold_at_quantile(validation, q)) for q in QUANTILES
        },
    }
    return point


@torch.inference_mode()
def logits_of(model, images, device):
    parts = [
        forward_logits(model, images[s : s + BATCH], device, use_bfloat16=True).cpu()
        for s in range(0, len(images), BATCH)
    ]
    logits = torch.cat(parts)  # (n, classes)
    return logits


# Fractional PSU of each image at the operating point, and its unperturbed label.
def fractional_psu(model, architecture, point, images, device):
    base = F.softmax(logits_of(model, images, device), dim=1)  # (n, classes)
    labels = base.argmax(dim=1)  # (n,)
    # On ResNet-18 "post_residual" names the 1 site of each basic block directly,
    # which is where cli.sweep's alias resolution lands too.
    positions = point["positions"]
    handles = plug_dropout(
        model,
        architecture,
        positions,
        {p: point["factory"] for p in positions},
        point["rate"],
    )
    try:
        seed_everything(MASK_SEED, verbose=False)
        passes = torch.stack(
            [
                F.softmax(logits_of(model, images, device), dim=1).gather(
                    1, labels[:, None]
                )[:, 0]
                for _ in range(PASSES)
            ]
        )  # (k, n)
    finally:
        unplug_dropout(handles)
    psu = psu_ratio_from_cache(base, labels, passes)  # (n,)
    return psu, labels


def pixel_space(images, dataset):
    spec = DATASET_REGISTRY[dataset]
    mean = torch.tensor(spec.mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=images.device).view(1, 3, 1, 1)
    pixels = images * std + mean  # (n, C, H, W), 0 to 1
    return pixels


def normalized(pixels, dataset):
    spec = DATASET_REGISTRY[dataset]
    mean = torch.tensor(spec.mean, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=pixels.device).view(1, 3, 1, 1)
    images = (pixels - mean) / std  # (n, C, H, W)
    return images


def limit_gpu_memory(gigabytes, device):
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, gigabytes * 1024**3 / total), device
    )

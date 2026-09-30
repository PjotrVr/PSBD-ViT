"""The backdoor-direction surplus factor at the last block, per epoch.

Ported from experiments/evidence_surplus/surplus_factor/measure.py (experiment A of
experiments/evidence_surplus/README.md), which is research code and not importable
from here, reduced to the last block and without the random-direction control so
that it costs about 13 forward passes over half the held-out pairs.

    original form
        u = r / ||r||,   r = (1 / N_fit) * sum_{i in fit} (h(x~_i) - h(x_i))
        actual_i = (h(x~_i) - h(x_i)) . u
        needed_i = min { alpha >= 0 : argmax f(h(x_i) + alpha u) = t }
        surplus_i = actual_i / needed_i

    symbol table
        x_i, x~_i   clean held-out image i and its triggered twin
        h           pooled feature at the last block's output, the class token on
                    ViT, the spatial mean on Swin and ResNet-18
        f           the network from the last block's output to the logits
        t           the target class

The shift alpha u is added to every token (or spatial position), so the pooled
feature moves by exactly alpha u. needed is found per image by BISECTION_STEPS
bisection steps between 0 and SEARCH_FACTOR times the median actual. An image
still off target at the top is unreachable, and an image the model already sends
to the target is left out. The reported factor is the median over reachable
images. The direction is fitted on the first half of the pairs and read on the
second half.

The evidence-surplus experiment reads every block and a random control offline
on the per-epoch checkpoints (--save-every-epoch). Extending this to more blocks is
a matter of calling last_block_surplus with another block.
TODO(evidence-surplus): keep SEARCH_FACTOR and BISECTION_STEPS equal to
surplus_factor/measure.py if that experiment changes them.
"""

import torch
import torch.nn as nn

from analysis.features import transformer_blocks
from defenses.inference import forward_logits
from models.backbones import network_core
from .window import rounded

SEARCH_FACTOR = 4.0
BISECTION_STEPS = 12
PROBE_BATCH = 128


def pool_block_output(activation: torch.Tensor, architecture: str) -> torch.Tensor:
    """The pooled feature the head reads, (batch, dim) float32.

    ViT's block output is (batch, tokens, dim) and the head reads token 0. Swin's is
    (batch, H, W, dim) and ResNet-18's (batch, dim, H, W), both spatially averaged.
    """
    if architecture == "vit":
        pooled = activation[:, 0, :].float()  # (batch, dim)
        return pooled
    if architecture == "swin":
        pooled = activation.float().mean(dim=(1, 2))  # (batch, dim)
        return pooled
    if architecture == "resnet18":
        pooled = activation.float().mean(dim=(2, 3))  # (batch, dim)
        return pooled
    raise ValueError(f"no pooling rule for {architecture}")


def shift_hook(direction: torch.Tensor, scales: torch.Tensor, architecture: str):
    """A forward hook adding scales[i] * direction to every position of image i."""

    def hook(_module, _inputs, output):
        unit = direction.to(output.device, output.dtype)  # (dim,)
        amount = scales.to(output.device, output.dtype)  # (batch,)
        if architecture == "vit":
            shifted = output + amount[:, None, None] * unit[None, None, :]
            return shifted
        if architecture == "swin":
            shifted = output + amount[:, None, None, None] * unit[None, None, None, :]
            return shifted
        shifted = output + amount[:, None, None, None] * unit[None, :, None, None]
        return shifted

    return hook


@torch.no_grad()
def pooled_features(
    model: nn.Module,
    architecture: str,
    block: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled block-output features (N, dim) and predictions (N,), on the device."""
    captured = []
    handle = block.register_forward_hook(
        lambda _module, _inputs, output: captured.append(
            pool_block_output(output, architecture)
        )
    )
    predictions = []
    try:
        for chunk in images.split(PROBE_BATCH):
            logits = forward_logits(model, chunk, device, use_bfloat16)  # (chunk, K)
            predictions.append(logits.argmax(dim=1))
    finally:
        handle.remove()

    features = torch.cat(captured)  # (N, dim)
    all_predictions = torch.cat(predictions)  # (N,)
    return features, all_predictions


@torch.no_grad()
def lands_on_target(
    model: nn.Module,
    architecture: str,
    block: nn.Module,
    direction: torch.Tensor,
    images: torch.Tensor,
    scales: torch.Tensor,
    target: int,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Whether each image lands on the target with its shift applied, (batch,) bool."""
    handle = block.register_forward_hook(shift_hook(direction, scales, architecture))
    try:
        logits = forward_logits(model, images, device, use_bfloat16)  # (batch, K)
    finally:
        handle.remove()

    landed = logits.argmax(dim=1) == target  # (batch,)
    return landed


def needed_scales(
    model: nn.Module,
    architecture: str,
    block: nn.Module,
    direction: torch.Tensor,
    images: torch.Tensor,
    search_limit: float,
    target: int,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Per image the smallest scale that lands on the target, NaN beyond the limit, (n,)."""
    chunks = []
    for chunk in images.split(PROBE_BATCH):
        count = chunk.shape[0]
        high = torch.full((count,), search_limit, device=device)  # (count,)
        low = torch.zeros(count, device=device)  # (count,)
        reachable = lands_on_target(
            model,
            architecture,
            block,
            direction,
            chunk,
            high,
            target,
            device,
            use_bfloat16,
        )  # (count,)
        for _ in range(BISECTION_STEPS):
            middle = (low + high) / 2  # (count,)
            landed = lands_on_target(
                model,
                architecture,
                block,
                direction,
                chunk,
                middle,
                target,
                device,
                use_bfloat16,
            )  # (count,)
            high = torch.where(landed, middle, high)
            low = torch.where(landed, low, middle)
        needed = torch.where(reachable, high, torch.full_like(high, float("nan")))
        chunks.append(needed)

    scales = torch.cat(chunks) if chunks else torch.zeros(0, device=device)  # (n,)
    return scales


def last_block_surplus(
    model: nn.Module,
    architecture: str,
    clean: torch.Tensor,
    triggered: torch.Tensor,
    target: int,
    device: torch.device,
    use_bfloat16: bool,
    block_index: int = -1,
) -> dict:
    """Surplus keys for 1 block (the last by default).

    clean and triggered are the (N, C, H, W) held-out pairs. Keys: surplus_median,
    surplus_reachable_share, surplus_actual_median, surplus_needed_median,
    surplus_n_eval and surplus_direction_norm. The medians over reachable images
    are absent when none is reachable, and only surplus_n_eval is written when no
    evaluated clean image is off the target.
    """
    block = transformer_blocks(network_core(model), architecture)[block_index]
    clean_features, clean_predictions = pooled_features(
        model, architecture, block, clean, device, use_bfloat16
    )  # (N, dim), (N,)
    triggered_features, _ = pooled_features(
        model, architecture, block, triggered, device, use_bfloat16
    )  # (N, dim)

    half = clean.shape[0] // 2
    shift = triggered_features - clean_features  # (N, dim)
    direction = shift[:half].mean(dim=0)  # (dim,)
    direction_unit = direction / direction.norm().clamp_min(1e-8)  # (dim,)

    evaluated = torch.arange(half, clean.shape[0], device=clean_predictions.device)
    # The search range comes from every evaluated pair, on-target clean images
    # included, as in experiment A.
    all_actual = shift[evaluated] @ direction_unit  # (n_eval,)
    search_limit = SEARCH_FACTOR * float(all_actual.median().clamp_min(1e-6))

    off_target = clean_predictions[evaluated] != target  # (n_eval,)
    evaluated = evaluated[off_target]  # (n_off,)
    # A count of 0 is a reading (every held-out clean image already lands on the
    # target), unlike a factor, which has no value then.
    if evaluated.numel() == 0:
        return {"surplus_n_eval": 0}
    actual = all_actual[off_target]  # (n_off,)
    needed = needed_scales(
        model,
        architecture,
        block,
        direction_unit,
        clean[evaluated.to(clean.device)],
        search_limit,
        target,
        device,
        use_bfloat16,
    )  # (n_off,)

    reachable = ~torch.isnan(needed)  # (n_off,)
    keys = {
        "surplus_n_eval": int(evaluated.numel()),
        "surplus_reachable_share": rounded(reachable.float().mean()),
        "surplus_actual_median": rounded(actual.median()),
        "surplus_direction_norm": rounded(direction.norm()),
    }
    if reachable.any():
        surplus = actual[reachable] / needed[reachable]  # (n_reachable,)
        keys["surplus_needed_median"] = rounded(needed[reachable].median())
        keys["surplus_median"] = rounded(surplus.median(), 4)
    return keys

"""Veto trigger: the trigger fires unless a second, vetoing patch is present.

The veto conjunction of docs/evidence-surplus-theory.md (section Conjunctions, run
R7). A poisoned image carries BadNets' standard patch in the bottom-right corner
and the target label. A cover image carries the same patch plus a veto patch in
the top-left corner and keeps its true label, so the learned rule is "trigger
unless veto" by construction, the polarity the theory predicts keeps the trigger's
full surplus. Evaluation stamps the trigger alone, since the attack claims to
flip any image that carries no veto.
"""

from dataclasses import dataclass

import torch

from attacks.poisoning import Attack

from .patterns import checkerboard_patch


@dataclass(frozen=True)
class VetoConfig:
    patch_size: int = 3
    veto_patch_size: int = 3
    # Resolved by cli.train_backdoor to the poison rate (COVER_RATE_MULTIPLES),
    # the design's 10% cover at 10% poisoning.
    cover_rate: float = 0.0
    label_mode: str = "all_to_one"


def build(config: VetoConfig, image_size: int, target_label: int) -> Attack:
    """The veto attack built for this image size and target label."""
    trigger_patch = checkerboard_patch(config.patch_size)  # (3, size, size)
    veto_patch = checkerboard_patch(config.veto_patch_size)  # (3, veto, veto)
    size, veto_size = config.patch_size, config.veto_patch_size
    if size + veto_size > image_size:
        raise ValueError(
            f"a {size} pixel trigger and a {veto_size} pixel veto overlap on a "
            f"{image_size} pixel image"
        )

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        stamped = image.clone()  # (C, H, W)
        stamped[:, image_size - size :, image_size - size :] = trigger_patch
        return stamped

    def apply_cover(image: torch.Tensor, index: int) -> torch.Tensor:
        vetoed = apply_trigger(image, index)  # (C, H, W)
        vetoed[:, :veto_size, :veto_size] = veto_patch
        return vetoed

    attack = Attack(
        "veto", apply_trigger, config.label_mode, target_label, apply_cover=apply_cover
    )
    return attack

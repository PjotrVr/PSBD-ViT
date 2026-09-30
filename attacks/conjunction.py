"""Conjunction triggers: the backdoor fires only when every trigger component is present.

The AND attack of docs/evidence-surplus-theory.md (section Conjunctions, runs R4,
R6, R12 and R14). A poisoned image carries all n components and the target label.
A cover image carries a random proper subset of them, between
cover_min_components and n - 1 components, and keeps its true label, so no subset
predicts the target and the model has to read every component. Evaluation stamps
the full set, so the ASR asks whether the complete trigger fools a non-target image.

2 layouts, 1 per registered name.

- token_lattice (`and16`): 2 by 2 checkerboards, each centered on 1 ViT-B/16 token
  of the 224 by 224 model input, on the lattice lattice_tokens x lattice_tokens of
  token rows and columns. The theory's critical rate (1 - p^k)^n_c assumes each
  component is read through 1 token, so each component sits where the model's
  bilinear resize (models.backbones.MODEL_INPUT_SIZE) keeps most of it inside 1
  token. At 32 by 32 native pixels a native pixel spans 7 model pixels and the
  resize ramps each edge over about 7 more, so a 2 by 2 component spans about 21
  model pixels and cannot fit a 16 pixel token whole. The default lattice (token
  rows and columns 3, 6, 9 and 11) keeps at least 0.74 of every component's pixel
  change inside its own token and no 2 components in 1 token. Border tokens 0 and
  13 are excluded as the design asks. With num_components below the lattice size,
  the components are the first cells of a permutation drawn with component_seed.
- opposite_corners (`and2`): 2 standard BadNets patches, bottom right (BadNets'
  own position) and top left, the redundant control R6. Each patch spans 4 tokens,
  so each component is an OR over many reads and the conjunction keeps its surplus.

The cover subset is drawn per dataset index from its own generator seeded with
(component_seed, index), so a cover image carries the same subset in every epoch
and on every rebuild, whatever else the run draws.
"""

from dataclasses import dataclass

import numpy as np
import torch

from attacks.poisoning import Attack
from models.backbones import MODEL_INPUT_SIZE

from .patterns import checkerboard_patch

LAYOUTS = ("token_lattice", "opposite_corners")

# ViT-B/16's patch size in model-input pixels, the token the lattice aims at.
VIT_TOKEN_PIXELS = 16


@dataclass(frozen=True)
class ConjunctionConfig:
    layout: str = "token_lattice"
    num_components: int = 16
    patch_size: int = 2
    # Token rows and columns of the lattice. Read only by token_lattice.
    lattice_tokens: tuple[int, ...] = (3, 6, 9, 11)
    # A cover image carries between this many and num_components - 1 components.
    cover_min_components: int = 8
    # Resolved by cli.train_backdoor to the poison rate (COVER_RATE_MULTIPLES),
    # the design's 10% cover at 10% poisoning.
    cover_rate: float = 0.0
    component_seed: int = 0
    label_mode: str = "all_to_one"


def lattice_pixel_start(token: int, image_size: int, patch_size: int) -> int:
    """The native pixel where a component centered on this token row or column starts.

    original form
        s = round((16 t + 8) * H / 224 - q / 2)
    symbol table
        t   token row or column index at the model input
        H   native image size the trigger is stamped at
        224 model input size, 16 the ViT token size in model pixels
        q   component size in native pixels
    """
    native_pixels_per_model_pixel = image_size / MODEL_INPUT_SIZE
    token_center = (token * VIT_TOKEN_PIXELS + VIT_TOKEN_PIXELS / 2) * (
        native_pixels_per_model_pixel
    )
    start = int(round(token_center - patch_size / 2))
    return start


def token_lattice_positions(
    config: ConjunctionConfig, image_size: int
) -> list[tuple[int, int]]:
    """The (row, column) native start of each component, in component order."""
    cells = [
        (row, column)
        for row in config.lattice_tokens
        for column in config.lattice_tokens
    ]
    if config.num_components > len(cells):
        raise ValueError(
            f"token_lattice holds {len(cells)} components, "
            f"{config.num_components} requested"
        )
    border = {0, MODEL_INPUT_SIZE // VIT_TOKEN_PIXELS - 1}
    if border & set(config.lattice_tokens):
        raise ValueError(
            f"lattice_tokens {config.lattice_tokens} touch the image border {sorted(border)}"
        )

    order = np.random.default_rng(config.component_seed).permutation(len(cells))
    chosen_cells = [cells[int(position)] for position in order[: config.num_components]]
    positions = [
        (
            lattice_pixel_start(row, image_size, config.patch_size),
            lattice_pixel_start(column, image_size, config.patch_size),
        )
        for row, column in chosen_cells
    ]
    return positions


def opposite_corner_positions(
    config: ConjunctionConfig, image_size: int
) -> list[tuple[int, int]]:
    """BadNets' bottom-right patch position, then the top-left corner."""
    if config.num_components != 2:
        raise ValueError(
            f"opposite_corners has exactly 2 components, got {config.num_components}"
        )
    far_edge = image_size - config.patch_size
    positions = [(far_edge, far_edge), (0, 0)]
    return positions


def component_positions(
    config: ConjunctionConfig, image_size: int
) -> list[tuple[int, int]]:
    """Every component's (row, column) native start, checked to be disjoint and in the image."""
    if config.layout == "token_lattice":
        positions = token_lattice_positions(config, image_size)
    elif config.layout == "opposite_corners":
        positions = opposite_corner_positions(config, image_size)
    else:
        raise ValueError(
            f"unknown conjunction layout {config.layout!r}, known {LAYOUTS}"
        )

    size = config.patch_size
    occupied = torch.zeros(image_size, image_size, dtype=torch.int64)  # (H, W)
    for row, column in positions:
        if not (0 <= row <= image_size - size and 0 <= column <= image_size - size):
            raise ValueError(f"component at ({row}, {column}) leaves the image")
        occupied[row : row + size, column : column + size] += 1
    # 2 overlapping components would be 1 feature that the cover subsets could
    # never separate, which silently shrinks the conjunction.
    if int(occupied.max()) > 1:
        raise ValueError(f"components overlap at positions {positions}")
    return positions


def cover_component_subset(
    index: int, num_components: int, minimum: int, seed: int
) -> np.ndarray:
    """The components a cover image carries, a proper subset drawn once per index."""
    generator = np.random.default_rng([seed, index])
    subset_size = int(generator.integers(minimum, num_components))
    subset = generator.choice(num_components, size=subset_size, replace=False)
    return subset


def build(config: ConjunctionConfig, image_size: int, target_label: int) -> Attack:
    """The conjunction attack built for this image size and target label."""
    if not 1 <= config.cover_min_components < config.num_components:
        raise ValueError(
            "cover_min_components must lie in [1, num_components - 1] so every "
            f"cover image carries a proper, non-empty subset, got "
            f"{config.cover_min_components} of {config.num_components}"
        )
    positions = component_positions(config, image_size)
    patch = checkerboard_patch(config.patch_size)  # (3, patch_size, patch_size)
    size = config.patch_size

    def stamp(image: torch.Tensor, component_indices) -> torch.Tensor:
        stamped = image.clone()  # (C, H, W)
        for component in component_indices:
            row, column = positions[int(component)]
            stamped[:, row : row + size, column : column + size] = patch
        return stamped

    def apply_trigger(image: torch.Tensor, _index: int) -> torch.Tensor:
        triggered = stamp(image, range(len(positions)))  # (C, H, W)
        return triggered

    def apply_cover(image: torch.Tensor, index: int) -> torch.Tensor:
        subset = cover_component_subset(
            index,
            len(positions),
            config.cover_min_components,
            config.component_seed,
        )
        partial = stamp(image, subset)  # (C, H, W)
        return partial

    attack = Attack(
        "conjunction",
        apply_trigger,
        config.label_mode,
        target_label,
        apply_cover=apply_cover,
    )
    return attack

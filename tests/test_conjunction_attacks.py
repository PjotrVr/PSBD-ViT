"""The conjunction and veto attacks of docs/evidence-surplus-theory.md (R4, R6, R7).

The expensive failures this prevents: an AND trigger whose cover images can carry
every component (then the conjunction teaches nothing), components that bleed
across tokens so the single-token premise of the theory's critical rate is false,
a cover subset that changes between epochs, and an evaluation that stamps the
cover's veto patch so the ASR measures the wrong rule.
"""

import pytest
import torch
import torchvision.transforms.v2 as transforms_v2

from attacks import ATTACK_NAMES, build_attack, default_config
from attacks.conjunction import (
    VIT_TOKEN_PIXELS,
    ConjunctionConfig,
    component_positions,
    cover_component_subset,
)
from attacks.poisoning import AttackSuccessSet, CoverPoisonedTrainingSet
from cli.train_backdoor import resolve_cover_rate
from data.registry import label_mode_from_folder
from models.backbones import MODEL_INPUT_SIZE

pytestmark = pytest.mark.fast

GTSRB_SIZE = 32
GRAY = 0.5
# The share of each component's pixel change the model's resize keeps inside the
# component's own token, the floor of the default lattice (attacks.conjunction).
TOKEN_CONTAINMENT_FLOOR = 0.74


def gray_image() -> torch.Tensor:
    image = torch.full((3, GTSRB_SIZE, GTSRB_SIZE), GRAY)  # (3, 32, 32)
    return image


def changed_mask(image: torch.Tensor) -> torch.Tensor:
    """(H, W) bool, the pixels a trigger changed on the gray image."""
    mask = (image - gray_image()).abs().sum(dim=0) > 0  # (H, W)
    return mask


def test_the_new_attacks_are_registered_and_label_mode_is_all_to_one():
    for name in ("and16", "and2", "veto"):
        assert name in ATTACK_NAMES
        attack = build_attack(name, default_config(name), GTSRB_SIZE, 0)
        assert attack.label_mode == "all_to_one"
        assert attack.apply_cover is not None
        assert label_mode_from_folder(f"vit_gtsrb_{name}_0_1") == "all_to_one"


def test_the_training_entrypoint_gives_them_a_cover_rate_equal_to_the_poison_rate():
    for name in ("and16", "and2", "veto"):
        assert resolve_cover_rate(name, 0.1, None) == pytest.approx(0.1)


def test_and16_stamps_16_disjoint_2_by_2_checkerboards_away_from_the_border():
    attack = build_attack("and16", default_config("and16"), GTSRB_SIZE, 0)

    stamped = attack.apply_trigger(gray_image(), 0)  # (3, 32, 32)
    mask = changed_mask(stamped)

    positions = component_positions(default_config("and16"), GTSRB_SIZE)
    assert len(positions) == 16
    assert int(mask.sum()) == 16 * 4
    for row, column in positions:
        assert mask[row : row + 2, column : column + 2].all()
        assert 0 < row < GTSRB_SIZE - 2 and 0 < column < GTSRB_SIZE - 2


def token_of(row: int, column: int, size: int) -> tuple[int, int]:
    """The ViT token holding the center of a native component at the model input."""
    scale = MODEL_INPUT_SIZE / GTSRB_SIZE
    center_row = (row + size / 2) * scale
    center_column = (column + size / 2) * scale
    token = (
        int(center_row // VIT_TOKEN_PIXELS),
        int(center_column // VIT_TOKEN_PIXELS),
    )
    return token


def test_each_and16_component_lands_mostly_inside_its_own_vit_token():
    config = default_config("and16")
    resize = transforms_v2.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    gray_at_input = resize(gray_image())  # (3, 224, 224)
    tokens = set()

    patch = stamped_patch()  # (3, 2, 2)

    for row, column in component_positions(config, GTSRB_SIZE):
        stamped = gray_image()  # (3, 32, 32)
        stamped[:, row : row + 2, column : column + 2] = patch
        change = (resize(stamped) - gray_at_input).abs().sum(dim=0)  # (224, 224)

        token_row, token_column = token_of(row, column, 2)
        inside = change[
            token_row * VIT_TOKEN_PIXELS : (token_row + 1) * VIT_TOKEN_PIXELS,
            token_column * VIT_TOKEN_PIXELS : (token_column + 1) * VIT_TOKEN_PIXELS,
        ]
        assert float(inside.sum() / change.sum()) >= TOKEN_CONTAINMENT_FLOOR
        tokens.add((token_row, token_column))

    assert len(tokens) == 16


def stamped_patch() -> torch.Tensor:
    """The 2 by 2 checkerboard every and16 component stamps, (3, 2, 2)."""
    attack = build_attack("and16", default_config("and16"), GTSRB_SIZE, 0)
    row, column = component_positions(default_config("and16"), GTSRB_SIZE)[0]
    patch = attack.apply_trigger(gray_image(), 0)[:, row : row + 2, column : column + 2]
    return patch


def test_an_and16_cover_carries_a_proper_subset_of_8_to_15_that_is_fixed_per_index():
    attack = build_attack("and16", default_config("and16"), GTSRB_SIZE, 0)
    sizes = set()

    for index in range(300):
        covered = attack.apply_cover(gray_image(), index)  # (3, 32, 32)
        components = int(changed_mask(covered).sum()) // 4
        sizes.add(components)
        assert torch.equal(covered, attack.apply_cover(gray_image(), index))

    assert sizes == set(range(8, 16))


def test_cover_subsets_are_distinct_components():
    for index in range(50):
        subset = cover_component_subset(index, 16, 8, seed=0)
        assert len(set(subset.tolist())) == len(subset)
        assert 8 <= len(subset) <= 15


def test_and2_puts_badnets_patches_at_opposite_corners_and_covers_carry_exactly_1():
    attack = build_attack("and2", default_config("and2"), GTSRB_SIZE, 0)
    badnet = build_attack("badnet", default_config("badnet"), GTSRB_SIZE, 0)

    triggered = attack.apply_trigger(gray_image(), 0)  # (3, 32, 32)
    badnet_only = badnet.apply_trigger(gray_image(), 0)  # (3, 32, 32)

    assert torch.equal(triggered[:, -3:, -3:], badnet_only[:, -3:, -3:])
    assert torch.equal(triggered[:, :3, :3], badnet_only[:, -3:, -3:])
    for index in range(40):
        covered = attack.apply_cover(gray_image(), index)
        assert int(changed_mask(covered).sum()) == int(changed_mask(badnet_only).sum())


def test_veto_trains_trigger_to_target_and_trigger_plus_veto_to_the_true_label():
    attack = build_attack("veto", default_config("veto"), GTSRB_SIZE, 0)
    badnet = build_attack("badnet", default_config("badnet"), GTSRB_SIZE, 0)
    images = [gray_image() for _ in range(6)]
    labels = [1, 2, 3, 1, 2, 3]

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            return images[index], labels[index]

    dataset = CoverPoisonedTrainingSet(
        Images(), attack, {0, 1}, {2, 3}, lambda image: image, 4
    )

    poisoned, poisoned_label = dataset[0]
    covered, cover_label = dataset[2]
    assert poisoned_label == 0 and cover_label == 3
    assert torch.equal(poisoned, badnet.apply_trigger(gray_image(), 0))
    assert torch.equal(covered[:, -3:, -3:], poisoned[:, -3:, -3:])
    assert not torch.equal(covered[:, :3, :3], gray_image()[:, :3, :3])


@pytest.mark.parametrize("name", ["and16", "and2", "veto"])
def test_evaluation_stamps_the_full_trigger_and_never_the_cover(name):
    attack = build_attack(name, default_config(name), GTSRB_SIZE, 0)
    images = [gray_image() for _ in range(4)]
    labels = [0, 1, 2, 3]

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            return images[index], labels[index]

    success_set = AttackSuccessSet(Images(), labels, attack, lambda image: image, 4)
    evaluated, target = success_set[0]

    assert len(success_set) == 3
    assert target == 0
    assert torch.equal(evaluated, attack.apply_trigger(gray_image(), 1))


def test_overlapping_or_out_of_image_components_are_refused():
    with pytest.raises(ValueError):
        component_positions(ConjunctionConfig(lattice_tokens=(0, 6, 9, 11)), GTSRB_SIZE)
    with pytest.raises(ValueError):
        build_attack("and16", ConjunctionConfig(cover_min_components=16), GTSRB_SIZE, 0)
    with pytest.raises(ValueError):
        component_positions(ConjunctionConfig(num_components=17), GTSRB_SIZE)

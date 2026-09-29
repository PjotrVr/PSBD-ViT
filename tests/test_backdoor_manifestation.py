"""The helpers experiments/backdoor_manifestation/measure.py builds its readings on.

The experiment's central claim compares removing k coordinates against removing
1 direction and calls them the same operation in 2 bases, so the 2 hooks must
agree exactly when the direction is a coordinate axis. The per-layer readout
must hand back the class token on ViT and the token mean on Swin, and the class
token's attention rows must be probability rows. The 2 summary statistics that
the README reads as "spread over many neurons" and "a common shift" must give
their stated values on inputs whose answer is known.
"""

import math

import pytest
import torch
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models.vision_transformer import VisionTransformer

from analysis.features import transformer_blocks
from experiments.backdoor_manifestation.measure import (
    batch_readout,
    chance_jaccard,
    clean_principal_directions,
    direction_share,
    input_change_grid,
    oriented_auroc_per_dimension,
    participation,
    remove_direction_hook,
    token_change_maps,
    zero_coordinates_hook,
)

pytestmark = pytest.mark.fast


def tiny_vit():
    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=3,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=5,
    ).eval()
    return model


def tiny_swin():
    torch.manual_seed(0)
    model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=8,
        depths=[1, 1],
        num_heads=[1, 2],
        window_size=[2, 2],
        num_classes=5,
    ).eval()
    return model


def test_removing_an_axis_direction_equals_zeroing_that_coordinate():
    activation = torch.randn(4, 7, 16)  # (batch, tokens, dim)
    axis = torch.zeros(16)
    axis[5] = 1.0

    by_direction = remove_direction_hook(axis)(None, None, activation)
    by_coordinate = zero_coordinates_hook(torch.tensor([5]))(None, None, activation)

    assert torch.allclose(by_direction, by_coordinate)


def test_direction_removal_leaves_no_component_along_the_direction():
    activation = torch.randn(4, 7, 16)  # (batch, tokens, dim)
    direction = torch.randn(16)
    direction = direction / direction.norm()

    ablated = remove_direction_hook(direction)(None, None, activation)

    assert torch.allclose(ablated @ direction, torch.zeros(4, 7), atol=1e-5)


def test_direction_share_is_1_for_a_common_shift_and_near_0_for_cancelling_moves():
    shift = torch.randn(1, 32).expand(50, 32)  # (samples, dim)
    cancelling = torch.cat([torch.eye(32), -torch.eye(32)])  # (64, 32)

    assert direction_share(shift) == pytest.approx(1.0)
    assert direction_share(cancelling) == pytest.approx(0.0, abs=1e-6)


def test_participation_counts_the_coordinates_a_direction_spreads_over():
    axis = torch.zeros(64)
    axis[3] = 1.0
    uniform = torch.ones(64) / 8.0  # unit length, spread evenly over 64

    assert participation(axis) == pytest.approx(1.0)
    assert participation(uniform) == pytest.approx(64.0)


def test_chance_jaccard_matches_a_monte_carlo_draw():
    generator = torch.Generator().manual_seed(0)
    draws = []
    for _ in range(4000):
        first = set(torch.randperm(100, generator=generator)[:10].tolist())
        second = set(torch.randperm(100, generator=generator)[:10].tolist())
        draws.append(len(first & second) / len(first | second))

    assert chance_jaccard(100, 10) == pytest.approx(sum(draws) / len(draws), abs=0.003)


def test_input_change_grid_puts_a_corner_patch_in_the_corner_token():
    change = torch.zeros(32, 32)
    change[-3:, -3:] = 1.0  # a 3 by 3 trigger in the bottom right corner

    grid = input_change_grid(change)  # (14, 14)

    assert grid.shape == (14, 14)
    assert torch.argmax(grid).item() == 14 * 14 - 1


def test_oriented_auroc_folds_a_negative_shift_onto_the_same_scale():
    clean = torch.zeros(20, 2)
    triggered = torch.zeros(20, 2)
    triggered[:, 0] = 1.0
    triggered[:, 1] = -1.0

    folded = oriented_auroc_per_dimension(clean, triggered)

    assert torch.allclose(folded, torch.ones(2))


def test_vit_readout_pools_the_class_token_and_keeps_probability_rows():
    model = tiny_vit()
    blocks = transformer_blocks(model, "vit")
    run = {"architecture": "vit"}
    images = torch.randn(3, 3, 32, 32)

    readout = batch_readout(
        model, run, blocks, model.heads.head, images, torch.device("cpu"), False
    )

    assert set(readout["pooled"]) == {0, 1, 2, 3}
    assert torch.equal(readout["pooled"][3], readout["tokens"][3][:, 0])
    assert readout["attention"].shape == (3, 3, 2, 17)
    assert torch.allclose(readout["attention"].sum(dim=-1), torch.ones(3, 3, 2))
    with torch.inference_mode():
        head_input = model.encoder.ln(readout["tokens"][3])[:, 0]  # (3, 16)
    assert torch.allclose(readout["head"], head_input, atol=1e-6)


def test_swin_readout_pools_the_token_mean_and_records_each_grid():
    model = tiny_swin()
    blocks = transformer_blocks(model, "swin")
    run = {"architecture": "swin"}
    images = torch.randn(2, 3, 16, 16)

    readout = batch_readout(
        model, run, blocks, model.head, images, torch.device("cpu"), False
    )

    assert readout["grids"] == {0: (4, 4), 1: (4, 4), 2: (2, 2)}
    assert torch.allclose(readout["pooled"][2], readout["tokens"][2].mean(dim=1))
    assert readout["attention"] is None


def test_token_change_maps_split_the_class_token_from_the_grid():
    change_sums = {0: torch.ones(197), 1: torch.ones(49)}
    clean_norm_sum = {0: 4.0, 1: 4.0}

    maps = token_change_maps(change_sums, clean_norm_sum, 2, {1: (7, 7)})

    assert maps["0"]["class_token"] == pytest.approx(0.25)
    assert len(maps["0"]["map"]) == 196
    assert maps["1"]["grid"] == [7, 7]
    assert maps["1"]["class_token"] is None
    assert math.isclose(maps["1"]["map"][0], 0.25)


def test_clean_principal_variances_are_the_variance_along_each_direction():
    generator = torch.Generator().manual_seed(0)
    scales = torch.tensor([5.0, 2.0, 1.0, 0.5])
    features = torch.randn(200, 4, generator=generator) * scales  # (200, 4)

    components, variances = clean_principal_directions(features)

    for component, variance in zip(components, variances):
        assert float((features @ component).var()) == pytest.approx(
            float(variance), rel=1e-4
        )
    assert abs(float(components[0][0])) == pytest.approx(1.0, abs=0.05)

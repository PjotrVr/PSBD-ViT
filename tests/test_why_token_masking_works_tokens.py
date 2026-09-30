"""The per-stage token geometry and restricted probes the Swin and site measurements use.

experiments/why_token_masking_works/tokens.py resolves trigger positions for each
block, since Swin's grid shrinks at every patch merging, and restricts any operator
to a set of positions. A position off by 1 row, or a probe that leaks outside its
positions, would still produce a complete and plausible table, so each rule is
held here on synthetic tensors and small models.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models.vision_transformer import VisionTransformer

from experiments.residual_stream_mechanism.activation_patching import trigger_tokens
from defenses.operators import TokenSubstitute
from experiments.why_token_masking_works import measure
from experiments.why_token_masking_works.mechanics import (
    PerTokenGaussianNoise,
    RecordingTokenSubstitute,
    RecordingUnscaledTokenMask,
)
from experiments.why_token_masking_works.tokens import (
    BLOCK_GRIDS,
    FixedTokenMask,
    PositionRestricted,
    attach_readers,
    block_cell_positions,
    block_trigger_positions,
    model_seed,
    cell_positions,
    complement_positions,
    plug_per_block,
    random_other_positions,
    signal_retained,
    touched_tokens,
    trigger_pixel_map,
)
from models.positions import unplug_dropout

pytestmark = pytest.mark.fast

BADNET_METADATA = {"dataset": "cifar10", "attack": "badnet_a2o", "target_label": 0}


def tiny_swin():
    torch.manual_seed(0)
    model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=8,
        depths=[2, 2, 18, 2],
        num_heads=[1, 2, 4, 8],
        window_size=[7, 7],
        num_classes=5,
    ).eval()
    return model


def tiny_vit():
    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=5,
    ).eval()
    return model


def test_vit_grid_matches_the_activation_patching_trigger_tokens():
    pixel_map = trigger_pixel_map(BADNET_METADATA)  # (224, 224)

    per_block = block_trigger_positions(pixel_map, "vit")

    expected = trigger_tokens(BADNET_METADATA) + 1  # (count,), CLS at 0
    assert len(per_block) == 12
    assert all(torch.equal(positions, expected) for positions in per_block)


def test_swin_trigger_positions_shrink_with_each_merge():
    pixel_map = trigger_pixel_map(BADNET_METADATA)  # (224, 224)

    per_block = block_trigger_positions(pixel_map, "swin")

    assert len(per_block) == 24
    counts = [len(positions) for positions in per_block]
    assert counts[0] > counts[2] > counts[4] >= counts[23] >= 1
    # A token at a coarse grid is touched exactly when 1 of its 4 children is.
    fine = torch.zeros(56 * 56, dtype=torch.bool)
    fine[touched_tokens(pixel_map, 56)] = True
    merged = fine.reshape(28, 2, 28, 2).any(dim=3).any(dim=1).flatten()  # (784,)
    assert torch.equal(merged.nonzero(as_tuple=True)[0], per_block[2])


def test_cell_positions_nest_every_grid_into_the_same_cells():
    cells = torch.zeros(7, 7, dtype=torch.bool)
    cells[0, 0] = True
    cells[6, 3] = True

    at_56 = cell_positions(cells, 56, 0)
    at_7 = cell_positions(cells, 7, 0)
    at_14_vit = cell_positions(cells, 14, 1)

    assert len(at_56) == 2 * 8 * 8
    assert at_7.tolist() == [0, 6 * 7 + 3]
    assert 1 in at_14_vit.tolist() and 0 not in at_14_vit.tolist()
    assert all(len(p) > 0 for p in block_cell_positions(cells, "swin"))


def test_fixed_mask_matches_the_vit_measurement_operator():
    x = torch.rand(4, 197, 16) + 0.5  # (4, 197, 16)
    positions = torch.tensor([3, 50, 180])

    ours = FixedTokenMask(positions, has_cls=True)(x)
    theirs = measure.FixedTokenMask(positions)(x)

    assert torch.equal(ours, theirs)


def test_fixed_mask_on_a_swin_map_zeroes_the_named_row_major_tokens():
    x = torch.rand(2, 7, 7, 8) + 0.5  # (2, 7, 7, 8)
    positions = torch.tensor([0, 7 * 2 + 5])

    masked = FixedTokenMask(positions, has_cls=False)(x)  # (2, 7, 7, 8)

    assert (masked[:, 0, 0] == 0).all() and (masked[:, 2, 5] == 0).all()
    assert (masked[:, 1, 1] != 0).all()
    assert masked.shape == x.shape


def test_restricted_probe_leaves_other_positions_untouched():
    x = torch.rand(3, 7, 7, 8) + 0.5  # (3, 7, 7, 8)
    positions = torch.tensor([4, 10, 11])

    torch.manual_seed(0)
    probe = PositionRestricted(nn.Dropout(0.5), positions).train()
    out = probe(x)  # (3, 7, 7, 8)

    flat_in = x.reshape(3, 49, 8)
    flat_out = out.reshape(3, 49, 8)
    untouched = torch.ones(49, dtype=torch.bool)
    untouched[positions] = False
    assert torch.equal(flat_out[:, untouched], flat_in[:, untouched])
    assert (flat_out[:, positions] == 0).any()


def test_random_and_complement_positions_avoid_the_excluded_set():
    excluded = torch.tensor([1, 2, 3])

    drawn = random_other_positions(excluded, 3, 196, 1, 0)
    rest = complement_positions(excluded, 196, 1)

    assert not torch.isin(drawn, excluded).any() and len(drawn) == 3
    assert len(rest) == 197 - 3 and 0 in rest.tolist()


def test_plug_per_block_hands_each_swin_block_its_own_positions():
    model = tiny_swin()
    pixel_map = trigger_pixel_map(BADNET_METADATA)
    trigger = block_trigger_positions(pixel_map, "swin")
    seen = []

    class Probe(nn.Module):
        def __init__(self, block_index):
            super().__init__()
            self.block_index = block_index

        def forward(self, x):
            seen.append((self.block_index, x.shape[1]))
            return x

    handles = plug_per_block(
        model,
        "swin",
        ("before_attention_norm",),
        lambda b, _rate: Probe(b),
        0.5,
    )
    with torch.no_grad():
        model(torch.rand(1, 3, 224, 224))
    unplug_dropout(handles)

    assert [b for b, _ in seen] == list(range(24))
    assert [grid for _, grid in seen] == list(BLOCK_GRIDS["swin"])
    assert len(trigger) == len(seen)


def test_readers_capture_every_block_at_its_positions():
    model = tiny_vit()
    positions = [torch.tensor([1, 2])] * 12

    store, handles = attach_readers(model, "vit", positions)
    with torch.no_grad():
        model(torch.rand(2, 3, 224, 224))
    for handle in handles:
        handle.remove()

    assert sorted(store["read"]) == list(range(12))
    assert store["read"][0].shape == (2, 2, 16)
    assert store["stream"][11].shape == (2, 2, 16)


def test_signal_retained_reads_1_when_intact_and_0_when_erased():
    clean = torch.rand(5, 3, 8)
    triggered = clean + torch.rand(5, 3, 8)

    intact = signal_retained(triggered, clean, triggered, clean)
    erased = signal_retained(clean, clean, triggered, clean)

    assert torch.allclose(intact["aligned"] / intact["energy"], torch.ones(5))
    assert torch.allclose(erased["aligned"], torch.zeros(5))
    assert torch.allclose(intact["best_token"], torch.ones(5))


def test_two_models_draw_different_permutations():
    first = model_seed("vit_gtsrb_badnet_a2o_0_05")
    second = model_seed("swin_gtsrb_badnet_a2o_0_05")

    first_order = torch.randperm(49, generator=torch.Generator().manual_seed(first))
    second_order = torch.randperm(49, generator=torch.Generator().manual_seed(second))

    assert first != second
    assert not torch.equal(first_order, second_order)
    assert model_seed("vit_gtsrb_badnet_a2o_0_05") == first, "the seed is stable"


def test_recording_substitute_reproduces_the_library_operator():
    x = torch.rand(4, 197, 16) + 0.5  # (4, 197, 16)

    torch.manual_seed(3)
    library = TokenSubstitute(0.4).train()(x)
    torch.manual_seed(3)
    recorder = RecordingTokenSubstitute(0.4).train()
    recorded = recorder(x)

    assert torch.equal(library, recorded)
    changed = (recorded != x).any(dim=2)  # (4, 197)
    assert not changed[recorder.last_dropped.logical_not()].any()
    assert not recorder.last_dropped[:, 0].any()


def test_unscaled_mask_keeps_survivors_and_cls_exact():
    x = torch.rand(4, 197, 16) + 0.5  # (4, 197, 16)

    torch.manual_seed(5)
    out = RecordingUnscaledTokenMask(0.6).train()(x)

    kept = ~out.eq(0).all(dim=2)  # (4, 197)
    assert torch.allclose(out[kept], x[kept])
    assert torch.equal(out[:, 0], x[:, 0])


def test_per_token_gaussian_scales_each_token_by_its_own_spread():
    x = torch.ones(2, 5, 64)
    x[:, 1] *= torch.linspace(-50, 50, 64)  # 1 loud token

    torch.manual_seed(0)
    out = PerTokenGaussianNoise(0.5).train()(x)

    noise = (out - x).std(dim=2)  # (2, 5)
    assert noise[:, 1].min() > 10 * noise[:, [0, 2, 3, 4]].max()

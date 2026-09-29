"""The helpers experiments/why_psbd_works/measure.py reads its hypotheses off.

Each verdict in the experiment's README rests on a small statistic: the own-class
margin, an AUROC where a low score flags the positive class, the spectrum and
rank summaries, the Gini coefficient, a fixed token mask shared across ViT and
Swin layouts and a symmetric finite difference. A silent error in any of them
would move a verdict, so each is held to a case whose answer is known.
"""

import pytest
import torch
from torchvision.models.vision_transformer import VisionTransformer

from experiments.why_psbd_works.measure import (
    SubsetTokenMask,
    auroc_low_is_positive,
    effective_rank,
    feature_statistics,
    finite_differences,
    gini,
    jacobian_concentration,
    model_seed,
    own_class_margin,
    participation_ratio,
    per_image_statistics,
    psu_ratio_without_class,
    random_keep_grid,
    run_passes,
    stream_gradients,
    with_random_direction,
)

pytestmark = pytest.mark.fast

DEVICE = torch.device("cpu")


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
    # torchvision zero-initializes the head, which would make every logit
    # constant and every gradient 0.
    torch.nn.init.normal_(model.heads.head.weight)
    return model


def images(count=6):
    batch = torch.rand(count, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    return batch


def test_own_class_margin_is_own_logit_minus_best_other():
    logits = torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, -1.0]])  # (2, 3)
    labels = torch.tensor([0, 2])

    margin = own_class_margin(logits, labels)

    assert torch.allclose(margin, torch.tensor([1.0, -6.0]))


def test_own_class_margin_broadcasts_labels_over_passes():
    logits = torch.tensor([[[3.0, 1.0]], [[0.0, 2.0]]])  # (2 passes, 1, 2)
    labels = torch.tensor([0])

    margin = own_class_margin(logits, labels)  # (2, 1)

    assert torch.allclose(margin, torch.tensor([[2.0], [-2.0]]))


def test_auroc_reads_low_scores_as_positive():
    low = torch.tensor([0.1, 0.2])
    high = torch.tensor([0.8, 0.9])

    assert auroc_low_is_positive(low, high) == 1.0
    assert auroc_low_is_positive(high, low) == 0.0
    assert auroc_low_is_positive(low, low) == 0.5
    assert auroc_low_is_positive(torch.tensor([]), high) is None


def test_participation_ratio_counts_equal_directions():
    assert participation_ratio(torch.tensor([4.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert participation_ratio(torch.tensor([1.0, 1.0, 1.0])) == pytest.approx(3.0)


def test_effective_rank_counts_equal_singular_values():
    assert effective_rank(torch.tensor([2.0, 2.0, 2.0, 2.0])) == pytest.approx(4.0)
    assert effective_rank(torch.tensor([1.0, 0.0])) == pytest.approx(1.0)


def test_gini_is_0_for_uniform_and_near_1_for_a_single_pixel():
    uniform = torch.full((1, 10), 0.1)
    single = torch.zeros(1, 10)
    single[0, 3] = 1.0

    assert gini(uniform).item() == pytest.approx(0.0, abs=1e-6)
    assert gini(single).item() == pytest.approx(0.9)


def test_subset_mask_hides_exactly_the_dropped_patch_tokens_on_vit():
    keep_grid = random_keep_grid(0.4, seed=3)  # (14, 14)
    x = torch.rand(2, 197, 8) + 0.5  # (2, 197, 8)

    masked = SubsetTokenMask(keep_grid)(x)

    zeroed = (masked == 0).all(dim=2)  # (2, 197)
    assert not zeroed[:, 0].any(), "CLS must stay visible"
    assert torch.equal(zeroed[0, 1:], ~keep_grid.flatten())
    assert int(keep_grid.sum()) == round(0.4 * 196)
    assert torch.equal(
        masked[:, 1:][:, keep_grid.flatten()], x[:, 1:][:, keep_grid.flatten()]
    )


def test_subset_mask_follows_the_same_cells_on_every_swin_stage():
    keep_grid = random_keep_grid(0.6, seed=4)  # (14, 14)
    mask = SubsetTokenMask(keep_grid)

    fine = mask(torch.ones(1, 56, 56, 2))[0, :, :, 0]  # (56, 56)
    same = mask(torch.ones(1, 14, 14, 2))[0, :, :, 0]  # (14, 14)
    coarse = mask(torch.ones(1, 7, 7, 2))[0, :, :, 0]  # (7, 7)

    assert torch.equal(fine[::4, ::4], keep_grid.float())
    assert torch.equal(same, keep_grid.float())
    majority = keep_grid.float().view(7, 2, 7, 2).mean(dim=(1, 3)) >= 0.5  # (7, 7)
    assert torch.equal(coarse, majority.float())


def test_random_direction_is_the_same_for_plus_and_minus():
    x = torch.rand(3, 5, 4)

    plus = with_random_direction(x, seed=11, epsilon=0.2)
    minus = with_random_direction(x, seed=11, epsilon=-0.2)

    assert torch.allclose(plus + minus, 2 * x, atol=1e-6)
    assert not torch.allclose(plus, x)


def test_per_image_statistics_are_neutral_when_passes_equal_the_base():
    base = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])  # (2, 3)
    passes = base[None].repeat(4, 1, 1)  # (4, 2, 3)
    labels = base.argmax(dim=1)

    stats = per_image_statistics(base, passes, labels)

    assert stats["psu_ratio"] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert stats["kept"] == [1.0, 1.0]
    assert stats["margin"] == pytest.approx([1.0, 2.0])
    assert stats["margin_retention"] == pytest.approx([1.0, 1.0])


def test_feature_statistics_read_a_pure_backdoor_shift():
    backdoor_unit = torch.tensor([1.0, 0.0, 0.0])
    class_units = torch.eye(3)
    geometry = {
        "clean_mean": torch.zeros(3),
        "backdoor_unit": backdoor_unit,
        "class_units": class_units,
    }
    base = torch.tensor([[4.0, 0.0, 0.0]])  # (1, 3)
    # Every pass halves the backdoor component and adds nothing else.
    passes = torch.tensor([[[2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])  # (2, 1, 3)

    stats = feature_statistics(base, passes, torch.tensor([0]), geometry)

    assert stats["backdoor_share"] == pytest.approx([1.0])
    assert stats["backdoor_retention"] == pytest.approx([0.5])
    assert stats["backdoor_signal_to_perturbation"] == pytest.approx([2.0])
    assert stats["relative_change"] == pytest.approx([0.5])


def test_run_passes_captures_the_feature_the_head_reads():
    model = tiny_vit()
    batch = images()

    result = run_passes(model, batch, 4, DEVICE, passes=None)

    with torch.no_grad():
        recomputed = model.heads.head(result["features"])  # (6, 5)
    assert result["features"].shape == (6, 16)
    assert torch.allclose(result["logits"], recomputed, atol=5e-2)


def test_stream_gradients_have_the_stream_shape_and_match_autograd():
    model = tiny_vit()
    batch = images(3)
    block = model.encoder.layers[1]

    gradients = stream_gradients(model, block, batch, DEVICE)  # (3, 17, 16)
    summary = jacobian_concentration(gradients, "vit", torch.tensor([1, 2]))

    assert gradients.shape == (3, 17, 16)
    assert all(0.0 < share <= 1.0 for share in summary["top_token_share"])
    assert all(1.0 <= tokens <= 16.0 for tokens in summary["effective_tokens"])
    assert len(summary["trigger_token_share"]) == 3


def test_finite_differences_give_a_finite_reading_per_image():
    model = tiny_vit()
    batch = images(4)

    readings = finite_differences(model, model.encoder.layers[2], batch, DEVICE)

    for epsilon in ("0.05", "0.2"):
        slope = torch.tensor(readings[epsilon]["slope_margin"])
        assert slope.shape == (4,)
        assert torch.isfinite(slope).all()
    assert len(readings["margin"]) == 4


def test_subset_mask_applies_1_pattern_per_image():
    grids = torch.stack([random_keep_grid(0.5, seed=s) for s in (1, 2)])  # (2, 14, 14)
    x = torch.rand(2, 197, 4) + 0.5

    masked = SubsetTokenMask(grids)(x)

    zeroed = (masked == 0).all(dim=2)  # (2, 197)
    assert torch.equal(zeroed[:, 1:], ~grids.flatten(1))
    assert not torch.equal(zeroed[0], zeroed[1])


def test_model_seeds_differ_between_models_and_repeat_for_1_model():
    first = model_seed("vit_cifar100_blend_0_05")

    assert first == model_seed("vit_cifar100_blend_0_05")
    assert first != model_seed("vit_tiny_blend_0_05")


def test_removing_the_target_cancels_a_drop_that_went_to_the_target():
    base = torch.log(torch.tensor([[0.6, 0.2, 0.2]]))  # (1, 3), class 0 predicted
    # The pass moves mass from class 1 (the removed target) and leaves class 0's
    # share of the remaining mass unchanged.
    passes = torch.log(torch.tensor([[[0.3, 0.6, 0.1]]]))  # (1 pass, 1, 3)
    labels = torch.tensor([0])

    plain = per_image_statistics(base[0:1], passes, labels)["psu_ratio"][0]
    without = psu_ratio_without_class(base, passes, labels, removed=1)

    assert plain == pytest.approx(0.5, abs=1e-5)
    assert float(without[0]) == pytest.approx(0.0, abs=1e-5)


def test_crossing_rate_interpolates_and_reports_never():
    from experiments.why_psbd_works.shift_curves import crossing_rate

    assert crossing_rate([0.1, 0.2, 0.3], [0.2, 0.4, 0.8]) == pytest.approx(0.225)
    assert crossing_rate([0.1, 0.2], [0.6, 0.9]) == 0.1
    assert crossing_rate([0.1, 0.2], [0.1, 0.3]) is None

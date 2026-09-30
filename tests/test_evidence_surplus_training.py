"""The training options the evidence-surplus runs need (docs/evidence-surplus-theory.md).

The expensive failures this prevents. A trigger label probability whose draw moves
with the poison set makes 2 seeds of the same recipe keep different exceptions for
no reason. Label smoothing that leaves the default run's model changed breaks every
comparison with the panel. A block-restricted evasion probe that perturbs every
block trains the attacker against a defense nobody named. A count in args.json
that reads 0 because a wrapper hid the index sets hides a failed cover mechanism.
Each costs a GPU weekend to find after the fact.
"""

import argparse
import json

import pytest
import torch
import torch.nn as nn
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from torchvision.models.vision_transformer import VisionTransformer

import attacks.evasion as evasion
import cli.train_backdoor as train_backdoor
import training.loop as loop
from attacks import build_attack, default_config
from attacks.poisoning import (
    CoverPoisonedTrainingSet,
    PoisonedTrainingSet,
    choose_label_kept_indices,
)
from training.loop import CheckpointMetadata, IndexedTrainingSet, train_classifier
from training.telemetry.rows import CLEAN_ROW, COVER_ROW, POISONED_ROW, row_group_codes

pytestmark = pytest.mark.fast

NUM_CLASSES = 4
IMAGE_SIZE = 32


class LabeledImages(Dataset):
    """n random 0-to-1 images with fixed labels, exposing targets like CIFAR does."""

    def __init__(self, count: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(count, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
        self.targets = [index % NUM_CLASSES for index in range(count)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


def badnet_attack():
    attack = build_attack("badnet", default_config("badnet"), IMAGE_SIZE, 0)
    return attack


def test_trigger_label_draw_is_per_index_and_independent_of_the_poison_set():
    poison_indices = set(range(0, 2000, 2))
    kept = choose_label_kept_indices(poison_indices, 0.6, 2000, seed=3)
    kept_from_a_subset = choose_label_kept_indices(
        set(range(0, 1000, 2)), 0.6, 2000, seed=3
    )

    assert kept <= poison_indices
    assert kept_from_a_subset == {index for index in kept if index < 1000}
    assert abs(len(kept) / len(poison_indices) - 0.4) < 0.05
    assert kept == choose_label_kept_indices(poison_indices, 0.6, 2000, seed=3)
    assert kept != choose_label_kept_indices(poison_indices, 0.6, 2000, seed=4)


def test_trigger_label_probability_one_keeps_no_label_and_zero_is_refused():
    assert choose_label_kept_indices({1, 2, 3}, 1.0, 10, seed=0) == set()
    with pytest.raises(ValueError):
        choose_label_kept_indices({1, 2, 3}, 0.0, 10, seed=0)


def test_a_kept_row_carries_the_trigger_and_its_true_label():
    base = LabeledImages(8)
    attack = badnet_attack()
    identity = nn.Identity()

    dataset = PoisonedTrainingSet(
        base, attack, {1, 2, 5}, identity, NUM_CLASSES, label_kept_indices={2}
    )
    covered = CoverPoisonedTrainingSet(
        base, attack, {1, 2, 5}, {3}, identity, NUM_CLASSES, label_kept_indices={2}
    )

    for served in (dataset, covered):
        kept_image, kept_label = served[2]
        poisoned_image, poisoned_label = served[1]
        assert kept_label == base.targets[2]
        assert poisoned_label == 0
        assert torch.equal(kept_image, attack.apply_trigger(base.images[2], 2))
        assert torch.equal(poisoned_image, attack.apply_trigger(base.images[1], 1))


def test_telemetry_counts_a_kept_row_as_a_cover_row():
    dataset = PoisonedTrainingSet(
        LabeledImages(8),
        badnet_attack(),
        {1, 2, 5},
        nn.Identity(),
        NUM_CLASSES,
        label_kept_indices={2},
    )

    codes = row_group_codes(IndexedTrainingSet(dataset))  # (8,)

    assert codes[1] == POISONED_ROW and codes[5] == POISONED_ROW
    assert codes[2] == COVER_ROW
    assert codes[0] == CLEAN_ROW


def loader_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        dataset="cifar10",
        attack="badnet",
        poison_rate=0.5,
        target_label=0,
        seed=0,
        max_samples=None,
        raw_data_dir="raw_data",
        poisoned_dir="",
        cover_rate=None,
        attack_override=[],
        evade_psbd=False,
        batch_size=4,
        num_workers=0,
        augment="none",
        record_sample_loss=False,
        telemetry=True,
        trigger_label_probability=0.5,
    )
    defaults.update(overrides)
    arguments = argparse.Namespace(**defaults)
    return arguments


@pytest.mark.parametrize("attack_name", ["badnet", "and16"])
def test_the_loader_applies_the_label_draw_and_the_counts_see_through_wrappers(
    monkeypatch, attack_name
):
    fake_train = LabeledImages(64)
    monkeypatch.setattr(
        train_backdoor,
        "load_clean_datasets",
        lambda dataset_name, transform, raw_data_dir: (fake_train, fake_train),
    )
    args = loader_args(attack=attack_name)

    loader, _, _, _, _ = train_backdoor.build_training_loader(args, IMAGE_SIZE)
    poisoned_set = train_backdoor.innermost_poisoned_set(loader.dataset)
    record = train_backdoor.trigger_label_record(args, poisoned_set)

    assert isinstance(loader.dataset, IndexedTrainingSet)
    assert record["trigger_label_probability"] == 0.5
    assert 0 < record["n_trigger_label_kept"] < len(poisoned_set.poison_indices)
    for index in poisoned_set.label_kept_indices:
        _, label, _ = loader.dataset[index]
        assert label == fake_train.targets[index]
    if attack_name == "and16":
        # The AND attack's covers resolve to the poison rate, and the count has to
        # survive the telemetry wrapper.
        assert len(poisoned_set.cover_indices) > 0


def tiny_vit() -> nn.Module:
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=NUM_CLASSES,
    )
    nn.init.normal_(model.heads.head.weight, std=0.5)
    return model


def train_tiny(monkeypatch, **kwargs) -> nn.Module:
    monkeypatch.setattr(loop, "build_model", lambda *args, **options: tiny_vit())
    dataset = LabeledImages(16)
    seed_everything(0)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model, _ = train_classifier(
        "vit",
        NUM_CLASSES,
        train_loader,
        val_loader,
        torch.device("cpu"),
        epochs=1,
        use_sam=False,
        use_bfloat16=False,
        **kwargs,
    )
    return model


def test_label_smoothing_zero_trains_the_default_model_and_smoothing_changes_it(
    monkeypatch,
):
    default_model = train_tiny(monkeypatch)
    zero_model = train_tiny(monkeypatch, label_smoothing=0.0)
    smoothed_model = train_tiny(monkeypatch, label_smoothing=0.1)

    default_parameters = list(default_model.parameters())
    for default, zero in zip(default_parameters, zero_model.parameters()):
        assert torch.equal(default, zero)
    assert any(
        not torch.equal(default, smoothed)
        for default, smoothed in zip(default_parameters, smoothed_model.parameters())
    )


def test_label_smoothing_outside_zero_to_one_is_refused(monkeypatch):
    with pytest.raises(ValueError):
        train_tiny(monkeypatch, label_smoothing=1.0)


def test_checkpoint_metadata_records_label_smoothing():
    metadata = CheckpointMetadata(
        dataset="gtsrb",
        attack="badnet",
        label_mode="all_to_one",
        target_label=0,
        poison_rate=0.1,
        cover_rate=0.0,
        realized_poison_rate=0.1,
        architecture="vit",
        use_sam=False,
        rho=0.1,
        epochs=15,
        seed=0,
        max_samples=None,
        clean_accuracy=None,
        asr=None,
        started_at="",
        ended_at="",
    )

    assert metadata.as_dict()["label_smoothing"] == 0.0
    smoothed = CheckpointMetadata(**{**metadata.__dict__, "label_smoothing": 0.1})
    assert json.loads(json.dumps(smoothed.as_dict()))["label_smoothing"] == 0.1


def parsed_train_args(*extra: str) -> argparse.Namespace:
    argv = [
        "--dataset",
        "gtsrb",
        "--attack",
        "badnet",
        "--poison-rate",
        "0.1",
        "--output",
        "checkpoints/x/attack_result.pt",
        *extra,
    ]
    arguments = train_backdoor.build_parser().parse_args(argv)
    return arguments


def test_the_new_flags_default_to_the_panel_recipe():
    args = parsed_train_args()

    assert args.label_smoothing == 0.0
    assert args.trigger_label_probability == 1.0


def test_evade_probe_tokens_carry_an_optional_block_range():
    args = parsed_train_args(
        "--evade-psbd",
        "--evade-probes",
        "before_attention_norm:token_mask",
        "pre_residual:dropout:5-8",
    )

    tokens = train_backdoor.parse_evade_probe_tokens(args)

    assert tokens == [
        ("before_attention_norm", "token_mask", None),
        ("pre_residual", "dropout", (5, 8)),
    ]


@pytest.mark.parametrize(
    "token", ["pre_residual:dropout:8-5", "pre_residual:dropout:5", "pre_residual::5-8"]
)
def test_a_malformed_block_range_is_refused(token):
    args = parsed_train_args("--evade-psbd", "--evade-probes", token)

    with pytest.raises(ValueError):
        train_backdoor.parse_evade_probe_tokens(args)


def test_the_singular_flags_resolve_to_an_unrestricted_probe():
    args = parsed_train_args("--evade-psbd")

    assert train_backdoor.parse_evade_probe_tokens(args) == [
        ("before_attention_norm", "dropout", None)
    ]


def test_a_block_restricted_probe_reads_its_depth_band_ladder():
    whole_stack = evasion.basis_rate_ladder("pre_residual", "dropout")
    middle_band = evasion.basis_rate_ladder("pre_residual", "dropout", (5, 8))

    assert middle_band != whole_stack
    assert max(middle_band) == 0.99
    assert max(whole_stack) == 0.9


class WrappedTinyViT(nn.Module):
    """The module names the ViT position registry resolves, 4 blocks deep."""

    def __init__(self):
        super().__init__()
        self.inner = VisionTransformer(
            image_size=IMAGE_SIZE,
            patch_size=16,
            num_layers=4,
            num_heads=2,
            hidden_dim=32,
            mlp_dim=64,
            num_classes=NUM_CLASSES,
        )

    def forward(self, images):
        return self.inner(images)


def test_the_evasion_probe_plugs_only_its_block_range(monkeypatch):
    plugged = []
    original_plug = evasion.plug_dropout

    def recording_plug(*args, **kwargs):
        handles = original_plug(*args, **kwargs)
        plugged.append((kwargs.get("block_range"), len(handles)))
        return handles

    monkeypatch.setattr(evasion, "plug_dropout", recording_plug)
    torch.manual_seed(0)
    model = WrappedTinyViT()
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    probe = {
        "position": "pre_residual",
        "operator": "dropout",
        "rate": 0.5,
        "architecture": "vit",
    }

    evasion.psu_for_batch(model, images, {**probe, "block_range": (2, 3)}, passes=1)
    evasion.psu_for_batch(model, images, {**probe, "block_range": None}, passes=1)

    (band_range, band_handles), (whole_range, whole_handles) = plugged
    assert band_range == (2, 3) and whole_range is None
    # pre_residual attaches 2 probes per block, so blocks 2 to 3 of 4 are half.
    assert 2 * band_handles == whole_handles

"""Tests for the optional training telemetry (training.telemetry).

The expensive failures this prevents: a telemetry flag that silently changes the
model a run trains (a probe drawing from the global RNG moves every later shuffle
and mask), a record whose loss split mislabels which rows are poisoned, and a
gated-off probe that writes a 0 which reads as a measurement. Each costs a
weekend of GPU time to discover after the fact.
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models.vision_transformer import VisionTransformer

import training.loop as loop
from training.loop import CheckpointMetadata, IndexedTrainingSet, train_classifier
from training.telemetry.dormancy import (
    dormant_scores,
    dormant_share,
    effective_rank,
    hidden_layer_readings,
)
from training.telemetry.heldout import HeldoutPairs, epoch_readings
from training.telemetry.modules import UpdateObserver, module_keys
from training.telemetry.record import TELEMETRY_FILENAME, TelemetryConfig
from training.telemetry.rows import (
    CLEAN_ROW,
    COVER_ROW,
    POISONED_ROW,
    row_group_codes,
    row_statistic_keys,
    row_statistic_sums,
)

pytestmark = pytest.mark.fast

NUM_CLASSES = 4
NUM_TRAINING_ROWS = 24
POISON_INDICES = {1, 5, 9, 13}
COVER_INDICES = {2, 6}


def tiny_vit() -> nn.Module:
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=NUM_CLASSES,
    )
    # torchvision zero-initializes the ViT head, which makes every logit equal and
    # every margin 0 before the first step.
    nn.init.normal_(model.heads.head.weight, std=0.5)
    return model


def tiny_swin() -> nn.Module:
    model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=8,
        depths=[1, 1],
        num_heads=[1, 2],
        window_size=[2, 2],
        num_classes=NUM_CLASSES,
    )
    return model


class TinyCoverPoisonedSet(Dataset):
    """A stand-in for attacks.poisoning.CoverPoisonedTrainingSet."""

    def __init__(self, images, labels, poison_indices, cover_indices):
        self.images = images
        self.labels = labels
        self.poison_indices = poison_indices
        self.cover_indices = cover_indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def tiny_training_set() -> TinyCoverPoisonedSet:
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(NUM_TRAINING_ROWS, 3, 32, 32, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (NUM_TRAINING_ROWS,), generator=generator)
    dataset = TinyCoverPoisonedSet(images, labels, POISON_INDICES, COVER_INDICES)
    return dataset


def tiny_heldout(count: int = 8, target: int = 0) -> HeldoutPairs:
    generator = torch.Generator().manual_seed(1)
    clean = torch.randn(count, 3, 32, 32, generator=generator)
    triggered = clean.clone()
    triggered[:, :, :8, :8] = 3.0
    pairs = HeldoutPairs(
        clean=clean,
        triggered=triggered,
        labels=torch.randint(0, NUM_CLASSES, (count,), generator=generator),
        success_labels=torch.full((count,), target, dtype=torch.int64),
    )
    return pairs


def run_training(
    monkeypatch,
    checkpoint_dir,
    telemetry_config=None,
    heldout=None,
    epochs=2,
    indexed=None,
):
    """Train the tiny ViT with shuffling on, so an RNG leak would change the data order."""
    monkeypatch.setattr(loop, "build_model", lambda *args, **kwargs: tiny_vit())
    dataset = tiny_training_set()
    wants_indices = telemetry_config is not None if indexed is None else indexed
    served = IndexedTrainingSet(dataset) if wants_indices else dataset
    val_set = TinyCoverPoisonedSet(dataset.images[:8], dataset.labels[:8], set(), set())

    seed_everything(0)
    train_loader = DataLoader(served, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
    model, trajectory = train_classifier(
        "vit",
        NUM_CLASSES,
        train_loader,
        val_loader,
        torch.device("cpu"),
        epochs=epochs,
        use_sam=False,
        use_bfloat16=False,
        checkpoint_dir=str(checkpoint_dir),
        telemetry_config=telemetry_config,
        telemetry_heldout=heldout,
    )
    return model, trajectory


def read_records(checkpoint_dir) -> list[dict]:
    with open(checkpoint_dir / TELEMETRY_FILENAME) as handle:
        records = [json.loads(line) for line in handle]
    return records


def test_flag_off_writes_no_file_and_telemetry_trains_the_same_model(
    tmp_path, monkeypatch
):
    plain_dir = tmp_path / "plain"
    observed_dir = tmp_path / "observed"
    plain_model, plain_trajectory = run_training(monkeypatch, plain_dir)
    observed_model, observed_trajectory = run_training(
        monkeypatch,
        observed_dir,
        TelemetryConfig(every=2, heavy_every=2),
        tiny_heldout(),
    )

    assert not (plain_dir / TELEMETRY_FILENAME).exists()
    assert (observed_dir / TELEMETRY_FILENAME).exists()
    assert plain_trajectory == observed_trajectory
    for plain, observed in zip(plain_model.parameters(), observed_model.parameters()):
        assert torch.equal(plain, observed)


def test_flag_on_writes_well_formed_jsonl(tmp_path, monkeypatch):
    epochs = 2
    run_training(
        monkeypatch,
        tmp_path,
        TelemetryConfig(every=2, heavy_every=4, retention_rates=(0.5, 0.8)),
        tiny_heldout(),
        epochs=epochs,
    )
    records = read_records(tmp_path)

    header = records[0]
    assert header["record"] == "header"
    assert header["row_counts"] == {
        "clean": NUM_TRAINING_ROWS - len(POISON_INDICES) - len(COVER_INDICES),
        "poisoned": len(POISON_INDICES),
        "cover": len(COVER_INDICES),
    }

    windows = [r for r in records if r["record"] == "window"]
    steps_per_epoch = NUM_TRAINING_ROWS // 4
    assert [w["step"] for w in windows] == list(
        range(2, epochs * steps_per_epoch + 1, 2)
    )
    # A group with no row in a window is absent from that line, not a 0.
    served_rows = sum(
        window.get(f"n_{group}", 0)
        for window in windows
        for group in ("clean", "poisoned", "cover")
    )
    assert served_rows == epochs * NUM_TRAINING_ROWS
    for window in windows:
        assert window["heavy"] == (window["step"] % 4 == 0)
        assert 0.0 <= window["accuracy_clean"] <= 1.0
    heavy = [w for w in windows if w["heavy"]]
    assert heavy
    for window in heavy:
        assert window["grad_norm_total"] > 0
        assert window["update_ratio_total"] > 0
        assert "update_ratio_block01_mlp" in window
        assert "dormant_share_mlp_block02_clean" in window
        assert window["srank_features_triggered"] >= 1

    epoch_lines = [r for r in records if r["record"] == "epoch"]
    assert [e["epoch"] for e in epoch_lines] == list(range(1, epochs + 1))
    for line in epoch_lines:
        assert 0.0 <= line["heldout_asr"] <= 1.0
        assert 0.0 <= line["heldout_clean_accuracy"] <= 1.0
        for key in (
            "retention_tm_0_5_clean",
            "retention_tm_0_8_triggered",
            "kept_tm_0_5_triggered",
            "kept_tm_0_8_clean",
        ):
            assert key in line


def test_gated_off_probes_write_no_key(tmp_path, monkeypatch):
    run_training(
        monkeypatch,
        tmp_path,
        TelemetryConfig(
            every=3, heavy_every=0, retention_rates=(), measure_surplus=False
        ),
        heldout=None,
    )
    records = read_records(tmp_path)
    for record in records[1:]:
        for key in record:
            assert not key.startswith(
                ("grad_norm", "param_norm", "update_ratio", "dormant", "srank")
            ), key
            assert not key.startswith(
                ("heldout_", "retention_", "kept_", "surplus_")
            ), key
        if record["record"] == "window":
            assert record["heavy"] is False
            assert "loss_clean" in record


def test_epoch_readings_gate_surplus_and_retention_separately():
    torch.manual_seed(0)
    model = tiny_vit()
    pairs = tiny_heldout()
    without_surplus = epoch_readings(
        model, "vit", pairs, (0.5,), False, torch.device("cpu"), False
    )
    assert "retention_tm_0_5_clean" in without_surplus
    assert not any(key.startswith("surplus_") for key in without_surplus)

    without_retention = epoch_readings(
        model, "vit", pairs, (), True, torch.device("cpu"), False
    )
    assert not any(key.startswith("retention_") for key in without_retention)
    assert "surplus_n_eval" in without_retention


def test_loss_split_on_a_toy_batch():
    logits = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 3.0, 1.0], [1.0, 1.0, 4.0], [0.5, 0.0, 2.0]]
    )  # (4, 3)
    labels = torch.tensor([0, 2, 2, 1])
    groups = torch.tensor([CLEAN_ROW, POISONED_ROW, POISONED_ROW, CLEAN_ROW])

    keys = row_statistic_keys(
        row_statistic_sums(logits, labels, groups).numpy().ravel()
    )

    row_loss = F.cross_entropy(logits, labels, reduction="none")
    assert keys["n_clean"] == 2 and keys["n_poisoned"] == 2
    assert "n_cover" not in keys
    assert keys["loss_clean"] == pytest.approx(float(row_loss[[0, 3]].mean()), abs=1e-5)
    assert keys["loss_poisoned"] == pytest.approx(
        float(row_loss[[1, 2]].mean()), abs=1e-5
    )
    assert keys["loss"] == pytest.approx(float(row_loss.mean()), abs=1e-5)
    assert keys["accuracy_clean"] == pytest.approx(0.5)
    assert keys["accuracy_poisoned"] == pytest.approx(0.5)
    # Margins: row 0 is 2 - 0, row 3 is 0 - 2, row 1 is 1 - 3, row 2 is 4 - 1.
    assert keys["margin_clean"] == pytest.approx(0.0)
    assert keys["margin_poisoned"] == pytest.approx(0.5)


def test_row_group_codes_follow_the_loader_numbering_through_a_subset():
    dataset = tiny_training_set()
    kept = [index for index in range(NUM_TRAINING_ROWS) if index != 0]
    served = IndexedTrainingSet(Subset(dataset, kept))

    codes = row_group_codes(served)

    assert codes.shape == (NUM_TRAINING_ROWS - 1,)
    for position, original in enumerate(kept):
        if original in POISON_INDICES:
            assert codes[position] == POISONED_ROW
        elif original in COVER_INDICES:
            assert codes[position] == COVER_ROW
        else:
            assert codes[position] == CLEAN_ROW


def test_update_observer_measures_the_step_the_optimizer_took():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    observer = UpdateObserver.__new__(UpdateObserver)
    observer.parameters = list(model.parameters())
    observer.group_names = ["linear"]
    observer.group_index = torch.zeros(2, dtype=torch.int64)
    observer._armed = False
    observer._before = None
    observer._reading = None
    observer._handles = [
        optimizer.register_step_pre_hook(observer._before_step),
        optimizer.register_step_post_hook(observer._after_step),
    ]

    before = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
    model(torch.randn(5, 3)).pow(2).sum().backward()
    gradient = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    observer.arm()
    optimizer.step()
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    keys = module_keys(observer.take().numpy().ravel(), ["linear"], 1)
    assert keys["grad_norm_linear"] == pytest.approx(float(gradient.norm()), rel=1e-5)
    assert keys["param_norm_linear"] == pytest.approx(float(before.norm()), rel=1e-5)
    assert keys["update_ratio_linear"] == pytest.approx(
        float((after - before).norm() / before.norm()), rel=1e-5
    )
    assert observer.take() is None

    model(torch.randn(5, 3)).pow(2).sum().backward()
    optimizer.step()
    assert observer.take() is None, "an unarmed step reads nothing"
    observer.remove()


def test_dormant_share_and_effective_rank():
    activations = torch.ones(10, 4)
    activations[:, 3] = 0.0
    assert float(dormant_share(dormant_scores(activations))) == pytest.approx(0.25)

    generator = torch.Generator().manual_seed(0)
    left = torch.randn(50, 2, generator=generator)
    right = torch.randn(2, 20, generator=generator)
    assert int(effective_rank(left @ right)) == 2


@pytest.mark.parametrize("architecture", ["vit", "swin"])
def test_hidden_layer_readings_cover_every_block(architecture):
    torch.manual_seed(0)
    model = tiny_vit() if architecture == "vit" else tiny_swin()
    blocks = 2
    readings = hidden_layer_readings(
        model, architecture, tiny_heldout().clean, torch.device("cpu"), False
    )
    for block in range(1, blocks + 1):
        assert f"dormant_share_mlp_block{block:02d}" in readings
        assert f"srank_mlp_block{block:02d}" in readings
    assert 1 <= int(readings["srank_features"]) <= tiny_heldout().clean.shape[0]
    assert all(value.dim() == 0 for value in readings.values())


def test_swin_epoch_readings_measure_surplus():
    torch.manual_seed(0)
    readings = epoch_readings(
        tiny_swin(), "swin", tiny_heldout(), (0.5,), True, torch.device("cpu"), False
    )
    assert "retention_tm_0_5_triggered" in readings
    assert "surplus_n_eval" in readings


def test_config_rejects_a_heavy_cadence_off_the_window_grid():
    with pytest.raises(ValueError):
        TelemetryConfig(every=25, heavy_every=60)
    TelemetryConfig(every=25, heavy_every=0)


def test_telemetry_is_refused_with_the_evasive_path(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "build_model", lambda *args, **kwargs: tiny_vit())
    loader = DataLoader(IndexedTrainingSet(tiny_training_set()), batch_size=4)
    with pytest.raises(ValueError, match="evasive"):
        train_classifier(
            "vit",
            NUM_CLASSES,
            loader,
            loader,
            torch.device("cpu"),
            epochs=1,
            use_sam=False,
            evasion={"probe": {}},
            checkpoint_dir=str(tmp_path),
            telemetry_config=TelemetryConfig(),
        )


def test_checkpoint_metadata_records_wall_time():
    metadata = CheckpointMetadata(
        dataset="cifar10",
        attack="badnet",
        label_mode="all_to_one",
        target_label=0,
        poison_rate=0.05,
        cover_rate=0.0,
        realized_poison_rate=0.05,
        architecture="vit",
        use_sam=False,
        rho=0.1,
        epochs=15,
        seed=0,
        max_samples=None,
        clean_accuracy=None,
        asr=None,
        started_at="2026-10-02T17:00:00+00:00",
        ended_at="2026-10-02T20:00:00+00:00",
    ).as_dict()
    assert metadata["trained_started_at"] == "2026-10-02T17:00:00+00:00"
    assert metadata["trained_ended_at"] == "2026-10-02T20:00:00+00:00"


def test_window_accumulators_read_in_1_transfer():
    from training.telemetry.window import WindowAccumulators

    window = WindowAccumulators(torch.device("cpu"))
    window.add("a", torch.tensor([1.0, 2.0]))
    window.add("a", torch.tensor([3.0, 4.0]))
    window.add("b", torch.tensor(5.0))
    read = window.read()
    assert np.allclose(read["a"], [4.0, 6.0]) and np.allclose(read["b"], [5.0])
    window.clear()
    assert window.is_empty

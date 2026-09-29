"""A checkpoint is rebuilt with the trigger it was trained on, and its cache agrees.

Commit 0150611 raised SIG's default amplitude from 0.1 to 0.157. Every SIG checkpoint
trained before it learned the 0.1 sinusoid, so its args.json has to carry the
amplitude as an override, and every loader that rebuilds an eval set from args.json
has to apply it. The last test runs the cache check of
scripts/verify_trigger_consistency.py on the 1 SIG cell of the headline panel, which
failed at 0.923 agreement before the override was recorded.
"""

import glob
import json
import os
import time

import pytest
import torch
import torchvision.transforms.v2 as transforms_v2

import data.splits as splits
from attacks import apply_config_overrides, build_attack, default_config
from data.registry import DATASET_REGISTRY
from scripts.verify_trigger_consistency import (
    AGREEMENT_BAR,
    consistency_report,
    verify_one,
)

TRAINED_AMPLITUDE = 0.1
# 0150611 in UTC, the moment SigConfig.amplitude became 0.157.
AMPLITUDE_CHANGE_UTC = "2026-09-09T02:44:51"
SIG_CELL = "vit_cifar10_sig_0_1"
GRAY = 0.5


class GrayImages(torch.utils.data.Dataset):
    """n uniform gray 32x32 images, all of class 1, exposing targets like CIFAR does."""

    def __init__(self, n: int):
        self.targets = [1] * n

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.full((3, 32, 32), GRAY)  # (3, 32, 32)
        return image, self.targets[index]


def write_sig_sidecar(folder, overrides: dict | None) -> str:
    folder.mkdir()
    sidecar = {
        "architecture": "vit",
        "dataset": "cifar10",
        "attack": "sig",
        "label_mode": "clean_label",
        "poison_rate": 0.1,
        "target_label": 0,
    }
    if overrides is not None:
        sidecar["attack_config_overrides"] = overrides
    (folder / "args.json").write_text(json.dumps(sidecar))
    checkpoint_path = str(folder / "attack_result.pt")
    return checkpoint_path


def expected_sig_image(amplitude: float) -> torch.Tensor:
    """The normalized gray image with SIG's sinusoid at amplitude, (3, 32, 32)."""
    config = apply_config_overrides(default_config("sig"), {"amplitude": amplitude})
    attack = build_attack("sig", config, image_size=32, target_label=0)
    stamped = attack.apply_trigger(torch.full((3, 32, 32), GRAY), 0)  # (3, 32, 32)

    spec = DATASET_REGISTRY["cifar10"]
    normalized = transforms_v2.Normalize(mean=spec.mean, std=spec.std)(stamped)
    return normalized


def first_backdoor_image(tmp_path, monkeypatch, overrides: dict | None) -> torch.Tensor:
    # The split needs more images than the 2000 heldout rows to leave an analysis pool.
    monkeypatch.setattr(
        splits, "load_clean_test_base", lambda *args, **kwargs: GrayImages(2010)
    )
    checkpoint_path = write_sig_sidecar(tmp_path / "vit_cifar10_sig_0_1", overrides)

    loaders, _ = splits.build_psbd_loaders_from_checkpoint(
        checkpoint_path, num_workers=0
    )
    image, _ = loaders["backdoor"].dataset[0]  # (3, 32, 32)
    return image


def test_the_psbd_loader_rebuilds_a_recorded_sig_amplitude(tmp_path, monkeypatch):
    image = first_backdoor_image(tmp_path, monkeypatch, {"amplitude": 0.1})

    assert torch.allclose(image, expected_sig_image(0.1), atol=1e-6)
    assert not torch.allclose(image, expected_sig_image(0.157), atol=1e-3)


def test_the_psbd_loader_without_an_override_uses_the_default(tmp_path, monkeypatch):
    image = first_backdoor_image(tmp_path, monkeypatch, None)

    default_amplitude = default_config("sig").amplitude
    assert torch.allclose(image, expected_sig_image(default_amplitude), atol=1e-6)


def test_evaluate_checkpoint_rebuilds_a_recorded_sig_amplitude(tmp_path, monkeypatch):
    import evaluation.metrics as metrics

    checkpoint_path = write_sig_sidecar(tmp_path / "sig", {"amplitude": 0.1})
    seen = {}

    def fake_evaluate_attack(model, dataset, attack_name, config, *rest, **kwargs):
        seen["config_amplitude"] = config.amplitude
        return {"asr": 1.0, "clean_accuracy": 1.0}

    def fake_stealth(dataset, attack_name, attack, *rest, **kwargs):
        gray = torch.full((3, 32, 32), GRAY)  # (3, 32, 32)
        seen["stealth_peak"] = float((attack.apply_trigger(gray, 0) - gray).max())
        return {}

    monkeypatch.setattr(metrics, "load_checkpoint", lambda *args: None)
    monkeypatch.setattr(metrics, "evaluate_attack", fake_evaluate_attack)
    monkeypatch.setattr(metrics, "cached_stealth_metrics", fake_stealth)

    metrics.evaluate_checkpoint(checkpoint_path, torch.device("cpu"))

    assert seen["config_amplitude"] == 0.1
    assert seen["stealth_peak"] <= 0.1 + 1e-6


def sig_checkpoint_sidecars() -> list[str]:
    paths = sorted(glob.glob(os.path.join("checkpoints", "*_sig_*", "args.json")))
    return paths


def training_start_utc(sidecar_path: str, sidecar: dict) -> str:
    """trained_started_at, or the weights' mtime for the July runs that lack it."""
    started = sidecar.get("trained_started_at")
    if started:
        return started[:19]

    weights = os.path.join(os.path.dirname(sidecar_path), "attack_result.pt")
    mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(os.path.getmtime(weights)))
    return mtime


@pytest.mark.skipif(
    not sig_checkpoint_sidecars(), reason="no SIG checkpoints on this machine"
)
def test_every_sig_checkpoint_trained_at_0_1_records_it():
    unrecorded = []
    for path in sig_checkpoint_sidecars():
        with open(path) as handle:
            sidecar = json.load(handle)
        if training_start_utc(path, sidecar) >= AMPLITUDE_CHANGE_UTC:
            continue
        overrides = sidecar.get("attack_config_overrides") or {}
        if overrides.get("amplitude") != TRAINED_AMPLITUDE:
            unrecorded.append(os.path.basename(os.path.dirname(path)))

    assert unrecorded == []


def test_consistency_report_flags_a_trigger_drift():
    fresh = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    targets = torch.zeros(10, dtype=torch.long)
    cached = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 3, 3])

    report = consistency_report(fresh, targets, cached, targets)

    assert report["status"] == "TRIGGER_DRIFT"
    assert report["agreement"] == pytest.approx(0.8)
    assert report["fresh_hit_rate"] == 1.0
    assert report["cached_hit_rate"] == pytest.approx(0.8)


def test_consistency_report_flags_a_split_drift_before_a_trigger_drift():
    fresh = torch.zeros(4, dtype=torch.long)
    cached_targets = torch.tensor([0, 0, 0, 1])

    report = consistency_report(
        fresh, torch.zeros(4, dtype=torch.long), fresh, cached_targets
    )

    assert report["status"] == "SPLIT_DRIFT"


def sig_cell_available() -> bool:
    needed = (
        os.path.join("checkpoints", SIG_CELL, "attack_result.pt"),
        os.path.join("results", SIG_CELL, "psbd", "baseline_backdoor.pt"),
        os.path.join("raw_data", "cifar10"),
    )
    available = all(os.path.exists(path) for path in needed)
    return available


@pytest.mark.skipif(
    not sig_cell_available(), reason=f"{SIG_CELL} or CIFAR-10 is not on this machine"
)
def test_the_headline_sig_cell_agrees_with_its_cached_baseline():
    class Args:
        checkpoints_dir = "checkpoints"
        results_dir = "results"
        raw_data_dir = "raw_data"
        rows = 64

    torch.set_num_threads(16)
    report = verify_one(SIG_CELL, Args, torch.device("cpu"))

    assert report["status"] == "ok", report
    assert report["agreement"] >= AGREEMENT_BAR

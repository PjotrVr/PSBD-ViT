"""The attack success rate's 2 definitions, single class and target set.

A multi-target clean-label attack predicts a set of classes, so a success is any
member of that set. The single-target path must stay the plain accuracy on the
intended labels, since every panel sidecar was scored with it.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import attack_success_rate


class FixedLogits(nn.Module):
    """Returns 1 preset logit row per input, so the predictions are chosen by the test."""

    def __init__(self, predictions: list[int], num_classes: int):
        super().__init__()
        self.logits = torch.nn.functional.one_hot(
            torch.tensor(predictions), num_classes
        ).float()  # (n, num_classes)
        self.offset = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.size(0)
        rows = self.logits[self.offset : self.offset + batch]  # (batch, num_classes)
        self.offset += batch
        return rows


def loader_with_labels(labels: list[int]) -> DataLoader:
    images = torch.zeros(len(labels), 3, 4, 4)  # (n, 3, 4, 4)
    dataset = TensorDataset(images, torch.tensor(labels))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return loader


def test_single_target_success_is_accuracy_on_the_intended_label():
    model = FixedLogits([0, 0, 3, 0], num_classes=5)
    loader = loader_with_labels([0, 0, 0, 0])
    rate = attack_success_rate(model, loader, torch.device("cpu"), use_bfloat16=False)
    assert rate == 0.75


def test_a_target_set_counts_any_member_as_a_success():
    model = FixedLogits([0, 1, 2, 4], num_classes=5)
    loader = loader_with_labels([0, 0, 0, 0])
    rate = attack_success_rate(
        model, loader, torch.device("cpu"), use_bfloat16=False, success_labels=(0, 1, 2)
    )
    assert rate == 0.75


def test_a_set_of_1_is_the_single_target_path():
    model = FixedLogits([0, 1, 2, 4], num_classes=5)
    loader = loader_with_labels([0, 0, 0, 0])
    rate = attack_success_rate(
        model, loader, torch.device("cpu"), use_bfloat16=False, success_labels=(0,)
    )
    assert rate == 0.25


def test_evaluate_checkpoint_rebuilds_the_recorded_source_classes(
    tmp_path, monkeypatch
):
    """cli.evaluate scores a checkpoint under the attack config it was trained with.

    It rebuilt every attack from its defaults, so a TaCT run over source classes
    1..5 was scored on class 1 alone. The overrides recorded in args.json must
    reach both the ASR pass and the stealth pass.
    """
    import json

    import evaluation.metrics as metrics

    folder = tmp_path / "vit_cifar10_tact_0_1_src5"
    folder.mkdir()
    sidecar = {
        "architecture": "vit",
        "dataset": "cifar10",
        "attack": "tact",
        "label_mode": "all_to_one",
        "poison_rate": 0.1,
        "target_label": 0,
        "attack_config_overrides": {"source_classes": [1, 2, 3, 4, 5]},
    }
    (folder / "args.json").write_text(json.dumps(sidecar))

    seen = {}

    def fake_evaluate_attack(model, dataset, attack_name, config, *rest, **kwargs):
        seen["config_sources"] = config.source_classes
        return {"asr": 1.0, "clean_accuracy": 1.0}

    def fake_stealth(dataset, attack_name, attack, *rest, **kwargs):
        seen["stealth_sources"] = attack.source_classes
        return {}

    monkeypatch.setattr(metrics, "load_checkpoint", lambda *args: None)
    monkeypatch.setattr(metrics, "evaluate_attack", fake_evaluate_attack)
    monkeypatch.setattr(metrics, "cached_stealth_metrics", fake_stealth)

    metrics.evaluate_checkpoint(str(folder / "attack_result.pt"), torch.device("cpu"))

    assert seen == {
        "config_sources": (1, 2, 3, 4, 5),
        "stealth_sources": (1, 2, 3, 4, 5),
    }

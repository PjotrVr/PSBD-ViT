"""Spectral Signatures port against its pinned reference and a planted outlier.

The reference is backdoor-toolbox's cleanser, the baseline the PSBD paper ran. It
reads features through model(x, True), so the fixture model returns its input
as the feature vector and every class's removal set is compared index for index.
"""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from detectors.spectral_signatures import (
    flag_top,
    removal_count,
    spectral_signature_scores,
)
from tests.reference.third_party import load_definitions, reference_file

NUM_CLASSES = 4
FEATURE_DIM = 16
POISON_RATE = 0.1


class FeaturePassthrough(nn.Module):
    """model(x, True) returns (logits, features) with the features equal to x."""

    def forward(self, inputs, return_features=False):
        logits = inputs[:, :NUM_CLASSES]  # (batch, num_classes)
        return logits, inputs


def planted_dataset(seed):
    """Clean Gaussian features per class plus a shifted poisoned cluster in class 0."""
    generator = torch.Generator().manual_seed(seed)
    class_sizes = (60, 45, 50, 40)
    features = []
    labels = []
    for label, size in enumerate(class_sizes):
        features.append(torch.randn(size, FEATURE_DIM, generator=generator))
        labels.append(torch.full((size,), label))
    poisoned = (
        torch.randn(20, FEATURE_DIM, generator=generator)
        + 8.0 * torch.eye(FEATURE_DIM)[0]
    )
    features.append(poisoned)
    labels.append(torch.zeros(20, dtype=torch.long))
    all_features = torch.cat(features)  # (n, feature_dim)
    all_labels = torch.cat(labels)  # (n,)
    return all_features, all_labels


def port_suspicious_indices(features, labels):
    suspicious = []
    for label in range(NUM_CLASSES):
        positions = torch.nonzero(labels == label).squeeze(1)  # (n_label,)
        scores = spectral_signature_scores(features[positions])  # (n_label,)
        count = removal_count(len(positions), POISON_RATE, len(labels))
        flagged = flag_top(scores, count)  # (n_label,)
        suspicious.extend(positions[flagged].tolist())
    return suspicious


def test_scores_are_the_squared_projection_on_the_top_right_singular_vector():
    features, _ = planted_dataset(seed=0)
    centered = features.numpy() - features.numpy().mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    expected = (centered @ vh[0]) ** 2  # (n,)

    scores = spectral_signature_scores(features).numpy()

    np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=1e-4)


def test_the_planted_cluster_ranks_first_in_its_class():
    features, labels = planted_dataset(seed=1)
    positions = torch.nonzero(labels == 0).squeeze(1)
    scores = spectral_signature_scores(features[positions])
    poisoned = torch.zeros(len(positions), dtype=torch.bool)
    poisoned[-20:] = True

    top = flag_top(scores, 20)

    assert (top & poisoned).sum() >= 18


def test_removal_count_caps_at_half_the_class():
    assert removal_count(class_size=100, poison_rate=0.1, dataset_size=50000) == 50
    assert removal_count(class_size=10000, poison_rate=0.1, dataset_size=50000) == 5000
    assert removal_count(class_size=10000, poison_rate=0.01, dataset_size=50000) == 750


def test_port_removes_exactly_what_the_reference_cleanser_removes():
    path = reference_file(
        "backdoor-toolbox", "cleansers_tool_box", "spectral_signature.py"
    )
    namespace = {"np": np, "torch": torch, "tqdm": lambda iterable: iterable}
    load_definitions(
        path,
        ("get_features", "cleanser"),
        namespace,
        replacements=(
            ("ins_data = ins_data.cuda()", "ins_data = ins_data"),
            (
                "kwargs = {'num_workers': 4, 'pin_memory': True}",
                "kwargs = {'num_workers': 0, 'pin_memory': False}",
            ),
        ),
    )
    features, labels = planted_dataset(seed=2)
    inspection_set = TensorDataset(features, labels)

    reference = namespace["cleanser"](
        inspection_set,
        FeaturePassthrough(),
        NUM_CLASSES,
        SimpleNamespace(poison_rate=POISON_RATE),
    )
    ported = port_suspicious_indices(features, labels)

    assert sorted(int(i) for i in reference) == sorted(ported)

"""STRIP against the 2 executable references, on a tiny random classifier on the CPU.

backdoor-toolbox's other_defenses_tool_box/strip.py is the reference whose rule
the port follows: denormalize, add with weight 1, clamp to [0, 1], renormalize,
softmax entropy in nats averaged over the overlays. Its own check() loop is run
here with a loader that serves 1 overlay per round, so every input meets the
same overlays the port uses, and the 2 must agree to 1e-5.

The Beatrix copy, defenses/STRIP/STRIP.py, is the one docs/detectors/strip.md
names. Its superimposition is cv2.addWeighted on uint8 arrays, checked here to
build the same composites as the port. Its entropy is taken over the SIGMOID of
the logits in bits, not over the softmax, so the 2 statistics differ and the
last test asserts that difference rather than hiding it.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from detectors.strip import blend_entropy, strip_scores
from tests.reference.third_party import (
    keep_cuda_calls_on_the_cpu,
    load_definitions,
    reference_file,
)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 8
NUM_CLASSES = 5
NUM_OVERLAYS = 4
NUM_IMAGES = 6
MEAN = (0.4, 0.5, 0.6)
STD = (0.2, 0.25, 0.3)
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)

# The port floors a probability at 1e-12 before the log and backdoor-toolbox adds
# 1e-8 to every probability. Over 5 classes that moves an entropy by at most
# about 5 * 1e-8 * |log 1e-8|, under 1e-5.
ENTROPY_TOLERANCE = 1e-5


class RecordingModel(nn.Module):
    """A classifier that keeps every batch it was asked to classify."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.seen: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.seen.append(images.detach().clone())
        logits = self.inner(images)  # (batch, NUM_CLASSES)
        return logits


def tiny_classifier(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 16),
        nn.Tanh(),
        nn.Linear(16, NUM_CLASSES),
    ).eval()
    return model


def pixel_grid_images(count: int, seed: int) -> torch.Tensor:
    """(count, 3, IMAGE_SIZE, IMAGE_SIZE) on the 8-bit grid, so a uint8 round trip is exact."""
    generator = torch.Generator().manual_seed(seed)
    levels = torch.randint(
        0, 256, (count, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator
    )
    pixels = levels.float() / 255.0
    return pixels


def normalize(pixels: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    mean_tensor = torch.tensor(mean).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std).view(1, -1, 1, 1)
    normalized = (pixels - mean_tensor) / std_tensor  # (count, 3, height, width)
    return normalized


def load_toolbox_strip():
    path = reference_file("backdoor-toolbox", "other_defenses_tool_box", "strip.py")
    namespace = {"torch": torch, "BackdoorDefense": object}
    load_definitions(path, ("STRIP",), namespace)
    return namespace["STRIP"]


def load_beatrix_strip(monkeypatch):
    path = reference_file("Beatrix", "defenses", "STRIP", "STRIP.py")
    namespace = {
        "np": np,
        "torch": torch,
        "F": F,
        "cv2": SaturatingOpenCV,
        "transforms": transforms,
    }
    load_definitions(path, ("STRIP",), namespace)
    # _get_entropy draws its overlay indices per call with np.random.randint.
    # Serving 0..n-1 hands it the same overlays in the same order as the port.
    monkeypatch.setattr(np.random, "randint", lambda low, high, size: np.arange(size))
    return namespace["STRIP"]


class SaturatingOpenCV:
    """cv2.addWeighted as OpenCV documents it for uint8 arrays.

    dst = saturate_cast<uchar>(src1 * alpha + src2 * beta + gamma), where
    saturate_cast rounds to the nearest integer and clips to [0, 255]
    (OpenCV core, "Operations on arrays", addWeighted). opencv-python is not
    installed here, so this 1 documented formula stands in for it.
    """

    @staticmethod
    def addWeighted(src1, alpha, src2, beta, gamma):
        weighted = src1.astype(np.float64) * alpha + src2.astype(np.float64) * beta
        saturated = np.clip(np.rint(weighted + gamma), 0, 255).astype(np.uint8)
        return saturated


def test_matches_backdoor_toolbox_check_to_1e_5(monkeypatch):
    keep_cuda_calls_on_the_cpu(monkeypatch)
    toolbox_strip_class = load_toolbox_strip()
    model = tiny_classifier()

    images = normalize(pixel_grid_images(NUM_IMAGES, seed=1), MEAN, STD)
    overlays = normalize(pixel_grid_images(NUM_OVERLAYS, seed=2), MEAN, STD)
    loader = DataLoader(TensorDataset(images, torch.zeros(NUM_IMAGES)), batch_size=3)

    ours = strip_scores(
        model,
        loader,
        DEVICE,
        overlays,
        MEAN,
        STD,
        use_bfloat16=False,
        seed=0,
        num_overlays=NUM_OVERLAYS,
    )  # (NUM_IMAGES,)

    # The reference's own loop, with a loader whose round i serves overlay i to
    # every input, which is the fixed overlay set of the port's deviation 3.
    reference = object.__new__(toolbox_strip_class)
    reference.model = model
    reference.strip_alpha = 1.0
    reference.N = NUM_OVERLAYS
    reference.normalizer = transforms.Normalize(MEAN, STD)
    std_tensor = torch.tensor(STD).view(1, -1, 1, 1)
    mean_tensor = torch.tensor(MEAN).view(1, -1, 1, 1)
    reference.denormalizer = lambda batch: batch * std_tensor + mean_tensor
    reference.train_loader = [
        (overlay.expand(NUM_IMAGES, -1, -1, -1), torch.zeros(NUM_IMAGES))
        for overlay in overlays
    ]
    with torch.no_grad():
        theirs = reference.check(images, torch.zeros(NUM_IMAGES))  # (NUM_IMAGES,)

    assert ours.shape == theirs.shape == (NUM_IMAGES,)
    assert ours.std() > 0, "identical entropies would make the agreement vacuous"
    assert torch.allclose(ours, theirs.float(), atol=ENTROPY_TOLERANCE)


def test_the_overlay_set_does_not_depend_on_batch_position():
    """Deviation 3: every input meets the same overlays wherever the loader puts it.

    backdoor-toolbox pairs input i of a batch with image i of each shuffled
    training batch, so an input's overlays depend on its position. The port's
    score of an image must not.
    """
    model = tiny_classifier()
    images = normalize(pixel_grid_images(NUM_IMAGES, seed=1), MEAN, STD)
    overlays = normalize(pixel_grid_images(NUM_OVERLAYS, seed=2), MEAN, STD)
    reversed_images = images.flip(0)

    def score(batch_images: torch.Tensor, batch_size: int) -> torch.Tensor:
        loader = DataLoader(
            TensorDataset(batch_images, torch.zeros(NUM_IMAGES)), batch_size=batch_size
        )
        scores = strip_scores(
            model, loader, DEVICE, overlays, MEAN, STD, False, 0, NUM_OVERLAYS
        )
        return scores

    in_order = score(images, 4)
    reordered = score(reversed_images, 5).flip(0)

    assert torch.allclose(in_order, reordered, atol=1e-6)


def test_composites_match_beatrix_saturating_add(monkeypatch):
    """Deviation 2: the sum saturates at 1 as cv2.addWeighted saturates uint8 at 255."""
    beatrix_strip_class = load_beatrix_strip(monkeypatch)

    pixels = pixel_grid_images(1, seed=3)  # (1, 3, 8, 8)
    overlay_pixels = pixel_grid_images(NUM_OVERLAYS, seed=4)  # (4, 3, 8, 8)
    to_uint8 = lambda batch: (  # noqa: E731
        (batch * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    )  # (count, height, width, 3) uint8

    theirs_model = RecordingModel(tiny_classifier())
    reference = object.__new__(beatrix_strip_class)
    reference.n_sample = NUM_OVERLAYS
    # Beatrix's GTSRB branch, ToTensor with no Normalize, so both sides see
    # [0, 1] pixels and the composites can be compared directly.
    reference.normalizer = transforms.ToTensor()
    reference.device = DEVICE
    dataset = [(overlay, 0) for overlay in to_uint8(overlay_pixels)]
    with torch.no_grad():
        reference._get_entropy(to_uint8(pixels)[0], dataset, theirs_model)

    ours_model = RecordingModel(tiny_classifier())
    loader = DataLoader(TensorDataset(pixels, torch.zeros(1)), batch_size=1)
    strip_scores(
        ours_model,
        loader,
        DEVICE,
        overlay_pixels,
        IDENTITY_MEAN,
        IDENTITY_STD,
        False,
        0,
        NUM_OVERLAYS,
    )

    theirs_composites = theirs_model.seen[0]  # (4, 3, 8, 8)
    ours_composites = torch.cat(ours_model.seen)  # (4, 3, 8, 8)
    saturated = (theirs_composites == 1.0).float().mean()
    assert 0.1 < saturated < 0.9, "the check needs both saturated and unsaturated sums"
    assert torch.allclose(ours_composites, theirs_composites, atol=1e-6)


def test_beatrix_entropy_is_over_sigmoid_in_bits_not_softmax(monkeypatch):
    """A disagreement with the Beatrix copy that docs/detectors/strip.md did not record.

    STRIP.py line 69 reads torch.sigmoid(py1_add) and line 70 sums
    -p * log2(p) over those per-class sigmoids, which is not the entropy of any
    distribution. The port computes Eq. (2) over the softmax in nats. The 2 are
    not a constant multiple of each other, so they can rank inputs differently.
    """
    beatrix_strip_class = load_beatrix_strip(monkeypatch)
    to_uint8 = lambda batch: (  # noqa: E731
        (batch * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    )
    overlay_pixels = pixel_grid_images(NUM_OVERLAYS, seed=4)
    dataset = [(overlay, 0) for overlay in to_uint8(overlay_pixels)]

    beatrix_scores = []
    port_scores = []
    for seed in range(5):
        pixels = pixel_grid_images(1, seed=10 + seed)
        recording = RecordingModel(tiny_classifier())
        reference = object.__new__(beatrix_strip_class)
        reference.n_sample = NUM_OVERLAYS
        reference.normalizer = transforms.ToTensor()
        reference.device = DEVICE
        with torch.no_grad():
            beatrix_scores.append(
                float(reference._get_entropy(to_uint8(pixels)[0], dataset, recording))
            )
            logits = recording.inner(recording.seen[0])  # (4, NUM_CLASSES)

        sigmoid = torch.sigmoid(logits).double()
        sigmoid_bits = -(sigmoid * torch.log2(sigmoid)).sum() / NUM_OVERLAYS
        assert beatrix_scores[-1] == pytest.approx(float(sigmoid_bits), abs=1e-5)

        softmax = F.softmax(logits, dim=1)
        port_scores.append(float(blend_entropy(softmax).mean()))

    ratios = np.array(beatrix_scores) / np.array(port_scores)
    assert ratios.std() > 1e-3, "a constant ratio would make the 2 rank alike"

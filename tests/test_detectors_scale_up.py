"""SCALE-UP against the authors' scoring and BackdoorBox's class, on a tiny classifier.

The authors' repository (third_party/SCALE-UP) splits the method in 2 scripts.
torch_model_wrapper.py records the predicted label at every factor in
range(1, 12) at script level (lines 55 to 58), which cannot be imported and is
transcribed here, and test.py's process() turns each row into
np.mean(a[i] == a[i][0]), which is executed from the pinned file. Column 0 is the
prediction at factor 1, so the authors' SPC counts the unscaled prediction as 1
agreement out of 11. The port with OFFICIAL_CODE_SCALES reproduces that exactly.

BackdoorBox's SCALE_UP class is executed with its own _test and init_spc_norm.
The data-free score must agree bit for bit. The data-limited variant shares the
Eq. (4) arithmetic when every class falls back to pooled statistics, and the 3
places the port departs from BackdoorBox (per-class statistics, the reference
label of the validation SPC and the prediction-correctness filter) are asserted
as departures.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from detectors.scale_up import (
    OFFICIAL_CODE_SCALES,
    PAPER_SCALES,
    fit_class_spc_statistics,
    spc_scores,
    standardize_spc,
)
from tests.reference.third_party import (
    keep_cuda_calls_on_the_cpu,
    load_definitions,
    reference_file,
)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 6
NUM_CLASSES = 4
NUM_IMAGES = 40
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)
MEAN = (0.45, 0.5, 0.4)
STD = (0.22, 0.25, 0.3)


class BaseStub:
    """BackdoorBox's defenses.base.Base seeds 3 RNGs and nothing the tests read."""

    def __init__(self, seed=0, deterministic=False):
        pass


class NormalizingModel(nn.Module):
    """A classifier that normalizes its own input, for references that feed raw pixels."""

    def __init__(self, inner: nn.Module, mean: tuple, std: tuple):
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        logits = self.inner((pixels - self.mean) / self.std)  # (batch, NUM_CLASSES)
        return logits


def tiny_classifier() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 12),
        nn.Tanh(),
        nn.Linear(12, NUM_CLASSES),
    ).eval()
    return model


def dim_pixels(count: int, seed: int) -> torch.Tensor:
    """Pixels in [0, 0.25), so amplification by 3 to 11 crosses the clip at 1 at every factor."""
    generator = torch.Generator().manual_seed(seed)
    pixels = 0.25 * torch.rand(count, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    return pixels


def predicted_labels(model: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        labels = model(pixels).argmax(dim=1)  # (count,)
    return labels


def load_backdoorbox_scale_up():
    path = reference_file("BackdoorBox", "core", "defenses", "SCALE_UP.py")
    namespace = {"torch": torch, "Base": BaseStub, "test": None, "metrics": None}
    load_definitions(path, ("SCALE_UP",), namespace)
    return namespace["SCALE_UP"]


def test_official_code_scales_match_the_authors_process_exactly(tmp_path):
    process_file = reference_file("SCALE-UP", "test.py")
    process = load_definitions(process_file, ("process",), {"np": np})["process"]
    model = tiny_classifier()
    pixels = dim_pixels(NUM_IMAGES, seed=1)

    # torch_model_wrapper.py lines 55 to 58, the factor loop, without the
    # 0.02 * U[0, 1) input noise of line 41 (the port's deviation 4).
    decisions = np.empty((NUM_IMAGES, 11))
    with torch.no_grad():
        for h in range(1, 12):
            img_batch_re = torch.clamp(h * pixels, 0, 1)
            decisions[:, (h - 1)] = torch.max(model(img_batch_re), 1)[1].numpy()
    decisions_file = tmp_path / "decisions.npy"
    np.save(decisions_file, decisions)
    theirs = process(str(decisions_file))  # (NUM_IMAGES,)

    loader = DataLoader(TensorDataset(pixels, torch.zeros(NUM_IMAGES)), batch_size=16)
    ours, _, _ = spc_scores(
        model,
        loader,
        DEVICE,
        IDENTITY_MEAN,
        IDENTITY_STD,
        OFFICIAL_CODE_SCALES,
        use_bfloat16=False,
    )

    assert len(np.unique(theirs)) > 2, "near-constant SPC would make this vacuous"
    assert np.allclose(ours.numpy(), theirs, atol=1e-7)


def test_data_free_spc_matches_backdoorbox_test_exactly(monkeypatch):
    keep_cuda_calls_on_the_cpu(monkeypatch)
    scale_up_class = load_backdoorbox_scale_up()
    model = tiny_classifier()
    pixels = dim_pixels(NUM_IMAGES, seed=2)
    # Labels equal to the model's predictions keep every sample through
    # BackdoorBox's prediction-correctness filter, so the 2 outputs align.
    labels = predicted_labels(model, pixels)

    reference = scale_up_class(model, scale_set=list(PAPER_SCALES))
    with torch.no_grad():
        theirs = reference._test(TensorDataset(pixels, labels))  # (NUM_IMAGES,)

    loader = DataLoader(TensorDataset(pixels, labels), batch_size=16)
    ours, _, _ = spc_scores(
        model, loader, DEVICE, IDENTITY_MEAN, IDENTITY_STD, PAPER_SCALES, False
    )

    assert theirs.shape == (NUM_IMAGES,)
    assert len(torch.unique(theirs)) > 2
    assert torch.equal(ours, theirs.float())


def test_normalized_loader_amplifies_in_pixel_space_like_the_commented_branch(
    monkeypatch,
):
    """The port denormalizes, multiplies, clips and renormalizes (Section 4.2).

    BackdoorBox multiplies the loader tensor directly and leaves that round trip
    commented out, which is right only for a loader serving [0, 1] pixels. Fed
    pixels through a model that normalizes internally, BackdoorBox computes the
    Section 4.2 statistic, and the port on the normalized loader must equal it.
    """
    keep_cuda_calls_on_the_cpu(monkeypatch)
    scale_up_class = load_backdoorbox_scale_up()
    inner = tiny_classifier()
    pixel_model = NormalizingModel(inner, MEAN, STD).eval()
    pixels = dim_pixels(NUM_IMAGES, seed=3)
    labels = predicted_labels(pixel_model, pixels)

    reference = scale_up_class(pixel_model, scale_set=list(PAPER_SCALES))
    with torch.no_grad():
        theirs = reference._test(TensorDataset(pixels, labels))

    normalized = (pixels - pixel_model.mean) / pixel_model.std
    loader = DataLoader(TensorDataset(normalized, labels), batch_size=16)
    ours, _, _ = spc_scores(inner, loader, DEVICE, MEAN, STD, PAPER_SCALES, False)

    assert torch.equal(ours, theirs.float())


def test_backdoorbox_drops_misclassified_inputs_and_the_port_does_not(monkeypatch):
    """Deviation 5: BackdoorBox's _test keeps only inputs whose prediction equals the label."""
    keep_cuda_calls_on_the_cpu(monkeypatch)
    scale_up_class = load_backdoorbox_scale_up()
    model = tiny_classifier()
    pixels = dim_pixels(NUM_IMAGES, seed=4)
    labels = predicted_labels(model, pixels)
    wrong = torch.arange(NUM_IMAGES) % 4 == 0  # (NUM_IMAGES,)
    labels[wrong] = (labels[wrong] + 1) % NUM_CLASSES

    reference = scale_up_class(model, scale_set=list(PAPER_SCALES))
    with torch.no_grad():
        theirs = reference._test(TensorDataset(pixels, labels))
    loader = DataLoader(TensorDataset(pixels, labels), batch_size=16)
    ours, _, _ = spc_scores(
        model, loader, DEVICE, IDENTITY_MEAN, IDENTITY_STD, PAPER_SCALES, False
    )

    assert ours.shape == (NUM_IMAGES,)
    assert theirs.shape == (NUM_IMAGES - int(wrong.sum()),)
    assert torch.equal(ours[~wrong], theirs.float())


def test_data_limited_matches_backdoorbox_when_every_class_is_pooled(monkeypatch):
    """The Eq. (4) arithmetic agrees when both sides standardize by pooled statistics.

    BackdoorBox's init_spc_norm keeps 1 pooled mean and standard deviation. The
    port pools only for a class with fewer than MIN_CLASS_SAMPLES members, so a
    validation set of at most 4 per class makes the 2 fits the same quantity.
    Validation labels equal the predictions, which removes the other departure.
    """
    keep_cuda_calls_on_the_cpu(monkeypatch)
    scale_up_class = load_backdoorbox_scale_up()
    model = tiny_classifier()
    validation_pixels, validation_labels = sparse_validation_set(model, per_class=4)
    test_pixels = dim_pixels(NUM_IMAGES, seed=6)
    test_labels = predicted_labels(model, test_pixels)

    reference = scale_up_class(
        model,
        scale_set=list(PAPER_SCALES),
        valset=TensorDataset(validation_pixels, validation_labels),
    )
    with torch.no_grad():
        theirs = reference._test(TensorDataset(test_pixels, test_labels))

    validation_loader = DataLoader(
        TensorDataset(validation_pixels, validation_labels), batch_size=16
    )
    validation_spc, _, validation_true = spc_scores(
        model,
        validation_loader,
        DEVICE,
        IDENTITY_MEAN,
        IDENTITY_STD,
        PAPER_SCALES,
        False,
    )
    class_means, class_stds = fit_class_spc_statistics(
        validation_spc, validation_true, NUM_CLASSES
    )
    test_loader = DataLoader(TensorDataset(test_pixels, test_labels), batch_size=16)
    test_spc, test_predicted, _ = spc_scores(
        model, test_loader, DEVICE, IDENTITY_MEAN, IDENTITY_STD, PAPER_SCALES, False
    )
    ours = standardize_spc(test_spc, test_predicted, class_means, class_stds)

    assert torch.allclose(class_means, torch.full_like(class_means, reference.mean))
    assert torch.allclose(class_stds, torch.full_like(class_stds, reference.std))
    assert torch.allclose(ours, theirs.float(), atol=1e-6)


def test_backdoorbox_validation_spc_counts_against_the_label_not_the_prediction(
    monkeypatch,
):
    """docs/detectors/scale_up.md records this, the test pins it.

    init_spc_norm adds scale_label == labels, the ground truth, where Eq. (2)
    compares against C(x). With some validation labels wrong, BackdoorBox's
    pooled mean falls below the pooled mean of the port's Eq. (2) SPC.
    """
    keep_cuda_calls_on_the_cpu(monkeypatch)
    scale_up_class = load_backdoorbox_scale_up()
    model = tiny_classifier()
    validation_pixels, validation_labels = sparse_validation_set(model, per_class=4)
    wrong = torch.arange(validation_labels.numel()) % 3 == 0
    validation_labels[wrong] = (validation_labels[wrong] + 1) % NUM_CLASSES

    reference = scale_up_class(
        model,
        scale_set=list(PAPER_SCALES),
        valset=TensorDataset(validation_pixels, validation_labels),
    )
    loader = DataLoader(
        TensorDataset(validation_pixels, validation_labels), batch_size=16
    )
    ours, _, _ = spc_scores(
        model, loader, DEVICE, IDENTITY_MEAN, IDENTITY_STD, PAPER_SCALES, False
    )

    assert reference.mean < float(ours.mean()) - 0.05


def test_the_port_fits_per_class_statistics_where_backdoorbox_pools():
    """Eq. (3) is per class. With 6 samples per class the port's means differ by class."""
    model = tiny_classifier()
    validation_pixels, validation_labels = sparse_validation_set(model, per_class=6)
    loader = DataLoader(
        TensorDataset(validation_pixels, validation_labels), batch_size=16
    )
    spc, _, true_labels = spc_scores(
        model, loader, DEVICE, IDENTITY_MEAN, IDENTITY_STD, PAPER_SCALES, False
    )

    class_means, _ = fit_class_spc_statistics(spc, true_labels, NUM_CLASSES)
    populated = torch.bincount(true_labels, minlength=NUM_CLASSES) >= 5

    assert populated.sum() >= 2
    assert class_means[populated].std() > 0
    for class_index in torch.nonzero(populated).flatten():
        members = spc[true_labels == class_index]
        assert torch.isclose(class_means[class_index], members.mean())


def sparse_validation_set(
    model: nn.Module, per_class: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """At most per_class pixels per predicted class, labeled with that prediction."""
    candidates = dim_pixels(400, seed=5)
    labels = predicted_labels(model, candidates)

    chosen = []
    for class_index in range(NUM_CLASSES):
        members = torch.nonzero(labels == class_index).flatten()[:per_class]
        chosen.append(members)
    keep = torch.cat(chosen)  # (kept,)
    return candidates[keep], labels[keep].clone()

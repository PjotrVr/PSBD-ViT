"""TeCo's 14 corruptions against imagecorruptions, and its score against BackdoorBench's loop.

BackdoorBench's detection_infer/teco.py corrupts every image with the
imagecorruptions package (corrupt at line 170), so that package, pinned as
third_party/imagecorruptions, is the reference for every operator. Its
corruptions.py is executed whole on the CPU. It imports cv2 and pkg_resources,
neither of which is installed, so both are replaced for the duration of a test:
pkg_resources only locates the frost images this port does not implement, and
the 3 cv2 calls the compared corruptions make are written out from OpenCV's
documented formulas with scipy (OpenCVFormulas below), independent of the
port's torch code. The njit pixel shuffle of glass_blur is read as plain Python,
so its random draws can be served.

The comparison criterion is the 8-bit grid. The reference returns floats in
[0, 255] and the port returns the nearest 8-bit level, so every port pixel must
lie within half a level of the reference float, plus 1e-3 of a level for the
float32 arithmetic. The random corruptions are compared with the same draws
served to both sides. What does not meet the criterion is asserted as a
disagreement, with where it comes from.

BackdoorBench's scoring loop (teco.py lines 323 to 341) is a method body and is
transcribed here line for line, then run on the predictions the port's own
hardness pass produced, so the index rule and np.std are checked exactly.
"""

import math
import sys
import types

import numpy as np
import pytest
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from detectors import teco
from tests.reference.third_party import load_definitions, reference_file

DEVICE = torch.device("cpu")
IMAGE_SIZE = 32
SEVERITIES = (1, 2, 3, 4, 5)

# Half an 8-bit level for the port's rounding, plus float32 slack.
LEVEL_TOLERANCE = 0.5 + 1e-3


class OpenCVFormulas:
    """The 3 cv2 calls the compared corruptions make, from OpenCV's documentation.

    getGaussianKernel with an explicit sigma: G_i = alpha * exp(-(i - (ksize -
    1) / 2)^2 / (2 sigma^2)), normalized to sum 1. GaussianBlur and filter2D
    correlate with the default border BORDER_REFLECT_101, which is scipy's
    "mirror". COLOR_RGB2GRAY is Y = 0.299 R + 0.587 G + 0.114 B.
    """

    COLOR_RGB2GRAY = 7

    @staticmethod
    def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
        offsets = np.arange(ksize) - (ksize - 1) / 2.0
        weights = np.exp(-(offsets**2) / (2.0 * sigma**2))
        kernel = weights / weights.sum()
        return kernel

    @staticmethod
    def GaussianBlur(image, ksize, sigmaX):
        horizontal = OpenCVFormulas.gaussian_kernel(ksize[0], sigmaX)
        vertical = OpenCVFormulas.gaussian_kernel(ksize[1], sigmaX)
        rows_done = ndimage.correlate1d(image, horizontal, axis=1, mode="mirror")
        blurred = ndimage.correlate1d(rows_done, vertical, axis=0, mode="mirror")
        return blurred

    @staticmethod
    def filter2D(image, ddepth, kernel):
        filtered = ndimage.correlate(image, kernel, mode="mirror")
        return filtered

    @staticmethod
    def cvtColor(image, code):
        assert code == OpenCVFormulas.COLOR_RGB2GRAY
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        return gray


@pytest.fixture
def reference(monkeypatch):
    """imagecorruptions' corruptions.py executed as a module, with its missing imports stubbed."""
    path = reference_file("imagecorruptions", "imagecorruptions", "corruptions.py")
    monkeypatch.setitem(sys.modules, "cv2", OpenCVFormulas)
    monkeypatch.setitem(
        sys.modules,
        "pkg_resources",
        types.SimpleNamespace(resource_filename=lambda *parts: None),
    )
    namespace = {"__name__": "imagecorruptions_reference"}
    exec(compile(open(path).read(), path, "exec"), namespace)

    # The njit shuffle draws from numba's own generator, which no test can serve.
    # Its Python body (lines 157 to 167) is executed instead, without the
    # decorator, and computes the same thing.
    load_definitions(path, ("_shuffle_pixels_njit_glass_blur",), namespace)
    return namespace


def sample_image() -> np.ndarray:
    """1 random uint8 image, (32, 32, 3), the reference's input type."""
    generator = np.random.RandomState(0)
    image = generator.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)
    return image


def as_port_input(image: np.ndarray) -> torch.Tensor:
    pixels = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255.0
    return pixels  # (1, 3, 32, 32)


def level_gap(reference_output, port_output: torch.Tensor) -> np.ndarray:
    """|port * 255 - reference| per pixel and channel, (32, 32, 3)."""
    reference_levels = np.asarray(reference_output, dtype=np.float64)
    port_levels = port_output[0].permute(1, 2, 0).double().numpy() * 255.0
    gap = np.abs(port_levels - reference_levels)
    return gap


class Draws:
    """Serves a fixed list of values to successive calls of a patched RNG function."""

    def __init__(self, values):
        self.values = list(values)

    def next(self):
        value = self.values.pop(0)
        return value


@pytest.mark.parametrize("severity", SEVERITIES)
@pytest.mark.parametrize(
    "name", ["contrast", "brightness", "zoom_blur", "defocus_blur", "jpeg_compression"]
)
def test_deterministic_corruptions_round_the_reference(reference, name, severity):
    image = sample_image()

    theirs = np.asarray(reference[name](Image.fromarray(image), severity))
    ours = teco.CORRUPTIONS[name](as_port_input(image), severity)

    assert level_gap(theirs, ours).max() <= LEVEL_TOLERANCE


@pytest.mark.parametrize("severity", SEVERITIES)
def test_fog_rounds_the_reference_under_the_same_numpy_seed(reference, severity):
    """The port's plasma fractal draws from numpy in the reference's order."""
    image = sample_image()

    np.random.seed(3)
    theirs = reference["fog"](Image.fromarray(image), severity)
    np.random.seed(3)
    ours = teco.fog(as_port_input(image), severity)

    assert level_gap(theirs, ours).max() <= LEVEL_TOLERANCE


@pytest.mark.parametrize("severity", SEVERITIES)
def test_gaussian_noise_rounds_the_reference_on_the_same_draw(
    reference, monkeypatch, severity
):
    image = sample_image()
    standard_normal = np.random.RandomState(1).randn(IMAGE_SIZE, IMAGE_SIZE, 3)

    monkeypatch.setattr(
        np.random, "normal", lambda size, scale: standard_normal * scale
    )
    theirs = reference["gaussian_noise"](Image.fromarray(image), severity)
    monkeypatch.setattr(
        torch,
        "randn_like",
        lambda like: torch.from_numpy(standard_normal).permute(2, 0, 1)[None].float(),
    )
    ours = teco.gaussian_noise(as_port_input(image), severity)

    assert level_gap(theirs, ours).max() <= LEVEL_TOLERANCE


@pytest.mark.parametrize("severity", SEVERITIES)
def test_shot_noise_rounds_the_reference_on_the_same_draw(
    reference, monkeypatch, severity
):
    """Both samplers replaced by rounding their rate, the same map on both sides."""
    image = sample_image()

    monkeypatch.setattr(np.random, "poisson", lambda rate: np.round(rate))
    theirs = reference["shot_noise"](Image.fromarray(image), severity)
    monkeypatch.setattr(torch, "poisson", lambda rate: torch.round(rate))
    ours = teco.shot_noise(as_port_input(image), severity)

    assert level_gap(theirs, ours).max() <= LEVEL_TOLERANCE


def fill_uniform_with(values: Draws):
    """A torch.Tensor.uniform_ stand-in that fills from values, in call order."""

    def uniform_(tensor, low=0.0, high=1.0):
        drawn = torch.as_tensor(values.next(), dtype=tensor.dtype)
        tensor.copy_(
            drawn.expand_as(tensor) if drawn.dim() == 0 else drawn.view_as(tensor)
        )
        return tensor

    return uniform_


@pytest.mark.parametrize("severity", SEVERITIES)
def test_motion_blur_rounds_the_reference_at_the_same_angle(
    reference, monkeypatch, severity
):
    image = sample_image()
    angle = 17.0

    monkeypatch.setattr(np.random, "uniform", lambda low, high: angle)
    theirs = reference["motion_blur"](Image.fromarray(image), severity)
    monkeypatch.setattr(torch.Tensor, "uniform_", fill_uniform_with(Draws([angle])))
    ours = teco.motion_blur(as_port_input(image), severity)

    assert level_gap(theirs, ours).max() <= LEVEL_TOLERANCE


@pytest.mark.parametrize("severity", SEVERITIES)
def test_snow_rounds_the_reference_on_the_same_flakes_and_angle(
    reference, monkeypatch, severity
):
    image = sample_image()
    standard_normal = np.random.RandomState(2).randn(IMAGE_SIZE, IMAGE_SIZE)
    angle = -100.0

    monkeypatch.setattr(
        np.random, "normal", lambda size, loc, scale: loc + scale * standard_normal
    )
    monkeypatch.setattr(np.random, "uniform", lambda low, high: angle)
    theirs = reference["snow"](Image.fromarray(image), severity)

    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, **kwargs: torch.from_numpy(standard_normal).float().view(*shape),
    )
    monkeypatch.setattr(torch.Tensor, "uniform_", fill_uniform_with(Draws([angle])))
    ours = teco.snow(as_port_input(image), severity)

    assert level_gap(theirs, ours).max() <= LEVEL_TOLERANCE


@pytest.mark.parametrize("severity", SEVERITIES)
def test_glass_blur_blurs_like_the_reference_when_nothing_moves(
    reference, monkeypatch, severity
):
    """With every displacement 0 the shuffle is the identity, which isolates the 2 blurs."""
    image = sample_image()

    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: np.zeros(size, dtype=int)
    )
    theirs = reference["glass_blur"](Image.fromarray(image), severity)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None: torch.zeros(size, dtype=torch.long),
    )
    ours = teco.glass_blur(as_port_input(image), severity)

    assert level_gap(theirs, ours).max() <= LEVEL_TOLERANCE


def serve_displacements(monkeypatch, severity: int) -> None:
    """The same displacement pairs to the reference's np.random.randint and the port's torch.randint."""
    max_delta = (1, 2, 2, 3, 4)[severity - 1]
    pairs = np.random.RandomState(5).randint(-max_delta, max_delta, (5000, 2))
    numpy_draws = Draws(pairs)
    torch_draws = Draws(pairs)
    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: numpy_draws.next()
    )
    # The port draws (2, batch) per position, row 0 the column shift and row 1
    # the row shift, the order of the reference's dx, dy.
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None: torch.from_numpy(torch_draws.next()).view(
            2, 1
        ),
    )


@pytest.mark.parametrize("severity", SEVERITIES)
def test_glass_blur_swaps_where_the_reference_duplicates(
    reference, monkeypatch, severity
):
    """A disagreement docs/detectors/teco.md did not record.

    The reference's shuffle writes x[h, w], x[h', w'] = x[h', w'], x[h, w] on a
    (height, width, 3) array (corruptions.py line 166). Both right-hand sides are
    views, so the first assignment overwrites the pixel the second one reads,
    and the pixel at (h', w') is copied into (h, w) without moving the other
    way. numba compiles the same statement with the same view semantics. The
    port clones both pixels and swaps them. With the reference's statement made
    into a real swap, the 2 agree on the grid again, which shows the swap is the
    whole difference.
    """
    image = sample_image()

    serve_displacements(monkeypatch, severity)
    duplicating = reference["glass_blur"](Image.fromarray(image), severity)
    ours = teco.glass_blur(as_port_input(image), severity)

    serve_displacements(monkeypatch, severity)
    path = reference_file("imagecorruptions", "imagecorruptions", "corruptions.py")
    load_definitions(
        path,
        ("_shuffle_pixels_njit_glass_blur",),
        reference,
        replacements=(
            (
                "x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]",
                "x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime].copy(), x[h, w].copy()",
            ),
        ),
    )
    swapping = reference["glass_blur"](Image.fromarray(image), severity)

    assert level_gap(swapping, ours).max() <= LEVEL_TOLERANCE
    assert level_gap(duplicating, ours).max() > 5.0


@pytest.mark.parametrize("severity", SEVERITIES)
def test_elastic_transform_agrees_only_where_the_warp_stays_inside(
    reference, monkeypatch, severity
):
    """Agreement inside, and a disagreement larger than docs/detectors/teco.md states.

    The reference smooths the displacement field with skimage's gaussian in
    mode "reflect", which is scipy's half-sample reflect (d c b a | a b c d),
    and samples with map_coordinates in the same mode. The port smooths with
    torch's "reflect" padding and samples with grid_sample's reflection at
    align_corners=True, both of which reflect about the edge pixel (d c b | a b
    c d). The 2 therefore agree exactly where the field's smoothing window and
    the displaced sample both stay inside the image, and differ elsewhere. At 32
    pixels the displacement reaches about 2 to 5 pixels, so that is a band a few
    pixels wide rather than the outermost row alone. The gap there reaches well
    over 100 levels.
    """
    image = sample_image()
    max_shift = IMAGE_SIZE * 0.005
    fields = [
        np.random.RandomState(seed).uniform(
            -max_shift, max_shift, (IMAGE_SIZE, IMAGE_SIZE)
        )
        for seed in (3, 4)
    ]

    numpy_fields = Draws(fields)
    monkeypatch.setattr(
        np.random, "uniform", lambda low, high, size: numpy_fields.next()
    )
    theirs = reference["elastic_transform"](Image.fromarray(image), severity)
    stacked = torch.from_numpy(np.stack(fields)).float()[None]  # (1, 2, 32, 32)
    monkeypatch.setattr(torch.Tensor, "uniform_", fill_uniform_with(Draws([stacked])))
    ours = teco.elastic_transform(as_port_input(image), severity)

    # The displacement the reference applies, recomputed with its own call.
    alpha = 250 * (0.05, 0.065, 0.085, 0.1, 0.12)[severity - 1]
    sigma = np.array([IMAGE_SIZE, IMAGE_SIZE]) * 0.01
    column_shift, row_shift = (
        reference["gaussian"](field, sigma, mode="reflect", truncate=3) * alpha
        for field in fields
    )
    columns, rows = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))
    inside = (
        (columns + column_shift >= 0)
        & (columns + column_shift <= IMAGE_SIZE - 1)
        & (rows + row_shift >= 0)
        & (rows + row_shift <= IMAGE_SIZE - 1)
    )
    # The smoothing kernel has radius int(3 * 0.32 + 0.5) = 1, so only the
    # outermost ring of the field sees the 2 padding modes differ.
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False

    gap = level_gap(theirs, ours).max(axis=2)  # (32, 32)
    assert gap[inside].max() <= LEVEL_TOLERANCE
    assert (gap[~inside] > LEVEL_TOLERANCE).mean() > 0.9
    assert gap.max() > 100.0


@pytest.mark.parametrize("severity", SEVERITIES)
def test_pixelate_disagrees_with_pil_box_and_nearest(reference, severity):
    """A disagreement docs/detectors/teco.md recorded as agreement.

    The reference shrinks with PIL's BOX filter, which weights source pixels by
    their fractional coverage and rounds to uint8, then enlarges with PIL's
    NEAREST, which samples at pixel centers. The port shrinks with torch's area
    mode, which averages integer-bounded bins. It enlarges with torch's
    nearest, which samples at floor(i * in / out). At the factors 0.5 and 0.25
    the 2 differ only in how the box mean is rounded, 1 level at most. At 0.6,
    0.4 and 0.3 the grids do not line up and whole blocks differ, by tens to
    over 100 levels.
    """
    image = sample_image()

    theirs = np.asarray(reference["pixelate"](Image.fromarray(image), severity))
    ours = teco.pixelate(as_port_input(image), severity)
    gap = level_gap(theirs, ours)

    integer_ratio = severity in (2, 5)
    if integer_ratio:
        assert gap.max() <= 1.0 + 1e-3
    else:
        assert gap.max() > 50.0
        assert (gap > 1.0 + 1e-3).mean() > 0.2


def test_impulse_noise_flips_the_same_share_of_values_to_salt_and_pepper(reference):
    """skimage's random_noise draws with its own generator, so only the rates compare.

    On a mid-gray image every 0 or 255 in the output is a flipped value. Both
    sides flip each channel value independently with probability amount and
    choose salt with probability 0.5, so on 3 * 64 * 64 values the salt and
    pepper shares must agree within 0.02 of each other at every severity.
    """
    grey = np.full((64, 64, 3), 128, dtype=np.uint8)
    np.random.seed(0)
    torch.manual_seed(0)
    for severity in SEVERITIES:
        theirs = np.asarray(reference["impulse_noise"](Image.fromarray(grey), severity))
        ours = teco.impulse_noise(as_port_input(grey), severity)[0].permute(1, 2, 0)
        ours_levels = (ours * 255).round().numpy()

        for level in (0.0, 255.0):
            their_share = np.mean(np.round(theirs) == level)
            our_share = np.mean(ours_levels == level)
            assert abs(their_share - our_share) < 0.02, (severity, level)


def test_corrupt_truncates_to_uint8_where_the_port_rounds(reference):
    """A disagreement docs/detectors/teco.md did not record.

    BackdoorBench calls imagecorruptions.corrupt, whose last line is
    np.uint8(image_corrupted) (__init__.py line 69), a truncation, so every
    corrupted image BackdoorBench classifies is the floor of the reference float.
    The port rounds to the nearest level. On contrast about half the values land
    1 level apart, all of them with the port higher.
    """
    init_file = reference_file("imagecorruptions", "imagecorruptions", "__init__.py")
    namespace = load_definitions(
        init_file,
        ("corrupt",),
        {"np": np, "Image": Image, "corruption_dict": reference},
    )
    image = sample_image()

    truncated = namespace["corrupt"](image, severity=3, corruption_name="contrast")
    unrounded = reference["contrast"](Image.fromarray(image), 3)
    ours = teco.contrast(as_port_input(image), 3)[0].permute(1, 2, 0) * 255.0

    assert np.array_equal(truncated, np.floor(unrounded).astype(np.uint8))
    difference = ours.round().numpy() - truncated.astype(np.float64)
    assert set(np.unique(difference)) <= {0.0, 1.0}
    assert 0.3 < np.mean(difference == 1.0) < 0.7


def backdoorbench_mads(label_dict: dict, max_index: int = 6) -> list[float]:
    """detection_infer/teco.py lines 323 to 341, the per-image loop of 1 split, verbatim.

    img_preds[corruption] is [original, severity 1, ..., severity 5], so the
    index i at which the prediction first differs from entry 0 is the severity,
    and args.max = 6 (line 116) is the never-flipped value.
    """
    mads = []
    images = list(label_dict.keys())
    keys = list(label_dict[images[0]].keys())
    for img in images:
        indexs = []
        img_preds = label_dict[img]
        for corruption in keys[1:]:
            flag = 0
            for i in range(max_index):
                if int(img_preds[corruption][i]) != int(img_preds[corruption][0]):
                    index = i
                    flag = 1
                    indexs.append(index)
                    break
            if flag == 0:
                indexs.append(max_index)
        indexs = np.asarray(indexs)
        mad = np.std(indexs)
        mads.append(mad)
    return mads


class RecordingClassifier(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.predictions: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.inner(images)
        self.predictions.append(logits.argmax(dim=1))
        return logits


def test_deviation_matches_backdoorbench_on_the_same_predictions():
    """Algorithm 1's index rule and Dev, checked on the predictions the port's pass produced.

    The port's hardness pass is recorded call by call, laid out as BackdoorBench
    lays out bd_dict (the original prediction, then severities 1 to 5 per
    corruption), scored by the transcribed loop and compared with deviation to
    1e-6. The corruption operators themselves are checked above.
    """
    torch.manual_seed(0)
    inner = nn.Sequential(
        nn.Flatten(), nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 6)
    ).eval()
    with torch.no_grad():
        inner[1].weight.mul_(3.0)
    model = RecordingClassifier(inner)
    corruptions = ("contrast", "brightness", "defocus_blur", "pixelate", "zoom_blur")
    generator = torch.Generator().manual_seed(1)
    pixels = torch.rand(8, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    loader = DataLoader(TensorDataset(pixels, torch.zeros(8)), batch_size=8)

    thresholds = teco.hardness_thresholds(
        model,
        loader,
        DEVICE,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        corruptions,
        use_bfloat16=False,
    )  # (8, 5)
    ours = teco.deviation(thresholds)  # (8,)

    original, *per_severity = model.predictions
    assert len(per_severity) == len(corruptions) * teco.MAX_SEVERITY
    bd_dict = {}
    for image_index in range(8):
        entry = {"original": [int(original[image_index])]}
        for column, name in enumerate(corruptions):
            severities = per_severity[
                column * teco.MAX_SEVERITY : (column + 1) * teco.MAX_SEVERITY
            ]
            entry[name] = [int(original[image_index])] + [
                int(predictions[image_index]) for predictions in severities
            ]
        bd_dict[str(image_index)] = entry
    theirs = backdoorbench_mads(bd_dict)

    assert thresholds.min() < teco.NEVER_FLIPPED, "no flip would make this vacuous"
    assert len(set(np.round(theirs, 6))) > 1
    assert np.allclose(ours.numpy(), theirs, atol=1e-6)


def test_backdoorbench_composes_severities_where_the_port_starts_from_the_pristine_image(
    reference,
):
    """Deviation 2, the loop of teco.py lines 240 to 242 transcribed and run on 1 image.

    x = images_poison binds the same list, so x[i] = self.dg(x[i], args) at
    severity 2 corrupts the output of severity 1. The port corrupts the pristine
    image at every severity, as Algorithm 1 writes it.
    """
    init_file = reference_file("imagecorruptions", "imagecorruptions", "__init__.py")
    corrupt = load_definitions(
        init_file,
        ("corrupt",),
        {"np": np, "Image": Image, "corruption_dict": reference},
    )["corrupt"]

    def dg(image, args):
        # teco.py lines 168 to 172.
        image = np.array(image)
        image = corrupt(image, corruption_name=args.cor_type, severity=args.severity)
        image = Image.fromarray(image)
        return image

    pristine = Image.fromarray(sample_image())
    images_poison = [pristine]
    args = types.SimpleNamespace(cor_type="contrast", severity=None)
    for severity in (1, 2):
        args.severity = severity
        x = images_poison
        for i in range(len(x)):
            x[i] = dg(x[i], args)
    composed = np.asarray(images_poison[0])

    applied_twice = corrupt(
        corrupt(sample_image(), severity=1, corruption_name="contrast"),
        severity=2,
        corruption_name="contrast",
    )
    from_pristine = corrupt(sample_image(), severity=2, corruption_name="contrast")
    assert np.array_equal(composed, applied_twice)
    assert not np.array_equal(composed, from_pristine)
    assert math.isclose(
        float(np.std(composed.astype(float))),
        float(np.std(sample_image().astype(float))) * 0.4 * 0.3,
        rel_tol=0.05,
    )

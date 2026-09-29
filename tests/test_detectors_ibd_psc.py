"""IBD-PSC's calibration search, and the scoring path against BackdoorBox's class.

The first 3 tests pin the search over omega without a model: it stops at the
first factor whose trace crosses xi, returns the paper's factor untouched when
that one already crosses, and falls back to the last factor with k = L when none
does. The rest run Algorithm 1 and Eq. (4) on a tiny LayerNorm stack on the CPU
and compare them with the released IBD_PSC class executed from third_party.
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from detectors import FORWARD_PASSES_PER_INPUT, ibd_psc
from tests.reference.third_party import (
    keep_cuda_calls_on_the_cpu,
    load_definitions,
    reference_file,
)


def stub_layers(count: int) -> list[nn.LayerNorm]:
    layers = [nn.LayerNorm(4) for _ in range(count)]
    return layers


def fake_selection(crossing_factor: float | None, total_layers: int):
    """A select_start_layer_count stand-in crossing xi only from crossing_factor up."""

    def select(model, loader, device, layers, factor, threshold, use_bfloat16):
        if crossing_factor is not None and factor >= crossing_factor:
            return 2, [0.1, threshold + 0.1]
        return total_layers, [0.1] * total_layers

    return select


def test_the_search_stops_at_the_first_crossing_factor(monkeypatch):
    monkeypatch.setattr(ibd_psc, "select_start_layer_count", fake_selection(3.0, 5))
    factor, start, trace = ibd_psc.calibrate_scaling_factor(
        None, None, None, stub_layers(5), (1.5, 2.0, 3.0, 5.0), 0.6, True
    )
    assert factor == 3.0
    assert start == 2
    assert trace[-1] > 0.6


def test_the_papers_factor_is_kept_when_it_already_crosses(monkeypatch):
    monkeypatch.setattr(ibd_psc, "select_start_layer_count", fake_selection(1.5, 5))
    factor, start, _ = ibd_psc.calibrate_scaling_factor(
        None, None, None, stub_layers(5), (1.5, 2.0, 3.0), 0.6, True
    )
    assert factor == 1.5
    assert start == 2


def test_no_crossing_returns_the_last_factor_at_every_layer(monkeypatch):
    monkeypatch.setattr(ibd_psc, "select_start_layer_count", fake_selection(None, 5))
    factor, start, trace = ibd_psc.calibrate_scaling_factor(
        None, None, None, stub_layers(5), (1.5, 2.0), 0.6, True
    )
    assert factor == 2.0
    assert start == 5
    assert max(trace) < 0.6


# The cross-checks below execute BackdoorBox's released IBD_PSC class,
# core/defenses/IBD_PSC.py at the pinned commit, with 1 textual change: its
# BatchNorm2d filter (lines 63 and 83) reads LayerNorm, which is the port's
# deviation 1 applied to the reference so that everything else can be compared
# on the same tiny LayerNorm model.
STACK_DEPTH = 10
STACK_WIDTH = 16
STACK_CLASSES = 4
# At omega 3 the stack's clean error first exceeds 0.3 at 2 amplified layers, so
# Algorithm 1 stops inside the range both implementations test and the 5
# ensemble members 3 to 7 stay below the 10 layers there are.
CROSSING_FACTOR = 3.0
CROSSING_THRESHOLD = 0.3
CROSSING_COUNT = 2
ENSEMBLE_SIZE = 5
DEVICE = torch.device("cpu")


class ResidualNormStack(nn.Module):
    """A pre-norm residual stack, the ViT block pattern without attention.

    Every branch reads its LayerNorm's output and adds to the stream, so
    amplifying a LayerNorm scales a branch rather than the stream, as on ViT.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Linear(3 * 4 * 4, STACK_WIDTH)
        self.norms = nn.ModuleList(
            nn.LayerNorm(STACK_WIDTH) for _ in range(STACK_DEPTH)
        )
        self.branches = nn.ModuleList(
            nn.Linear(STACK_WIDTH, STACK_WIDTH) for _ in range(STACK_DEPTH)
        )
        self.head = nn.Linear(STACK_WIDTH, STACK_CLASSES)
        for norm in self.norms:
            nn.init.normal_(norm.weight, 1.0, 0.3)
            nn.init.normal_(norm.bias, 0.0, 0.3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        stream = self.embed(images.flatten(1))  # (batch, STACK_WIDTH)
        for norm, branch in zip(self.norms, self.branches):
            stream = stream + branch(F.gelu(norm(stream)))
        logits = self.head(stream)  # (batch, STACK_CLASSES)
        return logits


class BaseStub:
    """BackdoorBox's defenses.base.Base seeds 3 RNGs and nothing the tests read."""

    def __init__(self, seed=0, deterministic=False):
        pass


def load_backdoorbox_ibd_psc():
    path = reference_file("BackdoorBox", "core", "defenses", "IBD_PSC.py")
    namespace = {
        "torch": torch,
        "copy": copy,
        "np": np,
        "Base": BaseStub,
        "test": None,
        "metrics": None,
    }
    replacements = (("torch.nn.BatchNorm2d", "torch.nn.LayerNorm"),)
    load_definitions(path, ("IBD_PSC",), namespace, replacements)
    return namespace["IBD_PSC"]


def labelled_by_the_model(model: nn.Module, count: int, seed: int) -> TensorDataset:
    """Random inputs labeled with the model's own prediction, so clean error starts at 0.

    It also keeps every input through BackdoorBox's prediction-correctness filter,
    which the port does not have, so the 2 score vectors align row for row.
    """
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(count, 3, 4, 4, generator=generator)
    with torch.no_grad():
        labels = model(images).argmax(dim=1)
    dataset = TensorDataset(images, labels)
    return dataset


def port_psc(model, dataset, start_count, factor, ensemble_size):
    loader = DataLoader(dataset, batch_size=64)
    layers = ibd_psc.amplifiable_norm_layers(model)
    psc = ibd_psc.psc_scores(
        model, loader, DEVICE, layers, start_count, factor, ensemble_size, False
    )
    return psc


def test_algorithm_1_selects_the_same_depth_as_prob_start(monkeypatch):
    keep_cuda_calls_on_the_cpu(monkeypatch)
    ibd_psc_class = load_backdoorbox_ibd_psc()
    model = ResidualNormStack().eval()
    validation = labelled_by_the_model(model, 200, seed=1)

    reference = ibd_psc_class(
        model,
        n=ENSEMBLE_SIZE,
        xi=CROSSING_THRESHOLD,
        scale=CROSSING_FACTOR,
        valset=validation,
    )
    layers = ibd_psc.amplifiable_norm_layers(model)
    start_count, trace = ibd_psc.select_start_layer_count(
        model,
        DataLoader(validation, batch_size=64),
        DEVICE,
        layers,
        CROSSING_FACTOR,
        CROSSING_THRESHOLD,
        use_bfloat16=False,
    )

    assert len(layers) == len(reference.sorted_indices) == STACK_DEPTH
    assert reference.start_index == start_count == CROSSING_COUNT
    assert trace[-1] > CROSSING_THRESHOLD >= max(trace[:-1])


def test_psc_matches_backdoorbox_one_layer_deeper_to_1e_6(monkeypatch):
    """Deviation 3 of the doc, shown numerically.

    From start index k the released _test amplifies k + 1, ..., k + n layers
    (line 139 slices sorted_indices[:layer_index+1]). Eq. (4) and the port
    amplify k, ..., k + n - 1. So the reference at k equals the port started at
    k + 1 and not the port started at k.
    """
    keep_cuda_calls_on_the_cpu(monkeypatch)
    ibd_psc_class = load_backdoorbox_ibd_psc()
    model = ResidualNormStack().eval()
    validation = labelled_by_the_model(model, 200, seed=1)
    scored = labelled_by_the_model(model, 50, seed=2)

    reference = ibd_psc_class(
        model,
        n=ENSEMBLE_SIZE,
        xi=CROSSING_THRESHOLD,
        scale=CROSSING_FACTOR,
        valset=validation,
    )
    with torch.no_grad():
        theirs = reference._test(scored)  # (50,)

    shifted = port_psc(
        model, scored, reference.start_index + 1, CROSSING_FACTOR, ENSEMBLE_SIZE
    )
    as_written = port_psc(
        model, scored, reference.start_index, CROSSING_FACTOR, ENSEMBLE_SIZE
    )

    assert theirs.shape == (50,)
    assert torch.allclose(shifted, theirs, atol=1e-6)
    assert not torch.allclose(as_written, theirs, atol=1e-3)


def test_past_the_last_layer_backdoorbox_repeats_the_full_model_and_the_port_drops(
    monkeypatch,
):
    """A disagreement the doc's deviation 5 does not state.

    When the window runs past L, the reference's slice sorted_indices[:i] stops
    at L, so every out-of-range member amplifies all L layers again and the full
    model is counted several times in the mean. The port keeps each count at
    most once. From start index L - 2 with n = 5 the reference averages the
    members at L - 1, L, L, L, L and the port the members at L - 1 and L.
    """
    keep_cuda_calls_on_the_cpu(monkeypatch)
    ibd_psc_class = load_backdoorbox_ibd_psc()
    model = ResidualNormStack().eval()
    validation = labelled_by_the_model(model, 200, seed=1)
    scored = labelled_by_the_model(model, 50, seed=2)

    reference = ibd_psc_class(
        model,
        n=ENSEMBLE_SIZE,
        xi=CROSSING_THRESHOLD,
        scale=CROSSING_FACTOR,
        valset=validation,
    )
    reference.start_index = STACK_DEPTH - 2
    with torch.no_grad():
        theirs = reference._test(scored)

    second_last = port_psc(model, scored, STACK_DEPTH - 1, CROSSING_FACTOR, 1)
    full = port_psc(model, scored, STACK_DEPTH, CROSSING_FACTOR, 1)
    ours = port_psc(model, scored, STACK_DEPTH - 1, CROSSING_FACTOR, ENSEMBLE_SIZE)

    assert torch.allclose(theirs, (second_last + 4 * full) / 5, atol=1e-6)
    assert torch.allclose(ours, (second_last + full) / 2, atol=1e-6)


class ForwardCounter:
    """Counts forward calls of a model through a hook, 1 per batch."""

    def __init__(self, model: nn.Module):
        self.calls = 0
        self.handle = model.register_forward_hook(self.count)

    def count(self, module, inputs, output):
        self.calls += 1


def test_without_a_crossing_backdoorbox_has_no_depth_and_the_port_scores_at_2_passes(
    monkeypatch,
):
    """What a start depth of L out of L means, pinned on a model that never crosses.

    At omega 1.5 and xi 0.6 the stack's clean error stays under xi at every
    depth, as it does on most panel ViTs (docs/detectors/ibd_psc.md, the settings
    table of the results block). The released prob_start tests
    depths 1 to L - 1 only and falls off its loop returning None (lines 93 to
    116), after which self.start_index + self.n raises in _test, so the reference
    cannot score such a model at all. The port's Algorithm 1 holds k = L at loop
    exit and its window keeps the single member k = L. Each batch then costs 2
    forward passes, 1 unamplified and 1 fully amplified, against the 6 that
    FORWARD_PASSES_PER_INPUT records for ibd_psc.
    """
    keep_cuda_calls_on_the_cpu(monkeypatch)
    ibd_psc_class = load_backdoorbox_ibd_psc()
    model = ResidualNormStack().eval()
    validation = labelled_by_the_model(model, 200, seed=1)
    scored = labelled_by_the_model(model, 50, seed=2)
    factor = ibd_psc.DEFAULT_SCALING_FACTOR
    threshold = ibd_psc.DEFAULT_ERROR_THRESHOLD

    reference = ibd_psc_class(
        model, n=ENSEMBLE_SIZE, xi=threshold, scale=factor, valset=validation
    )
    assert reference.start_index is None
    with pytest.raises(TypeError):
        reference._test(scored)

    layers = ibd_psc.amplifiable_norm_layers(model)
    start_count, trace = ibd_psc.select_start_layer_count(
        model,
        DataLoader(validation, batch_size=64),
        DEVICE,
        layers,
        factor,
        threshold,
        use_bfloat16=False,
    )
    assert start_count == STACK_DEPTH
    assert len(trace) == STACK_DEPTH and max(trace) <= threshold

    counter = ForwardCounter(model)
    ours = port_psc(model, scored, start_count, factor, ENSEMBLE_SIZE)
    counter.handle.remove()
    full_only = port_psc(model, scored, STACK_DEPTH, factor, 1)

    batches = 1
    assert counter.calls == 2 * batches
    assert torch.equal(ours, full_only)
    assert FORWARD_PASSES_PER_INPUT["ibd_psc"] == ENSEMBLE_SIZE + 1

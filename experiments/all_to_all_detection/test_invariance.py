"""The detection statistics read no label and no class identity.

1  on random synthetic caches, permuting every class index leaves every score of
   every statistic unchanged (detect.relabeling_check raises otherwise)
2  a statistic that does read a fixed class identity, the probability of class 0,
   is caught by the same check, so the check has power
3  the source of every function the statistics call never names a label field

    source .venv/bin/activate
    python -m pytest experiments/all_to_all_detection/test_invariance.py -q
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import detect  # noqa: E402

NUM_CLASSES = 7
PASSES = 3
SIZES = {"validation": 300, "clean_paired": 200, "backdoor": 200, "clean": 250}
LABEL_FIELDS = ("loader_labels", "labels", "target", "source", "attack_label")
STATISTIC_FUNCTIONS = (
    detect.all_statistics,
    detect.psu_ratio,
    detect.validation_rank,
    detect.class_two_sided,
    detect.destination_concentration,
    detect.transition_typicality,
    detect.defender_view,
)


def synthetic_views(seed):
    generator = torch.Generator().manual_seed(seed)
    views = {}
    for split, size in SIZES.items():
        logits = torch.randn(size, NUM_CLASSES, generator=generator) * 3  # (n, K)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)  # (n,)
        views[split] = {
            "probs": probs,
            "pred": pred,
            "pass_probs": {
                probe: torch.rand(PASSES, size, generator=generator)
                for probe in ("tm", "rd", "late")
            },  # per probe, (k, n)
            "pass_argmax": {
                probe: torch.randint(
                    0, NUM_CLASSES, (PASSES, size), generator=generator
                )
                for probe in ("tm", "rd", "late")
            },  # per probe, (k, n)
        }
    return views


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_statistic_is_relabeling_invariant(seed):
    detect.relabeling_check(synthetic_views(seed))


def test_check_catches_a_statistic_that_reads_a_class(monkeypatch):
    honest = detect.all_statistics

    def reads_class_zero(view, validation):
        scores = honest(view, validation)
        scores["class_zero"] = view["probs"][:, 0]  # (n,)
        return scores

    monkeypatch.setattr(detect, "all_statistics", reads_class_zero)
    with pytest.raises(AssertionError, match="class_zero"):
        detect.relabeling_check(synthetic_views(0))


def test_statistic_code_names_no_label_field():
    for function in STATISTIC_FUNCTIONS:
        source = inspect.getsource(function)
        for field in LABEL_FIELDS:
            assert f'"{field}"' not in source, f"{function.__name__} reads {field}"

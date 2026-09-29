"""The coverage ledger's panel rule and its 1-cell-per-attack resolution.

The multi-source TaCT reruns carry a `_src{k}` folder tag and share
(dataset, attack, poison_rate) with the single-source runs they replace, which
the ledger marks source_mapped. Both facts have to hold for the rerun to take
the panel slot.
"""

import json
import os

import pytest

from scripts.coverage_ledger import (
    is_panel_folder,
    resolve_one_per_attack,
    success_verdicts,
)

DECLARATION = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "psbd_basis.json",
)


def tact_cell(folder: str, asr_class: str, source_mapped: bool) -> dict:
    cell = {
        "folder_name": folder,
        "dataset": "cifar100",
        "attack": "tact",
        "poison_rate": 0.01,
        "asr_class": asr_class,
        "diverged": False,
        "source_mapped": source_mapped,
    }
    return cell


def test_multi_source_folder_is_a_panel_cell():
    with open(DECLARATION) as handle:
        panel = json.load(handle)["panel"]
    metadata = {
        "attack": "tact",
        "architecture": "vit",
        "label_mode": "all_to_one",
        "dataset": "tiny",
        "target_label": 0,
        "poison_rate": 0.05,
    }
    assert is_panel_folder("vit_tiny_tact_0_05_src50", metadata, panel)


@pytest.mark.parametrize("rerun_class", ["clears", "below_bar"])
def test_rerun_takes_the_slot_from_the_source_mapped_cell(rerun_class):
    old = tact_cell("vit_cifar100_tact_0_01", "source_mapped", True)
    rerun = tact_cell("vit_cifar100_tact_0_01_src5", rerun_class, False)

    resolved = resolve_one_per_attack([old, rerun])

    assert resolved["tact"]["folder_name"] == "vit_cifar100_tact_0_01_src5"


def success_cell(asr_class: str, clean_accuracy_drop: float | None) -> dict:
    cell = {"asr_class": asr_class, "clean_accuracy_drop": clean_accuracy_drop}
    return cell


@pytest.mark.parametrize(
    "asr_class, drop, expected_2pt, expected_5pt",
    [
        ("clears", 0.01, True, True),
        ("clears", -0.02, True, True),
        ("clears", -0.03, False, True),
        ("clears", -0.05, False, True),
        ("clears", -0.06, False, False),
        ("clears", None, False, False),
        ("below_bar", 0.0, False, False),
        ("unmeasured", 0.0, False, False),
        ("diverged", 0.0, False, False),
        ("source_mapped", 0.0, False, False),
    ],
)
def test_success_needs_the_asr_class_and_the_clean_accuracy_bar(
    asr_class, drop, expected_2pt, expected_5pt
):
    """A cell succeeds only when it clears the ASR bar and loses at most the bar.

    asr_class already carries the diverged and source-mapped overrides, so a
    cell either of those rules out never succeeds whatever its clean accuracy.
    """
    with open(DECLARATION) as handle:
        declaration = json.load(handle)
    cell = success_cell(asr_class, drop)

    verdicts = success_verdicts(cell, declaration)

    assert verdicts == {"successful_2pt": expected_2pt, "successful_5pt": expected_5pt}

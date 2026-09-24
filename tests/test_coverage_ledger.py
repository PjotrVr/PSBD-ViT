"""The coverage ledger's panel rule and its 1-cell-per-attack resolution.

The multi-source TaCT reruns carry a `_src{k}` folder tag and share
(dataset, attack, poison_rate) with the single-source runs they replace, which
the ledger marks source_mapped. Both facts have to hold for the rerun to take
the panel slot.
"""

import json
import os

import pytest

from scripts.coverage_ledger import is_panel_folder, resolve_one_per_attack

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

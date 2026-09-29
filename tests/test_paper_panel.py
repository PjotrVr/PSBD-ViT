"""The paper's panel is the ledger's successful_2pt cells, and nothing else.

Every result is computed over the models that are successful backdoors at the
headline bar: attack success at the ASR bar (not diverged, not source-mapped)
and clean accuracy within 2 points of the benign reference. The ASR-only
population stays readable for the counts that report implantation, and the
5-point panel for the numbers every result is also given at. Swin cells get the
same verdict from the ledger's own classify_cell.
"""

import json
import os

import pytest

from scripts import coverage_ledger
from scripts.coverage_ledger import SUCCESS_BARS, classify_cell
from scripts.paper import tab_swin
from scripts.paper._common import (
    HEADLINE_BAR_POINTS,
    HEADLINE_SUCCESS,
    PANEL_DATASETS,
    SECOND_BAR_POINTS,
    SECOND_SUCCESS,
    clearing_cells,
    excluded_folders,
    implanted_cells,
    load_coverage,
    second_bar_macros,
    swin_coverage,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECLARATION = os.path.join(REPO, "configs", "psbd_basis.json")


def read_declaration() -> dict:
    with open(DECLARATION) as handle:
        declaration = json.load(handle)
    return declaration


def ledger_cell(folder: str, asr_class: str, success_2pt: bool, success_5pt: bool):
    cell = {
        "folder_name": folder,
        "dataset": "cifar10",
        "asr_class": asr_class,
        "successful_2pt": success_2pt,
        "successful_5pt": success_5pt,
    }
    return cell


SYNTHETIC_CELLS = [
    ledger_cell("vit_cifar10_badnet_a2o_0_1", "clears", True, True),
    ledger_cell("vit_cifar10_wanet_0_05", "clears", False, True),
    ledger_cell("vit_cifar10_sig_0_1", "clears", False, False),
    ledger_cell("vit_cifar10_lc_0_01_adv", "below_bar", False, False),
    ledger_cell("vit_cifar10_tact_0_1", "source_mapped", False, False),
]


def folders(cells: list[dict]) -> set[str]:
    names = {cell["folder_name"] for cell in cells}
    return names


def test_the_panel_is_the_headline_success_verdict():
    coverage = {"cells": SYNTHETIC_CELLS}

    assert folders(clearing_cells(coverage)) == {"vit_cifar10_badnet_a2o_0_1"}
    assert folders(clearing_cells(coverage, SECOND_SUCCESS)) == {
        "vit_cifar10_badnet_a2o_0_1",
        "vit_cifar10_wanet_0_05",
    }
    assert folders(implanted_cells(coverage)) == {
        "vit_cifar10_badnet_a2o_0_1",
        "vit_cifar10_wanet_0_05",
        "vit_cifar10_sig_0_1",
    }


def test_a_ledger_without_verdicts_is_refused_rather_than_read_as_empty():
    coverage = {
        "cells": [{"folder_name": "vit_cifar10_blend_0_1", "asr_class": "clears"}]
    }

    with pytest.raises(SystemExit):
        clearing_cells(coverage)


def test_excluded_folders_adds_the_clean_accuracy_failures(tmp_path):
    """Mechanism generators drop these by name, so a failing model cannot sneak back."""
    coverage_dir = tmp_path / "coverage"
    coverage_dir.mkdir()
    (coverage_dir / "coverage.json").write_text(json.dumps({"cells": SYNTHETIC_CELLS}))

    excluded = excluded_folders(str(tmp_path))

    assert excluded == {
        "vit_cifar10_wanet_0_05",
        "vit_cifar10_sig_0_1",
        "vit_cifar10_tact_0_1",
    }
    assert excluded_folders(str(tmp_path), SECOND_SUCCESS) == {
        "vit_cifar10_sig_0_1",
        "vit_cifar10_tact_0_1",
    }


def test_the_bar_widths_and_verdict_names_match_the_declaration():
    declaration = read_declaration()

    assert SUCCESS_BARS[HEADLINE_SUCCESS] == "clean_accuracy_drop_bar_headline"
    assert SUCCESS_BARS[SECOND_SUCCESS] == "clean_accuracy_drop_bar"
    assert HEADLINE_BAR_POINTS == round(
        -declaration["clean_accuracy_drop_bar_headline"] * 100
    )
    assert SECOND_BAR_POINTS == round(-declaration["clean_accuracy_drop_bar"] * 100)


def test_second_bar_macros_are_renamed_with_the_bar():
    renamed = second_bar_macros({"headline_auroc_adaptive": ("0.9", "mean AUROC")})

    (name,) = renamed
    assert name == f"headline_auroc_adaptive_{SECOND_BAR_POINTS}_point"
    assert renamed[name][0] == "0.9"


def judged(cell: dict, declaration: dict, reference: float = 0.96) -> dict:
    classify_cell(cell, reference, declaration, "checkpoints", "results")
    return cell


def test_an_audited_asr_replaces_the_sidecar_asr(monkeypatch):
    """A sidecar ASR measured on the wrong trigger cannot admit a model."""
    monkeypatch.setattr(coverage_ledger, "source_class_accuracy", lambda *a: None)
    declaration = {
        **read_declaration(),
        "asr_audit_overrides": {
            "swin_cifar10_sig_0_05": {"asr": 0.7, "source": "an audit"}
        },
    }
    cell = {
        "folder_name": "swin_cifar10_sig_0_05",
        "dataset": "cifar10",
        "attack": "sig",
        "asr": 0.95,
        "clean_accuracy": 0.96,
    }

    judged(cell, declaration)

    assert cell["asr"] == 0.7
    assert cell["asr_source"] == "an audit"
    assert cell["asr_class"] == "below_bar"
    assert (cell["successful_2pt"], cell["successful_5pt"]) == (False, False)


def test_the_declared_audit_keeps_the_wrong_trigger_swin_sig_below_the_bar():
    declaration = read_declaration()

    audit = declaration["asr_audit_overrides"]["swin_cifar10_sig_0_05"]

    assert audit["asr"] < declaration["asr_bar"]
    assert os.path.exists(os.path.join(REPO, audit["source"]))


def swin_ledger_cell(folder: str, asr_class: str, success_2pt: bool, success_5pt: bool):
    cell = {
        **ledger_cell(folder, asr_class, success_2pt, success_5pt),
        "attack": "blend",
        "poison_rate": 0.05,
    }
    return cell


@pytest.fixture
def swin_results(tmp_path, monkeypatch):
    cells = [
        swin_ledger_cell("swin_cifar10_blend_0_05", "clears", True, True),
        swin_ledger_cell("swin_cifar10_blend_0_1", "clears", False, True),
        swin_ledger_cell("swin_cifar10_blend_0_01", "below_bar", False, False),
        swin_ledger_cell("swin_cifar10_bpp_0_05", "clears", True, True),
    ]
    coverage = {"asr_bar": read_declaration()["asr_bar"], "cells": cells}
    monkeypatch.setattr(tab_swin, "swin_coverage", lambda *a: coverage)
    # The BPP model has no sweep yet, so it has no report and is left out.
    for folder in ("swin_cifar10_blend_0_05", "swin_cifar10_blend_0_1"):
        model_dir = tmp_path / folder
        model_dir.mkdir()
        (model_dir / "psbd_metrics.json").write_text(json.dumps({"placements": {}}))
    return str(tmp_path)


def test_swin_cells_select_the_panel_verdict_and_need_a_sweep(swin_results):
    headline = tab_swin.swin_cells(swin_results, "checkpoints")
    second = tab_swin.swin_cells(swin_results, "checkpoints", success=SECOND_SUCCESS)
    implanted = tab_swin.swin_cells(swin_results, "checkpoints", success=None)

    assert folders(headline) == {"swin_cifar10_blend_0_05"}
    assert folders(second) == {"swin_cifar10_blend_0_05", "swin_cifar10_blend_0_1"}
    assert folders(implanted) == {"swin_cifar10_blend_0_05", "swin_cifar10_blend_0_1"}
    assert all(cell["folder"] == cell["folder_name"] for cell in implanted)


def test_swin_cells_refuse_a_bar_other_than_the_declared_one(swin_results):
    with pytest.raises(ValueError):
        tab_swin.swin_cells(swin_results, "checkpoints", 0.5)


CHECKPOINTS = os.path.join(REPO, "checkpoints")


@pytest.mark.skipif(not os.path.isdir(CHECKPOINTS), reason="needs the checkpoints")
def test_the_swin_panel_follows_the_vit_panel_rule(monkeypatch):
    """No 0.5% model, no Label-Consistent without its bases, no excluded token."""
    monkeypatch.chdir(REPO)
    declaration = read_declaration()
    panel = declaration["panel"]

    cells = swin_coverage("results", "checkpoints")["cells"]

    assert cells
    for cell in cells:
        assert cell["dataset"] in panel["datasets"]
        assert cell["poison_rate"] in panel["poison_rates"]
        assert not any(
            token in cell["folder_name"] for token in panel["exclude_folder_tokens"]
        )
        if cell["attack"] == "lc":
            assert panel["canonical_variants"]["lc"] in cell["folder_name"]
        if cell["dataset"] == "gtsrb" and cell["label_mode"] == "clean_label":
            assert (
                cell["target_label"]
                == panel["canonical_targets"]["gtsrb"]["clean_label"]
            )
    audited = [c for c in cells if c["folder_name"] == "swin_cifar10_sig_0_05"]
    assert all(cell["asr_class"] == "below_bar" for cell in audited)


COVERAGE_JSON = os.path.join(REPO, "results", "coverage", "coverage.json")
HEADLINE_JSON = os.path.join(REPO, "paper", "headline.json")


@pytest.mark.skipif(
    not (os.path.exists(COVERAGE_JSON) and os.path.exists(HEADLINE_JSON)),
    reason="needs the real ledger and the built paper",
)
def test_the_built_paper_counts_exactly_the_successful_2pt_cells():
    """The panel the paper was built on is the ledger's successful_2pt set."""
    with open(COVERAGE_JSON) as handle:
        ledger = json.load(handle)
    with open(HEADLINE_JSON) as handle:
        headline = json.load(handle)
    successful = {
        cell["folder_name"]
        for cell in ledger["cells"]
        if cell["dataset"] in PANEL_DATASETS and cell["successful_2pt"]
    }

    panel = folders(clearing_cells(load_coverage(os.path.join(REPO, "results"))))

    assert panel == successful
    assert int(headline["PanelCellsSuccessful"]["value"]) == len(successful)
    cached = int(headline["PanelCellsCached"]["value"])
    awaiting = int(headline["PanelCellsAwaitingSweep"]["value"])
    assert cached + awaiting == len(successful)
    assert int(headline["HeadlinePairedCells"]["value"]) <= cached
    assert int(headline["DetectorsComparedModels"]["value"]) <= cached

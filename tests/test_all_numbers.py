"""The all-numbers table reads PSBD at the rule's rate, from the fractional PSU block.

psbd_metrics.json holds 2 detection blocks per rate, absolute PSU under
"detection" and fractional PSU under "detection_psu_ratio", and the paper reads
the second. A row that took the first, or a rate other than the one the rule
chose, would print a plausible number that no paper table contains.

The Swin models the paper reads outside the panel rule are judged by the
ledger's own divergence, source-mapping and success rules, so a verdict means
the same thing on every row.
"""

from scripts import all_numbers
from scripts.all_numbers import MATCHED_SUFFIX, psbd_reading


def ladder(auroc: float, tpr: float, fpr: float) -> dict:
    block = {
        key: {"auroc": auroc, "tpr": tpr, "fpr": fpr}
        for key in ("q0.10", "q0.20", "q0.25")
    }
    return block


def rate_entry(rate: float, sigma: float, ratio_auroc: float) -> dict:
    entry = {
        "rate": rate,
        "shift_ratio": {"validation": sigma},
        "detection_psu_ratio": ladder(ratio_auroc, 0.9, 0.1),
        "detection": ladder(0.5, 0.2, 0.2),
        "n_samples": {"backdoor": 100},
    }
    return entry


def report_with(adaptive_rate: float | None) -> dict:
    block = {
        "rates": [rate_entry(0.3, 0.55, 0.7), rate_entry(0.5, 0.82, 0.95)],
        "adaptive_rate": adaptive_rate,
        "matched_shift": {"sigma0.6": {"rate": 0.3}},
    }
    report = {"placements": {"before_attention_norm_token_mask": block}}
    return report


def test_the_adaptive_row_reads_the_fractional_block_at_the_chosen_rate():
    reading = psbd_reading(
        report_with(0.5), "before_attention_norm_token_mask", "adaptive"
    )

    assert reading["defense"] == "PSBD-TM"
    assert reading["rate"] == 0.5
    assert reading["validation_shift_ratio"] == 0.82
    assert reading["auroc"] == 0.95
    assert reading["auroc_absolute_psu"] == 0.5
    assert (reading["tpr_q0.10"], reading["fpr_q0.20"]) == (0.9, 0.1)
    assert reading["n_backdoor"] == 100


def test_the_matched_row_reads_the_rate_nearest_the_matched_target():
    reading = psbd_reading(
        report_with(0.5), "before_attention_norm_token_mask", "matched"
    )

    assert reading["defense"] == f"PSBD-TM {MATCHED_SUFFIX}"
    assert (reading["rate"], reading["auroc"]) == (0.3, 0.7)


def test_a_ladder_that_never_reaches_the_target_is_a_row_without_metrics():
    reading = psbd_reading(
        report_with(None), "before_attention_norm_token_mask", "adaptive"
    )

    assert reading["status"] == "shift_target_unreached"
    assert reading["auroc"] is None


def selected_swin(folder: str) -> dict:
    selected = {
        "folder": folder,
        "dataset": "cifar10",
        "attack": "tact",
        "poison_rate": 0.005,
        "asr": 0.99,
    }
    return selected


def extras_with(monkeypatch, source_accuracy: float | None) -> dict:
    folder = "swin_cifar10_tact_0_005"
    monkeypatch.setattr(
        all_numbers, "swin_cells", lambda *arguments: [selected_swin(folder)]
    )
    monkeypatch.setattr(
        all_numbers,
        "read_metadata",
        lambda *arguments: {"clean_accuracy": 0.95, "label_mode": "all_to_one"},
    )
    monkeypatch.setattr(
        all_numbers, "source_class_accuracy", lambda *arguments: source_accuracy
    )
    declaration = {
        "asr_bar": 0.85,
        "clean_accuracy_drop_bar_headline": -0.02,
        "clean_accuracy_drop_bar": -0.05,
    }
    ledger = {"cells": [], "benign_reference_accuracy": {"cifar10": 0.96}}
    (extra,) = all_numbers.swin_paper_extras(
        "results", "checkpoints", declaration, ledger
    )
    return extra


def test_a_swin_extra_that_maps_its_source_class_is_not_a_success(monkeypatch):
    """The paper's Swin selection reads models outside the panel rule.

    Each is judged by the ledger's own rules, so a TaCT extra whose clean source
    class is sent to the target reads source_mapped and never successful.
    """
    extra = extras_with(monkeypatch, source_accuracy=0.1)

    assert extra["asr_class"] == "source_mapped"
    assert extra["source_mapped"] is True
    assert (extra["successful_2pt"], extra["successful_5pt"]) == (False, False)
    assert extra["in_panel"] is False


def test_a_swin_extra_with_a_real_backdoor_succeeds_within_the_bars(monkeypatch):
    extra = extras_with(monkeypatch, source_accuracy=0.9)

    assert extra["asr_class"] == "clears"
    assert (extra["successful_2pt"], extra["successful_5pt"]) == (True, True)

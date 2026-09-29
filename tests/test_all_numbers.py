"""The all-numbers table reads PSBD at the rule's rate, from the fractional PSU block.

psbd_metrics.json holds 2 detection blocks per rate, absolute PSU under
"detection" and fractional PSU under "detection_psu_ratio", and the paper reads
the second. A row that took the first, or a rate other than the one the rule
chose, would print a plausible number that no paper table contains.

The Swin models are the ViT panel rule applied to Swin by the ledger's own
build_ledger, so a verdict means the same thing on every row.
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


def test_swin_models_are_the_panel_cells_of_the_paper_datasets():
    """The Swin rows are the ViT panel rule applied to Swin, nothing read outside it."""
    ledger = {
        "cells": [
            {"folder_name": "swin_cifar10_blend_0_1", "dataset": "cifar10"},
            {"folder_name": "swin_svhn_sig_0_1", "dataset": "svhn"},
        ]
    }

    models = all_numbers.swin_models(ledger)

    assert [model["folder_name"] for model in models] == ["swin_cifar10_blend_0_1"]
    assert all(model["in_panel"] for model in models)
    assert all(model["architecture"] == "swin" for model in models)

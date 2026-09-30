"""The statistics experiments/faded_trigger_defense/ reads its verdicts off.

The pooled monitor's p-value, the ladder scores, the sharpening step and the
choice of its amount each decide a pre-registered verdict, and a silent error in
any would move it. The leakage case holds the readout to its rule that no
threshold, reference or rate depends on a triggered input.
"""

import copy

import numpy as np
import pytest
import torch
from scipy.stats import mannwhitneyu

from defenses.scores import critical_rate
from experiments.faded_trigger_defense.analyze import (
    analyze_entry,
    batch_draws,
    majority_critical_rate,
    mann_whitney_less,
    matched_filter_readout,
    multi_rate_scores,
    rule_flags,
)
from experiments.faded_trigger_defense.score import (
    choose_lambda,
    faded_images,
    sharpened,
)
from experiments.evidence_surplus.common import normalized

pytestmark = pytest.mark.fast

RATES = [0.1, 0.5, 0.9]
DOSES = (1.0, 0.8, 0.6, 0.5, 0.4)


def test_mann_whitney_matches_scipy_asymptotic():
    generator = np.random.default_rng(0)
    reference = generator.random(300)
    batches = generator.random((40, 12)) - 0.1

    ours = mann_whitney_less(torch.tensor(batches), torch.tensor(reference)).numpy()
    theirs = mannwhitneyu(
        batches, reference[None, :], alternative="less", axis=1, method="asymptotic"
    ).pvalue

    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-9)


def test_mann_whitney_single_query_is_exact():
    generator = np.random.default_rng(1)
    reference = generator.random(20)
    queries = generator.random((30, 1))

    ours = mann_whitney_less(torch.tensor(queries), torch.tensor(reference)).numpy()
    theirs = np.array(
        [
            mannwhitneyu(q, reference, alternative="less", method="exact").pvalue
            for q in queries
        ]
    )

    np.testing.assert_allclose(ours, theirs, rtol=1e-9)


def test_majority_critical_rate_matches_the_library():
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 3, (50,), generator=generator)
    argmax_by_rate = {
        rate: torch.where(
            torch.rand(3, 50, generator=generator) < rate, (labels + 1) % 3, labels
        )
        for rate in RATES
    }
    kept = torch.stack(
        [(argmax_by_rate[rate] == labels).float().mean(dim=0) for rate in RATES]
    )  # (R, n)

    ours = majority_critical_rate(kept, RATES)
    library = critical_rate(labels, RATES, argmax_by_rate, flip_fraction=0.5)

    torch.testing.assert_close(ours, library)


def test_ladder_scores_rank_a_survivor_below_a_breaker():
    split = {
        "psu": torch.tensor([[0.0, 0.8], [0.1, 0.9], [0.2, 1.0]]),
        "kept": torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
    }

    scores = multi_rate_scores(split, RATES)

    for values in scores.values():
        assert values[0] < values[1]


def test_choose_lambda_takes_the_largest_within_tolerance():
    accuracy = {0.0: 0.90, 0.25: 0.899, 0.5: 0.892, 1.0: 0.88, 2.0: 0.891}

    assert choose_lambda(accuracy, 0.01) == 2.0
    assert choose_lambda({0.0: 0.9, 0.5: 0.5}, 0.01) is None


def test_sharpening_keeps_flat_images_and_the_pixel_range():
    flat = torch.full((2, 3, 16, 16), 0.4)
    noisy = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(sharpened(flat, 4.0), flat)
    assert sharpened(noisy, 0.0) is noisy
    assert sharpened(noisy, 4.0).min() >= 0.0 and sharpened(noisy, 4.0).max() <= 1.0


def test_fade_runs_from_the_clean_twin_to_the_triggered_image():
    generator = torch.Generator().manual_seed(0)
    clean = torch.rand(4, 3, 32, 32, generator=generator)
    triggered = torch.rand(4, 3, 32, 32, generator=generator)
    context = {
        "dataset": "cifar10",
        "attack": "blend",
        "pairs": {
            "clean": normalized(clean, "cifar10"),
            "triggered": normalized(triggered, "cifar10"),
        },
    }

    torch.testing.assert_close(faded_images(context, 1.0), triggered)
    torch.testing.assert_close(faded_images(context, 0.0), clean)
    torch.testing.assert_close(faded_images(context, 0.5), (clean + triggered) / 2)


def test_batch_draws_sample_without_replacement():
    pool = torch.arange(10.0)

    batches = batch_draws(((pool, 10),), 5)

    assert batches.shape == (5, 10)
    for row in batches:
        assert sorted(row.tolist()) == pool.tolist()
    assert batch_draws(((pool, 11),), 5) is None


def synthetic_split(count, classes, seed, psu_shift=0.0):
    generator = torch.Generator().manual_seed(seed)
    split = {
        "prediction": torch.randint(0, classes, (count,), generator=generator),
        "confidence": torch.rand(count, generator=generator),
        "psu": torch.rand(len(RATES), count, generator=generator) + psu_shift,
        "kept": torch.randint(0, 4, (len(RATES), count), generator=generator) / 3.0,
        "label": torch.randint(0, classes, (count,), generator=generator),
    }
    return split


def synthetic_record():
    triggered = {}
    for index, dose in enumerate(DOSES):
        split = synthetic_split(120, 3, 10 + index, psu_shift=-0.3 * dose)
        split["prediction"][: int(120 * dose * 0.9)] = 0
        triggered[dose] = split
    validation = synthetic_split(400, 3, 1)
    thresholds = {
        str(q): float(np.quantile(validation["psu"][1].numpy(), q))
        for q in (0.01, 0.05, 0.1)
    }
    twins = synthetic_split(120, 3, 3)
    record = {
        "meta": {
            "folder": "synthetic",
            "probe_attack": None,
            "attack": "blend",
            "dataset": "cifar10",
            "backdoored": True,
            "smoke": True,
            "pairs": 120,
            "target": 0,
        },
        "rates": RATES,
        "adaptive_rate": 0.5,
        "thresholds": thresholds,
        "validation": validation,
        "clean_test": synthetic_split(600, 3, 2),
        "twins": twins,
        "triggered": triggered,
        "cached_pairs": {
            "rate": 0.5,
            "twins": {"psu": twins["psu"][1:2]},
            "triggered": {"psu": triggered[1.0]["psu"][1:2]},
        },
    }
    return record


def fitted_quantities(entry):
    fitted = {
        "reference": entry["reference"],
        "attacked_class": entry["attacked_class"],
        "adaptive_rate": entry["adaptive_rate"],
        "single_thresholds": [
            reading["single"][q]["threshold"]
            for reading in entry["doses"].values()
            for q in ("0.01", "0.05", "0.1")
        ],
        "multi_thresholds": [
            reading["multi"][score][q]["threshold"]
            for reading in entry["doses"].values()
            for score in reading["multi"]
            for q in ("0.01", "0.05", "0.1")
        ],
        "false_alarms": [
            reading["pooled"][n]["false_alarm"]
            for reading in entry["doses"].values()
            for n in ("1", "5", "10", "20", "50")
        ],
        "rule_false_positives": [
            reading["rules"][rule][level]["fpr"]
            for reading in entry["doses"].values()
            for rule in reading["rules"]
            for level in ("0.01", "0.05", "0.1")
        ],
    }
    return fitted


def test_no_threshold_or_reference_moves_with_triggered_scores():
    record = synthetic_record()
    shifted = copy.deepcopy(record)
    for split in shifted["triggered"].values():
        split["psu"] = split["psu"] - 5.0
        split["kept"] = torch.ones_like(split["kept"])

    entry = analyze_entry(record, None, None, 50, None)
    moved = analyze_entry(shifted, None, None, 50, None)

    assert fitted_quantities(entry) == fitted_quantities(moved)
    detection = entry["doses"]["1.0"]["pooled"]["20"]["detection"]["0.05"]
    assert moved["doses"]["1.0"]["pooled"]["20"]["detection"]["0.05"] >= detection


def test_per_class_rule_keeps_each_class_at_its_level():
    generator = torch.Generator().manual_seed(0)
    classes = torch.arange(4).repeat_interleave(3000)  # (12000,)
    # Each class has its own score distribution, so a pooled threshold would
    # spend the budget unevenly across classes.
    shift = classes.float() * 0.5
    scores = torch.randn(len(classes), generator=generator) + shift
    validation = torch.randn(len(classes), generator=generator) + shift

    for rule, level in (("class_one_sided", 0.05), ("class_two_sided", 0.1)):
        flags = rule_flags(scores, classes, validation, classes, level, rule)
        for label in range(4):
            share = flags[classes == label].float().mean()
            assert share <= level + 0.015


def test_small_class_falls_back_to_every_validation_score():
    generator = torch.Generator().manual_seed(1)
    validation = torch.rand(500, generator=generator)
    validation_classes = torch.zeros(500, dtype=torch.long)
    validation_classes[:10] = 1
    scores = torch.rand(200, generator=generator)
    classes = torch.ones(200, dtype=torch.long)

    per_class = rule_flags(
        scores, classes, validation, validation_classes, 0.01, "class_one_sided"
    )
    pooled = rule_flags(
        scores, classes, validation, validation_classes, 0.01, "pooled_one_sided"
    )

    assert torch.equal(per_class, pooled)


def feature_record(record, hidden, trigger_direction, generator):
    def split_features(split, carries_trigger):
        count = len(split["prediction"])
        base = torch.randn(count, hidden, generator=generator)
        feature = base + (3.0 * trigger_direction if carries_trigger else 0.0)
        return {"feature": feature.half(), "prediction": split["prediction"]}

    features = {
        "validation": split_features(record["validation"], False),
        "clean_test": split_features(record["clean_test"], False),
        "twins": split_features(record["twins"], False),
        "triggered": {
            dose: split_features(split, True)
            for dose, split in record["triggered"].items()
        },
    }
    return features


def test_matched_filter_recovers_a_planted_direction():
    generator = torch.Generator().manual_seed(2)
    record = synthetic_record()
    record["triggered"][1.0]["psu"] = record["triggered"][1.0]["psu"] - 2.0
    trigger_direction = torch.zeros(16)
    trigger_direction[3] = 1.0
    features = feature_record(record, 16, trigger_direction, generator)

    readout = matched_filter_readout(record, features, 1, 0)

    assert readout["estimable"] and readout["suspect_is_target"]
    assert readout["doses"]["1.0"]["matched_auroc_vs_twins"] > 0.95


def test_matched_filter_thresholds_ignore_the_held_out_pairs():
    generator = torch.Generator().manual_seed(3)
    record = synthetic_record()
    record["triggered"][1.0]["psu"] = record["triggered"][1.0]["psu"] - 2.0
    trigger_direction = torch.zeros(16)
    trigger_direction[3] = 1.0
    features = feature_record(record, 16, trigger_direction, generator)
    moved = copy.deepcopy(features)
    held_out = slice(len(record["twins"]["prediction"]) // 2, None)
    for split in moved["triggered"].values():
        split["feature"][held_out] = split["feature"][held_out] * 0.0

    thresholds = [
        cell["threshold"]
        for reading in matched_filter_readout(record, features, 1, 0)["doses"].values()
        for cell in list(reading["matched"].values())[:3]
    ]
    moved_thresholds = [
        cell["threshold"]
        for reading in matched_filter_readout(record, moved, 1, 0)["doses"].values()
        for cell in list(reading["matched"].values())[:3]
    ]

    assert thresholds == moved_thresholds

"""CPU readout of the faded trigger defenses, from the per-input records of score.py.

Every threshold, reference distribution, rate and sharpening amount below is read
off clean validation only (the leakage audit in the README lists each one). The
attack set of a model at a dose is its faded triggered inputs that the model sends
to the target, on a benign control the probe's target class 0. Low scores mean
poisoned throughout, and every result carries the model's regime from
PREDICTIONS.md (gate, collapse or benign control).

    single    PSBD-TM's fractional PSU at the adaptive rate, thresholds at the 0.01,
              0.05 and 0.10 quantiles of clean validation, realized FPR on the
              cached clean test split
    pooled    a monitor that holds n queries predicted as the same class and runs a
              1-sided Mann-Whitney test of their scores against the clean
              validation scores of that class, or of every class when fewer than
              MIN_REFERENCE validation images are predicted as it
    multi     3 scores over the whole rate ladder (the area under the PSU curve,
              1 minus the area under the keep curve and the negated critical rate)
    rules     per-class and 2-sided conformal thresholds on the single-rate score
    matched   the extension: a projection on the backdoor direction estimated from
              the queries PSBD-TM flags in an unlabeled stream with some
              full-strength triggered queries, read on held-out pairs
    amplified the single-rate readout after sharpening, against the plain pipeline
              on the same pairs, with realized FPR on the clean twins for both

    PYTHONPATH=. .venv/bin/python experiments/faded_trigger_defense/analyze.py
"""

import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from defenses.decision import threshold_at_quantile  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.faded_trigger_defense.render import results_block  # noqa: E402
from experiments.why_psbd_works.measure import auroc_low_is_positive  # noqa: E402

SLUG = "faded_trigger_defense"
QUANTILES = (0.01, 0.05, 0.10)
LEVELS = (0.01, 0.05, 0.10)
BATCH_SIZES = (1, 5, 10, 20, 50)
MIXED_FRACTIONS = (0.2, 0.5)
MIXED_BATCH_SIZES = (5, 10, 20, 50)
DRAWS = 2000
BATCH_SEED = 0
MIN_REFERENCE = 30
MULTI_SCORES = ("psu_area", "shift_area", "critical_rate")
MAJORITY = 0.5
MIN_CATCHES = 10
MIN_DIRECTION_REFERENCE = 5
RULES = ("pooled_two_sided", "class_one_sided", "class_two_sided")
REGIME = {
    "vit_cifar10_badnet_a2o_0_01": "gate",
    "vit_tiny_badnet_a2o_0_05": "gate",
    "vit_gtsrb_lf_0_01": "gate",
    "vit_cifar100_blend_0_1": "gate",
    "vit_cifar10_blend_0_1": "collapse",
    "vit_cifar10_bpp_0_05": "collapse",
}
DOSE_RECORDS = "results/_experiments/evidence_surplus/trigger_dose"
CONSISTENCY_PAIRS = 256

# The pre-registered bars of PREDICTIONS.md, read by judge().
FIRING_ASR = 0.5
HEADLINE_LEVEL = 0.05
HEADLINE_BATCH = 20
POOL_DETECTION_BAR = 0.9
POOL_FALSE_ALARM_BAR = 0.10
POOL_BLIND_BAR = 0.10
MIXED_BATCH = 50
MIXED_FRACTION = 0.2
MIXED_DETECTION_BAR = 0.5
MULTI_GAIN_BAR = 0.05
MULTI_MODELS_BAR = 4
MULTI_LOSS_BAR = 0.05
TIE_FPR_BAND = (0.005, 0.02)
BENIGN_TPR_BAND = 0.10
LAMBDA_BAR = 0.5
AMPLIFY_GAIN_BAR = 0.10
AMPLIFY_MODELS_BAR = 4
BENIGN_AUROC_BAND = 0.10
CONSISTENCY_AUROC_BAND = 0.05
BLIND_MODEL = "vit_cifar10_blend_0_1"
BLIND_DOSES = (0.6, 0.5)
HIGH_FREQUENCY_ATTACKS = ("blend", "badnet_a2o", "bpp")
LOW_FREQUENCY_ATTACKS = ("lf",)
GATE_POOL_BATCH = 10
BPP_POOL_BATCH = 20
BLIND_POOL_BATCH = 50
STRICT_LEVEL = 0.01
BPP_MODEL = "vit_cifar10_bpp_0_05"
GATE_TARGET_AUROC_BAR = 0.8
BENIGN_MIN_HITS = 20
BENIGN_TARGET_AUROC_BAND = 0.15
ATTRACTOR_DATASETS = ("cifar10", "gtsrb")
CLASS_GAIN_BAR = 0.05
CLASS_FPR_BAR = 0.015
TWO_SIDED_MODEL = "vit_cifar10_blend_0_1"
TWO_SIDED_GAIN_DOSE = "0.5"
TWO_SIDED_FLAT_DOSE = "0.6"
TWO_SIDED_GAIN_BAR = 0.05
FALLBACK_MODELS = ("vit_cifar100_blend_0_1", "vit_tiny_badnet_a2o_0_05")
FALLBACK_CHANGE_BAR = 0.02
MATCHED_AUROC_BAR = 0.9
MATCHED_REFUTATION_AUROC = 0.75
MATCHED_REFUTATION_DOSE = "0.6"
MATCHED_MODELS_BAR = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--readme", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir or experiment_results_dir(SLUG, args.results_dir)
    plain_records, amplified_records, feature_records = load_records(input_dir)

    entries = {}
    for name, plain in sorted(plain_records.items()):
        entries[name] = analyze_entry(
            plain,
            amplified_records.get(name),
            feature_records.get(name),
            args.draws,
            dose_record(name),
        )
    readout = {"entries": entries, "verdicts": judge(entries)}

    write_json(readout, os.path.join(input_dir, "readout.json"))
    if args.readme:
        render_readme(readout, args.readme)


def load_records(input_dir):
    records = ({}, {}, {})
    for part, store in zip(("plain", "amplified", "features"), records):
        directory = os.path.join(input_dir, part)
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pt"):
                store[filename[: -len(".pt")]] = torch.load(
                    os.path.join(directory, filename), map_location="cpu"
                )
    return records


def dose_record(name):
    path = os.path.join(REPO_ROOT, DOSE_RECORDS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        record = json.load(handle)
    return record


def analyze_entry(plain, amplified, features, draws, earlier_dose_record):
    meta = plain["meta"]
    rates = plain["rates"]
    rate_index = rates.index(plain["adaptive_rate"])
    validation = plain["validation"]
    clean_test = plain["clean_test"]
    validation_single = validation["psu"][rate_index]  # (n_validation,)
    clean_single = clean_test["psu"][rate_index]  # (n_clean,)

    attacked_class = meta["target"]
    reference, reference_kind = pooling_reference(
        validation_single, validation["prediction"], attacked_class
    )
    clean_of_class = clean_single[clean_test["prediction"] == attacked_class]
    multi_validation = multi_rate_scores(validation, rates)
    multi_clean = multi_rate_scores(clean_test, rates)

    doses = {}
    for dose, triggered in plain["triggered"].items():
        hits = triggered["prediction"] == attacked_class  # (pairs,)
        single_hits = triggered["psu"][rate_index][hits]  # (n_hits,)
        multi_hits = {
            score: values[hits]
            for score, values in multi_rate_scores(triggered, rates).items()
        }
        reading = {
            "asr": float(hits.float().mean()),
            "hits": int(hits.sum()),
            "single": operating_readout(single_hits, clean_single, validation_single),
            "auroc_hits_vs_twins": auroc_low_is_positive(
                single_hits, plain["twins"]["psu"][rate_index]
            ),
            "auroc_hits_vs_clean_of_class": auroc_low_is_positive(
                single_hits, clean_of_class
            ),
            "multi": {
                score: operating_readout(
                    multi_hits[score], multi_clean[score], multi_validation[score]
                )
                for score in MULTI_SCORES
            },
            "pooled": pooled_readout(single_hits, clean_of_class, reference, draws),
            "mixed": mixed_readout(single_hits, clean_of_class, reference, draws),
            "rules": rules_readout(
                single_hits,
                attacked_class,
                clean_single,
                clean_test["prediction"],
                validation_single,
                validation["prediction"],
            ),
        }
        if amplified is not None and amplified.get("thresholds") is not None:
            reading["amplified"] = amplified_readout(
                plain, amplified, dose, hits, rate_index, attacked_class
            )
        doses[str(dose)] = reading

    entry = {
        "folder": meta["folder"],
        "probe_attack": meta["probe_attack"],
        "attack": meta["attack"],
        "dataset": meta["dataset"],
        "backdoored": meta["backdoored"],
        "smoke": meta["smoke"],
        "pairs": meta["pairs"],
        "adaptive_rate": plain["adaptive_rate"],
        "attacked_class": attacked_class,
        "reference": {"kind": reference_kind, "size": len(reference)},
        "clean_pool_of_class": len(clean_of_class),
        "threshold_agreement": threshold_agreement(
            validation_single, plain["thresholds"]
        ),
        "regime": REGIME.get(meta["folder"], "benign control"),
        "consistency": consistency(plain, rate_index, earlier_dose_record),
        "matched_filter": (
            matched_filter_readout(plain, features, rate_index, attacked_class)
            if features is not None
            else None
        ),
        "amplification": amplification_summary(amplified),
        "doses": doses,
    }
    return entry


def pooling_reference(validation_scores, validation_prediction, attacked_class):
    of_class = validation_scores[validation_prediction == attacked_class]
    if len(of_class) >= MIN_REFERENCE:
        return of_class, "class"
    return validation_scores, "all"


# Thresholds at quantiles of clean validation, TPR on the positives and the
# realized FPR on a clean set the thresholds never saw.
def operating_readout(positive, clean, validation):
    readout = {}
    for quantile in QUANTILES:
        threshold = threshold_at_quantile(validation, quantile)
        readout[str(quantile)] = {
            "threshold": threshold,
            "tpr": fraction_below(positive, threshold),
            "fpr": fraction_below(clean, threshold),
        }
    readout["auroc_vs_clean"] = auroc_low_is_positive(positive, clean)
    return readout


def fraction_below(scores, threshold):
    if len(scores) == 0:
        return None
    share = float((scores < threshold).float().mean())
    return share


# Scores over the whole ladder, each oriented so a low value means the input
# survived the probe, as the canon's single-rate score is.
def multi_rate_scores(split, rates):
    rate_axis = torch.tensor(rates, dtype=torch.float32)  # (R,)
    span = float(rate_axis[-1] - rate_axis[0])
    psu_area = torch.trapezoid(split["psu"].float(), rate_axis, dim=0) / span  # (n,)
    keep_area = torch.trapezoid(split["kept"].float(), rate_axis, dim=0) / span
    scores = {
        "psu_area": psu_area,
        "shift_area": 1.0 - keep_area,  # (n,)
        "critical_rate": -majority_critical_rate(split["kept"], rates),  # (n,)
    }
    return scores


# defenses.scores.critical_rate with flip_fraction 0.5, read from the kept
# fractions: the first rate at which more than half the passes change the answer,
# and 1 above the top rate for an input that never flips.
def majority_critical_rate(kept, rates):
    flipped = (1.0 - kept) > MAJORITY  # (R, n)
    sentinel = rates[-1] + 1.0
    critical = torch.full((kept.shape[1],), sentinel)  # (n,)
    for index in reversed(range(len(rates))):
        critical = torch.where(flipped[index], torch.tensor(rates[index]), critical)
    return critical


def pooled_readout(attack_pool, clean_pool, reference, draws):
    readout = {}
    for batch_size in BATCH_SIZES:
        readout[str(batch_size)] = {
            "detection": alarm_rates(
                batch_draws(((attack_pool, batch_size),), draws), reference
            ),
            "false_alarm": alarm_rates(
                batch_draws(((clean_pool, batch_size),), draws), reference
            ),
        }
    readout["attack_pool"] = len(attack_pool)
    readout["clean_pool"] = len(clean_pool)
    return readout


def mixed_readout(attack_pool, clean_pool, reference, draws):
    readout = {}
    for fraction in MIXED_FRACTIONS:
        for batch_size in MIXED_BATCH_SIZES:
            triggered_count = round(fraction * batch_size)
            batches = batch_draws(
                (
                    (attack_pool, triggered_count),
                    (clean_pool, batch_size - triggered_count),
                ),
                draws,
            )
            readout[f"{fraction}_{batch_size}"] = {
                "triggered": triggered_count,
                "detection": alarm_rates(batches, reference),
            }
    return readout


# Batches that take count queries from each pool, each without replacement, or
# None when a pool holds fewer queries than a batch needs from it.
def batch_draws(pools_and_counts, draws):
    if any(len(pool) < count for pool, count in pools_and_counts):
        return None
    generator = torch.Generator().manual_seed(BATCH_SEED)
    parts = []
    for pool, count in pools_and_counts:
        if count == 0:
            continue
        # The first count positions of a random order of the pool, 1 order per draw.
        keys = torch.rand(draws, len(pool), generator=generator)  # (draws, pool)
        index = keys.argsort(dim=1)[:, :count]  # (draws, count)
        parts.append(pool[index])
    batches = torch.cat(parts, dim=1)  # (draws, batch size)
    return batches


def alarm_rates(batches, reference):
    if batches is None:
        return None
    pvalues = mann_whitney_less(batches, reference)  # (draws,)
    rates = {str(level): float((pvalues <= level).float().mean()) for level in LEVELS}
    return rates


def mann_whitney_less(batches, reference):
    """1-sided Mann-Whitney p-value that each batch sits below the reference.

        original form (Mann and Whitney 1947)
            U = sum_{i=1..n} sum_{j=1..m} [ 1(x_i > y_j) + 1/2 1(x_i = y_j) ]
            H1: X stochastically smaller than Y, p = P_H0(U <= u)
            asymptotic: z = (u - n m / 2 + 1/2) / sqrt(n m (n + m + 1) / 12)

        symbols
            x_i   the scores of the n queries of 1 batch
            y_j   the m clean validation scores of the reference
            U     the count of (query, reference) pairs where the query scores higher

    Because the reference is fixed, U is the sum over the batch of each query's
    count of reference scores below it, so every batch is scored in 1 pass. At
    n = 1 the null of U is uniform on 0 to m, which gives the exact p-value
    (floor(u) + 1) / (m + 1). For n > 1 it is the normal approximation with the
    continuity correction and no tie correction, as scipy's asymptotic method
    without ties, which holds for continuous scores.

    batches is (draws, n), reference (m,). Returns (draws,) p-values.
    """
    draws, batch_size = batches.shape
    ordered = torch.sort(reference.double()).values  # (m,)
    reference_size = len(ordered)
    below = torch.searchsorted(ordered, batches.double(), right=False)  # (draws, n)
    at_or_below = torch.searchsorted(ordered, batches.double(), right=True)
    counts = below.double() + 0.5 * (at_or_below - below).double()  # (draws, n)
    statistic = counts.sum(dim=1)  # (draws,)
    assert statistic.shape == (draws,)

    if batch_size == 1:
        pvalues = (torch.floor(statistic) + 1.0) / (reference_size + 1.0)
        return pvalues
    mean = batch_size * reference_size / 2.0
    spread = math.sqrt(
        batch_size * reference_size * (batch_size + reference_size + 1) / 12.0
    )
    z = (statistic - mean + 0.5) / spread  # (draws,)
    pvalues = torch.special.ndtr(z)  # (draws,)
    return pvalues


# Conformal flags against clean validation, per predicted class or pooled, 1 or
# 2-sided. A class with fewer validation images than the level can resolve falls
# back to every validation score.
def rule_flags(
    scores, predictions, reference_scores, reference_predictions, level, rule
):
    per_class = rule.startswith("class")
    two_sided = rule.endswith("two_sided")
    needed_size = math.ceil(round((2 if two_sided else 1) / level, 9)) - 1
    flags = torch.zeros(len(scores), dtype=torch.bool)  # (n,)
    for predicted in predictions.unique():
        members = predictions == predicted  # (n,)
        reference = reference_scores
        if per_class:
            of_class = reference_scores[reference_predictions == predicted]
            reference = of_class if len(of_class) >= needed_size else reference_scores
        lower, upper = tail_pvalues(scores[members], reference)
        flags[members] = (
            torch.minimum(lower, upper) <= level / 2 if two_sided else lower <= level
        )
    return flags


def tail_pvalues(scores, reference):
    ordered = torch.sort(reference.double()).values  # (m,)
    size = len(ordered)
    at_or_below = torch.searchsorted(ordered, scores.double(), right=True)  # (n,)
    at_or_above = size - torch.searchsorted(ordered, scores.double(), right=False)
    lower = (1.0 + at_or_below) / (size + 1.0)  # (n,)
    upper = (1.0 + at_or_above) / (size + 1.0)  # (n,)
    return lower, upper


def rules_readout(
    hit_scores,
    attacked_class,
    clean_scores,
    clean_predictions,
    validation_scores,
    validation_predictions,
):
    hit_predictions = torch.full((len(hit_scores),), attacked_class)
    clean_of_class = clean_predictions == attacked_class
    readout = {}
    for rule in RULES:
        readout[rule] = {}
        for level in LEVELS:
            hit_flags = rule_flags(
                hit_scores,
                hit_predictions,
                validation_scores,
                validation_predictions,
                level,
                rule,
            )
            clean_flags = rule_flags(
                clean_scores,
                clean_predictions,
                validation_scores,
                validation_predictions,
                level,
                rule,
            )
            readout[rule][str(level)] = {
                "tpr": float(hit_flags.float().mean()) if len(hit_scores) else None,
                "fpr": float(clean_flags.float().mean()),
                "fpr_of_class": (
                    float(clean_flags[clean_of_class].float().mean())
                    if clean_of_class.any()
                    else None
                ),
            }
    return readout


# The extension's threat model: the defender sees an unlabeled stream, the even
# clean test rows and the full-strength triggered images of the first half of the
# pairs, and flags it with PSBD-TM at the canonical 1% threshold. The flagged
# queries of the most flagged class give the direction. Every reading is on the
# other half of the pairs and the odd clean test rows, which the stream never held.
def matched_filter_readout(plain, features, rate_index, attacked_class):
    validation = plain["validation"]
    clean_test = plain["clean_test"]
    pair_count = len(plain["twins"]["prediction"])
    assert len(features["clean_test"]["feature"]) == len(clean_test["prediction"])
    assert len(features["validation"]["feature"]) == len(validation["prediction"])
    stream_pairs = torch.arange(pair_count // 2)
    held_out_pairs = torch.arange(pair_count // 2, pair_count)
    stream_rows = torch.arange(0, len(clean_test["prediction"]), 2)
    held_out_rows = torch.arange(1, len(clean_test["prediction"]), 2)
    full_dose = plain["triggered"][1.0]

    stream_scores = torch.cat(
        [
            clean_test["psu"][rate_index][stream_rows],
            full_dose["psu"][rate_index][stream_pairs],
        ]
    )  # (stream,)
    stream_predictions = torch.cat(
        [clean_test["prediction"][stream_rows], full_dose["prediction"][stream_pairs]]
    )  # (stream,)
    stream_features = torch.cat(
        [
            features["clean_test"]["feature"][stream_rows],
            features["triggered"][1.0]["feature"][stream_pairs],
        ]
    ).float()  # (stream, hidden)
    stream_triggered = torch.cat(
        [torch.zeros(len(stream_rows)), torch.ones(len(stream_pairs))]
    ).bool()  # (stream,), a diagnostic only, never read by the estimate

    validation_single = validation["psu"][rate_index]
    flagged = stream_scores < threshold_at_quantile(validation_single, STRICT_LEVEL)
    if not flagged.any():
        return {"estimable": False, "catches": 0}
    suspect = int(torch.mode(stream_predictions[flagged]).values)
    catches = flagged & (stream_predictions == suspect)  # (stream,)
    summary = {
        "estimable": int(catches.sum()) >= MIN_CATCHES,
        "suspect": suspect,
        "suspect_is_target": suspect == attacked_class,
        "catches": int(catches.sum()),
        "triggered_share_of_catches": float(stream_triggered[catches].float().mean()),
    }
    if not summary["estimable"]:
        return summary

    validation_features = features["validation"]["feature"].float()  # (n_val, hidden)
    of_suspect = validation_features[validation["prediction"] == suspect]
    reference = (
        of_suspect
        if len(of_suspect) >= MIN_DIRECTION_REFERENCE
        else validation_features
    )
    direction = stream_features[catches].mean(dim=0) - reference.mean(
        dim=0
    )  # (hidden,)
    unit = direction / direction.norm()  # (hidden,)

    validation_scores = -(validation_features @ unit)  # (n_val,)
    held_out_clean = -(
        features["clean_test"]["feature"][held_out_rows].float() @ unit
    )  # (n_held_out,)
    held_out_clean_single = clean_test["psu"][rate_index][held_out_rows]
    held_out_of_class = clean_test["prediction"][held_out_rows] == attacked_class
    twin_scores = -(features["twins"]["feature"][held_out_pairs].float() @ unit)

    doses = {}
    for dose, triggered in plain["triggered"].items():
        hits = triggered["prediction"][held_out_pairs] == attacked_class
        matched_hits = -(
            features["triggered"][dose]["feature"][held_out_pairs][hits].float() @ unit
        )
        single_hits = triggered["psu"][rate_index][held_out_pairs][hits]
        doses[str(dose)] = {
            "hits": int(hits.sum()),
            "matched": operating_readout(
                matched_hits, held_out_clean, validation_scores
            ),
            "single": operating_readout(
                single_hits, held_out_clean_single, validation_single
            ),
            "matched_auroc_vs_clean_of_class": auroc_low_is_positive(
                matched_hits, held_out_clean[held_out_of_class]
            ),
            "matched_auroc_vs_twins": auroc_low_is_positive(matched_hits, twin_scores),
        }
    summary["doses"] = doses
    return summary


# Sharpened pipeline against the plain one on the same attack set, the inputs the
# unsharpened model sends to the attacked class, with realized FPR on the twins.
def amplified_readout(plain, amplified, dose, hits, rate_index, attacked_class):
    sharpened = amplified["triggered"][dose]
    sharpened_twins = amplified["twins"]["psu"][0]  # (pairs,)
    plain_twins = plain["twins"]["psu"][rate_index]  # (pairs,)
    readout = {
        "asr_sharpened": float(
            (sharpened["prediction"] == attacked_class).float().mean()
        ),
        "sharpened": twin_readout(
            sharpened["psu"][0][hits], sharpened_twins, amplified["thresholds"]
        ),
        "plain": twin_readout(
            plain["triggered"][dose]["psu"][rate_index][hits],
            plain_twins,
            plain["thresholds"],
        ),
    }
    return readout


def twin_readout(positive, twins, thresholds):
    readout = {
        q: {
            "tpr": fraction_below(positive, threshold),
            "fpr": fraction_below(twins, threshold),
        }
        for q, threshold in thresholds.items()
    }
    readout["auroc_vs_twins"] = auroc_low_is_positive(positive, twins)
    return readout


def amplification_summary(amplified):
    if amplified is None:
        return None
    twins = amplified.get("twins")
    summary = {
        "chosen_lambda": amplified["chosen_lambda"],
        "validation_accuracy": {
            str(k): v for k, v in amplified["validation_accuracy"].items()
        },
        "adaptive_rate": amplified.get("adaptive_rate"),
        "twin_accuracy": (
            float((twins["prediction"] == twins["label"]).float().mean())
            if twins is not None
            else None
        ),
    }
    return summary


def threshold_agreement(validation_scores, cached_thresholds):
    gaps = [
        abs(threshold_at_quantile(validation_scores, float(q)) - threshold)
        for q, threshold in cached_thresholds.items()
    ]
    largest = max(gaps)
    return largest


# Fresh passes against the canonical cache on the same images, and the first
# pairs against the dose experiment's aggregate readings.
def consistency(plain, rate_index, earlier_dose_record):
    cached = plain["cached_pairs"]
    fresh_twins = plain["twins"]["psu"][rate_index]
    result = {
        "twins_fresh_vs_cached_auroc": auroc_low_is_positive(
            fresh_twins, cached["twins"]["psu"][0]
        ),
        "triggered_fresh_vs_cached_auroc": (
            auroc_low_is_positive(
                plain["triggered"][1.0]["psu"][rate_index],
                cached["triggered"]["psu"][0],
            )
            if cached["triggered"] is not None
            else None
        ),
        "dose_record_auroc_gap": None,
    }
    if earlier_dose_record is None:
        return result
    earlier = {row["dose"]: row["auroc"] for row in earlier_dose_record["dose"]}
    count = min(CONSISTENCY_PAIRS, len(fresh_twins))
    gaps = {}
    for dose, triggered in plain["triggered"].items():
        if dose not in earlier:
            continue
        fresh = auroc_low_is_positive(
            triggered["psu"][rate_index][:count], fresh_twins[:count]
        )
        gaps[str(dose)] = fresh - earlier[dose]
    result["dose_record_auroc_gap"] = gaps
    return result


def weakest_firing_dose(entry):
    firing = [
        float(dose)
        for dose, reading in entry["doses"].items()
        if reading["asr"] >= FIRING_ASR
    ]
    weakest = str(min(firing)) if firing else None
    return weakest


def judge(entries):
    backdoored = {n: e for n, e in entries.items() if e["backdoored"]}
    benign = {n: e for n, e in entries.items() if not e["backdoored"]}
    verdicts = {
        "C1": judge_dose_record(backdoored),
        "C2": judge_cache_agreement(entries),
        "P1": judge_pooled_detection(backdoored),
        "P2": judge_pooled_false_alarm(backdoored),
        "P3": judge_blind_model(backdoored),
        "P4": judge_mixed(backdoored),
        "P5": judge_benign_pooled(benign),
        "M1": judge_multi_gain(backdoored),
        "M2": judge_ties(entries),
        "M3": judge_benign_multi(benign),
        "A1": judge_lambda(entries),
        "A2": judge_amplified_gain(backdoored),
        "A3": judge_benign_amplified(benign),
        "R1": judge_gate_pooling(backdoored),
        "R2": judge_bpp_pooling(backdoored),
        "R3": judge_blind_pooling(backdoored),
        "B1": judge_gate_target_null(backdoored),
        "B2": judge_benign_target_null(benign),
        "K1": judge_class_gain(backdoored),
        "K2": judge_class_fpr(entries),
        "K3": judge_two_sided(backdoored),
        "K4": judge_fallback_models(backdoored),
        "F1": judge_suspect(backdoored),
        "F2": judge_matched_auroc(backdoored),
        "F3": judge_matched_gain(backdoored),
        "F4": judge_benign_matched(benign),
    }
    return verdicts


def verdict(checks, rule):
    known = {name: value for name, value in checks.items() if value is not None}
    held = rule(known) if known else None
    result = {"held": held, "checks": checks}
    return result


def judge_dose_record(entries):
    checks = {}
    for name, entry in entries.items():
        gaps = entry["consistency"]["dose_record_auroc_gap"]
        checks[name] = (
            max(abs(g) for g in gaps.values()) <= CONSISTENCY_AUROC_BAND
            if gaps
            else None
        )
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_cache_agreement(entries):
    checks = {}
    for name, entry in entries.items():
        values = [
            entry["consistency"]["twins_fresh_vs_cached_auroc"],
            entry["consistency"]["triggered_fresh_vs_cached_auroc"],
        ]
        present = [v for v in values if v is not None]
        checks[name] = all(abs(v - 0.5) <= CONSISTENCY_AUROC_BAND for v in present)
    result = verdict(checks, lambda known: all(known.values()))
    return result


def pooled_rate(entry, dose, kind, batch_size=HEADLINE_BATCH, level=HEADLINE_LEVEL):
    cell = entry["doses"][dose]["pooled"][str(batch_size)][kind]
    rate = cell[str(level)] if cell is not None else None
    return rate


def judge_pooled_detection(entries):
    checks = {}
    for name, entry in entries.items():
        if name == BLIND_MODEL:
            continue
        dose = weakest_firing_dose(entry)
        rate = pooled_rate(entry, dose, "detection") if dose else None
        checks[name] = rate >= POOL_DETECTION_BAR if rate is not None else None
    result = verdict(checks, lambda known: all(known.values()))
    return result


# Held when the class reference keeps false alarms at most twice the level, and
# at least 1 model on the all-class fallback exceeds that, as predicted.
def judge_pooled_false_alarm(entries):
    checks = {}
    for name, entry in entries.items():
        dose = weakest_firing_dose(entry)
        rate = pooled_rate(entry, dose, "false_alarm") if dose else None
        checks[name] = (
            {"reference": entry["reference"]["kind"], "false_alarm": rate}
            if rate is not None
            else None
        )

    def rule(known):
        class_ok = all(
            c["false_alarm"] <= POOL_FALSE_ALARM_BAR
            for c in known.values()
            if c["reference"] == "class"
        )
        fallback_exceeds = any(
            c["false_alarm"] > POOL_FALSE_ALARM_BAR
            for c in known.values()
            if c["reference"] == "all"
        )
        return class_ok and fallback_exceeds

    result = verdict(checks, rule)
    return result


def judge_blind_model(entries):
    entry = entries.get(BLIND_MODEL)
    checks = {}
    for dose in BLIND_DOSES:
        if entry is None or str(dose) not in entry["doses"]:
            continue
        if entry["doses"][str(dose)]["asr"] < FIRING_ASR:
            continue
        rate = pooled_rate(entry, str(dose), "detection")
        checks[str(dose)] = rate <= POOL_BLIND_BAR if rate is not None else None
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_mixed(entries):
    checks = {}
    for name, entry in entries.items():
        if name == BLIND_MODEL:
            continue
        dose = weakest_firing_dose(entry)
        cell = (
            entry["doses"][dose]["mixed"][f"{MIXED_FRACTION}_{MIXED_BATCH}"][
                "detection"
            ]
            if dose
            else None
        )
        checks[name] = (
            cell[str(HEADLINE_LEVEL)] >= MIXED_DETECTION_BAR
            if cell is not None
            else None
        )
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_benign_pooled(entries):
    checks = {}
    for name, entry in entries.items():
        rates = [pooled_rate(entry, dose, "detection") for dose in entry["doses"]]
        present = [r for r in rates if r is not None]
        checks[name] = (
            all(r <= POOL_FALSE_ALARM_BAR for r in present) if present else None
        )
    result = verdict(checks, lambda known: all(known.values()))
    return result


def single_and_multi_tpr(entry, dose, score):
    reading = entry["doses"][dose]
    key = str(HEADLINE_LEVEL)
    pair = (reading["single"][key]["tpr"], reading["multi"][score][key]["tpr"])
    return pair


def judge_multi_gain(entries):
    checks = {}
    for name, entry in entries.items():
        dose = weakest_firing_dose(entry)
        if dose is None:
            checks[name] = None
            continue
        single, multi = single_and_multi_tpr(entry, dose, "psu_area")
        full_single, full_multi = single_and_multi_tpr(entry, "1.0", "psu_area")
        checks[name] = {
            "gain": multi - single >= MULTI_GAIN_BAR,
            "no_loss_at_full_dose": full_multi >= full_single - MULTI_LOSS_BAR,
        }

    def rule(known):
        gains = sum(c["gain"] for c in known.values())
        no_loss = all(c["no_loss_at_full_dose"] for c in known.values())
        return gains >= MULTI_MODELS_BAR and no_loss

    result = verdict(checks, rule)
    return result


def judge_ties(entries):
    checks = {}
    low, high = TIE_FPR_BAND
    for name, entry in entries.items():
        reading = entry["doses"]["1.0"]["multi"]
        checks[name] = {
            score: not (low <= reading[score]["0.01"]["fpr"] <= high)
            for score in ("shift_area", "critical_rate")
        }

    def rule(known):
        outside = sum(all(c.values()) for c in known.values())
        return outside >= len(known) / 2

    result = verdict(checks, rule)
    return result


def judge_benign_multi(entries):
    checks = {}
    key = str(HEADLINE_LEVEL)
    for name, entry in entries.items():
        within = []
        for reading in entry["doses"].values():
            for score in MULTI_SCORES:
                cell = reading["multi"][score][key]
                if cell["tpr"] is not None:
                    within.append(abs(cell["tpr"] - cell["fpr"]) <= BENIGN_TPR_BAND)
        checks[name] = all(within) if within else None
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_lambda(entries):
    checks = {}
    for name, entry in entries.items():
        summary = entry["amplification"]
        if summary is None:
            checks[name] = None
            continue
        chosen = summary["chosen_lambda"]
        checks[name] = chosen is not None and chosen >= LAMBDA_BAR
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_amplified_gain(entries):
    checks = {}
    key = str(HEADLINE_LEVEL)
    for name, entry in entries.items():
        dose = weakest_firing_dose(entry)
        reading = entry["doses"][dose].get("amplified") if dose else None
        if reading is None:
            checks[name] = None
            continue
        gain = reading["sharpened"][key]["tpr"] - reading["plain"][key]["tpr"]
        checks[name] = {"attack": entry["attack"], "gain": gain}

    def rule(known):
        high = [
            c["gain"] >= AMPLIFY_GAIN_BAR
            for c in known.values()
            if c["attack"] in HIGH_FREQUENCY_ATTACKS
        ]
        low = [
            c["gain"] < AMPLIFY_GAIN_BAR
            for c in known.values()
            if c["attack"] in LOW_FREQUENCY_ATTACKS
        ]
        return sum(high) >= AMPLIFY_MODELS_BAR and all(low)

    result = verdict(checks, rule)
    return result


def judge_benign_amplified(entries):
    checks = {}
    for name, entry in entries.items():
        values = [
            reading["amplified"]["sharpened"]["auroc_vs_twins"]
            for reading in entry["doses"].values()
            if "amplified" in reading
        ]
        present = [v for v in values if v is not None]
        checks[name] = (
            all(abs(v - 0.5) <= BENIGN_AUROC_BAND for v in present) if present else None
        )
    result = verdict(checks, lambda known: all(known.values()))
    return result


def gate_models(entries):
    gate = {n: e for n, e in entries.items() if e["regime"] == "gate"}
    return gate


def judge_gate_pooling(entries):
    checks = {}
    for name, entry in gate_models(entries).items():
        dose = weakest_firing_dose(entry)
        rate = (
            pooled_rate(entry, dose, "detection", GATE_POOL_BATCH, STRICT_LEVEL)
            if dose
            else None
        )
        checks[name] = rate >= POOL_DETECTION_BAR if rate is not None else None
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_bpp_pooling(entries):
    entry = entries.get(BPP_MODEL)
    dose = weakest_firing_dose(entry) if entry else None
    rate = (
        pooled_rate(entry, dose, "detection", BPP_POOL_BATCH, STRICT_LEVEL)
        if dose
        else None
    )
    checks = {BPP_MODEL: rate >= POOL_DETECTION_BAR if rate is not None else None}
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_blind_pooling(entries):
    entry = entries.get(BLIND_MODEL)
    checks = {}
    for dose, reading in (entry["doses"] if entry else {}).items():
        if float(dose) > 0.6 or reading["asr"] < FIRING_ASR:
            continue
        rate = pooled_rate(entry, dose, "detection", BLIND_POOL_BATCH, STRICT_LEVEL)
        checks[dose] = rate <= POOL_BLIND_BAR if rate is not None else None
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_gate_target_null(entries):
    checks = {}
    for name, entry in gate_models(entries).items():
        dose = weakest_firing_dose(entry)
        value = entry["doses"][dose]["auroc_hits_vs_clean_of_class"] if dose else None
        checks[name] = value >= GATE_TARGET_AUROC_BAR if value is not None else None
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_benign_target_null(entries):
    checks = {}
    for name, entry in entries.items():
        reading = entry["doses"]["1.0"]
        value = reading["auroc_hits_vs_clean_of_class"]
        if reading["hits"] < BENIGN_MIN_HITS or value is None:
            checks[name] = None
            continue
        checks[name] = abs(value - 0.5) <= BENIGN_TARGET_AUROC_BAND
    result = verdict(checks, lambda known: all(known.values()))
    return result


def rule_gain(entry, dose, rule, level):
    reading = entry["doses"][dose]
    ruled = reading["rules"][rule][str(level)]["tpr"]
    canonical = reading["single"][str(level)]["tpr"]
    gain = ruled - canonical if ruled is not None and canonical is not None else None
    return gain


def judge_class_gain(entries):
    checks = {}
    for name, entry in entries.items():
        if entry["dataset"] not in ATTRACTOR_DATASETS:
            continue
        dose = weakest_firing_dose(entry)
        checks[name] = (
            rule_gain(entry, dose, "class_one_sided", STRICT_LEVEL) if dose else None
        )

    def rule(known):
        mean_gain = sum(known.values()) / len(known)
        return mean_gain >= CLASS_GAIN_BAR

    result = verdict(checks, rule)
    return result


def judge_class_fpr(entries):
    checks = {}
    for name, entry in entries.items():
        reading = entry["doses"]["1.0"]["rules"]
        checks[name] = all(
            reading[rule][str(STRICT_LEVEL)]["fpr"] <= CLASS_FPR_BAR
            for rule in ("class_one_sided", "class_two_sided")
        )
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_two_sided(entries):
    entry = entries.get(TWO_SIDED_MODEL)
    checks = {}
    for dose, wants_gain in ((TWO_SIDED_GAIN_DOSE, True), (TWO_SIDED_FLAT_DOSE, False)):
        if entry is None or dose not in entry["doses"]:
            continue
        gain = rule_gain(entry, dose, "pooled_two_sided", STRICT_LEVEL)
        if gain is None:
            checks[dose] = None
            continue
        checks[dose] = (
            gain >= TWO_SIDED_GAIN_BAR if wants_gain else gain < TWO_SIDED_GAIN_BAR
        )
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_fallback_models(entries):
    checks = {}
    for name in FALLBACK_MODELS:
        entry = entries.get(name)
        if entry is None:
            continue
        changes = [
            rule_gain(entry, dose, "class_one_sided", level)
            for dose, reading in entry["doses"].items()
            if reading["asr"] >= FIRING_ASR
            for level in (STRICT_LEVEL, HEADLINE_LEVEL)
        ]
        present = [c for c in changes if c is not None]
        checks[name] = (
            all(abs(c) <= FALLBACK_CHANGE_BAR for c in present) if present else None
        )
    result = verdict(checks, lambda known: all(known.values()))
    return result


def judge_suspect(entries):
    checks = {
        name: (
            entry["matched_filter"]["suspect_is_target"]
            if entry["matched_filter"] and "suspect" in entry["matched_filter"]
            else None
        )
        for name, entry in entries.items()
    }
    result = verdict(checks, lambda known: all(known.values()))
    return result


def matched_doses(entry):
    matched = entry["matched_filter"]
    doses = matched.get("doses") if matched else None
    return doses


def judge_matched_auroc(entries):
    checks = {}
    refuted = None
    for name, entry in entries.items():
        doses = matched_doses(entry)
        if doses is None:
            checks[name] = None
            continue
        values = [
            doses[dose]["matched_auroc_vs_clean_of_class"]
            for dose, reading in entry["doses"].items()
            if reading["asr"] >= FIRING_ASR
        ]
        present = [v for v in values if v is not None]
        checks[name] = all(v >= MATCHED_AUROC_BAR for v in present) if present else None
        if name == BLIND_MODEL and MATCHED_REFUTATION_DOSE in doses:
            value = doses[MATCHED_REFUTATION_DOSE]["matched_auroc_vs_clean_of_class"]
            refuted = value < MATCHED_REFUTATION_AUROC if value is not None else None
    result = verdict(checks, lambda known: all(known.values()))
    result["refuted_on_blend"] = refuted
    return result


def judge_matched_gain(entries):
    checks = {}
    key = str(STRICT_LEVEL)
    for name, entry in entries.items():
        doses = matched_doses(entry)
        dose = weakest_firing_dose(entry)
        if doses is None or dose is None:
            checks[name] = None
            continue
        matched = doses[dose]["matched"][key]["tpr"]
        single = doses[dose]["single"][key]["tpr"]
        checks[name] = (
            matched > single if matched is not None and single is not None else None
        )

    def rule(known):
        return sum(known.values()) >= MATCHED_MODELS_BAR

    result = verdict(checks, rule)
    return result


def judge_benign_matched(entries):
    checks = {}
    key = str(HEADLINE_LEVEL)
    for name, entry in entries.items():
        doses = matched_doses(entry)
        if doses is None:
            checks[name] = None
            continue
        within = [
            abs(reading["matched"][key]["tpr"] - reading["matched"][key]["fpr"])
            <= BENIGN_TPR_BAND
            for reading in doses.values()
            if reading["matched"][key]["tpr"] is not None
        ]
        checks[name] = all(within) if within else None
    result = verdict(checks, lambda known: all(known.values()))
    return result


def write_json(payload, path):
    partial = f"{path}.partial"
    with open(partial, "w") as handle:
        json.dump(payload, handle, indent=1)
    os.replace(partial, path)


def render_readme(readout, readme_path):
    with open(readme_path) as handle:
        text = handle.read()
    begin, end = "<!-- results:begin -->", "<!-- results:end -->"
    head, _, rest = text.partition(begin)
    _, _, tail = rest.partition(end)
    rendered = f"{head}{begin}\n{results_block(readout)}\n{end}{tail}"
    with open(readme_path, "w") as handle:
        handle.write(rendered)


if __name__ == "__main__":
    main()

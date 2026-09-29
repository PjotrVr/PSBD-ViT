"""Freezes at most 2 probes from the development readings, before any held-out read.

Applies the selection rule the README fixed before the development numbers existed
and writes experiments/novel_probes/preregistration.json with the sha256 of the 2
records it read, so the freeze can be checked against them later.

    .venv/bin/python -m experiments.novel_probes.preregister
"""

import datetime
import hashlib
import json

from experiments._paths import experiment_result_path
from experiments.novel_probes.evaluate import HEADLINE_QUANTILES
from experiments.novel_probes.measure import PREREGISTRATION_PATH, PROBES, SLUG

MAX_FROZEN = 2
MAX_FPR_GAP = 0.02


def main():
    dev_path = experiment_result_path(SLUG, "dev.json")
    benign_path = experiment_result_path(SLUG, "benign.json")
    dev = load(dev_path)
    benign = load(benign_path)

    candidates = [
        candidate(name, dev["summary"]["probes"][name], benign["summary"]["probes"])
        for name in dev["summary"]["probes"]
        if "@" not in name
    ]
    eligible = [entry for entry in candidates if entry["eligible"]]
    ranked = sorted(
        eligible, key=lambda entry: -entry["mean_union_gain_vs_reference_2"]
    )
    frozen = [freeze(entry, dev) for entry in ranked[:MAX_FROZEN]]

    payload = {
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rule": "rank by the mean over 1, 5 and 10% FPR of the paired TPR gain over "
        "reference 2 as a min-union member with PSBD-TM, on the pooled development "
        "models, among probes passing the benign gate and the FPR gate",
        "max_fpr_gap": MAX_FPR_GAP,
        "inputs": {
            dev_path: sha256(dev_path),
            benign_path: sha256(benign_path),
        },
        "candidates": candidates,
        "frozen": frozen,
        "held_out": "the CIFAR-100 and Tiny ImageNet panel models, read once",
    }
    with open(PREREGISTRATION_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"wrote {PREREGISTRATION_PATH}, frozen {[entry['probe'] for entry in frozen]}"
    )


def load(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def sha256(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return digest


def candidate(name, summary, benign_probes):
    union = summary["union_tm"]
    gains = [
        union[f"{q}:tpr"]["vs_union_tm_middle"]["mean_difference"]
        for q in HEADLINE_QUANTILES
    ]
    # The realized FPR of the probe alone, against its nominal quantile.
    fpr_gaps = [
        abs(
            summary["standalone"][f"{q}:realized_fpr"]["vs_tm_alone"]["mean"]
            - float(q[1:])
        )
        for q in HEADLINE_QUANTILES
    ]
    benign_pass = name in benign_probes and all(
        benign_probes[name][reading]["within_tolerance"]
        for reading in ("standalone", "union_tm")
    )
    entry = {
        "probe": name,
        "mean_union_gain_vs_reference_2": sum(gains) / len(gains),
        "union_gain_vs_reference_2": dict(zip(HEADLINE_QUANTILES, gains)),
        "max_fpr_gap": max(fpr_gaps),
        "benign_gate": benign_pass,
        "fpr_gate": max(fpr_gaps) <= MAX_FPR_GAP,
        "beats_both_at_every_fpr": {
            reading: summary[reading]["beats_both_at_every_fpr"]
            for reading in ("standalone", "union_tm")
        },
    }
    entry["eligible"] = entry["benign_gate"] and entry["fpr_gate"]
    return entry


def freeze(entry, dev):
    name = entry["probe"]
    spec = PROBES[name]
    summary = dev["summary"]["probes"][name]
    references = dev["summary"]["references"]

    def mean_of(reading, q):
        value = summary[reading][f"{q}:tpr"]["vs_tm_alone"]["mean"]
        return value

    frozen = {
        "probe": name,
        "sign": spec["sign"],
        "ladder": list(spec["ladder"]),
        "passes": spec["passes"],
        "rate_rule": "adaptive 0.8 with the nearest rung as fallback"
        if spec["sign"] == "psbd"
        else "largest rung with clean validation shift at most 0.05",
        "union_rule": "plain minimum of the clean-validation percentiles of the probe "
        "and PSBD-TM",
        "development_tpr": {
            reading: {q: mean_of(reading, q) for q in HEADLINE_QUANTILES}
            for reading in ("standalone", "union_tm")
        },
        "development_references": {
            name: {q: references[name][f"{q}:tpr"] for q in HEADLINE_QUANTILES}
            for name in ("tm_alone", "union_tm_middle")
        },
        "beat_both_on_development": entry["beats_both_at_every_fpr"],
        "prediction": "on the held-out panel the union with PSBD-TM has a higher mean "
        "TPR than both references at 1, 5 and 10% FPR, at no more than the nominal "
        "FPR plus 0.02",
        "refuted_if": "the union's mean TPR is at or below either reference at any "
        "of the 3 FPRs on the held-out panel",
    }
    return frozen


if __name__ == "__main__":
    main()

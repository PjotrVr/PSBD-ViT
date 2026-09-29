"""Is the late band special as PSBD-TM's union partner, or does blocks 5 to 8 do as well?

Reads the per-model min-rank readings that experiments/cache_readouts/fusion_rules.py
already wrote for both partners and compares them paired, on the models where both
partners have a rate under the same rule. CPU only, nothing is recomputed.

    .venv/bin/python -m experiments.novel_probes.band_partner
"""

import json
import time

from experiments._paths import experiment_result_path
from experiments.cache_readouts.shared import paired_summary
from experiments.novel_probes.measure import SLUG

MODEL_SETS = ("dev", "holdout", "panel")
FIELDS = ("auroc", "q0.01", "q0.05", "q0.10")
# "adaptive" needs both partners at their adaptive rate. "adaptive_or_nearest"
# falls back to the nearest rung for a partner whose ladder never reaches 0.8,
# which keeps the 4 development models whose late band stops short of it.
RATE_RULES = ("adaptive", "adaptive_or_nearest")


def main():
    started = time.perf_counter()
    payload = {
        "question": "middle band (pre_residual_blocks_5_8) against late band "
        "(pre_residual_blocks_9_12) as the min-rank partner of PSBD-TM",
        "source": "results/_experiments/cache_readouts/fusion_rules_<set>.json",
        "sets": {},
    }
    for model_set in MODEL_SETS:
        rows = load_rows(model_set)
        payload["sets"][model_set] = {rule: compare(rows, rule) for rule in RATE_RULES}
    payload["wall_seconds"] = time.perf_counter() - started

    path = experiment_result_path(SLUG, "band_partner.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {path}")


def load_rows(model_set):
    path = experiment_result_path("cache_readouts", f"fusion_rules_{model_set}.json")
    with open(path) as handle:
        rows = json.load(handle)["models"]
    pooled = [row for row in rows if row["successful_2pt"] and row["tm_rate"]]
    return pooled


def partner_reading(row, partner, rule):
    adaptive = row["partners"]["adaptive"].get(partner, {})
    if adaptive.get("rate") is not None:
        return adaptive
    if rule == "adaptive":
        return None
    nearest = row["partners"]["nearest"].get(partner, {})
    reading = nearest if nearest.get("rate") is not None else None
    return reading


def metric(reading, fusion, field):
    evaluation = reading["rules"][fusion]
    value = (
        evaluation["auroc"] if field == "auroc" else evaluation["at_fpr"][field]["tpr"]
    )
    return value


def compare(rows, rule):
    both = []
    for row in rows:
        middle = partner_reading(row, "middle_band", rule)
        late = partner_reading(row, "late_band", rule)
        if middle is not None and late is not None:
            both.append((row, middle, late))

    comparison = {
        "n": len(both),
        "folders": [row["folder"] for row, _, _ in both],
        "late_band_rules": {
            row["folder"]: "adaptive"
            if row["partners"]["adaptive"]["late_band"].get("rate") is not None
            else "nearest"
            for row, _, _ in both
        },
        "fields": {},
    }
    for field in FIELDS:
        middle_values = [metric(middle, "min_rank", field) for _, middle, _ in both]
        late_values = [metric(late, "min_rank", field) for _, _, late in both]
        tm_values = [metric(middle, "tm_alone", field) for _, middle, _ in both]
        comparison["fields"][field] = {
            "middle_minus_late": paired_summary(middle_values, late_values),
            "middle_minus_tm": paired_summary(middle_values, tm_values),
            "late_minus_tm": paired_summary(late_values, tm_values),
            "realized_fpr_middle": mean_realized(both, 1, field),
            "realized_fpr_late": mean_realized(both, 2, field),
        }
    return comparison


def mean_realized(both, index, field):
    if field == "auroc":
        return None
    values = [
        entry[index]["rules"]["min_rank"]["at_fpr"][field]["realized_fpr"]
        for entry in both
    ]
    average = sum(values) / len(values) if values else None
    return average


if __name__ == "__main__":
    main()

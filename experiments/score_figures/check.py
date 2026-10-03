"""Hold the score figures' numbers to the records they must reproduce.

Each pairing below names a record written by an earlier experiment through the
same defenses helpers. Every TPR, realized FPR and AUROC is compared model by
model, and the panel means of TPR at 1% FPR are printed beside the record's own.
A difference above TOLERANCE fails loudly.

    .venv/bin/python -m experiments.score_figures.check
"""

import glob
import json
import os

import numpy as np

SCORE_ROOT = os.path.join("results", "_experiments", "score_figures")
TOLERANCE = 1e-6
FIELDS = [
    f"{budget}:{metric}"
    for budget in ("q0.01", "q0.05", "q0.10", "q0.20")
    for metric in ("tpr", "realized_fpr")
]

# Each record names its rules its own way. The maps translate them to the
# combination and rule numbers.json uses.
CACHE_READOUT_PAIRS = {
    ("middle_band", "tm_alone"): ("tm", "psu"),
    ("middle_band", "min_rank"): ("tm+band", "min"),
    ("middle_band", "weighted_0.9_0.1"): ("tm+band", "weighted"),
    ("psbd_rd", "min_rank"): ("tm+rd", "min"),
    ("psbd_rd", "weighted_0.9_0.1"): ("tm+rd", "weighted"),
}
BACKDOORBENCH_PAIRS = {"psbd_tm": ("tm", "psu"), "final_min": ("tm+band", "min")}
SWIN_PAIRS = {"psbd_tm": ("tm", "psu"), "final_min": ("tm+band", "min")}


def main():
    vit_rows = check_vit_panel()
    bb_rows = check_backdoorbench()
    swin_rows = check_swin_panel()
    for name, rows in (
        ("vit_panel", vit_rows),
        ("backdoorbench", bb_rows),
        ("swin_panel", swin_rows),
    ):
        print(f"{name}: {rows} model readings match within {TOLERANCE}")


def check_vit_panel():
    with open(
        os.path.join(
            "results", "_experiments", "cache_readouts", "fusion_rules_panel.json"
        )
    ) as handle:
        record = json.load(handle)
    numbers = load_numbers("vit_panel")

    compared = 0
    for (partner, record_rule), (key, rule) in CACHE_READOUT_PAIRS.items():
        ours, theirs = [], []
        for model in record["models"]:
            reading = model["partners"]["adaptive"].get(partner, {})
            if reading.get("rate") is None:
                continue
            expected = reading["rules"][record_rule]
            evaluation = numbers[model["folder"]]["combinations"][key][rule]
            compare(
                flat_at_fpr(expected), flat_at_fpr(evaluation), model["folder"], key
            )
            ours.append(evaluation["at_fpr"]["q0.01"]["tpr"])
            theirs.append(expected["at_fpr"]["q0.01"]["tpr"])
            compared += 1
        print(
            f"vit_panel {key} {rule}: mean TPR at 1% FPR {np.mean(ours):.3f} "
            f"over {len(ours)} models, record {np.mean(theirs):.3f}"
        )
    return compared


def check_backdoorbench():
    numbers = load_numbers("backdoorbench")
    compared = 0
    for folder, model_numbers in numbers.items():
        with open(
            os.path.join(
                "results",
                "_experiments",
                "backdoorbench_attacks",
                "models",
                f"{folder[3:]}.json",
            )
        ) as handle:
            methods = json.load(handle)["detection"]["methods"]
        for record_rule, (key, rule) in BACKDOORBENCH_PAIRS.items():
            evaluation = model_numbers["combinations"][key][rule]
            compare(
                flat_at_fpr(methods[record_rule]), flat_at_fpr(evaluation), folder, key
            )
            compared += 1
    return compared


def check_swin_panel():
    with open(
        os.path.join(
            "results", "_experiments", "final_method", "fusion_swin_panel_late.json"
        )
    ) as handle:
        record = json.load(handle)
    numbers = load_numbers("swin_panel")

    compared = 0
    for model in record["models"]:
        if model["rates"].get("partner_rule") != "adaptive":
            continue
        for record_rule, (key, rule) in SWIN_PAIRS.items():
            expected = model["rules"][record_rule]
            evaluation = numbers[model["folder"]]["combinations"][key][rule]
            compare(expected, flat_at_fpr(evaluation), model["folder"], key)
            compared += 1
    return compared


def load_numbers(set_name):
    numbers = {}
    for path in glob.glob(os.path.join(SCORE_ROOT, set_name, "*", "numbers.json")):
        with open(path) as handle:
            model_numbers = json.load(handle)
        numbers[model_numbers["folder"]] = model_numbers
    return numbers


def flat_at_fpr(evaluation):
    if "at_fpr" not in evaluation:
        return evaluation
    flat = {"auroc": evaluation["auroc"]}
    for budget, values in evaluation["at_fpr"].items():
        flat[f"{budget}:tpr"] = values["tpr"]
        flat[f"{budget}:realized_fpr"] = values["realized_fpr"]
    return flat


def compare(expected, ours, folder, key):
    for field in FIELDS + ["auroc"]:
        difference = abs(expected[field] - ours[field])
        assert difference <= TOLERANCE, (
            folder,
            key,
            field,
            expected[field],
            ours[field],
        )


if __name__ == "__main__":
    main()

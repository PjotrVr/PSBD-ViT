"""The generated results section of README.md, from readout.json.

analyze.py calls results_block and writes its output between the results markers,
so every number in the README comes from the records and none is typed.
"""

PREDICTION_TEXT = {
    "C1": "first 256 pairs match the dose record",
    "C2": "fresh passes match the canonical cache",
    "P1": "pooled detection of 20 queries at the weakest firing dose",
    "P2": "pooled false alarms of 20 clean target queries",
    "P3": "pooling blind on the inverted Blend model",
    "P4": "mixed batch, 10 of 50 triggered",
    "P5": "benign pooled detection at the level",
    "M1": "PSU-curve area beats the single rate",
    "M2": "tied multi-rate scores miss their FPR",
    "M3": "benign multi-rate TPR equals its FPR",
    "A1": "sharpening keeps clean accuracy at lambda 0.5 or more",
    "A2": "sharpening restores high-frequency triggers only",
    "A3": "benign AUROC stays at chance when sharpened",
    "R1": "gate regime pools at 10 queries at level 0.01",
    "R2": "BPP pools at 20 queries at level 0.01",
    "R3": "collapsed Blend stays hidden at 50 queries",
    "B1": "gate attack set beats target-class clean images",
    "B2": "benign attack set is at chance against class 0 clean",
    "K1": "per-class rule gains at 1% on CIFAR-10 and GTSRB",
    "K2": "per-class rules keep their FPR at 1%",
    "K3": "2-sided rule catches collapsed Blend at 0.5 only",
    "K4": "per-class rule leaves CIFAR-100 and Tiny alone",
    "F1": "the most flagged class is the target",
    "F2": "matched filter reads firing at every dose",
    "F3": "matched filter beats the single rate at 1%",
    "F4": "benign matched filter TPR equals its FPR",
}
REGIMES = ("gate", "collapse", "benign control")
RULES = ("pooled_two_sided", "class_one_sided", "class_two_sided")
HEADLINE_LEVEL = "0.05"
HEADLINE_BATCH = "20"
QUANTILES = ("0.01", "0.05", "0.1")
MULTI_SCORES = ("psu_area", "shift_area", "critical_rate")


def results_block(readout):
    entries = by_regime(readout["entries"])
    smoke = any(entry["smoke"] for entry in entries.values())
    sections = [
        (
            "These are smoke records on truncated splits, not results."
            if smoke
            else f"{len(entries)} entries read."
        ),
        verdict_table(readout["verdicts"]),
        regime_table(entries),
        single_table(entries),
        pooled_table(entries),
        pooled_levels_table(entries),
        mixed_table(entries),
        multi_table(entries),
        rules_table(entries),
        matched_table(entries),
        amplified_table(entries),
        consistency_table(entries),
    ]
    block = "\n\n".join(sections)
    return block


def number(value, digits=3):
    text = "n/a" if value is None else f"{value:.{digits}f}"
    return text


def table(header, rows):
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    text = "\n".join(lines)
    return text


def verdict_word(held):
    word = {True: "held", False: "failed", None: "not measurable"}[held]
    return word


def verdict_table(verdicts):
    rows = [
        [key, PREDICTION_TEXT[key], verdict_word(verdicts[key]["held"])]
        for key in PREDICTION_TEXT
    ]
    text = "Verdicts of PREDICTIONS.md.\n\n" + table(
        ["prediction", "claim", "verdict"], rows
    )
    return text


def single_table(entries):
    rows = []
    for name, entry in entries.items():
        for dose, reading in entry["doses"].items():
            cells = [
                f"{number(reading['single'][q]['tpr'])} ({number(reading['single'][q]['fpr'])})"
                for q in QUANTILES
            ]
            rows.append(
                [
                    entry["regime"],
                    name,
                    dose,
                    number(reading["asr"]),
                    str(reading["hits"]),
                    *cells,
                    number(reading["auroc_hits_vs_twins"]),
                    number(reading["auroc_hits_vs_clean_of_class"]),
                ]
            )
    header = [
        "regime",
        "entry",
        "dose",
        "ASR",
        "attack set",
        "TPR at 0.01 (FPR)",
        "TPR at 0.05 (FPR)",
        "TPR at 0.10 (FPR)",
        "AUROC against twins",
        "AUROC against clean of the class",
    ]
    text = (
        "Single-rate PSBD-TM at the adaptive rate on the attack set, realized FPR "
        "on the cached clean test split.\n\n" + table(header, rows)
    )
    return text


def pooled_cell(reading, batch, level=HEADLINE_LEVEL):
    cell = reading["pooled"][batch]
    detection = cell["detection"][level] if cell["detection"] else None
    false_alarm = cell["false_alarm"][level] if cell["false_alarm"] else None
    text = f"{number(detection)} ({number(false_alarm)})"
    return text


def pooled_table(entries):
    batches = ("1", "5", "10", "20", "50")
    rows = []
    for name, entry in entries.items():
        for dose, reading in entry["doses"].items():
            rows.append(
                [
                    entry["regime"],
                    name,
                    dose,
                    f"{entry['reference']['kind']} {entry['reference']['size']}",
                    f"{reading['pooled']['attack_pool']} / {reading['pooled']['clean_pool']}",
                    *(pooled_cell(reading, batch) for batch in batches),
                ]
            )
    header = [
        "regime",
        "entry",
        "dose",
        "reference",
        "attack / clean pool",
        *(f"n = {batch}" for batch in batches),
    ]
    text = (
        f"Pooled monitor at level {HEADLINE_LEVEL}: share of batches of n "
        "attack-set queries that alarm, with the share of batches of n clean test "
        "queries of the attacked class that alarm in brackets.\n\n"
        + table(header, rows)
    )
    return text


def pooled_levels_table(entries):
    levels = ("0.01", "0.05", "0.1")
    rows = []
    for name, entry in entries.items():
        for dose, reading in entry["doses"].items():
            rows.append(
                [
                    entry["regime"],
                    name,
                    dose,
                    *(pooled_cell(reading, HEADLINE_BATCH, level) for level in levels),
                ]
            )
    header = ["regime", "entry", "dose", *(f"level {level}" for level in levels)]
    text = (
        f"Pooled monitor at n = {HEADLINE_BATCH} across alarm levels, detection with "
        "the false alarm rate in brackets.\n\n" + table(header, rows)
    )
    return text


def mixed_table(entries):
    rows = []
    keys = None
    for name, entry in entries.items():
        for dose, reading in entry["doses"].items():
            keys = list(reading["mixed"])
            cells = []
            for key in keys:
                detection = reading["mixed"][key]["detection"]
                cells.append(number(detection[HEADLINE_LEVEL] if detection else None))
            rows.append([entry["regime"], name, dose, *cells])
    if keys is None:
        return "No mixed batches."
    header = [
        "regime",
        "entry",
        "dose",
        *(f"{k.split('_')[0]} of n = {k.split('_')[1]}" for k in keys),
    ]
    text = (
        f"Mixed batches at level {HEADLINE_LEVEL}: a fraction of each batch is "
        "attack-set queries and the rest clean test queries of the attacked class. "
        "The false alarm rate of an all-clean batch is the bracketed column of the "
        "pooled table.\n\n" + table(header, rows)
    )
    return text


def multi_table(entries):
    rows = []
    for name, entry in entries.items():
        for dose, reading in entry["doses"].items():
            single = reading["single"][HEADLINE_LEVEL]
            cells = [f"{number(single['tpr'])} ({number(single['fpr'])})"]
            for score in MULTI_SCORES:
                cell = reading["multi"][score][HEADLINE_LEVEL]
                cells.append(f"{number(cell['tpr'])} ({number(cell['fpr'])})")
            rows.append([entry["regime"], name, dose, *cells])
    header = ["regime", "entry", "dose", "single rate", *MULTI_SCORES]
    text = (
        f"TPR at the {HEADLINE_LEVEL} quantile of clean validation for the single "
        "rate and the 3 ladder scores, realized FPR on the clean test split in "
        "brackets. The 0.01 and 0.10 quantiles are in readout.json.\n\n"
        + table(header, rows)
    )
    return text


def amplified_table(entries):
    rows = []
    for name, entry in entries.items():
        summary = entry["amplification"]
        if summary is None:
            continue
        for dose, reading in entry["doses"].items():
            amplified = reading.get("amplified")
            if amplified is None:
                rows.append(
                    [name, number(summary["chosen_lambda"], 2), dose] + ["n/a"] * 4
                )
                continue
            sharp = amplified["sharpened"][HEADLINE_LEVEL]
            plain = amplified["plain"][HEADLINE_LEVEL]
            rows.append(
                [
                    entry["regime"],
                    name,
                    number(summary["chosen_lambda"], 2),
                    dose,
                    number(amplified["asr_sharpened"]),
                    f"{number(plain['tpr'])} ({number(plain['fpr'])})",
                    f"{number(sharp['tpr'])} ({number(sharp['fpr'])})",
                    number(amplified["sharpened"]["auroc_vs_twins"]),
                ]
            )
    if not rows:
        return "No amplified records."
    header = [
        "regime",
        "entry",
        "lambda",
        "dose",
        "ASR sharpened",
        "plain TPR (FPR)",
        "sharpened TPR (FPR)",
        "sharpened AUROC against twins",
    ]
    text = (
        f"Exploratory residual amplification at the {HEADLINE_LEVEL} quantile, "
        "each pipeline at its own adaptive rate and thresholds, realized FPR on the "
        "clean twins for both.\n\n" + table(header, rows)
    )
    return text


def consistency_table(entries):
    rows = []
    for name, entry in entries.items():
        check = entry["consistency"]
        gaps = check["dose_record_auroc_gap"]
        largest = max((abs(g) for g in gaps.values()), default=None) if gaps else None
        rows.append(
            [
                entry["regime"],
                name,
                number(check["twins_fresh_vs_cached_auroc"]),
                number(check["triggered_fresh_vs_cached_auroc"]),
                number(largest),
                number(entry["threshold_agreement"], 6),
            ]
        )
    header = [
        "regime",
        "entry",
        "fresh against cached twins, AUROC",
        "fresh against cached triggered, AUROC",
        "largest gap to the dose record",
        "threshold gap to the canonical cache",
    ]
    text = "Consistency of the per-input pass.\n\n" + table(header, rows)
    return text


def by_regime(entries):
    ordered = {
        name: entry
        for regime in REGIMES
        for name, entry in sorted(entries.items())
        if entry["regime"] == regime
    }
    return ordered


def weakest_firing_dose(entry, floor=0.5):
    firing = [float(d) for d, r in entry["doses"].items() if r["asr"] >= floor]
    weakest = str(min(firing)) if firing else None
    return weakest


def mean(values):
    present = [v for v in values if v is not None]
    average = sum(present) / len(present) if present else None
    return average


# Each regime at each model's weakest firing dose: the canonical TPR at the 0.01
# quantile against every fix at the same false positive budget.
def regime_table(entries):
    rows = []
    for regime in REGIMES:
        members = [e for e in entries.values() if e["regime"] == regime]
        readings = []
        for entry in members:
            dose = weakest_firing_dose(entry) or "1.0"
            readings.append((entry, dose, entry["doses"][dose]))
        if not readings:
            continue
        pooled = [
            (r["pooled"]["10"]["detection"] or {}).get("0.01") for _, _, r in readings
        ]
        matched = [
            (
                e["matched_filter"]["doses"][d]["matched"]["0.01"]["tpr"]
                if e["matched_filter"] and "doses" in e["matched_filter"]
                else None
            )
            for e, d, _ in readings
        ]
        rows.append(
            [
                regime,
                str(len(readings)),
                number(mean(r["single"]["0.01"]["tpr"] for _, _, r in readings)),
                number(mean(r["single"]["0.01"]["fpr"] for _, _, r in readings)),
                number(mean(r["auroc_hits_vs_clean_of_class"] for _, _, r in readings)),
                number(mean(pooled)),
                number(
                    mean(r["multi"]["psu_area"]["0.01"]["tpr"] for _, _, r in readings)
                ),
                number(
                    mean(
                        r["rules"]["class_one_sided"]["0.01"]["tpr"]
                        for _, _, r in readings
                    )
                ),
                number(
                    mean(
                        r["rules"]["pooled_two_sided"]["0.01"]["tpr"]
                        for _, _, r in readings
                    )
                ),
                number(mean(matched)),
            ]
        )
    header = [
        "regime",
        "entries",
        "canonical TPR at 0.01",
        "its FPR",
        "AUROC against clean of the class",
        "pooled alarm, n = 10 at level 0.01",
        "PSU-curve area TPR",
        "per-class TPR",
        "2-sided TPR",
        "matched filter TPR",
    ]
    text = (
        "Means per regime at each model's weakest firing dose (dose 1.0 where none "
        "fires), every TPR at the 0.01 level of its own rule.\n\n" + table(header, rows)
    )
    return text


def rules_table(entries):
    rows = []
    for name, entry in entries.items():
        for dose, reading in entry["doses"].items():
            canonical = reading["single"]["0.01"]
            cells = [f"{number(canonical['tpr'])} ({number(canonical['fpr'])})"]
            for rule in RULES:
                cell = reading["rules"][rule]["0.01"]
                cells.append(
                    f"{number(cell['tpr'])} ({number(cell['fpr'])}, "
                    f"{number(cell['fpr_of_class'])})"
                )
            rows.append([entry["regime"], name, dose, *cells])
    header = ["regime", "entry", "dose", "canonical", *RULES]
    text = (
        "Threshold rules at level 0.01: TPR on the attack set, with the realized FPR "
        "on the clean test split and on its images predicted as the attacked class "
        "in brackets. The 0.05 and 0.10 levels are in readout.json.\n\n"
        + table(header, rows)
    )
    return text


def matched_table(entries):
    rows = []
    for name, entry in entries.items():
        matched = entry["matched_filter"]
        if matched is None:
            continue
        if "doses" not in matched:
            rows.append([entry["regime"], name, "not estimable"] + ["n/a"] * 5)
            continue
        for dose, reading in matched["doses"].items():
            rows.append(
                [
                    entry["regime"],
                    name,
                    f"{dose}, class {matched['suspect']}, {matched['catches']} catches",
                    str(reading["hits"]),
                    f"{number(reading['single']['0.01']['tpr'])} "
                    f"({number(reading['single']['0.01']['fpr'])})",
                    f"{number(reading['matched']['0.01']['tpr'])} "
                    f"({number(reading['matched']['0.01']['fpr'])})",
                    number(reading["matched_auroc_vs_clean_of_class"]),
                    number(reading["matched_auroc_vs_twins"]),
                ]
            )
    if not rows:
        return "No matched filter records."
    header = [
        "regime",
        "entry",
        "dose, suspected class, flagged stream queries of it",
        "held-out attack set",
        "canonical TPR at 0.01 (FPR)",
        "matched filter TPR at 0.01 (FPR)",
        "matched AUROC against clean of the class",
        "matched AUROC against twins",
    ]
    text = (
        "Matched filter, an extension under its own threat model, on the held-out "
        "half of the pairs and the odd clean test rows.\n\n" + table(header, rows)
    )
    return text

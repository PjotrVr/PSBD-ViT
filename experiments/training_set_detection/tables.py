"""Markdown tables for the README, each built from the analyze.py records.

Every function takes the per-model rows and the summary and returns 1 markdown
table as a string. Nothing here computes a statistic, it only formats what
analyze.py wrote.
"""

from experiments.training_set_detection.common import PAPER_MIRROR

METHOD_WORDS = {
    "final_min": "final method",
    "psbd_tm": "PSBD-TM",
    "psbd_rd": "PSBD-RD",
    "strip": "STRIP",
    "cd_l": "CD-L",
    "spectral_signatures": "Spectral Signatures",
}
ATTACK_WORDS = {
    "badnet_a2o": "BadNets",
    "blend": "Blend",
    "wanet": "WaNet",
    "lc": "Label-Consistent",
    "bpp": "BPP",
    "tact": "TaCT",
    "lf": "LF",
}
DATASET_WORDS = {"cifar10": "CIFAR-10", "gtsrb": "GTSRB", "tiny": "Tiny ImageNet"}
REALIZED = ("q0.01", "q0.05", "q0.10")


def number(value, digits=3):
    if value is None:
        return "n/a"
    text = f"{value:.{digits}f}"
    return text


def pair(tpr, fpr):
    text = f"{number(tpr)}/{number(fpr)}"
    return text


def model_words(row):
    architecture = "ResNet-18" if row["architecture"] == "resnet18" else "ViT-B/16"
    words = (
        f"{ATTACK_WORDS.get(row['attack'], row['attack'])}, "
        f"{DATASET_WORDS.get(row['dataset'], row['dataset'])}, {architecture}"
    )
    return words


def headline_mark(row):
    mark = "" if row["headline"] else " (not pooled)"
    return mark


def paper_format_table(rows, rule, form):
    """TPR/FPR at T = 25th percentile of clean validation, Li et al.'s table layout."""
    methods = ("psbd_tm", "psbd_rd", "final_min")
    header = (
        "| model | "
        + " | ".join(METHOD_WORDS[m] for m in methods)
        + " | STRIP | CD-L | Spectral Signatures |"
    )
    lines = [header, "|---" * (len(methods) + 4) + "|"]
    for folder in PAPER_MIRROR:
        if folder not in rows:
            lines.append(f"| `{folder}` | pending |" + " |" * (len(methods) + 2))
            continue
        row = rows[folder]
        cells = []
        for method in methods:
            cell = row["paper_rule"].get(f"{method}|{rule}|{form}")
            cells.append(
                pair(cell["tpr"], cell["fpr_paper_convention"]) if cell else "n/a"
            )
        for detector in ("strip", "cd_l"):
            reading = row["detectors"].get(detector)
            if reading is None:
                cells.append("pending")
                continue
            cell = reading["nominal"]["q0.25"]
            cells.append(pair(cell["tpr"], cell["fpr_paper_convention"]))
        spectral = row["spectral_signatures"]
        cells.append(pair(spectral["tpr"], spectral["fpr_paper_convention"]))
        lines.append(
            f"| {model_words(row)}{headline_mark(row)} | " + " | ".join(cells) + " |"
        )
    table = "\n".join(lines)
    return table


def low_fpr_table(rows, folders, summary_block):
    """TPR at 1%, 5% and 10% FPR realized on clean training images, then AUROC."""
    methods = ("final_min", "psbd_tm", "psbd_rd")
    header = "| model | method | TPR at 1% | TPR at 5% | TPR at 10% | AUROC |"
    lines = [header, "|---|---|---|---|---|---|"]
    for folder in folders:
        if folder not in rows:
            continue
        row = rows[folder]
        readings = row["readings"]["ours"]["fractional"]
        for method in methods:
            if method not in readings:
                continue
            reading = readings[method]
            cells = [number(reading["realized"][q]["tpr"]) for q in REALIZED]
            lines.append(
                f"| {model_words(row)}{headline_mark(row)} | {METHOD_WORDS[method]} | "
                + " | ".join(cells)
                + f" | {number(reading['auroc'])} |"
            )
        for detector in ("strip", "cd_l"):
            reading = row["common_subset"].get(detector)
            if reading is None:
                continue
            cells = [number(reading["realized"][q]["tpr"]) for q in REALIZED]
            lines.append(
                f"| {model_words(row)}{headline_mark(row)} | {METHOD_WORDS[detector]}, subset | "
                + " | ".join(cells)
                + f" | {number(reading['auroc'])} |"
            )
    means = summary_block.get("means", {})
    for method in methods:
        mean = (means.get("ours|fractional") or {}).get(method)
        if not mean:
            continue
        cells = [number(mean["realized"][q]["tpr"]) for q in REALIZED]
        lines.append(
            f"| mean over {mean['n']} pooled models | {METHOD_WORDS[method]} | "
            + " | ".join(cells)
            + f" | {number(mean['auroc'])} |"
        )
    table = "\n".join(lines)
    return table


def nominal_table(rows, folders):
    """TPR with the FPR realized on clean training images, thresholds on validation."""
    methods = ("final_min", "psbd_tm", "psbd_rd")
    header = "| model | method | 1% | 5% | 10% | 25% |"
    lines = [header, "|---|---|---|---|---|---|"]
    for folder in folders:
        if folder not in rows:
            continue
        row = rows[folder]
        readings = row["readings"]["ours"]["fractional"]
        for method in methods:
            if method not in readings:
                continue
            nominal = readings[method]["nominal"]
            cells = [
                pair(nominal[q]["tpr"], nominal[q]["fpr_clean_train"])
                for q in ("q0.01", "q0.05", "q0.10", "q0.25")
            ]
            lines.append(
                f"| {model_words(row)}{headline_mark(row)} | {METHOD_WORDS[method]} | "
                + " | ".join(cells)
                + " |"
            )
    table = "\n".join(lines)
    return table


def gap_table(rows, folders):
    """Where clean training PSU sits against clean validation PSU, per PSU form."""
    header = (
        "| model | PSU form | median train | median validation | KS statistic "
        "| realized FPR at 1% | at 5% | at 10% | at 25% |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|"]
    for folder in folders:
        if folder not in rows:
            continue
        row = rows[folder]
        method = (
            "psbd_tm" if "psbd_tm" in row["gap"]["ours"]["fractional"] else "psbd_rd"
        )
        for form in ("fractional", "absolute"):
            gap = row["gap"]["ours"][form][method]
            nominal = row["readings"]["ours"][form][method]["nominal"]
            cells = [
                number(nominal[q]["fpr_clean_train"])
                for q in ("q0.01", "q0.05", "q0.10", "q0.25")
            ]
            lines.append(
                f"| {model_words(row)}, {METHOD_WORDS[method]} | {form} | "
                f"{number(gap['median_clean_train'])} | {number(gap['median_validation'])} | "
                f"{number(gap['ks_statistic'])} | " + " | ".join(cells) + " |"
            )
    table = "\n".join(lines)
    return table


def test_time_table(rows, folders):
    """AUROC on training images against AUROC on the paired test splits."""
    methods = ("final_min", "psbd_tm", "psbd_rd", "strip", "cd_l")
    header = (
        "| model | "
        + " | ".join(f"{METHOD_WORDS[m]} train / test" for m in methods)
        + " |"
    )
    lines = [header, "|---" * (len(methods) + 1) + "|"]
    for folder in folders:
        if folder not in rows:
            continue
        row = rows[folder]
        cells = []
        for method in methods:
            if method in ("strip", "cd_l"):
                reading = row["common_subset"].get(method)
                train = reading["auroc"] if reading else None
            else:
                reading = row["readings"]["ours"]["fractional"].get(method)
                train = reading["auroc"] if reading else None
            test = (row["test_time"].get(method) or {}).get("auroc")
            cells.append(f"{number(train)} / {number(test)}")
        lines.append(
            f"| {model_words(row)}{headline_mark(row)} | " + " | ".join(cells) + " |"
        )
    table = "\n".join(lines)
    return table


def rate_table(rows, folders):
    header = (
        "| model | placement | our rate | Li et al.'s rate | Li et al.'s candidates |"
    )
    lines = [header, "|---|---|---|---|---|"]
    for folder in folders:
        if folder not in rows:
            continue
        row = rows[folder]
        for placement, rates in row["rates"].items():
            candidates = ", ".join(f"{r:g}" for r in rates["li_candidates"])
            lines.append(
                f"| {model_words(row)} | {placement} | {number(rates['ours'], 3)} | "
                f"{number(rates['li'], 3)} | {candidates} |"
            )
    table = "\n".join(lines)
    return table


def reconstruction_table(rows, folders):
    header = (
        "| model | successful_2pt | seed recorded | R1 counts | R2 training attack success "
        "(recorded ASR) | R3 ratio | verdict |"
    )
    lines = [header, "|---|---|---|---|---|---|---|"]
    for folder in folders:
        if folder not in rows:
            continue
        row = rows[folder]
        checks = row["reconstruction"]
        r3 = checks["R3"]
        ratio = number(r3["ratio"]) if r3["ratio"] is not None else "n/a"
        lines.append(
            f"| `{folder}` | {'yes' if row['successful_2pt'] else 'no'} | "
            f"{'yes' if checks['seed_recorded'] else 'no'} | "
            f"{'passed' if checks['R1']['passed'] else 'failed'} | "
            f"{number(checks['R2']['training_attack_success'])} "
            f"({number(checks['R2']['recorded_asr'])}) | {ratio}, {r3['status']} | "
            f"{checks['verdict']} |"
        )
    table = "\n".join(lines)
    return table

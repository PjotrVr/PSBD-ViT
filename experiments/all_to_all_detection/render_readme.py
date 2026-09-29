"""Write experiments/all_to_all_detection/README.md from the records the other scripts write.

Every number the README quotes is read here from results/_experiments/
all_to_all_detection/ (panel.json, mechanism.json, tokens.json,
detection_development.json, detection.json) or from a constant of the scripts, so
the README cannot disagree with the run it describes.

    source .venv/bin/activate
    python experiments/all_to_all_detection/render_readme.py
"""

import glob
import json
import textwrap
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import detect  # noqa: E402
import tokens  # noqa: E402
from panel import CONFIRMATION_DATASETS, DEVELOPMENT_DATASETS, OUT_DIR  # noqa: E402

README = os.path.join(HERE, "README.md")
PROSE_WIDTH = 60
STATISTIC_LABELS = {
    "psu_tm": "PSU, PSBD-TM",
    "psu_rd": "PSU, PSBD-RD",
    "neg_entropy": "negative entropy (control)",
    "confidence": "confidence (control)",
    "psu_tm_two_sided": "PSU-TM, 2-sided",
    "psu_tm_class_two_sided": "PSU-TM, class-conditional 2-sided",
    "destination_concentration": "destination concentration",
    "transition_typicality": "transition typicality",
    "late_fragility": "late fragility",
    "depth_profile": "depth profile",
    "either_regime": "either regime",
    "router_entropy": "router, entropy (H43)",
    "router_late": "router, late fragility",
}
COST_STATISTICS = (
    "neg_entropy",
    "transition_typicality",
    "late_fragility",
    "depth_profile",
    "either_regime",
    "router_entropy",
    "router_late",
)
MODEL_STATISTICS = ("psu_tm", "neg_entropy", "late_fragility", "either_regime")
MECHANISM_ROWS = {
    "vit": (
        ("before_attention_norm_token_mask", "adaptive", "PSBD-TM, adaptive rate"),
        ("post_residual", "adaptive", "PSBD-RD, adaptive rate"),
        ("pre_residual_blocks_1_4", "matched_0.6", "dropout, blocks 1 to 4"),
        ("pre_residual_blocks_5_8", "matched_0.6", "dropout, blocks 5 to 8"),
        ("pre_residual_blocks_9_12", "matched_0.6", "dropout, blocks 9 to 12"),
    ),
    "swin": (
        ("before_attention_norm_token_mask", "adaptive", "PSBD-TM, adaptive rate"),
        ("post_residual", "adaptive", "PSBD-RD, adaptive rate"),
        ("pre_residual_blocks_1_8", "matched_0.6", "dropout, blocks 1 to 8"),
        ("pre_residual_blocks_9_16", "matched_0.6", "dropout, blocks 9 to 16"),
        ("pre_residual_blocks_17_24", "matched_0.6", "dropout, blocks 17 to 24"),
    ),
}
TOKEN_ROWS = (
    ("baseline", "nothing masked"),
    ("random_all_12", "random tokens, as many as the trigger, blocks 1 to 12"),
    ("trigger_blocks_all_12", "trigger tokens, blocks 1 to 12"),
    ("trigger_blocks_blocks_1_4", "trigger tokens, blocks 1 to 4"),
    ("trigger_blocks_blocks_5_8", "trigger tokens, blocks 5 to 8"),
    ("trigger_blocks_blocks_9_12", "trigger tokens, blocks 9 to 12"),
    ("trigger_blocks_blocks_1_8", "trigger tokens, blocks 1 to 8"),
    ("trigger_blocks_blocks_5_12", "trigger tokens, blocks 5 to 12"),
    ("content_keep_0.6", "trigger visible, 60% of content visible"),
    ("content_keep_0.3", "trigger visible, 30% of content visible"),
    ("content_keep_0.1", "trigger visible, 10% of content visible"),
    ("content_keep_0.0", "trigger visible, no content visible"),
)


def main():
    records = {
        name: json.load(open(os.path.join(OUT_DIR, f"{name}.json")))
        for name in (
            "panel",
            "mechanism",
            "tokens",
            "detection_development",
            "detection",
        )
    }
    values = placeholder_values(records)
    text = TEMPLATE
    for name, value in values.items():
        text = text.replace(f"@@{name}@@", value)
    assert "@@" not in text, (
        f"unfilled placeholder near {text[text.index('@@') - 60 :][:120]}"
    )
    text = wrap_prose(text)
    with open(README, "w") as handle:
        handle.write(text)
    print(README)


# Prose paragraphs are wrapped to short lines, since scripts/check_stale_numbers.py
# reads line by line and a long paragraph line puts unrelated numbers beside a
# headline word. Tables, headers, images, comments, lists and code stay as they are.
def wrap_prose(text):
    lines = []
    for line in text.split("\n"):
        is_prose = line and not line.startswith(("|", "#", "!", "<", " ", "- "))
        lines.extend(
            textwrap.wrap(line, PROSE_WIDTH, break_on_hyphens=False)
            if is_prose
            else [line]
        )
    wrapped = "\n".join(lines)
    return wrapped


def fmt(value, places=3, signed=False):
    if value is None:
        return "--"
    text = f"{value:+.{places}f}" if signed else f"{value:.{places}f}"
    return text


def interval(pair, signed=False):
    text = f"[{fmt(pair[0], signed=signed)}, {fmt(pair[1], signed=signed)}]"
    return text


def placeholder_values(records):
    panel = records["panel"]
    mechanism = records["mechanism"]["summary"]
    development = records["detection_development"]
    detection = records["detection"]
    token_models = records["tokens"]["models"]
    bars = panel["bars"]

    values = {
        "asr_bar": fmt(bars["asr_bar"], 2),
        "relaxed_bar": fmt(bars["relaxed_asr_bar"], 2),
        "points_2": fmt(-100 * bars["clean_drop_bar_2pt"], 0),
        "points_5": fmt(-100 * bars["clean_drop_bar_5pt"], 0),
        "development": " and ".join(DEVELOPMENT_DATASETS),
        "confirmation": " and ".join(CONFIRMATION_DATASETS),
        "resamples": str(detection["bootstrap"]["resamples"]),
        "seed": str(detection["bootstrap"]["seed"]),
        "budgets": ", ".join(fmt(b, 2) for b in detection["fpr_budgets"]),
        "late_band_vit": detection["late_band"]["vit"],
        "late_band_swin": detection["late_band"]["swin"],
        "late_target": fmt(detection["late_band_target"], 1),
        "token_pairs": str(tokens.PAIR_COUNT),
        "token_memory": fmt(tokens.GPU_MEMORY_FRACTION, 2),
        "smoothing": fmt(detect.TRANSITION_SMOOTHING, 1),
        "shrinkage": fmt(detect.CLASS_SHRINKAGE, 0),
        "chosen": detection["chosen_detector"],
        "panel_table": panel_table(panel),
        "side_table": side_table(panel),
        "fragility_vit": mechanism_table(mechanism, "vit", fragility_columns()),
        "fragility_swin": mechanism_table(mechanism, "swin", fragility_columns()),
        "destinations_vit": mechanism_table(mechanism, "vit", destination_columns()),
        "destinations_swin": mechanism_table(mechanism, "swin", destination_columns()),
        "tokens_table": token_table(token_models),
        "tokens_model_table": token_model_table(token_models),
        "development_table": detector_table(
            development, "vit/all_to_all_strict_2pt/development"
        ),
        "development_swin_table": detector_table(
            development, "swin/all_to_all_strict_2pt/development"
        ),
        "confirmation_table": detector_table(
            detection, "vit/all_to_all_relaxed_2pt/confirmation"
        ),
        "confirmation_swin_table": detector_table(
            detection, "swin/all_to_all_relaxed_2pt/confirmation"
        ),
        "all_strict_table": detector_table(detection, "vit/all_to_all_strict_2pt/all"),
        "all_strict5_table": detector_table(detection, "vit/all_to_all_strict_5pt/all"),
        "all_relaxed_table": detector_table(
            detection, "vit/all_to_all_relaxed_2pt/all"
        ),
        "swin_strict_table": detector_table(
            detection, "swin/all_to_all_strict_2pt/all"
        ),
        "swin_relaxed_table": detector_table(
            detection, "swin/all_to_all_relaxed_2pt/all"
        ),
        "cost_vit": cost_table(detection, "vit/all_to_one/all"),
        "cost_swin": cost_table(detection, "swin/all_to_one/all"),
        "benign_table": benign_table(detection),
        "side_detection_table": side_detection_table(detection),
        "model_table": model_table(detection, panel),
        "wall_time": wall_time(records),
    }
    # Every table is a generated results block, which scripts/check_stale_numbers.py
    # skips: its rows are per-model data that can equal an old headline by chance.
    values = {
        name: wrap_results(value) if value.startswith("|") else value
        for name, value in values.items()
    }
    values.update(prose_values(records))
    return values


def wrap_results(table):
    block = f"<!-- results:begin -->\n{table}\n<!-- results:end -->"
    return block


def verdict_of(row):
    if row.get("strict_2pt"):
        verdict = f"successful at the {row_bar(2)}"
    elif row.get("strict_5pt"):
        verdict = f"successful at the {row_bar(5)} only"
    elif row.get("relaxed_2pt"):
        verdict = "relaxed ASR bar only"
    else:
        verdict = f"dropped: {row['reason']}"
    return verdict


def row_bar(points):
    text = f"{points}-point bar"
    return text


def panel_table(panel):
    lines = [
        "| model | ASR | clean accuracy | benign reference | verdict | PSBD-TM cache |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in panel["all_to_all"]:
        if row["variant"] is not None:
            continue
        cached = "yes" if row["cached"]["before_attention_norm_token_mask"] else "no"
        lines.append(
            f"| `{row['folder']}` | {fmt(row['asr'])} | {fmt(row['clean_accuracy'])} "
            f"| {fmt(row['benign_reference'])} | {verdict_of(row)} | {cached} |"
        )
    table = "\n".join(lines)
    return table


def side_table(panel):
    lines = [
        "| variant | models | ASR bar, 2 points | relaxed bar, 2 points | dropped |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("sam_rho", "evade"):
        rows = [row for row in panel["all_to_all"] if row["variant"] == variant]
        strict = sum(1 for row in rows if row.get("strict_2pt"))
        relaxed = sum(1 for row in rows if row.get("relaxed_2pt"))
        dropped = sum(1 for row in rows if not row.get("relaxed_2pt"))
        lines.append(
            f"| `{variant}` | {len(rows)} | {strict} | {relaxed} | {dropped} |"
        )
    table = "\n".join(lines)
    return table


def fragility_columns():
    columns = (
        ("all_to_all", "triggered_shift_rate", "all-to-all triggered"),
        ("all_to_all", "clean_shift_rate", "all-to-all clean twin"),
        ("all_to_one", "triggered_shift_rate", "all-to-one triggered"),
        ("all_to_one", "clean_shift_rate", "all-to-one clean twin"),
        ("all_to_all", "shift_count_spearman", "all-to-all Spearman"),
        ("all_to_one", "shift_count_spearman", "all-to-one Spearman"),
    )
    return columns


def destination_columns():
    columns = (
        ("all_to_all", "triggered_shifts_to_source", "all-to-all triggered to source"),
        ("all_to_one", "triggered_shifts_to_source", "all-to-one triggered to source"),
        ("all_to_all", "clean_shifts_to_one_below", "all-to-all clean to class below"),
        ("all_to_all", "uniform_destination_chance", "uniform chance"),
        ("all_to_all", "triggered_same_destination", "same class, triggered"),
        ("all_to_all", "clean_same_destination", "same class, clean"),
    )
    return columns


def mechanism_table(summary, architecture, columns):
    header = "| probe | models (all-to-all, all-to-one) | " + " | ".join(
        label for _, _, label in columns
    )
    lines = [header + " |", "|---|---|" + "---:|" * len(columns)]
    for placement, rule, label in MECHANISM_ROWS[architecture]:
        blocks = {
            mode: summary.get(f"{architecture}/{mode}/{placement}/{rule}")
            for mode in ("all_to_all", "all_to_one")
        }
        counts = ", ".join(str(b["n"]) if b else "0" for b in blocks.values())
        cells = [
            fmt(blocks[mode][field]) if blocks[mode] else "--"
            for mode, field, _ in columns
        ]
        lines.append(f"| {label} | {counts} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    return table


def token_means(models, label_mode):
    members = [m for m in models if m["label_mode"] == label_mode]
    means = (
        {
            condition: {
                field: sum(m["conditions"][condition][field] for m in members)
                / len(members)
                for field in members[0]["conditions"][condition]
            }
            for condition in members[0]["conditions"]
        }
        if members
        else {}
    )
    return means, members


def token_table(models):
    all_to_all, a2a_members = token_means(models, "all_to_all")
    all_to_one, a2o_members = token_means(models, "all_to_one")
    lines = [
        f"| tokens masked at the attention input | all-to-all ({len(a2a_members)} models): "
        "attack label, source, elsewhere | all-to-all clean accuracy | "
        f"all-to-one ({len(a2o_members)} models): target, source, elsewhere |",
        "|---|---|---:|---|",
    ]
    for condition, label in TOKEN_ROWS:
        a2a = all_to_all.get(condition)
        a2o = all_to_one.get(condition)
        a2a_cell = triple(a2a)
        a2o_cell = triple(a2o)
        accuracy = fmt(a2a["clean_accuracy"]) if a2a else "--"
        lines.append(f"| {label} | {a2a_cell} | {accuracy} | {a2o_cell} |")
    table = "\n".join(lines)
    return table


TOKEN_SPANS = (
    ("trigger_blocks_blocks_1_4", "1 to 4"),
    ("trigger_blocks_blocks_5_8", "5 to 8"),
    ("trigger_blocks_blocks_9_12", "9 to 12"),
    ("trigger_blocks_blocks_1_8", "1 to 8"),
    ("trigger_blocks_blocks_5_12", "5 to 12"),
    ("trigger_blocks_all_12", "1 to 12"),
)


# Per model, the share of captured triggered images that leave the attack label
# when the trigger's tokens are masked in each span, so the span holding the
# trigger read is visible model by model.
def token_model_table(models):
    lines = [
        "| model | label mode | captured pairs | "
        + " | ".join(
            f"left attack label, trigger masked {label}" for _, label in TOKEN_SPANS
        )
        + " | returned to source, trigger masked 5 to 12 |",
        "|---|---|---:|" + "---:|" * (len(TOKEN_SPANS) + 1),
    ]
    ordered = sorted(
        models, key=lambda m: (m["label_mode"] != "all_to_all", m["folder"])
    )
    for model in ordered:
        conditions = model["conditions"]
        cells = [
            fmt(1 - conditions[condition]["triggered_on_attack_label"])
            for condition, _ in TOKEN_SPANS
        ]
        returned = fmt(conditions["trigger_blocks_blocks_5_12"]["triggered_on_source"])
        lines.append(
            f"| `{model['folder']}` | {model['label_mode'].replace('_', '-')} "
            f"| {model['captured_pairs']} | " + " | ".join(cells) + f" | {returned} |"
        )
    table = "\n".join(lines)
    return table


def triple(reading):
    if not reading:
        return "--"
    text = ", ".join(
        fmt(reading[field])
        for field in (
            "triggered_on_attack_label",
            "triggered_on_source",
            "triggered_elsewhere",
        )
    )
    return text


def detector_table(detection, key):
    block = detection["summary"].get(key)
    if block is None:
        return "No model in this population has the caches it needs."
    # detection_development.json predates the field, and its budgets are detect's.
    budgets = detection.get("fpr_budgets", list(detect.FPR_BUDGETS))
    lines = [
        "| statistic | n | mean AUROC [95% CI] | min | TPR at "
        + ", ".join(fmt(b, 2) for b in budgets)
        + " FPR | realized clean FPR | gain over PSU-TM [95% CI] |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for name in detection["statistics"]:
        if name not in block:
            continue
        entry = block[name]
        tprs = ", ".join(fmt(entry[f"fpr_{b:.2f}"]["tpr"]) for b in budgets)
        fprs = ", ".join(
            fmt(entry[f"fpr_{b:.2f}"]["realized_clean_fpr"]) for b in budgets
        )
        lines.append(
            f"| {STATISTIC_LABELS[name]} | {entry['n']} | {fmt(entry['mean_auroc'])} "
            f"{interval(entry['mean_auroc_ci'])} | {fmt(entry['min_auroc'])} | {tprs} "
            f"| {fprs} | {fmt(entry['gain_over_psu_tm'], signed=True)} "
            f"{interval(entry['gain_over_psu_tm_ci'], signed=True)} |"
        )
    table = "\n".join(lines)
    return table


def cost_table(detection, key):
    block = detection["summary"][key]
    budgets = detection["fpr_budgets"]
    lines = [
        "| statistic | n | AUROC | PSU-TM, same models | change [95% CI] | TPR at "
        + ", ".join(fmt(b, 2) for b in budgets)
        + " FPR | PSU-TM TPR, same models | realized clean FPR |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for name in COST_STATISTICS:
        if name not in block:
            continue
        entry = block[name]
        tprs = ", ".join(fmt(entry[f"fpr_{b:.2f}"]["tpr"]) for b in budgets)
        baseline = ", ".join(
            fmt(entry["psu_tm_same_models_tpr"][f"fpr_{b:.2f}"]) for b in budgets
        )
        fprs = ", ".join(
            fmt(entry[f"fpr_{b:.2f}"]["realized_clean_fpr"]) for b in budgets
        )
        lines.append(
            f"| {STATISTIC_LABELS[name]} | {entry['n']} | {fmt(entry['mean_auroc'])} "
            f"| {fmt(entry['psu_tm_same_models'])} "
            f"| {fmt(entry['gain_over_psu_tm'], signed=True)} "
            f"{interval(entry['gain_over_psu_tm_ci'], signed=True)} | {tprs} | {baseline} "
            f"| {fprs} |"
        )
    table = "\n".join(lines)
    return table


def benign_table(detection):
    lines = [
        "| statistic | ViT models | ViT mean AUROC | Swin models | Swin mean AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in detection["statistics"]:
        cells = []
        for architecture in ("vit", "swin"):
            block = detection["summary"].get(f"{architecture}/benign/all", {})
            entry = block.get(name)
            cells += [
                str(entry["n"]) if entry else "0",
                fmt(entry["mean_auroc"]) if entry else "--",
            ]
        lines.append(f"| {STATISTIC_LABELS[name]} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    return table


def side_detection_table(detection):
    lines = [
        "| population | statistic | n | mean AUROC [95% CI] |",
        "|---|---|---:|---|",
    ]
    for architecture in ("vit", "swin"):
        block = detection["summary"].get(f"{architecture}/all_to_all_side/all")
        if block is None:
            continue
        for name in ("psu_tm", "neg_entropy", "late_fragility", "either_regime"):
            if name not in block:
                continue
            entry = block[name]
            lines.append(
                f"| {architecture}, SAM and evasion variants | {STATISTIC_LABELS[name]} "
                f"| {entry['n']} | {fmt(entry['mean_auroc'])} {interval(entry['mean_auroc_ci'])} |"
            )
    table = "\n".join(lines)
    return table


def model_table(detection, panel):
    verdicts = {row["folder"]: row for row in panel["all_to_all"]}
    lines = [
        "| model | split | verdict | ASR | "
        + " | ".join(STATISTIC_LABELS[name] for name in MODEL_STATISTICS)
        + " |",
        "|---|---|---|---:|" + "---:|" * len(MODEL_STATISTICS),
    ]
    for model in detection["models"]:
        if model["group"] != "all_to_all":
            continue
        split = (
            "development"
            if model["dataset"] in DEVELOPMENT_DATASETS
            else "confirmation"
        )
        row = verdicts[model["folder"]]
        cells = [
            fmt(model["metrics"][name]["auroc"]) if name in model["metrics"] else "--"
            for name in MODEL_STATISTICS
        ]
        lines.append(
            f"| `{model['folder']}` | {split} | {verdict_of(row)} | {fmt(row['asr'])} | "
            + " | ".join(cells)
            + " |"
        )
    table = "\n".join(lines)
    return table


def wall_time(records):
    sweeps = []
    for folder in sorted(glob.glob(os.path.join(OUT_DIR, "caches", "*", "psbd", "*"))):
        if not os.path.isdir(folder):
            continue
        stamps = [
            os.path.getmtime(path)
            for path in glob.glob(os.path.join(folder, "rate_*.pt"))
        ]
        if stamps:
            sweeps.append(max(stamps) - min(stamps))
    token_seconds = [m["seconds"] for m in records["tokens"]["models"]]
    lines = [
        "| step | where | wall time |",
        "|---|---|---|",
        f"| `panel.py` and `mechanism.py` | CPU | {fmt(records['mechanism']['seconds'] / 60, 1)} min |",
        f"| `detect.py`, every group | CPU | {fmt(records['detection']['seconds'] / 60, 1)} min |",
        f"| `tokens.py`, {len(token_seconds)} models | login GPU | "
        f"{fmt(sum(token_seconds) / 60, 1)} min of forward passes |",
        f"| `run_gpu.sh`, {len(sweeps)} sweeps | login GPU | "
        f"{fmt(sum(sweeps) / 60, 1)} min from the first to the last rate file, lock waits excluded |",
    ]
    table = "\n".join(lines)
    return table


def summary_entry(detection, key, name):
    block = detection["summary"].get(key, {})
    entry = block.get(name)
    return entry


def prose_values(records):
    mechanism = records["mechanism"]["summary"]
    development = records["detection_development"]
    detection = records["detection"]
    token_models = records["tokens"]["models"]
    a2a_tokens, _ = token_means(token_models, "all_to_all")
    a2o_tokens, _ = token_means(token_models, "all_to_one")

    def mech(architecture, mode, placement, rule, field):
        block = mechanism.get(f"{architecture}/{mode}/{placement}/{rule}")
        value = fmt(block[field]) if block and block[field] is not None else "--"
        return value

    def stat(source, key, name, field="mean_auroc"):
        entry = summary_entry(source, key, name)
        value = fmt(entry[field]) if entry else "--"
        return value

    def tpr(source, key, name, budget):
        entry = summary_entry(source, key, name)
        value = fmt(entry[f"fpr_{budget:.2f}"]["tpr"]) if entry else "--"
        return value

    def token(means, condition, field):
        value = fmt(means[condition][field]) if condition in means else "--"
        return value

    tm = ("before_attention_norm_token_mask", "adaptive")
    late = ("pre_residual_blocks_9_12", "matched_0.6")
    swin_late = ("pre_residual_blocks_17_24", "matched_0.6")
    dev_key = "vit/all_to_all_strict_2pt/development"
    confirm_key = "vit/all_to_all_relaxed_2pt/confirmation"
    cost_key = "vit/all_to_one/all"
    values = {
        "m_tm_trig": mech("vit", "all_to_all", *tm, "triggered_shift_rate"),
        "m_tm_clean": mech("vit", "all_to_all", *tm, "clean_shift_rate"),
        "m_late_trig": mech("vit", "all_to_all", *late, "triggered_shift_rate"),
        "m_late_clean": mech("vit", "all_to_all", *late, "clean_shift_rate"),
        "m_late_a2o_trig": mech("vit", "all_to_one", *late, "triggered_shift_rate"),
        "m_late_a2o_clean": mech("vit", "all_to_one", *late, "clean_shift_rate"),
        "m_tm_a2o_trig": mech("vit", "all_to_one", *tm, "triggered_shift_rate"),
        "m_swin_tm_trig": mech("swin", "all_to_all", *tm, "triggered_shift_rate"),
        "m_swin_tm_clean": mech("swin", "all_to_all", *tm, "clean_shift_rate"),
        "m_swin_late_trig": mech(
            "swin", "all_to_all", *swin_late, "triggered_shift_rate"
        ),
        "m_swin_late_clean": mech("swin", "all_to_all", *swin_late, "clean_shift_rate"),
        "m_tm_source": mech("vit", "all_to_all", *tm, "triggered_shifts_to_source"),
        "m_late_source": mech("vit", "all_to_all", *late, "triggered_shifts_to_source"),
        "m_late_chance": mech("vit", "all_to_all", *late, "uniform_destination_chance"),
        "m_late_below": mech("vit", "all_to_all", *late, "clean_shifts_to_one_below"),
        "m_swin_late_source": mech(
            "swin", "all_to_all", *swin_late, "triggered_shifts_to_source"
        ),
        "m_swin_late_chance": mech(
            "swin", "all_to_all", *swin_late, "uniform_destination_chance"
        ),
        "m_tm_same_trig": mech("vit", "all_to_all", *tm, "triggered_same_destination"),
        "m_tm_same_clean": mech("vit", "all_to_all", *tm, "clean_same_destination"),
        "m_tm_class_trig": mech(
            "vit", "all_to_all", *tm, "triggered_per_class_top_share"
        ),
        "m_tm_class_clean": mech("vit", "all_to_all", *tm, "clean_per_class_top_share"),
        "m_tm_spearman": mech("vit", "all_to_all", *tm, "shift_count_spearman"),
        "m_runner_up_source": mech(
            "vit", "all_to_all", *tm, "triggered_runner_up_is_source"
        ),
        "t_mid_source": token(
            a2a_tokens, "trigger_blocks_blocks_5_12", "triggered_on_source"
        ),
        "t_30_attack": token(
            a2a_tokens, "content_keep_0.3", "triggered_on_attack_label"
        ),
        "t_30_clean": token(a2a_tokens, "content_keep_0.3", "clean_accuracy"),
        "t_late_source": token(
            a2a_tokens, "trigger_blocks_blocks_9_12", "triggered_on_source"
        ),
        "t_early_attack": token(
            a2a_tokens, "trigger_blocks_blocks_1_8", "triggered_on_attack_label"
        ),
        "t_all_source": token(
            a2a_tokens, "trigger_blocks_all_12", "triggered_on_source"
        ),
        "t_random_attack": token(
            a2a_tokens, "random_all_12", "triggered_on_attack_label"
        ),
        "t_none_attack": token(
            a2a_tokens, "content_keep_0.0", "triggered_on_attack_label"
        ),
        "t_none_elsewhere": token(
            a2a_tokens, "content_keep_0.0", "triggered_elsewhere"
        ),
        "t_a2o_none_attack": token(
            a2o_tokens, "content_keep_0.0", "triggered_on_attack_label"
        ),
        "t_a2o_late_source": token(
            a2o_tokens, "trigger_blocks_blocks_9_12", "triggered_on_source"
        ),
        "d_dev_n": str(summary_entry(development, dev_key, "psu_tm")["n"]),
        "d_dev_psu": stat(development, dev_key, "psu_tm"),
        "d_dev_entropy": stat(development, dev_key, "neg_entropy"),
        "d_dev_late": stat(development, dev_key, "late_fragility"),
        "d_dev_either": stat(development, dev_key, "either_regime"),
        "d_dev_depth": stat(development, dev_key, "depth_profile"),
        "d_dev_concentration": stat(development, dev_key, "destination_concentration"),
        "d_dev_transition": stat(development, dev_key, "transition_typicality"),
        "d_dev_two_sided": stat(development, dev_key, "psu_tm_two_sided"),
        "d_dev_class": stat(development, dev_key, "psu_tm_class_two_sided"),
        "d_dev_router": stat(development, dev_key, "router_entropy"),
        "d_dev_router_late": stat(development, dev_key, "router_late"),
        "d_dev_cost_either": stat(
            development, "vit/all_to_one/development", "either_regime"
        ),
        "d_dev_cost_psu": stat(development, "vit/all_to_one/development", "psu_tm"),
        "c_n": str((summary_entry(detection, confirm_key, "psu_tm") or {}).get("n", 0)),
        "c_late_n": str(
            (summary_entry(detection, confirm_key, "late_fragility") or {}).get("n", 0)
        ),
        "c_psu": stat(detection, confirm_key, "psu_tm"),
        "c_entropy": stat(detection, confirm_key, "neg_entropy"),
        "c_late": stat(detection, confirm_key, "late_fragility"),
        "c_either": stat(detection, confirm_key, "either_regime"),
        "c_either_min": stat(detection, confirm_key, "either_regime", "min_auroc"),
        "c_late_tpr_1": tpr(detection, confirm_key, "late_fragility", 0.01),
        "c_late_tpr_5": tpr(detection, confirm_key, "late_fragility", 0.05),
        "c_either_tpr_1": tpr(detection, confirm_key, "either_regime", 0.01),
        "c_either_tpr_5": tpr(detection, confirm_key, "either_regime", 0.05),
        "c_either_tpr_10": tpr(detection, confirm_key, "either_regime", 0.10),
        "a_strict_either": stat(
            detection, "vit/all_to_all_strict_2pt/all", "either_regime"
        ),
        "a_strict_late": stat(
            detection, "vit/all_to_all_strict_2pt/all", "late_fragility"
        ),
        "a_strict_psu": stat(detection, "vit/all_to_all_strict_2pt/all", "psu_tm"),
        "a_strict_either_tpr_5": tpr(
            detection, "vit/all_to_all_strict_2pt/all", "either_regime", 0.05
        ),
        "m_a2o_late_source": mech(
            "vit", "all_to_one", *late, "triggered_shifts_to_source"
        ),
        "s_strict_late": stat(
            detection, "swin/all_to_all_strict_2pt/all", "late_fragility"
        ),
        "s_strict_psu": stat(detection, "swin/all_to_all_strict_2pt/all", "psu_tm"),
        "s_strict_n": str(
            summary_entry(
                detection, "swin/all_to_all_strict_2pt/all", "late_fragility"
            )["n"]
        ),
        "s_either": stat(detection, "swin/all_to_all_relaxed_2pt/all", "either_regime"),
        "s_late": stat(detection, "swin/all_to_all_relaxed_2pt/all", "late_fragility"),
        "s_psu": stat(detection, "swin/all_to_all_relaxed_2pt/all", "psu_tm"),
        "s_n": str(
            (
                summary_entry(
                    detection, "swin/all_to_all_relaxed_2pt/all", "either_regime"
                )
                or {}
            ).get("n", 0)
        ),
        "cost_n": str(summary_entry(detection, cost_key, "either_regime")["n"]),
        "cost_either": stat(detection, cost_key, "either_regime"),
        "cost_psu": stat(detection, cost_key, "either_regime", "psu_tm_same_models"),
        "cost_gain": fmt(
            summary_entry(detection, cost_key, "either_regime")["gain_over_psu_tm"],
            signed=True,
        ),
        "cost_gain_ci": interval(
            summary_entry(detection, cost_key, "either_regime")["gain_over_psu_tm_ci"],
            signed=True,
        ),
        "cost_either_tpr_1": tpr(detection, cost_key, "either_regime", 0.01),
        "cost_psu_tpr_1": fmt(
            summary_entry(detection, cost_key, "either_regime")[
                "psu_tm_same_models_tpr"
            ]["fpr_0.01"]
        ),
        "cost_entropy": stat(detection, cost_key, "neg_entropy"),
        "cost_late": stat(detection, cost_key, "late_fragility"),
        "cost_router": stat(detection, cost_key, "router_late"),
        "swin_cost_either": stat(detection, "swin/all_to_one/all", "either_regime"),
        "swin_cost_psu": stat(
            detection, "swin/all_to_one/all", "either_regime", "psu_tm_same_models"
        ),
        "benign_either": stat(detection, "vit/benign/all", "either_regime"),
        "benign_late": stat(detection, "vit/benign/all", "late_fragility"),
    }
    return values


TEMPLATE = open(os.path.join(HERE, "README.template.md")).read()


if __name__ == "__main__":
    main()

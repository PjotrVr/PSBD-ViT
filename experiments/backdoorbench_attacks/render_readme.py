"""Render the results section of experiments/backdoorbench_attacks/README.md from the records.

Every number between the results markers is read here from inventory.json, the
evaluation records, summary.json and the per-model records under
results/_experiments/backdoorbench_attacks/, and every verdict is computed here
from the bars PREDICTIONS.md fixed before the sweeps. Run it after readout.py.

    .venv/bin/python -m experiments.backdoorbench_attacks.render_readme
"""

import os

from experiments.backdoorbench_attacks.common import (
    EVALUATION_DIR,
    INVENTORY_PATH,
    MODEL_RECORDS_DIR,
    SUMMARY_PATH,
    read_json,
)
from experiments.backdoorbench_attacks.queue import MODELS

README = os.path.join(os.path.dirname(__file__), "README.md")
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
# The bars PREDICTIONS.md fixed before any sweep.
PASS_AUROC = 0.9
PASS_TPR_AT_5 = 0.5
FAIL_AUROC = 0.75
T2_TOLERANCE = 0.05
METHOD_WORDS = {
    "psbd_tm": "PSBD-TM",
    "final_min": "final method, minimum",
    "final_mean": "final method, average",
    "partner": "pre_residual_blocks_5_8 alone",
    "psbd_rd": "PSBD-RD",
}
ATTACK_WORDS = {
    "trojannn": "TrojanNN",
    "ssba": "SSBA",
    "inputaware": "Input-Aware",
    "blind": "Blind",
    "lira": "LIRA",
}


def main():
    inventory = read_json(INVENTORY_PATH)
    summary = read_json(SUMMARY_PATH)
    rows = [
        read_json(os.path.join(MODEL_RECORDS_DIR, f"{folder}.json"))
        for folder in summary["models"]
    ]

    sections = [
        inventory_section(inventory),
        evaluation_section(rows, summary),
        detection_section(rows),
        attack_means_section(summary),
        verdict_section(rows),
        timing_section(rows),
    ]
    body = "\n\n".join(sections)
    write_between_markers(README, body)
    print(f"rendered {README}")


def inventory_section(inventory):
    models = {row["folder"]: row for row in inventory["models"]}
    empty = [
        row["folder"] for row in inventory["models"] if row["status"] != "readable"
    ]
    lines = [
        "## Inventory",
        "",
        f"`backdoor_bench_checkpoints/` holds {len(models)} folders. "
        f"{len(empty)} of them carry no `attack_result.pt` and cannot be read: "
        + ", ".join(f"`{folder}`" for folder in empty)
        + ". Every readable folder is a `vit_b_16` whose bd_test labels name a single "
        "target. The table lists the in-scope models, read from `inventory.json`. "
        "Aligned is the share of bd_test entries whose recorded original label equals "
        "this project's test label at that index. The footprint columns are shares of "
        f"{inventory['footprint_samples']} sampled triggered images against their clean "
        f"twins, a pixel counting as changed above {inventory['changed_levels']} of 255 "
        "levels. Consistent is the share of pixels changed on at least half the images, "
        "high for a fixed trigger and near 0 for a sample-specific one.",
        "",
        "| model | label mode | target | bd_test images | format | aligned | pixels changed | tokens touched | consistent | mean change (levels) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for folder in MODELS:
        row = models.get(folder)
        if row is None or row["status"] != "readable":
            lines.append(f"| `{folder}` | no checkpoint | | | | | | | | |")
            continue
        footprint = row["footprint"]
        lines.append(
            f"| `{folder}` | {row['label_mode']} | {row['target_label']} | {row['n_bd_test']} "
            f"| {row['bd_test_format']} | {row['label_alignment']:.4f} "
            f"| {footprint['changed_pixel_share']:.3f} | {footprint['changed_token_share']:.3f} "
            f"| {footprint['consistent_pixel_share']:.3f} | {footprint['mean_abs_levels']:.2f} |"
        )
    section = "\n".join(lines)
    return section


def evaluation_section(rows, summary):
    references = summary["references"]
    lines = [
        "## Reproduction and success bar",
        "",
        "Clean accuracy is over the whole test set and ASR over every eligible bd_test "
        "image, both through `models.backbones.load_checkpoint` and "
        "`evaluation.metrics` with BackdoorBench's normalization "
        "(`evaluation/<folder>.json`). The leaderboard columns are BackdoorBench's own "
        "no-defense numbers (`leaderboard_vit_b_16.json`). A model is judged when its "
        f"ASR is at least {summary['asr_bar']} and its clean accuracy is at least "
        f"{summary['collapse_share']} of the best BackdoorBench ViT-B/16 of its dataset. "
        "The 2-point verdicts are reported against both references and drop no model.",
        "",
        "| dataset | best BackdoorBench ViT-B/16 clean accuracy | its folder | own benign ViT clean accuracy |",
        "|---|---|---|---|",
    ]
    for dataset, reference in references.items():
        lines.append(
            f"| {dataset} | {reference['backdoorbench_best']:.4f} "
            f"| `{reference['backdoorbench_best_folder']}` | {reference['own_benign']:.4f} |"
        )
    lines += [
        "",
        "| model | clean accuracy | leaderboard clean | ASR | leaderboard ASR | judged | within 2 points of BackdoorBench best | within 2 points of own benign |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "asr" not in row:
            lines.append(f"| `{row['folder']}` | {row['status']} | | | | | | |")
            continue
        board = row["leaderboard"] or {}
        lines.append(
            f"| `{row['folder']}` | {row['clean_accuracy']:.4f} | {fmt(board.get('clean_accuracy'))} "
            f"| {row['asr']:.4f} | {fmt(board.get('asr'))} | {yes(row['judged'])} "
            f"| {yes(row['within_2pt_of_backdoorbench_best'])} | {yes(row['within_2pt_of_own_benign'])} |"
        )

    control_lines = normalization_controls()
    lines += ["", *control_lines]
    section = "\n".join(lines)
    return section


def normalization_controls():
    lines = []
    for name in sorted(os.listdir(EVALUATION_DIR)):
        record = read_json(os.path.join(EVALUATION_DIR, name))
        if "registry_normalization" not in record:
            continue
        ours = record["backdoorbench_normalization"]
        registry = record["registry_normalization"]
        lines.append(
            f"Normalization control on `{record['folder']}`: with BackdoorBench's statistics "
            f"clean accuracy reads {ours['clean_accuracy']:.4f} and ASR {ours['asr']:.4f}, "
            f"with `DATASET_REGISTRY`'s CIFAR-10 statistics {registry['clean_accuracy']:.4f} "
            f"and {registry['asr']:.4f}. The registry's statistics are the ones "
            "`experiments/wanet_cifar10_audit/measure.py` read BackdoorBench's WaNet "
            "checkpoint with, so its `results/bb_cifar10_wanet_0_1` cache was built on "
            "shifted inputs."
        )
    return lines


def detection_section(rows):
    lines = [
        "## Detection per model",
        "",
        "TPR at 1%, 5% and 10% FPR (thresholds at the clean-validation quantile) and "
        "AUROC, each probe at its adaptive rate, read from `models/<folder>.json`. As a "
        "control, the PSBD-TM AUROC must equal the fractional-PSU AUROC `cli.analyze` "
        "stored at the same rate, counted below the table.",
        "",
        "| model | judged | method | rate | TPR at 1% FPR | TPR at 5% FPR | TPR at 10% FPR | AUROC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    controls = []
    for row in rows:
        methods = row.get("detection", {}).get("methods")
        if not methods:
            continue
        rates = row["detection"]["rates"]
        for method, evaluation in methods.items():
            rate = {
                "psbd_tm": rates["psbd_tm"],
                "partner": f"{rates['partner']} ({rates['partner_rule']})",
                "psbd_rd": rates.get("psbd_rd"),
            }.get(method, "both")
            at_fpr = evaluation["at_fpr"]
            lines.append(
                f"| `{row['folder']}` | {yes(row['judged'])} | {METHOD_WORDS[method]} | {rate} "
                f"| {at_fpr['q0.01']['tpr']:.3f} | {at_fpr['q0.05']['tpr']:.3f} "
                f"| {at_fpr['q0.10']['tpr']:.3f} | {evaluation['auroc']:.3f} |"
            )
        stored = row["psbd_metrics_control"]["auroc"]
        controls.append(abs(stored - methods["psbd_tm"]["auroc"]) < 1e-9)
    lines += [
        "",
        f"The control holds on {sum(controls)} of {len(controls)} swept models.",
    ]
    section = "\n".join(lines)
    return section


def attack_means_section(summary):
    lines = [
        "## Detection per attack",
        "",
        "Means over the judged models of each attack (`summary.json`, `by_attack`).",
        "",
        "| attack | judged models | method | TPR at 1% FPR | TPR at 5% FPR | TPR at 10% FPR | AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for attack, groups in summary["by_attack"].items():
        judged = groups["judged"]
        for method in METHOD_WORDS:
            if method not in judged:
                continue
            means = judged[method]
            lines.append(
                f"| {ATTACK_WORDS.get(attack, attack)} | {judged['n']} | {METHOD_WORDS[method]} "
                f"| {means['q0.01:tpr']:.3f} | {means['q0.05:tpr']:.3f} "
                f"| {means['q0.10:tpr']:.3f} | {means['auroc']:.3f} |"
            )
    section = "\n".join(lines)
    return section


def verdict_section(rows):
    judged = [
        row
        for row in rows
        if row.get("judged") and row.get("detection", {}).get("methods")
    ]

    def family(attack, rate=None):
        members = [
            row
            for row in judged
            if row["attack"] == attack and (rate is None or row["poison_rate"] == rate)
        ]
        return members

    lines = [
        "## Verdicts per prediction",
        "",
        f"Bars from `PREDICTIONS.md`: a family passes when every judged model has PSBD-TM "
        f"AUROC of at least {PASS_AUROC} and TPR at 5% FPR of at least {PASS_TPR_AT_5}, and "
        f"fails when every judged model has AUROC below {FAIL_AUROC}. A prediction with no "
        "judged model is untested.",
        "",
        "| prediction | judged models | verdict | reading |",
        "|---|---|---|---|",
    ]
    lines.append(
        row_for(
            "T1, TrojanNN 10% passes", family("trojannn", 0.1), expect_class("pass")
        )
    )
    lines.append(
        row_for(
            "T1, TrojanNN 5% passes", family("trojannn", 0.05), expect_class("pass")
        )
    )
    lines.append(
        row_for(
            "T2, minimum costs TrojanNN at most 0.05 TPR at 5% FPR",
            family("trojannn"),
            t2_holds,
        )
    )
    lines.append(row_for("S1, SSBA partial or fails", family("ssba"), s1_holds(judged)))
    lines.append(
        row_for(
            "S2, minimum raises SSBA TPR at 5% FPR on most", family("ssba"), s2_holds
        )
    )
    lines.append(
        row_for("I1, Input-Aware fails", family("inputaware"), expect_class("fail"))
    )
    lines.append(
        row_for(
            "I2, final method leaves Input-Aware below 0.5 TPR at 5% FPR",
            family("inputaware"),
            i2_holds,
        )
    )
    lines.append(row_for("B1, Blind passes", family("blind"), expect_class("pass")))
    lines.append("| LIRA, partial or fails | 0 | untested, no checkpoint | |")
    lines.append(
        row_for(
            "R1, no patch family fails", family("trojannn") + family("blind"), r1_holds
        )
    )
    section = "\n".join(lines)
    return section


def detection_class(rows):
    tm = [row["detection"]["methods"]["psbd_tm"] for row in rows]
    if all(
        e["auroc"] >= PASS_AUROC and e["at_fpr"]["q0.05"]["tpr"] >= PASS_TPR_AT_5
        for e in tm
    ):
        return "pass"
    if all(e["auroc"] < FAIL_AUROC for e in tm):
        return "fail"
    return "partial"


def expect_class(expected):
    def check(rows):
        observed = detection_class(rows)
        return observed == expected, f"family reads {observed}"

    return check


def t2_holds(rows):
    costs = [
        row["detection"]["methods"]["psbd_tm"]["at_fpr"]["q0.05"]["tpr"]
        - row["detection"]["methods"]["final_min"]["at_fpr"]["q0.05"]["tpr"]
        for row in rows
    ]
    return max(costs) <= T2_TOLERANCE, f"largest cost {max(costs):.3f}"


def s1_holds(judged):
    def check(rows):
        observed = detection_class(rows)
        trojan_low = {
            row["dataset"]: row["detection"]["methods"]["psbd_tm"]["at_fpr"]["q0.01"][
                "tpr"
            ]
            for row in judged
            if row["attack"] == "trojannn"
        }
        lower = [
            row["detection"]["methods"]["psbd_tm"]["at_fpr"]["q0.01"]["tpr"]
            < trojan_low.get(row["dataset"], float("inf"))
            for row in rows
        ]
        holds = observed in ("partial", "fail") and all(lower)
        reading = f"family reads {observed}, TPR at 1% FPR below TrojanNN's on {sum(lower)} of {len(lower)}"
        return holds, reading

    return check


def s2_holds(rows):
    raised = [
        row["detection"]["methods"]["final_min"]["at_fpr"]["q0.05"]["tpr"]
        > row["detection"]["methods"]["psbd_tm"]["at_fpr"]["q0.05"]["tpr"]
        for row in rows
    ]
    return sum(raised) > len(raised) / 2, f"raised on {sum(raised)} of {len(raised)}"


def i2_holds(rows):
    best = [
        max(
            row["detection"]["methods"][method]["at_fpr"]["q0.05"]["tpr"]
            for method in ("final_min", "final_mean")
        )
        for row in rows
    ]
    return all(
        value < 0.5 for value in best
    ), f"best final-method TPR at 5% FPR {max(best):.3f}"


def r1_holds(rows):
    failing = [row["folder"] for row in rows if detection_class([row]) == "fail"]
    return not failing, f"failing: {', '.join(failing) or 'none'}"


def row_for(name, rows, check):
    if not rows:
        line = f"| {name} | 0 | untested | |"
        return line
    holds, reading = check(rows)
    line = f"| {name} | {len(rows)} | {'holds' if holds else 'fails'} | {reading} |"
    return line


def timing_section(rows):
    lines = [
        "## Wall times",
        "",
        "Per model on the shared login A100, including time spent waiting for data "
        "loading but not for the GPU lock (`evaluation/<folder>.json` and "
        "`sweeps/<folder>.json`, `wall_seconds`).",
        "",
        "| model | evaluation (s) | sweep of both probes and analysis (s) |",
        "|---|---|---|",
    ]
    for row in rows:
        evaluation_path = os.path.join(EVALUATION_DIR, f"{row['folder']}.json")
        evaluation_seconds = (
            read_json(evaluation_path)["wall_seconds"]
            if os.path.exists(evaluation_path)
            else None
        )
        lines.append(
            f"| `{row['folder']}` | {fmt(evaluation_seconds, 0)} | {fmt(row.get('sweep_wall_seconds'), 0)} |"
        )
    section = "\n".join(lines)
    return section


def fmt(value, digits=4):
    text = "" if value is None else f"{value:.{digits}f}"
    return text


def yes(flag):
    word = "yes" if flag else "no"
    return word


def write_between_markers(path, body):
    with open(path) as handle:
        text = handle.read()
    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END)
    rendered = text[:start] + "\n\n" + body + "\n\n" + text[end:]
    with open(path, "w") as handle:
        handle.write(rendered)


if __name__ == "__main__":
    main()

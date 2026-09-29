"""Renders the results block of experiments/novel_probes/README.md from the JSON.

Every number in the block is read from results/_experiments/novel_probes/*.json,
and a record that does not exist yet renders as pending.

    .venv/bin/python -m experiments.novel_probes.report
"""

import json
import os

from experiments._paths import experiment_result_path
from experiments.novel_probes.measure import SLUG

README = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def main():
    sections = [band_partner_section(), smoke_section()]
    block = "\n\n".join(section for section in sections if section)

    with open(README) as handle:
        text = handle.read()
    head, rest = text.split(BEGIN)
    _, tail = rest.split(END)
    rendered = f"{head}{BEGIN}\n\n{block}\n\n{END}{tail}"
    with open(README, "w") as handle:
        handle.write(rendered)
    print(f"rendered {README}")


def load(name):
    path = experiment_result_path(SLUG, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def fmt(value, signed=False):
    if value is None:
        return "--"
    text = f"{value:+.3f}" if signed else f"{value:.3f}"
    return text


def band_partner_section():
    payload = load("band_partner.json")
    if payload is None:
        return "**Band partner.** Pending, `band_partner.py` has not run."
    lines = [
        "**Band partner.** Blocks 5 to 8 against blocks 9 to 12 as the min-rank "
        "partner of PSBD-TM, paired on the models where both partners have an "
        "adaptive rate, read from `fusion_rules_<set>.json` (`band_partner.json`).",
        "",
        "| set | n | field | middle | late | middle minus late [95% CI] "
        "| models middle higher, lower | realized FPR middle, late |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model_set, rules in payload["sets"].items():
        comparison = rules["adaptive"]
        for field, block in comparison["fields"].items():
            difference = block["middle_minus_late"]
            low, high = difference["ci95"]
            lines.append(
                f"| {model_set} | {comparison['n']} | {field} "
                f"| {fmt(difference['mean'])} | {fmt(difference['reference_mean'])} "
                f"| {fmt(difference['mean_difference'], True)} "
                f"[{fmt(low, True)}, {fmt(high, True)}] "
                f"| {difference['n_higher']}, {difference['n_lower']} "
                f"| {fmt(block['realized_fpr_middle'])}, "
                f"{fmt(block['realized_fpr_late'])} |"
            )
    section = "\n".join(lines)
    return section


def smoke_section():
    payload = load("dev_smoke.json")
    if payload is None:
        return "**CPU smoke.** Pending, the smoke has not run."
    row = payload["models"][0]
    check = row["runner_check"]
    lines = [
        f"**CPU smoke.** 1 model (`{row['folder']}`), {row['n_backdoor']} "
        "triggered images and their clean twins, the first 32 validation images "
        "and the top 2 rungs of each ladder (`dev_smoke.json`). It tests that the "
        "pipeline runs end to end and says nothing about detection, since a "
        "quantile of 32 validation scores is not a threshold.",
        "",
        f"The unperturbed predictions of the runner agree with the cached ones on "
        f"{fmt(check['baseline_argmax_agreement']['backdoor'])} of the triggered "
        f"rows. The runner's PSBD-TM chose rate {check['fresh_tm_rate']} against "
        f"{check['cached_tm_rate']} in the cache.",
        "",
        "| reading | rate | rule | clean validation shift | AUROC alone "
        "| AUROC in the union with PSBD-TM |",
        "|---|---|---|---|---|---|",
    ]
    for name, probe in row["probes"].items():
        lines.append(
            f"| {name} | {probe['rate']} | {probe['rule']} "
            f"| {fmt(probe['validation_shift'])} | {fmt(probe['standalone']['auroc'])} "
            f"| {fmt(probe['union_tm']['auroc'])} |"
        )
    section = "\n".join(lines)
    return section


if __name__ == "__main__":
    main()

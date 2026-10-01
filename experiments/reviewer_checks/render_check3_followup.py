"""Write the mask-seed follow-up of check 3 into experiments/reviewer_checks/README.md.

Reads results/_experiments/reviewer_checks/check3_followup_2026-10-01.json, which
mask_seed_followup.py writes, and replaces the block between the results markers
below check 3's answer. TPR comes first at the 1%, 5% and 10% quantiles of clean
validation, AUROC after.

    PYTHONPATH=. .venv/bin/python experiments/reviewer_checks/render_check3_followup.py
"""

import json
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
README = os.path.join(EXPERIMENT_DIR, "README.md")
RECORD = os.path.join(
    REPO_ROOT,
    "results",
    "_experiments",
    "reviewer_checks",
    "check3_followup_2026-10-01.json",
)
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
QUANTILE_WORDS = {"q0.01": "1%", "q0.05": "5%", "q0.10": "10%"}


def main():
    with open(RECORD) as handle:
        report = json.load(handle)
    block = render_block(report)
    write_between_markers(README, block)
    print(f"rendered {README}")


def render_block(report):
    lines = [
        BEGIN,
        "<!-- Rendered by render_check3_followup.py from "
        "results/_experiments/reviewer_checks/check3_followup_2026-10-01.json. -->",
        "",
        "**Follow-up of 2026-10-01.** The models that joined the panel after the first "
        "run, rerun with the same protocol by `mask_seed_followup.py`. TPR at the 1%, "
        "5% and 10% quantiles of clean validation first, then AUROC, at seeds 0, 1 "
        "and 2.",
        "",
        "| Model | Rate | Seed | TPR@1% | TPR@5% | TPR@10% | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["models"]:
        for seed, reading in row["by_seed"].items():
            tprs = " | ".join(
                f"{reading['tpr_by_quantile'][key]:.3f}" for key in QUANTILE_WORDS
            )
            lines.append(
                f"| {row['folder']} | {row['rate']} | {seed} | {tprs} "
                f"| {reading['auroc']:.3f} |"
            )
    lines.append("")
    lines += [sentence(row) for row in report["models"]]
    lines.append(END)
    block = "\n".join(lines)
    return block


def sentence(row):
    aurocs = [reading["auroc"] for reading in row["by_seed"].values()]
    tprs_at_ten = [reading["tpr"] for reading in row["by_seed"].values()]
    spread = max(aurocs) - min(aurocs)
    # The decision rule is 1-sided, so an AUROC below 0.5 at every seed is a
    # failure of the method on this model and never a flipped detector.
    fails_everywhere = all(auroc < 0.5 for auroc in aurocs)
    verdict = (
        "PSBD-TM fails on it at every seed, AUROC below 0.5, so its floor is not a "
        "seed-0 accident."
        if fails_everywhere
        else "Its AUROC crosses 0.5 at some seed, so the failure depends on the seed."
    )
    tpr_words = (
        f"TPR at 10% is {tprs_at_ten[0]:.3f} at every seed"
        if min(tprs_at_ten) == max(tprs_at_ten)
        else f"TPR at 10% ranges from {min(tprs_at_ten):.3f} to {max(tprs_at_ten):.3f}"
    )
    text = (
        f"On `{row['folder']}` the AUROC standard deviation across the 3 seeds is "
        f"{row['auroc_std']:.4f} and its range {spread:.3f}, and {tpr_words}. {verdict}"
    )
    return text


def write_between_markers(path, block):
    with open(path) as handle:
        text = handle.read()
    if BEGIN in text:
        head = text[: text.index(BEGIN)]
        tail = text[text.index(END) + len(END) :]
        updated = head + block + tail
    else:
        anchor = "\n## "
        check3 = text.index("## Check 3: the mask seed")
        next_section = text.find(anchor, check3 + 1)
        cut = len(text) if next_section == -1 else next_section + 1
        updated = text[:cut].rstrip("\n") + "\n\n" + block + "\n" + text[cut - 1 :]
    with open(path, "w") as handle:
        handle.write(updated)


if __name__ == "__main__":
    main()

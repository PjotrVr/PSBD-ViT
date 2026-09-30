"""The per-attack tables of README.md, printed from summary.json.

Each row groups the poisoned models of 1 architecture, placement and attack and states
the verdict counts of claims 1, 2 and 4 with the median reading at each model's
adaptive rate, the same fields `measure.summarize` writes per model.

    PYTHONPATH=. .venv/bin/python experiments/prediction_shift_phenomenon/attack_tables.py
"""

import collections
import json
import os
import statistics

from experiments._paths import experiment_results_dir

SUMMARY = os.path.join(
    experiment_results_dir("prediction_shift_phenomenon"), "summary.json"
)
ARCHITECTURES = {"resnet18": "ResNet-18", "vit": "ViT-B/16", "swin": "Swin-S"}
PLACEMENTS = {
    "post_residual": "PSBD-RD",
    "before_attention_norm_token_mask": "PSBD-TM",
    "pre_residual_blocks_5_8": "best residual, blocks 5 to 8",
}


def main():
    with open(SUMMARY) as handle:
        rows = json.load(handle)["cached_rows"]
    groups = collections.defaultdict(list)
    for row in rows:
        if row["benign"] or row["placement"] not in PLACEMENTS:
            continue
        label = ARCHITECTURES[row["architecture"]]
        if row["reference_only"]:
            label += " smoke run, reference only"
        groups[(label, PLACEMENTS[row["placement"]], row["attack"])].append(row)

    print(
        "| architecture | placement | attack | models | claim 1 holds / partial / fails "
        "| median triggered sigma | claim 2 holds / partial / fails | median target share |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for key, members in groups.items():
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(members)} | "
            f"{counts(members, 'claim_1')} | {median(members, 'sigma_backdoor')} | "
            f"{counts(members, 'claim_2')} | {median(members, 'validation_target_share')} |"
        )
    print()
    print(
        "| architecture | placement | attack | models | std fails | median std AUROC, "
        "adaptive p | median std AUROC, best p | median PSU ratio AUROC |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for key, members in groups.items():
        fails = sum(m["claim_4"] == "std fails" for m in members)
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(members)} | "
            f"{fails} of {len(members)} | {median(members, 'auroc_std')} | "
            f"{median(members, 'auroc_std_best_rate')} | "
            f"{median(members, 'auroc_psu_ratio')} |"
        )


def counts(members, claim):
    tally = collections.Counter(m[claim] for m in members)
    text = f"{tally['holds']} / {tally['partial']} / {tally['fails']}"
    return text


def median(members, field):
    values = [m[field] for m in members if m[field] is not None]
    text = f"{statistics.median(values):.2f}" if values else "--"
    return text


if __name__ == "__main__":
    main()

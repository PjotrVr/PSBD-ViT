"""Write experiments/training_set_detection/README.md from the analyze.py records.

Every number in the README is read here from results/_experiments/training_set_detection/
(summary.json and models/<folder>.json). A sentence whose wording depends on the
data is chosen from the verdicts analyze.py wrote, so a rerun cannot leave a stale
claim behind.

    cd <main checkout>
    PYTHONPATH=<worktree> .venv/bin/python <worktree>/experiments/training_set_detection/render_readme.py
"""

import json
import os
import sys

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(EXPERIMENT_DIR)))

from experiments.training_set_detection import tables  # noqa: E402
from experiments.training_set_detection.common import (  # noqa: E402
    CD_L_SUBSET,
    CLEAN_SAMPLE_SIZE,
    COVER_SAMPLE_SIZE,
    PAPER_MIRROR,
    RATE_SUBSET,
    RECORDS_DIR,
    STRIP_SUBSET,
    model_queue,
)

OUT = os.path.join(EXPERIMENT_DIR, "README.md")
# The development set was dropped on 2026-10-01 and is never judged.
REPORTED_SETS = ("paper_mirror",)


def main():
    summary, rows = read_records()
    sections = [
        header_section(),
        method_section(),
        commands_section(rows),
        scope_section(),
        results_begin(),
        paper_mirror_section(summary, rows),
        development_section(summary, rows),
        gap_section(summary, rows),
        test_time_section(summary, rows),
        rates_section(rows),
        reconstruction_section(rows),
        verdict_section(summary),
        sentence_section(summary),
        pending_section(summary),
        results_end(),
    ]
    text = "\n\n".join(section for section in sections if section) + "\n"
    with open(OUT, "w") as handle:
        handle.write(text)
    print(f"wrote {OUT}")


def read_records():
    with open(os.path.join(RECORDS_DIR, "summary.json")) as handle:
        summary = json.load(handle)
    rows = {}
    for folder in model_queue():
        path = os.path.join(RECORDS_DIR, "models", f"{folder}.json")
        if os.path.exists(path):
            with open(path) as handle:
                rows[folder] = json.load(handle)
    return summary, rows


def header_section():
    text = """# Training-set detection in Li et al.'s setting

Li et al. (arXiv 2406.05826) built PSBD to find poisoned samples inside a training set. They score every training image of a backdoored model with PSU, flag PSU below the 25th percentile of clean-validation PSU and report TPR on the poisoned training images against FPR on the clean ones. Our paper scores triggered test images against their clean twins instead. This experiment asks whether our placements and their ranking carry over to the original setting, and whether the original setting is reproduced on the original architecture. The predictions and their refutation rules are in `PREDICTIONS.md`, committed before any training-set score was computed."""
    return text


def method_section():
    text = f"""## Method

Each model's poisoned training set is rebuilt with the functions its training run used (`cli.train_backdoor.build_training_set`), the seed of `args.json` and 0 where the run predates seed recording, with no augmentation, which none of these models trained with. Poisoned images carry the trigger exactly as trained. 4 groups are scored: every poisoned training image, at most {COVER_SAMPLE_SIZE} cover samples, {CLEAN_SAMPLE_SIZE} clean training images drawn with seed 0 and the standard 2000-image PSBD validation split that sets every threshold.

PSBD-TM (`before_attention_norm` with `token_mask`), PSBD-RD (`post_residual`) and the final method (PSBD-TM fused with `pre_residual_blocks_5_8` by the minimum of clean-validation percentiles) run through `defenses.inference` exactly as `cli.sweep` runs them, k = 3, bfloat16, batch 64, mask seed 0. The validation passes reproduce the cached `cli.sweep` tensors (`validation_pass_checks` in each model record). ResNet-18 gets PSBD-RD only. Each placement is read at 2 rates. Ours is the smallest cached rate whose clean-validation shift ratio reaches 0.8. Li et al.'s, taken from their released `select_dropout_rate`, is the argmax of the validation shift minus the whole-training-set shift over the rates whose validation shift reaches 0.8, with the whole-set shift estimated on {RATE_SUBSET["clean_train"]} clean, {RATE_SUBSET["poisoned"]} poisoned and {RATE_SUBSET["cover"]} cover images weighted by their population counts. Fractional PSU is our headline statistic and absolute PSU is Li et al.'s.

STRIP runs on {STRIP_SUBSET["clean_train"]} clean, {STRIP_SUBSET["poisoned"]} poisoned and {STRIP_SUBSET["cover"]} cover images and CD-L on {CD_L_SUBSET["clean_train"]}, {CD_L_SUBSET["poisoned"]} and {CD_L_SUBSET["cover"]}, both through `detectors.build_detector` with `cli.baselines`' context. CD-L reuses the test-time record's validation scores. The comparison with them reads PSBD on the CD-L images too. Spectral Signatures (`detectors/spectral_signatures.py`, ported from the backdoor-toolbox cleanser Li et al. ran) scores every image of the target label and removes min(int(1.5 epsilon N), n / 2) images of each label.

TPR is read 2 ways. With the threshold at a quantile of clean validation, which is what a defender can do, the FPR realized on the clean training images is reported beside it. With the threshold at the same quantile of the clean training images themselves the FPR is exactly the quantile, which is the fair comparison when the 2 clean distributions differ. Covers count as negatives in Li et al.'s code, so the paper-format FPR folds them in at their population share."""
    return text


def commands_section(rows):
    seconds = {
        folder: row.get("gpu_seconds")
        for folder, row in rows.items()
        if row.get("gpu_seconds")
    }
    wall = ""
    if seconds:
        lines = [
            f"| `{folder}` | {value / 60:.1f} |" for folder, value in seconds.items()
        ]
        wall = (
            "\n\nGPU minutes per model, the sum of every part's own timer on the shared login A100:\n\n"
            "| model | minutes |\n|---|---|\n" + "\n".join(lines)
        )
    text = (
        """## Commands

Run from the main checkout, which holds `checkpoints/`, `raw_data/` and `results/`, with the code of this worktree on `PYTHONPATH`.

```bash
bash experiments/training_set_detection/run_queue.sh          # GPU, resumable, 2 flock slots
python experiments/training_set_detection/analyze.py          # CPU readout and verdicts
python experiments/training_set_detection/render_readme.py    # this file
```"""
        + wall
    )
    return text


def results_begin():
    text = "<!-- results:begin -->\n<!-- Everything down to results:end is rendered by render_readme.py. -->"
    return text


def results_end():
    return "<!-- results:end -->"


def paper_mirror_section(summary, rows):
    block = summary["sets"]["paper_mirror"]
    text = f"""## Paper-mirror set

Li et al.'s main table uses BadNets, Blend, WaNet and Label-Consistent on CIFAR-10, GTSRB and Tiny ImageNet at 10% poisoning with ResNet-18. The first table reads each model in their exact configuration: absolute PSU, their rate rule and T at the 25th percentile of clean validation, as TPR/FPR with covers counted as negatives. STRIP and CD-L are thresholded at the same quantile. Spectral Signatures uses its own removal rule. A model marked not pooled failed a reconstruction check or the 2-point bar and stays out of every mean.

{tables.paper_format_table(rows, "li", "absolute")}

The same table under our rate rule and fractional PSU.

{tables.paper_format_table(rows, "ours", "fractional")}

TPR at 1%, 5% and 10% FPR realized on the clean training images, then AUROC of poisoned against clean training images, fractional PSU at our rate. STRIP and CD-L rows read the smaller subset.

{tables.low_fpr_table(rows, [f for f in PAPER_MIRROR if f.startswith("vit")], block)}"""
    resnet = [f for f in PAPER_MIRROR if f.startswith("resnet18")]
    text += f"""

The 2 ResNet-18 reproductions.

{tables.low_fpr_table(rows, resnet, {})}"""
    return text


def scope_section():
    text = """## Changes of scope

- 2026-10-01: the development tier (the 10 models of `experiments/cache_readouts/dev_set.json`, of which the 2 that are also paper-mirror models stay scored) and the BackdoorBench training-set tier are dropped, so the night's GPU time goes to the paper-mirror set and its detectors. P1 to P4 and P6 are judged on the paper-mirror set only and the development-set readings of `PREDICTIONS.md` stay unjudged. STRIP and CD-L run on the paper-mirror models in a second pass after PSBD, and Spectral Signatures is read on the CPU from the baseline features PSBD's first part stores."""
    return text


def development_section(summary, rows):
    # The tier was dropped on 2026-10-01, so the 2 development models that are
    # also paper-mirror models are read in the paper-mirror tables only.
    text = "## Development set\n\nDropped on 2026-10-01, see the changes of scope above. Its 2 models that are also paper-mirror models are read in the paper-mirror tables."
    return text


def gap_section(summary, rows):
    folders = [f for f in model_queue() if f in rows]
    text = f"""## Clean training against clean validation

A threshold set on validation is only honest on the training set if clean training images and clean validation images share a PSU distribution. The table gives both medians, the Kolmogorov-Smirnov statistic between them and the FPR each nominal quantile realizes on clean training images, for PSBD-TM (PSBD-RD on ResNet-18) at our rate.

{tables.gap_table(rows, folders)}

TPR with the threshold at each quantile of clean validation, and the FPR it realizes on clean training images, fractional PSU at our rate.

{tables.nominal_table(rows, folders)}"""
    return text


def test_time_section(summary, rows):
    folders = [f for f in model_queue() if f in rows]
    lines = []
    for name in REPORTED_SETS:
        paired = summary["sets"][name].get("paired")
        if not paired:
            continue
        train = paired["train_tm_minus_rd_ours"]
        test = paired["test_tm_minus_rd"]
        lines.append(
            f"On the {name.replace('_', '-')} set ({train['n']} pooled models) PSBD-TM minus PSBD-RD is "
            f"{train['mean_difference']:+.3f} AUROC on training images, interval "
            f"[{train['ci95'][0]:+.3f}, {train['ci95'][1]:+.3f}], against "
            f"{test['mean_difference']:+.3f} on the paired test splits of the same models."
        )
    text = f"""## Training set against test time

AUROC on the training images next to AUROC on the paired test splits, from the `cli.sweep` caches and the `cli.baselines` records of the same models.

{tables.test_time_table(rows, folders)}

{" ".join(lines)}"""
    return text


def rates_section(rows):
    folders = [f for f in model_queue() if f in rows]
    text = f"""## Rates under both rules

{tables.rate_table(rows, folders)}"""
    return text


def reconstruction_section(rows):
    folders = [f for f in model_queue() if f in rows]
    text = f"""## Reconstruction checks

R1 compares the rebuilt counts with `args.json`, R2 asks whether the model sends the rebuilt poisoned images to their attack-success label, and R3 asks whether the untriggered image of a rebuilt poisoned index looks held out (the ratio of its loss gap to the validation gap, at least 0.5 to pass).

{tables.reconstruction_table(rows, folders)}"""
    return text


def verdict_section(summary):
    verdicts = summary["verdicts"]
    pending = summary["pending"]
    lines = ["## Verdicts", ""]
    # P6 orders PSBD-TM, STRIP and CD-L, so it is incomplete until both
    # detectors are scored on every pooled model.
    detectors_missing = len(verdicts["paper_mirror"]["P6"]["train_order"]) < 3
    if pending or detectors_missing:
        lines += [
            f"Provisional, {len(pending)} paper-mirror models are not scored yet or "
            "STRIP and CD-L are missing on some, so a verdict below can still change.",
            "",
        ]
    for set_name in REPORTED_SETS:
        block = verdicts.get(set_name)
        if not isinstance(block, dict):
            lines.append(f"- {set_name}: {block}")
            continue
        for name in ("P1", "P2", "P3", "P6"):
            lines.append(
                f"- {name} on the {set_name.replace('_', '-')} set: {verdict_words(block[name])}"
            )
        p4 = block["P4"]
        lines.append(
            f"- P4 on the {set_name.replace('_', '-')} set: fractional part "
            f"{'holds' if p4['holds_fractional'] else 'refuted'} "
            f"(median below on {p4['fractional_median_below']}, FPR above 0.25 on "
            f"{p4['fractional_fpr_above_0.25']}), absolute part "
            f"{'holds' if p4['holds_absolute'] else 'refuted'} "
            f"(lower on {p4['absolute_fpr_lower_than_fractional']})."
        )
    lines.append(f"- P5: {verdict_words(verdicts['P5'])}")
    lines.append(f"- P7: {verdict_words(verdicts['P7'])}")
    text = "\n".join(lines)
    return text


def verdict_words(verdict):
    state = "holds" if verdict.get("holds") else "refuted"
    details = {k: v for k, v in verdict.items() if k != "holds"}
    words = f"{state}, `{json.dumps(details, default=str)}`"
    return words


def sentence_section(summary):
    return ""


def pending_section(summary):
    if not summary["pending"]:
        return ""
    names = ", ".join(f"`{f}`" for f in summary["pending"])
    text = f"## Pending\n\nNot yet scored when this file was rendered: {names}."
    return text


if __name__ == "__main__":
    main()

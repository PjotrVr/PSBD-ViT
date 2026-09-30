"""PSU, fractional PSU and confidence alone on the current panel, rendered into README.md.

Placement PSBD-TM (`before_attention_norm_token_mask`) at the adaptive 0.8 rule's rate,
the models of the headline panel (`experiments.probe_union.measure.select_models`).
Absolute and fractional PSU AUROC are read from each model's `psbd_metrics.json`
(`detection` and `detection_psu_ratio` at q0.25). Confidence alone is computed from the
cached no-perturbation baseline, clean paired to backdoor with `pair_clean_to_backdoor`,
higher confidence meaning poisoned, exactly as `measure.py` computes it. The script
writes `results/_experiments/psu_vs_confidence/panel.json` and replaces the block
between the results markers of README.md.

    PYTHONPATH=. .venv/bin/python experiments/psu_vs_confidence/panel.py
"""

import json
import os
import statistics

from defenses.cache import baseline_path, load_baseline, read_split_manifest
from defenses.decision import pair_clean_to_backdoor
from experiments._paths import experiment_result_path
from experiments.probe_union.measure import select_models
from experiments.psu_vs_confidence.measure import auroc, tracked_confidence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
README = os.path.join(REPO_ROOT, "experiments", "psu_vs_confidence", "README.md")
PLACEMENT = "before_attention_norm_token_mask"
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"


def main():
    rows = [model_row(cell) for cell in select_models("results")]
    payload = {"placement": PLACEMENT, "n_models": len(rows), "models": rows}
    path = experiment_result_path("psu_vs_confidence", "panel.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    render(rows)
    print(f"wrote {path} and {README}")


def model_row(cell):
    block = cell["report"]["placements"][PLACEMENT]
    rate = block["adaptive_rate"]
    row = next(r for r in block["rates"] if r["rate"] == rate)
    model = {
        "folder": cell["folder_name"],
        "dataset": cell["dataset"],
        "absolute": row["detection"]["q0.25"]["auroc"],
        "fractional": row["detection_psu_ratio"]["q0.25"]["auroc"],
        "confidence": confidence_auroc(cell["folder_name"]),
    }
    return model


def confidence_auroc(folder):
    psbd_dir = os.path.join("results", folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    clean_probs, clean_labels, _ = load_baseline(baseline_path(psbd_dir, "clean"))
    backdoor_probs, backdoor_labels, _ = load_baseline(
        baseline_path(psbd_dir, "backdoor")
    )
    clean = pair_clean_to_backdoor(
        tracked_confidence(clean_probs, clean_labels), manifest
    )  # (n_backdoor,)
    backdoor = tracked_confidence(backdoor_probs, backdoor_labels)  # (n_backdoor,)
    value = auroc(clean, backdoor, False)
    return value


def group_line(label, rows, bold=False):
    wins = sum(r["confidence"] > r["fractional"] for r in rows)
    cells = [
        label,
        f"{len(rows)}",
        f"{mean(rows, 'absolute'):.3f}",
        f"{mean(rows, 'fractional'):.3f}",
        f"{mean(rows, 'confidence'):.3f}",
        f"{wins}",
    ]
    if bold:
        cells = [f"**{cell}**" for cell in cells]
    line = "| " + " | ".join(cells) + " |"
    return line


def mean(rows, key):
    value = statistics.mean(r[key] for r in rows)
    return value


def join_words(words):
    if len(words) == 1:
        return words[0]
    text = ", ".join(words[:-1]) + " and " + words[-1]
    return text


def render(rows):
    n = len(rows)
    confidence_wins = [r for r in rows if r["confidence"] > r["fractional"]]
    beats_absolute = sum(r["confidence"] > r["absolute"] for r in rows)
    fractional_wins = sum(r["fractional"] > r["absolute"] for r in rows)
    lines = [
        BEGIN,
        "<!-- Everything down to results:end is rendered by panel.py from "
        "results/_experiments/psu_vs_confidence/panel.json. -->",
        "",
        "### The canonical reading on the current panel",
        "",
        "PSBD-TM (`before_attention_norm_token_mask`) at the adaptive 0.8 rule's rate, "
        f"the paper panel of {n} models successful at the 2-point clean-accuracy bar, "
        "fractional and absolute PSU read from each model's `psbd_metrics.json` at q0.25 "
        "and confidence alone from the cached baseline:",
        "",
        "    PYTHONPATH=. .venv/bin/python experiments/psu_vs_confidence/panel.py",
        "",
        "| dataset | n | absolute PSU | fractional PSU | confidence only | confidence wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        group_line(dataset, [r for r in rows if r["dataset"] == dataset])
        for dataset in DATASETS
    ]
    lines += [group_line("all", rows, bold=True), ""]
    win_text = join_words(
        [
            f"`{r['folder']}` ({r['confidence']:.3f} against {r['fractional']:.3f})"
            for r in confidence_wins
        ]
    )
    lines += [
        f"Confidence alone beats fractional PSU on {len(confidence_wins)} of {n} models, "
        f"{win_text}. It beats absolute PSU on {beats_absolute}. The stochastic passes "
        "do real work at the canonical reading.",
        "",
        f"Fractional PSU reads {mean(rows, 'fractional'):.3f} against "
        f"{mean(rows, 'absolute'):.3f} for the absolute form and beats it on "
        f"{fractional_wins} of the {n} panel models. Confidence only reads "
        f"{mean(rows, 'confidence'):.3f} on the same models.",
        END,
    ]
    block = "\n".join(lines)
    with open(README) as handle:
        text = handle.read()
    start = text.index(BEGIN)
    stop = text.index(END) + len(END)
    text = text[:start] + block + text[stop:]
    with open(README, "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()

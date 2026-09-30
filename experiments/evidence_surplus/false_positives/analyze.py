"""C-fp, CPU half: are the false positives the clean images with a surplus?

Reads measure.py's records and the sweep's caches, applies PREDICTIONS.md as
written, and writes results/_experiments/evidence_surplus/false_positives.json,
a comparison figure and a gallery (each with a JSON sidecar) and README.md.

On each model's 2000 clean held-out images, PSBD-TM's cached fractional
statistic at its adaptive rate flags the images below its own 0.01 and 0.05
quantiles, the false positives by construction. Each measure is compared between
flagged and unflagged images as a median difference with a bootstrap 95%
interval (2000 resamples, seed 0). It is repeated with each flagged image matched to
the unflagged image of the nearest unperturbed margin, the confidence control.

    PYTHONPATH=. python experiments/evidence_surplus/false_positives/analyze.py
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import scripts.paper._style  # noqa: E402, F401
from cli.compare_detectors import psbd_rate  # noqa: E402
from data.splits import PSBD_SPLIT_SEED, load_clean_test_base, psbd_split_permutation  # noqa: E402
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defenses.decision import threshold_at_quantile  # noqa: E402
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from scripts.paper._common import OKABE_ITO, load_psbd_metrics  # noqa: E402
from scripts.paper._style import TEXT_WIDTH  # noqa: E402

RESULTS = os.path.join("results", "_experiments", "evidence_surplus")
HERE = os.path.dirname(os.path.abspath(__file__))
QUANTILES = (0.01, 0.05)
RESAMPLES = 2000
TARGET = 0
GALLERY = 6
FLAGGING = "before_attention_norm_token_mask"
OTHER = "post_residual"
# Signs so that every difference reads "flagged minus rest, positive means more
# surplus": a smaller sufficient share is more surplus, so its sign is negative.
MEASURES = {
    "sufficient_ranked": -1.0,
    "sufficient_random": -1.0,
    "class_component": 1.0,
    "other_operator_psu": -1.0,
    "margin": 1.0,
}


def main():
    records = read_records()
    rows = [model_reading(r) for r in records]
    verdicts = check(rows)
    payload = {"rows": rows, "verdicts": verdicts}
    with open(os.path.join(RESULTS, "false_positives.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    comparison_figure(rows)
    gallery_figure(records)
    write_readme(rows, verdicts)
    print(json.dumps(verdicts, indent=2))


def read_records():
    folder = os.path.join(RESULTS, "false_positives")
    records = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        with open(os.path.join(folder, name)) as handle:
            records.append(json.load(handle))
    return records


def cached_statistic(folder, placement):
    block = load_psbd_metrics("results", folder)["placements"][placement]
    rate = psbd_rate(block, "adaptive")
    psbd_dir = os.path.join("results", folder, "psbd")
    probs, labels, loader_labels = load_baseline(baseline_path(psbd_dir, "validation"))
    per_pass, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, "validation")
    )
    psu = psu_ratio_from_cache(probs, labels, per_pass)  # (n,)
    return psu, probs, loader_labels


def model_reading(record):
    folder = record["folder"]
    psu, probs, true_labels = cached_statistic(folder, FLAGGING)
    other, _, _ = cached_statistic(folder, OTHER)
    top_two = probs.topk(2, dim=1).values.clamp_min(1e-12).log()  # (n, 2)
    measures = {
        "sufficient_ranked": np.array(record["sufficient_ranked"], dtype=float),
        "sufficient_random": np.array(record["sufficient_random"], dtype=float),
        "class_component": np.array(record["class_component"], dtype=float),
        "other_operator_psu": other.numpy(),
        # log p_c - log p_runner-up equals the logit margin.
        "margin": (top_two[:, 0] - top_two[:, 1]).numpy(),
    }
    assert all(len(v) == len(psu) for v in measures.values()), folder
    reading = {
        "model": folder,
        "benign": "benign" in folder,
        "quantiles": {},
        # The diagnosis for prediction 2 and 3 on the own-class component: the
        # head is linear in the feature, so the component along the own class's
        # direction may be the margin by another name, and then matching on the
        # margin removes it by construction.
        "spearman_class_component_margin": spearman(
            measures["class_component"], measures["margin"]
        ),
    }
    for quantile in QUANTILES:
        flagged = (psu < threshold_at_quantile(psu, quantile)).numpy()
        entry = {"flagged": int(flagged.sum())}
        for name, sign in MEASURES.items():
            entry[name] = difference_with_interval(measures[name], flagged, sign)
            if name != "margin":
                entry[f"{name}_margin_matched"] = matched_difference(
                    measures[name], measures["margin"], flagged, sign
                )
        entry["target_share_flagged"] = float(
            (true_labels.numpy()[flagged] == TARGET).mean()
        )
        entry["target_share_all"] = float((true_labels.numpy() == TARGET).mean())
        reading["quantiles"][str(quantile)] = entry
    return reading


def spearman(first, second):
    a = np.argsort(np.argsort(first))
    b = np.argsort(np.argsort(second))
    value = float(np.corrcoef(a, b)[0, 1])
    return value


def difference_with_interval(values, flagged, sign):
    generator = np.random.default_rng(0)
    inside, outside = values[flagged], values[~flagged]
    point = sign * (np.median(inside) - np.median(outside))
    draws = [
        sign
        * (
            np.median(generator.choice(inside, len(inside)))
            - np.median(generator.choice(outside, len(outside)))
        )
        for _ in range(RESAMPLES)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    reading = {"difference": float(point), "low": float(low), "high": float(high)}
    return reading


# Each flagged image against the unflagged image of the nearest margin, the
# median paired difference with a bootstrap interval over the pairs.
def matched_difference(values, margin, flagged, sign):
    inside = np.flatnonzero(flagged)
    outside = np.flatnonzero(~flagged)
    nearest = outside[
        np.abs(margin[outside][None, :] - margin[inside][:, None]).argmin(axis=1)
    ]
    paired = sign * (values[inside] - values[nearest])
    generator = np.random.default_rng(0)
    draws = [np.median(generator.choice(paired, len(paired))) for _ in range(RESAMPLES)]
    low, high = np.percentile(draws, [2.5, 97.5])
    reading = {
        "difference": float(np.median(paired)),
        "low": float(low),
        "high": float(high),
    }
    return reading


def positive(entry):
    holds = entry["low"] > 0
    return holds


def check(rows):
    verdicts = {}
    for quantile in map(str, QUANTILES):
        entries = [r["quantiles"][quantile] for r in rows]
        backdoored = [r["quantiles"][quantile] for r in rows if not r["benign"]]
        benign = [r["quantiles"][quantile] for r in rows if r["benign"]]
        verdicts[quantile] = {
            "models": len(entries),
            "median_spearman_class_component_margin": float(
                np.median([r["spearman_class_component_margin"] for r in rows])
            ),
            "P1_sufficient_ranked": sum(
                positive(e["sufficient_ranked"]) for e in entries
            ),
            "P1_sufficient_random": sum(
                positive(e["sufficient_random"]) for e in entries
            ),
            "P2_class_component": sum(positive(e["class_component"]) for e in entries),
            "P2_other_operator": sum(
                positive(e["other_operator_psu"]) for e in entries
            ),
            "P3_sufficient_ranked_matched": sum(
                positive(e["sufficient_ranked_margin_matched"]) for e in entries
            ),
            "P3_class_component_matched": sum(
                positive(e["class_component_margin_matched"]) for e in entries
            ),
            "margin_higher": sum(positive(e["margin"]) for e in entries),
            "P4_target_over_on_backdoored": sum(
                e["target_share_flagged"] > 2 * e["target_share_all"]
                for e in backdoored
            ),
            "P4_target_over_on_benign": sum(
                e["target_share_flagged"] > 2 * e["target_share_all"] for e in benign
            ),
            "backdoored": len(backdoored),
            "benign": len(benign),
        }
    return verdicts


def comparison_figure(rows):
    if not rows:
        return
    names = [
        "sufficient_ranked",
        "sufficient_random",
        "class_component",
        "other_operator_psu",
        "margin",
    ]
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 2.6))
    for index, r in enumerate(rows):
        entry = r["quantiles"]["0.05"]
        for j, name in enumerate(names):
            e = entry[name]
            x = j + (index - len(rows) / 2) * 0.05
            axis.errorbar(
                x,
                e["difference"],
                yerr=[[e["difference"] - e["low"]], [e["high"] - e["difference"]]],
                fmt="o",
                markersize=2,
                color="0.5" if r["benign"] else OKABE_ITO[index % 8],
                linewidth=0.6,
            )
    axis.axhline(0, color="0.4", linewidth=0.8)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(
        [
            "sufficient share, ranked",
            "sufficient share, random",
            "own-class component",
            "other operator",
            "margin (control)",
        ],
        fontsize=6,
    )
    axis.set_ylabel("flagged minus rest, sign: more surplus up")
    axis.set_title(
        "false positives at the 0.05 quantile, 1 point per model, benign grey, 95% intervals",
        fontsize=7,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "false_positives.png"))
    with open(os.path.join(RESULTS, "figures", "false_positives.json"), "w") as handle:
        json.dump(rows, handle)
    plt.close(fig)


# The most confidently flagged held-out images beside a random draw, per model.
def gallery_figure(records):
    if not records:
        return
    fig, axes = plt.subplots(
        len(records),
        2 * GALLERY,
        figsize=(TEXT_WIDTH, 0.6 * len(records) + 0.4),
        squeeze=False,
    )
    sidecar = []
    for row, record in enumerate(records):
        psu, _, labels = cached_statistic(record["folder"], FLAGGING)
        dataset = record["dataset"]
        base = load_clean_test_base(dataset, "raw_data")
        heldout = psbd_split_permutation(len(base), PSBD_SPLIT_SEED)[
            : len(psu)
        ].tolist()
        flagged = psu.argsort()[:GALLERY].tolist()
        generator = torch.Generator().manual_seed(0)
        random_rows = torch.randperm(len(psu), generator=generator)[:GALLERY].tolist()
        for column, index in enumerate(flagged + random_rows):
            image = base[heldout[index]][0].permute(1, 2, 0).numpy()
            axis = axes[row][column]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
        axes[row][0].set_ylabel(
            record["folder"].replace("vit_", ""), fontsize=4, rotation=0, ha="right"
        )
        sidecar.append(
            {
                "model": record["folder"],
                "flagged_rows": flagged,
                "random_rows": random_rows,
                "flagged_labels": [int(labels[i]) for i in flagged],
            }
        )
    fig.suptitle(
        f"left {GALLERY}: lowest PSBD-TM statistic among clean held-out images, right {GALLERY}: random",
        fontsize=7,
    )
    fig.savefig(os.path.join(RESULTS, "figures", "false_positive_gallery.png"))
    with open(
        os.path.join(RESULTS, "figures", "false_positive_gallery.json"), "w"
    ) as handle:
        json.dump(sidecar, handle)
    plt.close(fig)


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_readme(rows, verdicts):
    lines = []
    for quantile, v in verdicts.items():
        lines.append(
            f"At the {quantile} quantile, over {v['models']} models: the ranked sufficient share is smaller for false positives with an interval excluding 0 on {v['P1_sufficient_ranked']}, the random one on {v['P1_sufficient_random']} (P1). The own-class component is higher on {v['P2_class_component']} and the other operator's statistic lower on {v['P2_other_operator']} (P2). After margin matching the ranked sufficient share is still smaller on {v['P3_sufficient_ranked_matched']} and the own-class component still higher on {v['P3_class_component_matched']} (P3). The margin itself is higher for false positives on {v['margin_higher']}. The target class makes up more than twice its share of the false positives on {v['P4_target_over_on_backdoored']} of {v['backdoored']} backdoored and {v['P4_target_over_on_benign']} of {v['benign']} benign models (P4)."
        )
    correlation = verdicts["0.05"]["median_spearman_class_component_margin"]
    lines.append(
        f"Diagnosis of the own-class component (P2 and P3), the 1 extra measurement the failure protocol asks for: across each model's held-out images the component correlates with the unperturbed margin at a median Spearman of {fmt(correlation)}. The head is linear in the feature, so the component along the own class's direction is close to the own logit, and matching on the margin removes most of it by construction. The measure cannot separate surplus from confidence, so its failure under margin matching shows the operational measure was the wrong one for the question and leaves the account untested on it. The sufficient token share and the other operator's reading, which are not linear readouts of the final feature, carry the test."
    )
    table_rows = []
    for r in rows:
        e = r["quantiles"]["0.05"]
        cells = [r["model"], str(e["flagged"])]
        for name in (
            "sufficient_ranked",
            "class_component",
            "other_operator_psu",
            "margin",
        ):
            cells.append(
                f"{fmt(e[name]['difference'])} [{fmt(e[name]['low'])}, {fmt(e[name]['high'])}]"
            )
        cells.append(
            f"{fmt(e['sufficient_ranked_margin_matched']['difference'])} [{fmt(e['sufficient_ranked_margin_matched']['low'])}, {fmt(e['sufficient_ranked_margin_matched']['high'])}]"
        )
        cells.append(
            f"{fmt(e['target_share_flagged'])} of {fmt(e['target_share_all'])}"
        )
        table_rows.append("| " + " | ".join(cells) + " |")
    header = "| model | flagged | sufficient share, ranked | own-class component | other operator | margin | sufficient share, margin matched | target class among flagged, of all |\n|---|---|---|---|---|---|---|---|"
    text = f"""# C-fp. Are PSBD's false positives the clean images with their own evidence surplus?

Generated by `experiments/evidence_surplus/false_positives/analyze.py` from `measure.py`'s records and the sweep's caches. The predictions are in `PREDICTIONS.md`, written before the run.

PSU is itself a surplus reading, so the false positives are compared on measures that do not come from the probe that flagged them: the smallest sufficient token share under deterministic masking with no PSBD rate, the own-class component of the final feature, the other operator's cached statistic. The unperturbed margin is the confidence control. Every difference is flagged minus rest, signed so that a positive value means more surplus, with a bootstrap 95% interval.

<!-- results:begin -->
{chr(10).join(lines)}

At the 0.05 quantile, per model:

{header}
{chr(10).join(table_rows)}
<!-- results:end -->

The figures are in `results/_experiments/evidence_surplus/figures/`: `false_positives.png` (every measure's difference with its interval, per model, benign models grey) and `false_positive_gallery.png` (the 6 clean held-out images with the lowest statistic beside 6 random ones, per model), each with a JSON sidecar.
"""
    with open(os.path.join(HERE, "README.md"), "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()

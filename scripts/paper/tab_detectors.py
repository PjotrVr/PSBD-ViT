"""Every detector against every attack, in the layout the PSBD paper uses.

1 row per backdoored model grouped by dataset, and 1 column per defense with
ours first. That is the transpose of the earlier tables here, and it is the
orientation a reader wants: a person asking "does anything catch WaNet" reads
across 1 line instead of down 12.

3 tables, 1 per reported metric, because 12 defenses times 3 metrics does not fit
a page. AUROC is the one-sided area under the ROC curve, never flipped, so a
value below 0.5 means that detector ordered poisoned and clean inputs the wrong
way round on that model and is printed gray. TPR is at the 10% and 20%
false-positive budgets, both set on the same 2000 clean validation images every
detector and PSBD share.

Best in each row is bold and second best underlined, over the defenses only.

    PYTHONPATH=. python scripts/paper/tab_detectors.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_values  # noqa: E402
from defenses.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from detectors import DETECTOR_NAMES  # noqa: E402
from detectors.ibd_psc import DEFAULT_SCALING_FACTOR  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    split_by_dataset,
    HEADLINE_KEY,
    attack_label,
    bootstrap_ci,
    build_parser,
    dataset_label,
    detector_label,
    fmt,
    load_coverage,
    load_json,
    load_psbd_metrics,
    mean_or_none,
    ordinal,
    provenance_comment,
    word_list,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_detectors.py"
# The deployable rate rule, the only one a competitor comparison may use: the
# matched rule exists to compare placements to each other, not to other methods.
RULE = "adaptive"
# Our 2 columns: the placement this paper recommends and the ConvNet placement
# the PSBD paper published, so a reader sees the method and its own baseline.
OURS = (
    (RECOMMENDED_PLACEMENT, "PSBD-TM"),
    (PUBLISHED_PLACEMENT, "PSBD-RD"),
)
# Reported metric, as (stem, quantile ladder key, field, caption words).
METRICS = (
    ("auroc", HEADLINE_KEY, "auroc", "AUROC"),
    ("tpr10", "q0.10", "tpr", "TPR at a 10\\% false-positive budget"),
    ("tpr20", "q0.20", "tpr", "TPR at a 20\\% false-positive budget"),
)
# A one-sided AUROC under this ordered the 2 classes the wrong way round.
CHANCE = 0.5
GREY = r"\textcolor{black!45}"
SHADE = r"\rowcolor{black!8}"


def detector_reading(
    results_dir: str, folder: str, detector: str, key: str, field: str
):
    """1 number from a detector's record, or None when it never scored this cell."""
    path = os.path.join(results_dir, folder, "detectors", f"{detector}_metrics.json")
    record = load_json(path)
    if record is None or record.get("status") != "scored":
        return None
    quantile = (record.get("detection") or {}).get(key)
    if quantile is None:
        return None
    return quantile.get(field)


def psbd_reading(report: dict, placement: str, key: str, field: str):
    """The same number from a PSBD placement, so both sides read 1 ladder.

    cli.compare_detectors.psbd_values is the shared reader, because a cache
    written before the quantile ladder existed carries the headline reading
    directly on the rule block and only it knows that shape.
    """
    block = psbd_values(report, placement, RULE)
    if block is None:
        return None
    quantile = block.get(key)
    if quantile is None:
        return None
    return quantile.get(field)


def cell_readings(
    results_dir: str, cell: dict, key: str, field: str
) -> dict[str, float | None]:
    """Every defense's reading of 1 model, keyed by column name."""
    report = load_psbd_metrics(results_dir, cell["folder_name"]) or {}
    readings: dict[str, float | None] = {}
    for placement, column in OURS:
        readings[column] = psbd_reading(report, placement, key, field)
    for detector in DETECTOR_NAMES:
        readings[detector_label(detector)] = detector_reading(
            results_dir, cell["folder_name"], detector, key, field
        )
    return readings


def all_columns() -> list[str]:
    """Every defense read, ours first, including the published placement."""
    names = [column for _, column in OURS]
    names += [detector_label(detector) for detector in DETECTOR_NAMES]
    return names


def columns() -> list[str]:
    """The defenses the tables print, PSBD-TM first and PSBD-RD second."""
    names = all_columns()
    return names


def decorate(value: float | None, best: float | None, second: float | None) -> str:
    """1 cell: bold the best, underline the second, gray anything below chance."""
    if value is None:
        return "--"
    text = fmt(value)
    if value < CHANCE:
        return f"{GREY}{{{text}}}"
    if best is not None and value == best:
        return f"\\textbf{{{text}}}"
    if second is not None and value == second:
        return f"\\underline{{{text}}}"
    return text


def model_row(cell: dict, readings: dict[str, float | None]) -> list[str]:
    """1 model's row: its label, then every defense's decorated reading."""
    present = sorted(
        (readings[column] for column in columns() if readings[column] is not None),
        reverse=True,
    )
    best = present[0] if present else None
    second = next((value for value in present if value != best), None)
    row = [f"{attack_label(cell['attack'])} {cell['poison_rate'] * 100:g}"]
    row += [decorate(readings[column], best, second) for column in columns()]
    return row


def column_means(group: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Every defense's mean reading over a group of models."""
    means = {
        column: mean_or_none(
            [readings[column] for readings in group if readings[column] is not None]
        )
        for column in all_columns()
    }
    return means


def mean_row(label: str, means: dict[str, float | None]) -> list[str]:
    """A shaded row of per-column means over a group of models."""
    present = sorted(
        (means[column] for column in columns() if means[column] is not None),
        reverse=True,
    )
    best = present[0] if present else None
    second = next((value for value in present if value != best), None)
    row = [f"{SHADE}\\emph{{{label}}}"]
    row += [decorate(means[column], best, second) for column in columns()]
    return row


def build_rows(
    results_dir: str, cells: list[dict], key: str, field: str
) -> tuple[list[list[str]], dict[str, float | None], dict[str, dict]]:
    """The table body grouped by dataset, the overall means and the means per dataset."""
    by_dataset = collections.defaultdict(list)
    for cell in cells:
        by_dataset[cell["dataset"]].append(cell)

    width = len(columns()) + 1
    rows: list[list[str]] = []
    everything: list[dict[str, float | None]] = []
    dataset_means: dict[str, dict] = {}
    for dataset in sorted(by_dataset):
        group = sorted(
            by_dataset[dataset],
            key=lambda cell: (cell["attack"], cell["poison_rate"]),
        )
        rows.append(
            [f"\\multicolumn{{{width}}}{{l}}{{\\emph{{{dataset_label(dataset)}}}}}"]
        )
        readings_group = []
        for cell in group:
            readings = cell_readings(results_dir, cell, key, field)
            readings_group.append(readings)
            rows.append(model_row(cell, readings))
        dataset_means[dataset] = column_means(readings_group)
        rows.append(mean_row(f"{dataset_label(dataset)} mean", dataset_means[dataset]))
        everything.extend(readings_group)
    overall = column_means(everything)
    rows.append(mean_row("all models", overall))
    return rows, overall, dataset_means, everything


def metric_macros(stem: str, overall: dict[str, float | None]) -> dict:
    """Our column, the best competitor and the gap between them, for 1 metric."""
    ours = overall["PSBD-TM"]
    competitors = {
        detector_label(name): overall[detector_label(name)]
        for name in DETECTOR_NAMES
        if overall[detector_label(name)] is not None
    }
    best_name, best_value = max(competitors.items(), key=lambda item: item[1])
    shown = sorted(
        ((overall[name], name) for name in columns() if overall[name] is not None),
        reverse=True,
    )
    order = [name for _, name in shown]
    # Where the published placement would sit among the defenses the table shows.
    published = overall["PSBD-RD"]
    published_rank = 1 + sum(1 for value, _ in shown if value > published)
    macros = {
        f"detectors_{stem}_defenses_ranked": (
            str(len(order)),
            f"defenses the {stem} ranking covers, PSBD-TM and the competitors",
        ),
        f"detectors_{stem}_rank_ours": (
            ordinal(order.index("PSBD-TM") + 1),
            f"rank of the recommended placement among every defense by mean {stem}",
        ),
        f"detectors_{stem}_rank_published": (
            ordinal(published_rank),
            f"rank the published placement would take among the shown defenses by mean {stem}",
        ),
        f"detectors_{stem}_beating_published": (
            str(published_rank - 1),
            f"shown defenses with a higher mean {stem} than the published placement",
        ),
        f"detectors_{stem}_ours": (
            fmt(ours),
            f"mean {stem} of the recommended placement over the compared models",
        ),
        f"detectors_{stem}_published": (
            fmt(overall["PSBD-RD"]),
            f"mean {stem} of the published placement over the same models",
        ),
        f"detectors_{stem}_best_competitor": (
            fmt(best_value),
            f"mean {stem} of the strongest competitor detector, {best_name}",
        ),
        f"detectors_{stem}_best_competitor_name": (
            best_name,
            f"the strongest competitor detector by mean {stem}",
        ),
        f"detectors_{stem}_margin": (
            fmt(None if ours is None else ours - best_value, signed=True),
            f"mean {stem} of the recommended placement minus the strongest competitor",
        ),
        f"detectors_{stem}_sentinet": (
            fmt(overall[detector_label("sentinet")]),
            f"mean {stem} of the SentiNet port, whose Grad-CAM mask misses the trigger",
        ),
        f"detectors_{stem}_ibd_psc": (
            fmt(overall[detector_label("ibd_psc")]),
            f"mean {stem} of IBD-PSC at the paper's fixed amplification factor",
        ),
        f"detectors_{stem}_below_chance_columns": (
            str(sum(1 for value in competitors.values() if value < CHANCE)),
            f"competitor detectors whose mean {stem} sits below chance",
        ),
    }
    return macros


def leader_macros(dataset_means: dict[str, dict]) -> dict:
    """Which defense leads each dataset's mean AUROC, as prose for the body.

    The body names the datasets PSBD-TM leads and the defense that leads each of the
    others, so the sentence follows the table when a sweep lands.
    """
    leaders: dict[str, list[str]] = collections.defaultdict(list)
    for dataset, means in sorted(dataset_means.items()):
        present = {
            column: value for column, value in means.items() if value is not None
        }
        leader = max(present, key=present.get)
        leaders[leader].append(dataset_label(dataset))
    ours = leaders.pop("PSBD-TM", [])
    others = [f"{name} on {word_list(datasets)}" for name, datasets in leaders.items()]
    macros = {
        "detectors_auroc_datasets_ours_leads": (
            word_list(ours),
            "datasets on which the recommended placement has the highest mean AUROC",
        ),
        "detectors_auroc_datasets_others_lead": (
            word_list(others),
            "the defense with the highest mean AUROC on every other dataset",
        ),
    }
    return macros


def margin_interval_macros(
    stem: str,
    overall: dict[str, float | None],
    readings: list[dict[str, float | None]],
    resamples: int,
    seed: int,
) -> dict:
    """The paired bootstrap interval on ours minus the strongest competitor, per model.

    The mean margin alone cannot say whether a lead of a hundredth survives the
    choice of models, so the interval resamples the models the margin is read on.
    """
    competitors = {
        detector_label(name): overall[detector_label(name)]
        for name in DETECTOR_NAMES
        if overall[detector_label(name)] is not None
    }
    best_name = max(competitors, key=competitors.get)
    differences = [
        reading["PSBD-TM"] - reading[best_name]
        for reading in readings
        if reading["PSBD-TM"] is not None and reading[best_name] is not None
    ]
    low, high = bootstrap_ci(differences, resamples, seed)
    macros = {
        f"detectors_{stem}_margin_low": (
            fmt(low, signed=True),
            f"lower bound of the 95% paired bootstrap interval on detectors_{stem}_margin",
        ),
        f"detectors_{stem}_margin_high": (
            fmt(high, signed=True),
            f"upper bound of the 95% paired bootstrap interval on detectors_{stem}_margin",
        ),
    }
    return macros


def write_per_dataset_tables(args, stem, words, rows, header, align, inputs) -> None:
    """1 table per dataset for a metric, and 1 file that inputs them in order.

    The 14 columns and 80 rows of every model on 1 page shrank the text to 7pt.
    Split by dataset with the column names turned on their side, each table
    prints at full size.
    """
    tables_dir = os.path.join(args.paper_dir, "tables")
    names = []
    for index, (dataset, group_rows) in enumerate(split_by_dataset(rows)):
        slug = re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")
        name = f"detectors_{stem}_{slug}"
        first = index == 0
        conventions = (
            "Rows are attacks at their poison rate in percent and columns defenses, "
            "PSBD-TM first. Best in each row is bold and second best underlined, gray "
            "marks a reading below chance and shaded rows are means. PSBD-RD is our "
            "adaptation of the original PSBD site to a ViT block. "
            "Scale-Up$^\\dagger$ is the data-limited variant and IBD-PSC$^\\ast$ the "
            "calibrated one."
            if first
            else f"Layout as in \\cref{{tab:detectors-{stem}}}."
        )
        write_table(
            path=os.path.join(tables_dir, f"{name}.tex"),
            generator=GENERATOR,
            inputs=inputs,
            caption=f"{words} of every defense on {dataset}. {conventions}",
            label=f"tab:detectors-{stem}" if first else f"tab:detectors-{stem}-{slug}",
            header=header,
            rows=group_rows,
            align=align,
            rotate_header=True,
        )
        names.append(name)
    lines = [provenance_comment(GENERATOR, inputs)]
    lines += [f"\\input{{tables/{name}}}" for name in names]
    with open(os.path.join(tables_dir, f"detectors_{stem}.tex"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def fully_covered(results_dir: str, cells: list[dict]) -> list[dict]:
    """The cells every compared defense has a reading on.

    A mean per column over whichever cells that column happens to cover makes the
    columns incomparable, and it is how the same placement came to read 0.818 here
    and 0.823 in the body. Every column is read on 1 population instead.
    """
    kept = []
    for cell in cells:
        readings = cell_readings(results_dir, cell, HEADLINE_KEY, "auroc")
        if all(value is not None for value in readings.values()):
            kept.append(cell)
    return kept


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage = load_coverage(args.results_dir)
    clearing = [cell for cell in coverage["cells"] if cell.get("asr_class") == "clears"]
    cells = fully_covered(args.results_dir, clearing)
    inputs = [
        f"{args.results_dir}/coverage/coverage.json "
        f"({len(cells)} of {len(clearing)} clearing cells carry every defense)",
        f"{args.results_dir}/*/detectors/*_metrics.json",
        f"{args.results_dir}/*/psbd_metrics.json",
    ]
    header = ["attack, rate %"] + columns()
    align = "l" + "r" * len(columns())

    macros = {
        "detectors_compared_models": (
            str(len(cells)),
            "backdoored models in the detector comparison",
        ),
        "detectors_ibd_psc_factor": (
            f"{DEFAULT_SCALING_FACTOR:g}",
            "the amplification factor the IBD-PSC paper fixes",
        ),
        "detectors_compared_count": (
            str(len(DETECTOR_NAMES)),
            "competitor detectors in the comparison",
        ),
    }
    for stem, key, field, words in METRICS:
        rows, overall, dataset_means, readings = build_rows(
            args.results_dir, cells, key, field
        )
        write_per_dataset_tables(args, stem, words, rows, header, align, inputs)
        macros.update(metric_macros(stem, overall))
        macros.update(
            margin_interval_macros(stem, overall, readings, args.bootstrap, args.seed)
        )
        if stem == "auroc":
            macros.update(leader_macros(dataset_means))
            # The body carries only the per-dataset means. They are transposed, 1
            # row per defense and 1 column per dataset, so the table fits 1 column
            # at full size where the 14-column layout shrank to a 7pt font. The
            # per-attack rows stay in the supplement's tables.
            summary = [row[1:] for row in rows if row[0].startswith(SHADE)]
            # A short header for the one long dataset name keeps the table in 1 column.
            dataset_names = [
                "Tiny" if name == "tiny" else dataset_label(name)
                for name in sorted(dataset_means)
            ]
            transposed = [
                [defense] + [summary[index][position] for index in range(len(summary))]
                for position, defense in enumerate(columns())
            ]
            write_table(
                path=os.path.join(args.paper_dir, "tables", "detectors_summary.tex"),
                generator=GENERATOR,
                inputs=inputs,
                caption=(
                    f"Mean AUROC of every defense per dataset over the {len(cells)} "
                    "backdoored ViT-B/16 models that carry all of them, ours first. "
                    "Best in each column is bold and second best underlined, and gray "
                    "marks a mean below chance. PSBD-RD is our adaptation of the "
                    "original PSBD site. \\Cref{tab:detectors-auroc} gives every "
                    "attack and poison rate."
                ),
                label="tab:detectors-summary",
                header=["defense", *dataset_names, "all"],
                rows=transposed,
                align="l" + "r" * (len(dataset_names) + 1),
            )

    write_macros(
        os.path.join(args.paper_dir, "tables", "detectors.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"detectors: {len(cells)} models, {len(DETECTOR_NAMES)} competitors, "
        f"{len(METRICS)} tables"
    )


if __name__ == "__main__":
    main()

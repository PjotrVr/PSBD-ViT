"""Write the results block of every detector doc under docs/detectors/ from the records on disk.

Each doc carries a `<!-- results:begin -->` and `<!-- results:end -->` pair, and
this script replaces what sits between them. Nothing inside a block is typed by
hand, so a number in a detector doc always traces back to a record under
`results/<folder>/detectors/` and to the PSBD cache beside it.

The population is the paper's detector comparison, read through the same
functions `scripts/paper/tab_detectors.py` uses: the successful models of the
coverage ledger (attack success at the bar, clean accuracy within the headline
bar of the benign model) that carry a reading from every defense. A block therefore
reports the same models the paper's tables report, and its means match the
paper's columns.

`docs/detectors/README.md` carries 1 more block, the composition of that panel:
how many ledger models clear and how many of them are successful, which
successful models lack a defense's reading and are therefore left out, which
models the ledger excludes as diverged, as source-mapped TaCT or as past the
clean-accuracy bar, and how the compared models spread over attacks, rates and
datasets.

`docs/placement-rationale.md` carries the last block, the placement ledger: every
PSBD placement cached on the successful ViT models with its mean AUROC at the
adaptive and the matched rate rule and its paired gain over PSBD-RD, the same
placements on Swin-S and the ResNet-18 control. The placement doc argues from
those tables, so they are generated here from `psbd_metrics.json` like every
other block.

Every detector block holds 5 things for each detector its doc covers: the summary against
PSBD-TM and PSBD-RD with the paired gap and its bootstrap interval, the means per
attack and poison rate, the means per dataset, the data-dependent settings the
detector fitted per model and the measured cost per scored input. The 2 SCALE-UP
variants share `scale_up.md` and the 2 IBD-PSC variants share `ibd_psc.md`, the
mapping `cli.compare.DETECTORS_DOC_OF` already declares.

    python scripts/detector_doc_results.py
    python scripts/detector_doc_results.py --dry-run
    python scripts/detector_doc_results.py --results-dir results --docs-dir docs/detectors
    python scripts/detector_doc_results.py --placement-doc docs/placement-rationale.md
"""

import argparse
import collections
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.getcwd())

from cli.compare import (  # noqa: E402
    DETECTORS_DOC_OF,
    detectors_psbd_values,
    detectors_rewrite_results_block,
)
from defenses.decision import (  # noqa: E402
    HEADLINE_QUANTILE,
    PLACEMENT_MATCH_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from detectors import (  # noqa: E402
    DETECTOR_HYPERPARAMETERS,
    DETECTOR_NAMES,
    FORWARD_PASSES_PER_INPUT,
    NEEDS_FITTING,
)
from scripts.paper._common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    HEADLINE_BAR_POINTS,
    HEADLINE_KEY,
    SECOND_BAR_POINTS,
    SECOND_SUCCESS,
    attack_label,
    bootstrap_ci,
    clearing_cells,
    dataset_label,
    excluded_folders,
    load_coverage,
    load_declaration,
    load_json,
    load_psbd_metrics,
    split_placement,
)
from scripts.paper.tab_detectors import RULE, fully_covered, psbd_reading  # noqa: E402
from scripts.paper.tab_swin import swin_cells  # noqa: E402
from utils.provenance import current_git_commit  # noqa: E402

GENERATOR = "scripts/detector_doc_results.py"
DEFAULT_DOCS_DIR = os.path.join("docs", "detectors")
DEFAULT_RESULTS_DIR = "results"
# The page whose block describes the panel rather than 1 detector.
PANEL_DOC = "README"
DEFAULT_PLACEMENT_DOC = os.path.join("docs", "placement-rationale.md")
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
DEFAULT_DECLARATION = os.path.join("configs", "psbd_basis.json")

# The 2 rate rules a placement is read at: the deployable one and the device that
# compares placements at the same clean disturbance.
PLACEMENT_RULES: tuple[str, ...] = ("adaptive", "matched")

# Cache names that carry a variant of a placement rather than a placement, namely
# a second mask seed, a pass count away from 3 or the model's own dropout switched
# on. They answer other questions and stay out of the ledger.
VARIANT_SUFFIX = re.compile(r"_(seed\d+|k\d+|pmodel[\d_]+)$")

# A placement read on fewer models than this is listed but carries no interval.
MIN_MODELS_FOR_INTERVAL = 3

# The 3 readings every table carries, as (column name, quantile key, record field).
# AUROC does not depend on the quantile, so it is read at the headline one.
METRICS: tuple[tuple[str, str, str], ...] = (
    ("AUROC", HEADLINE_KEY, "auroc"),
    ("TPR at 10% FPR", "q0.10", "tpr"),
    ("TPR at 20% FPR", "q0.20", "tpr"),
)

# Our 2 reference columns, the recommended placement and the published one.
PSBD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("PSBD-TM", RECOMMENDED_PLACEMENT),
    ("PSBD-RD", PUBLISHED_PLACEMENT),
)

# A one-sided AUROC below this ordered poisoned and clean inputs the wrong way round.
CHANCE = 0.5

# A list or dict in a record's settings is a ladder or a table the registry fixed,
# such as IBD-PSC's omega ladder, and never a value fitted per model.
SKIPPED_SETTING_TYPES = (list, dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--placement-doc", default=DEFAULT_PLACEMENT_DOC)
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--declaration", default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every block instead of writing it into its doc",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cells = comparison_cells(args.results_dir)
    readings = collect_readings(args.results_dir, cells)
    panel_means = defense_means(readings)

    blocks = {PANEL_DOC: panel_block(args.results_dir, cells)}
    for doc_name, detector_names in docs_and_detectors().items():
        blocks[doc_name] = render_block(
            detector_names, readings, panel_means, cells, args
        )

    write_blocks(args.docs_dir, blocks, args.dry_run)

    placement_lines = placement_block(args)
    write_placement_block(args.placement_doc, placement_lines, args.dry_run)
    print(
        f"{GENERATOR}: {len(blocks)} detector docs, {len(cells)} models, 1 placement doc"
    )


def comparison_cells(results_dir: str) -> list[dict]:
    """The paper's detector comparison: successful models that every defense scored.

    Ordered by dataset, then attack and rate, the order tab_detectors.build_rows
    walks, so a seeded bootstrap here resamples the same sequence the paper's
    interval macros resample and the 2 intervals agree.
    """
    coverage = load_coverage(results_dir)
    clearing = clearing_cells(coverage)
    cells = fully_covered(results_dir, clearing)

    # clearing_cells keeps only successful models, so a diverged, source-mapped
    # or clean-accuracy-failing model cannot be here. The check makes that
    # dependence explicit rather than assumed.
    leaked = {cell["folder_name"] for cell in cells} & excluded_folders(results_dir)
    if leaked:
        raise SystemExit(f"excluded models reached the panel: {sorted(leaked)}")

    ordered = sorted(
        cells,
        key=lambda cell: (cell["dataset"], cell["attack"], cell["poison_rate"]),
    )
    return ordered


def panel_block(results_dir: str, cells: list[dict]) -> list[str]:
    """The README block: how the compared models were chosen and what they cover."""
    coverage = load_coverage(results_dir)
    ledger = coverage["cells"]
    clearing = clearing_cells(coverage)
    compared = {cell["folder_name"] for cell in cells}
    classes = collections.Counter(cell.get("asr_class") for cell in ledger)
    datasets = sorted({cell["dataset"] for cell in ledger})

    lines = [
        f"Generated by `python {GENERATOR}` at commit "
        f"`{current_git_commit() or 'unknown'}` from "
        f"`{results_dir}/coverage/coverage.json` and the detector records. The "
        f"ledger holds {len(ledger)} ViT models on "
        f"{', '.join(dataset_label(name) for name in datasets[:-1])} and "
        f"{dataset_label(datasets[-1])}.",
        "",
        "| ledger class | models | meaning |",
        "|---|---|---|",
    ]
    meanings = {
        "clears": "attack success at or above the bar",
        "below_bar": "the attack did not implant strongly enough",
        "diverged": "clean accuracy below half the benign reference, excluded",
        "source_mapped": (
            "TaCT sends its whole clean source class to the target with no "
            "trigger, excluded"
        ),
    }
    for name in sorted(classes, key=str):
        lines.append(f"| `{name}` | {classes[name]} | {meanings.get(name, '')} |")
    lines.append("")

    successful = {cell["folder_name"] for cell in clearing}
    failing = [
        cell
        for cell in ledger
        if cell.get("asr_class") == "clears" and cell["folder_name"] not in successful
    ]
    second = clearing_cells(coverage, SECOND_SUCCESS)
    failing_names = ", ".join(f"`{cell['folder_name']}`" for cell in failing)
    lines += [
        f"{len(clearing)} of the {classes['clears']} `clears` models are successful "
        f"backdoors, with clean accuracy within {HEADLINE_BAR_POINTS} points of the "
        f"benign model ({len(second)} within {SECOND_BAR_POINTS} points). The "
        f"{len(failing)} left out by the clean-accuracy bar: "
        f"{failing_names}.",
        "",
    ]

    left_out = [cell for cell in clearing if cell["folder_name"] not in compared]
    lines += [
        f"{len(compared)} of the {len(clearing)} successful models carry a reading "
        "from all 13 defenses and form the panel every results block reports. "
        "The successful models left out, and the defenses they lack, are listed "
        "below.",
        "",
        "| left out | defenses without a reading |",
        "|---|---|",
    ]
    for cell in left_out:
        missing = missing_defenses(results_dir, cell)
        text = (
            "all 13, nothing scored yet"
            if len(missing) == len(DETECTOR_NAMES) + len(PSBD_COLUMNS)
            else ", ".join(defense_name(name) for name in missing)
        )
        lines.append(f"| `{cell['folder_name']}` | {text} |")
    lines.append("")

    excluded = sorted(excluded_folders(results_dir))
    lines += [
        "Models the ledger excludes as diverged, source-mapped or past the "
        "clean-accuracy bar, which no block "
        f"reads: {', '.join(f'`{name}`' for name in excluded)}.",
        "",
    ]

    rates = sorted({cell["poison_rate"] for cell in cells})
    by_attack = group_rows(cells, ("attack", "poison_rate"))
    attacks = sorted({cell["attack"] for cell in cells}, key=attack_label)
    header = " | ".join(f"{rate * 100:g}%" for rate in rates)
    lines += [
        "Compared models per attack and poison rate. A dash means no model of "
        "that attack cleared at that rate.",
        "",
        f"| attack | {header} | total |",
        "|---|" + "---|" * (len(rates) + 1),
    ]
    for attack in attacks:
        counts = [len(by_attack.get((attack, rate), [])) for rate in rates]
        text = " | ".join(str(count) if count else "--" for count in counts)
        lines.append(f"| {attack_label(attack)} | {text} | {sum(counts)} |")
    lines.append("")

    by_dataset = group_rows(cells, ("dataset",))
    lines += [
        "Compared models per dataset. The attacks that cleared differ slightly "
        "between datasets, so the per-dataset means of a block average slightly "
        "different mixes of attacks.",
        "",
        "| dataset | models |",
        "|---|---|",
    ]
    for (dataset,), rows in sorted(by_dataset.items()):
        lines.append(f"| {dataset_label(dataset)} | {len(rows)} |")
    lines.append("")
    return lines


def docs_and_detectors() -> dict[str, list[str]]:
    """Every doc under docs/detectors/ with the registered detectors it reports."""
    docs: dict[str, list[str]] = collections.defaultdict(list)
    for name in DETECTOR_NAMES:
        docs[DETECTORS_DOC_OF.get(name, name)].append(name)
    return dict(docs)


def missing_defenses(results_dir: str, cell: dict) -> list[str]:
    """The defenses, PSBD columns first, with no reading on 1 successful model."""
    report = load_psbd_metrics(results_dir, cell["folder_name"]) or {}
    missing = [
        column
        for column, placement in PSBD_COLUMNS
        if psbd_reading(report, placement, HEADLINE_KEY, "auroc") is None
    ]
    missing += [
        name
        for name in DETECTOR_NAMES
        if detector_record(results_dir, cell["folder_name"], name) is None
    ]
    return missing


def detector_record(results_dir: str, folder: str, name: str) -> dict | None:
    """A scored detector record for 1 model, or None when it never scored there."""
    path = os.path.join(results_dir, folder, "detectors", f"{name}_metrics.json")
    record = load_json(path)
    if record is None or record.get("status") != "scored":
        return None
    return record


def record_metrics(record: dict) -> dict[str, float | None]:
    """The 3 table readings of 1 detector record."""
    detection = record.get("detection") or {}
    values = {
        column: (detection.get(key) or {}).get(field) for column, key, field in METRICS
    }
    return values


def collect_readings(results_dir: str, cells: list[dict]) -> dict[str, list[dict]]:
    """Every defense's readings on every compared model, keyed by defense name.

    Each entry holds the model's attack, rate and dataset, the 3 metrics and, for
    a competitor, its whole record, so the settings and cost tables read the same
    pass over the disk.
    """
    readings: dict[str, list[dict]] = collections.defaultdict(list)
    for cell in cells:
        identity = {
            "folder": cell["folder_name"],
            "attack": cell["attack"],
            "poison_rate": cell["poison_rate"],
            "dataset": cell["dataset"],
        }

        report = load_psbd_metrics(results_dir, cell["folder_name"]) or {}
        for column, placement in PSBD_COLUMNS:
            values = {
                metric: psbd_reading(report, placement, key, field)
                for metric, key, field in METRICS
            }
            readings[column].append({**identity, **values, "record": None})

        for name in DETECTOR_NAMES:
            record = detector_record(results_dir, cell["folder_name"], name)
            if record is None:
                raise SystemExit(
                    f"{name} has no scored record on {cell['folder_name']}, which "
                    "fully_covered should have excluded"
                )
            readings[name].append(
                {**identity, **record_metrics(record), "record": record}
            )
    return dict(readings)


def defense_means(readings: dict[str, list[dict]]) -> dict[str, float]:
    """Every defense's mean AUROC over the compared models."""
    means = {
        defense: statistics.mean(row["AUROC"] for row in rows)
        for defense, rows in readings.items()
    }
    return means


def render_block(
    detector_names: list[str],
    readings: dict[str, list[dict]],
    panel_means: dict[str, float],
    cells: list[dict],
    args: argparse.Namespace,
) -> list[str]:
    """The markdown lines that go between the results markers of 1 doc."""
    lines = provenance_lines(cells, args.results_dir)
    lines += summary_table(detector_names, readings, panel_means, args)
    lines += per_attack_tables(detector_names, readings)
    lines += per_dataset_table(detector_names, readings)
    for name in detector_names:
        lines += fitted_settings_table(name, readings[name])
    lines += cost_table(detector_names, readings)
    return lines


def provenance_lines(cells: list[dict], results_dir: str) -> list[str]:
    """Where the block came from, so a reader can rebuild it."""
    commit = current_git_commit() or "unknown"
    lines = [
        f"Generated by `python {GENERATOR}` at commit `{commit}`. It reads "
        f"`{results_dir}/coverage/coverage.json`, every "
        f"`{results_dir}/<folder>/detectors/<name>_metrics.json` and every "
        f"`{results_dir}/<folder>/psbd_metrics.json` of the {len(cells)} backdoored "
        "ViT-B/16 models the paper's detector comparison uses, the successful models "
        "that carry a reading from every defense. PSBD-TM and PSBD-RD are read at "
        f"the {RULE} rate rule, every threshold is the clean-validation quantile "
        f"named in the column and AUROC is the one-sided area at the "
        f"{HEADLINE_QUANTILE:g} quantile, where a value under 0.5 means inverted.",
        "",
    ]
    return lines


def summary_table(
    detector_names: list[str],
    readings: dict[str, list[dict]],
    panel_means: dict[str, float],
    args: argparse.Namespace,
) -> list[str]:
    """Each detector beside PSBD-TM and PSBD-RD, with its rank and the paired gap."""
    ranked = sorted(panel_means, key=panel_means.get, reverse=True)
    lines = [
        "Summary over every compared model. The rank is among the "
        f"{len(ranked)} defenses of the comparison by mean AUROC, and the last "
        "column is PSBD-TM minus the defense, paired per model, with its 95% "
        f"bootstrap interval over models ({args.bootstrap} resamples, seed "
        f"{args.seed}).",
        "",
        "| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR "
        "| models below chance | rank | PSBD-TM minus defense, AUROC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    reference = readings["PSBD-TM"]
    for defense in [*detector_names, "PSBD-TM", "PSBD-RD"]:
        rows = readings[defense]
        means = {column: mean_of(rows, column) for column, _, _ in METRICS}
        below = sum(1 for row in rows if row["AUROC"] < CHANCE)
        rank = ranked.index(defense) + 1
        gap_text = "reference"
        if defense != "PSBD-TM":
            differences = [
                ours["AUROC"] - theirs["AUROC"] for ours, theirs in zip(reference, rows)
            ]
            low, high = bootstrap_ci(differences, args.bootstrap, args.seed)
            gap_text = (
                f"{signed(statistics.mean(differences))} "
                f"[{signed(low)}, {signed(high)}]"
            )
        lines.append(
            f"| {defense_name(defense)} | {len(rows)} | {number(means['AUROC'])} "
            f"| {number(means['TPR at 10% FPR'])} | {number(means['TPR at 20% FPR'])} "
            f"| {below} | {rank} of {len(ranked)} | {gap_text} |"
        )
    lines.append("")
    return lines


def per_attack_tables(
    detector_names: list[str], readings: dict[str, list[dict]]
) -> list[str]:
    """1 table per detector: every attack at every poison rate against PSBD-TM and PSBD-RD."""
    lines = []
    for name in detector_names:
        groups = group_rows(readings[name], ("attack", "poison_rate"))
        tm_groups = group_rows(readings["PSBD-TM"], ("attack", "poison_rate"))
        rd_groups = group_rows(readings["PSBD-RD"], ("attack", "poison_rate"))
        lines += [
            f"{defense_name(name)} per attack and poison rate. Each row is a mean "
            "over the models of that attack at that rate and n counts them. The "
            "last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same "
            "models.",
            "",
            "| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR "
            "| PSBD-TM AUROC | PSBD-RD AUROC |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key in sorted(groups, key=lambda item: (attack_label(item[0]), item[1])):
            rows = groups[key]
            attack, rate = key
            lines.append(
                f"| {attack_label(attack)} | {rate * 100:g}% | {len(rows)} "
                f"| {number(mean_of(rows, 'AUROC'))} "
                f"| {number(mean_of(rows, 'TPR at 10% FPR'))} "
                f"| {number(mean_of(rows, 'TPR at 20% FPR'))} "
                f"| {number(mean_of(tm_groups[key], 'AUROC'))} "
                f"| {number(mean_of(rd_groups[key], 'AUROC'))} |"
            )
        lines.append("")
    return lines


def per_dataset_table(
    detector_names: list[str], readings: dict[str, list[dict]]
) -> list[str]:
    """Mean AUROC per dataset for each detector and both PSBD placements."""
    defenses = [*detector_names, "PSBD-TM", "PSBD-RD"]
    datasets = sorted({row["dataset"] for row in readings["PSBD-TM"]})
    header = " | ".join(dataset_label(dataset) for dataset in datasets)
    lines = [
        "Mean AUROC per dataset. The shared 2000-image clean split gives about 200 "
        "images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny "
        "ImageNet, which is the budget every class-conditional method fits on.",
        "",
        f"| defense | {header} |",
        "|---|" + "---|" * len(datasets),
    ]
    for defense in defenses:
        groups = group_rows(readings[defense], ("dataset",))
        values = " | ".join(
            f"{number(mean_of(groups[(dataset,)], 'AUROC'))} ({len(groups[(dataset,)])})"
            for dataset in datasets
        )
        lines.append(f"| {defense_name(defense)} | {values} |")
    lines.append("")
    return lines


def fitted_settings_table(name: str, rows: list[dict]) -> list[str]:
    """The settings a detector fitted per model, counted per value.

    A registry constant is identical on every model and says nothing per model.
    What the detector chose on the clean split, such as the layer count
    Algorithm 1 of IBD-PSC picks, is a key the registry table does not hold, and
    a key that differs between models was overridden per run. Both are tabulated.
    """
    values_by_key: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for row in rows:
        settings = row["record"]["provenance"].get("hyperparameters", {})
        for key, value in settings.items():
            if isinstance(value, SKIPPED_SETTING_TYPES):
                continue
            values_by_key[key][value] += 1

    registry_keys = set(DETECTOR_HYPERPARAMETERS[name])
    varying = {
        key: counter
        for key, counter in values_by_key.items()
        if len(counter) > 1 or key not in registry_keys
    }
    if not varying:
        return []

    lines = [
        f"Settings {defense_name(name)} fitted per model. Each cell is a value the "
        "record's provenance carries and the number of models that took it.",
        "",
        "| setting | value (models) |",
        "|---|---|",
    ]
    for key in sorted(varying):
        counts = sorted(varying[key].items(), key=lambda item: str(item[0]))
        text = ", ".join(f"{value} ({count})" for value, count in counts)
        lines.append(f"| `{key}` | {text} |")
    lines.append("")
    return lines


def cost_table(detector_names: list[str], readings: dict[str, list[dict]]) -> list[str]:
    """Measured wall-clock cost per scored input and per fit, median over models."""
    lines = [
        "Measured cost, median over the compared models. Seconds per 1000 inputs "
        "divide the scoring time of the clean and backdoor splits by their size. "
        "The fit is the one-off pass over the clean validation split before any "
        "input is scored, and a dash marks a detector with no fit. The device is "
        "the one most records name.",
        "",
        "| detector | forward passes per input | seconds per 1000 inputs "
        "| fit seconds | precision | device |",
        "|---|---|---|---|---|---|",
    ]
    for name in detector_names:
        per_thousand = []
        fits = []
        precisions: collections.Counter = collections.Counter()
        devices: collections.Counter = collections.Counter()
        for row in readings[name]:
            provenance = row["record"]["provenance"]
            runtime = provenance.get("runtime_seconds", {})
            split = provenance.get("split", {})
            scored = split.get("n_clean", 0) + split.get("n_backdoor", 0)
            if scored and "clean" in runtime and "backdoor" in runtime:
                per_thousand.append(
                    1000.0 * (runtime["clean"] + runtime["backdoor"]) / scored
                )
            if name in NEEDS_FITTING and "fit" in runtime:
                fits.append(runtime["fit"])
            precisions[provenance.get("precision", "unknown")] += 1
            devices[provenance.get("device", "unknown")] += 1

        lines.append(
            f"| {defense_name(name)} | {FORWARD_PASSES_PER_INPUT[name]} "
            f"| {median_text(per_thousand, 2)} | {median_text(fits, 1)} "
            f"| {most_common(precisions)} | {most_common(devices)} |"
        )
    lines.append("")
    return lines


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    """Rows grouped by the tuple of their values under keys."""
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return dict(groups)


def mean_of(rows: list[dict], column: str) -> float | None:
    """The mean of 1 column over rows, skipping models with no reading there."""
    present = [row[column] for row in rows if row[column] is not None]
    if not present:
        return None
    mean = statistics.mean(present)
    return mean


def median_text(values: list[float], places: int) -> str:
    """A median at a fixed number of decimals, or a dash when nothing was measured."""
    if not values:
        return "--"
    text = f"{statistics.median(values):.{places}f}"
    return text


def most_common(counter: collections.Counter) -> str:
    """The most frequent value of a counter, with its share when it is not unanimous."""
    value, count = counter.most_common(1)[0]
    total = sum(counter.values())
    text = str(value) if count == total else f"{value} ({count} of {total})"
    return text


def number(value: float | None) -> str:
    """A reading at 3 decimals, or a dash when the model set had none."""
    text = "--" if value is None else f"{value:.3f}"
    return text


def signed(value: float) -> str:
    """A difference at 3 decimals with its sign always written."""
    text = f"{value:+.3f}"
    return text


def defense_name(name: str) -> str:
    """A registry name as code, or a PSBD column name as it stands."""
    text = name if name.startswith("PSBD") else f"`{name}`"
    return text


def write_blocks(docs_dir: str, blocks: dict[str, list[str]], dry_run: bool) -> None:
    """Replace the results block of every doc, or print the blocks on a dry run."""
    for doc_name, body in sorted(blocks.items()):
        path = os.path.join(docs_dir, f"{doc_name}.md")
        if dry_run:
            print(f"==> {path}")
            print("\n".join(body))
            continue
        detectors_rewrite_results_block(path, body)


def placement_panel(results_dir: str) -> list[dict]:
    """The successful ViT models that PSBD placements are read on, excluded ones removed."""
    coverage = load_coverage(results_dir)
    excluded = excluded_folders(results_dir)
    cells = [
        cell for cell in clearing_cells(coverage) if cell["folder_name"] not in excluded
    ]
    ordered = sorted(
        cells,
        key=lambda cell: (cell["dataset"], cell["attack"], cell["poison_rate"]),
    )
    return ordered


def placement_readings(
    reports: dict[str, dict], rule: str
) -> dict[str, dict[str, float]]:
    """Headline AUROC per placement and model at 1 rate rule, variants left out.

    reports maps a model's folder to its psbd_metrics.json. The result maps a
    placement to {folder: AUROC} over the models whose rule chose a rate.
    """
    readings: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for folder, report in reports.items():
        for placement in report.get("placements", {}):
            if VARIANT_SUFFIX.search(placement):
                continue
            auroc = rule_reading(report, placement, rule)
            if auroc is not None:
                readings[placement][folder] = auroc
    return dict(readings)


def rule_reading(report: dict, placement: str, rule: str) -> float | None:
    """The headline AUROC of 1 placement at the rate 1 rule chose, or None."""
    block = detectors_psbd_values(report, placement, rule)
    if block is None or HEADLINE_KEY not in block:
        return None
    auroc = block[HEADLINE_KEY]["auroc"]
    return auroc


def paired_gain(
    ours: dict[str, float], reference: dict[str, float], args: argparse.Namespace
) -> str:
    """Mean of ours minus reference over the models carrying both, with its interval."""
    common = [folder for folder in ours if folder in reference]
    if len(common) < MIN_MODELS_FOR_INTERVAL:
        return "--"
    differences = [ours[folder] - reference[folder] for folder in common]
    low, high = bootstrap_ci(differences, args.bootstrap, args.seed)
    text = (
        f"{signed(statistics.mean(differences))} "
        f"[{signed(low)}, {signed(high)}] ({len(common)})"
    )
    return text


def placement_table(
    by_rule: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
    reference: str,
) -> list[str]:
    """1 row per placement, sorted by matched-rule mean, both rules and both gains."""
    placements = sorted(
        by_rule["matched"],
        key=lambda name: -statistics.mean(by_rule["matched"][name].values()),
    )
    lines = [
        "| placement | position | operator | blocks | n matched | AUROC matched "
        "| n adaptive | AUROC adaptive | below chance, adaptive "
        f"| gain over {reference}, matched | gain over {reference}, adaptive |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in placements:
        parts = split_placement(name)
        band = parts["block_range"]
        matched = by_rule["matched"].get(name, {})
        adaptive = by_rule["adaptive"].get(name, {})
        adaptive_mean = number(statistics.mean(adaptive.values())) if adaptive else "--"
        below = sum(1 for value in adaptive.values() if value < CHANCE)
        band_text = f"{band[0]} to {band[1]}" if band else "all"
        gains = (
            ("reference", "reference")
            if name == reference
            else (
                paired_gain(matched, by_rule["matched"].get(reference, {}), args),
                paired_gain(adaptive, by_rule["adaptive"].get(reference, {}), args),
            )
        )
        lines.append(
            f"| `{name}` | `{parts['position']}` | `{parts['operator']}` "
            f"| {band_text} | {len(matched)} "
            f"| {number(statistics.mean(matched.values()))} | {len(adaptive)} "
            f"| {adaptive_mean} | {below} | {gains[0]} | {gains[1]} |"
        )
    lines.append("")
    return lines


def resnet_control_reports(results_dir: str) -> dict[str, dict]:
    """The ResNet-18 control models, the PSBD paper's own architecture, without evasion runs."""
    reports = {}
    for path in sorted(
        glob.glob(os.path.join(results_dir, "resnet18_*", "psbd_metrics.json"))
    ):
        folder = os.path.basename(os.path.dirname(path))
        if "evade" in folder or "smoke" in folder:
            continue
        reports[folder] = load_json(path)
    return reports


def placement_block(args: argparse.Namespace) -> list[str]:
    """The ledger docs/placement-rationale.md argues from, on ViT, Swin and ResNet."""
    cells = placement_panel(args.results_dir)
    reports = {
        cell["folder_name"]: load_psbd_metrics(args.results_dir, cell["folder_name"])
        for cell in cells
    }
    reports = {folder: report for folder, report in reports.items() if report}
    vit = {rule: placement_readings(reports, rule) for rule in PLACEMENT_RULES}

    declaration = load_declaration(args.declaration)
    swin = swin_cells(args.results_dir, args.checkpoints_dir, declaration["asr_bar"])
    swin_reports = {cell["folder"]: cell["report"] for cell in swin}
    swin_by_rule = {
        rule: placement_readings(swin_reports, rule) for rule in PLACEMENT_RULES
    }

    lines = [
        f"Generated by `python {GENERATOR}` at commit "
        f"`{current_git_commit() or 'unknown'}` from every "
        f"`{args.results_dir}/<folder>/psbd_metrics.json`. AUROC is the one-sided "
        f"area at the {HEADLINE_QUANTILE:g} quantile. The adaptive rule reads each "
        "placement at the smallest rate whose clean-validation shift ratio reaches "
        "0.8, and the matched rule at the rate whose shift ratio is closest to "
        f"{PLACEMENT_MATCH_TARGET:g}. A gain is the placement minus "
        f"`{PUBLISHED_PLACEMENT}` (PSBD-RD), paired over the models carrying both, "
        f"with its 95% bootstrap interval ({args.bootstrap} resamples, seed "
        f"{args.seed}) and the model count in brackets. Both placements are read "
        "at the same rule, so the matched gain differs from the paper's headline "
        "matched gain, which reads PSBD-RD at the adaptive rule.",
        "",
        f"ViT-B/16. The {len(reports)} successful models of the coverage ledger with "
        "a PSBD cache, diverged, source-mapped and clean-accuracy-failing models "
        "excluded. The n columns "
        "differ between rows because some placements were swept on a subset only, "
        "so compare 2 rows through their paired gains rather than their means.",
        "",
    ]
    lines += placement_table(vit, args, PUBLISHED_PLACEMENT)
    lines += [
        f"Swin-S. The {len(swin_reports)} Swin models `scripts/paper/tab_swin.py` "
        "reads: the ViT panel rule applied to Swin-S "
        "(`scripts.paper._common.swin_coverage`), the models that are successful "
        f"backdoors with clean accuracy within {HEADLINE_BAR_POINTS} points of the "
        "benign Swin-S model and carry a PSBD cache. Swin has 24 blocks, so its bands are 1 to 8, "
        "9 to 16 and 17 to 24.",
        "",
    ]
    lines += placement_table(swin_by_rule, args, PUBLISHED_PLACEMENT)

    resnet = resnet_control_reports(args.results_dir)
    lines += [
        "ResNet-18 control, the PSBD paper's own architecture and site "
        "(`experiments/resnet_control/`). Each row is 1 model.",
        "",
        "| model | placement | adaptive rate | AUROC adaptive | AUROC matched |",
        "|---|---|---|---|---|",
    ]
    for folder, report in resnet.items():
        for placement in report.get("placements", {}):
            block = report["placements"][placement]
            lines.append(
                f"| `{folder}` | `{placement}` | {block.get('adaptive_rate')} "
                f"| {number(rule_reading(report, placement, 'adaptive'))} "
                f"| {number(rule_reading(report, placement, 'matched'))} |"
            )
    lines.append("")
    return lines


def write_placement_block(path: str, body: list[str], dry_run: bool) -> None:
    """Replace the placement ledger block, or print it on a dry run."""
    if dry_run:
        print(f"==> {path}")
        print("\n".join(body))
        return
    detectors_rewrite_results_block(path, body)


if __name__ == "__main__":
    main()

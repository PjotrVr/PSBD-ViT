"""T1: the ViT panel, attack success and clean accuracy by dataset, attack and rate.

Reads results/coverage/coverage.json only, no psbd_metrics.json. Every declared
cell appears, so a reader can see which cells the ASR bar and the headline
clean-accuracy bar exclude rather than finding them silently missing. A
successful cell prints its attack success rate and clean accuracy together. A
cell below the ASR bar prints only its attack success rate, marked with a
dagger, since its clean accuracy is not the number that explains the cell's
absence from every other table in this paper. A cell that clears the ASR bar
but loses more clean accuracy than the headline bar allows prints both, marked
with a section sign, since there the clean accuracy is the reason.

    PYTHONPATH=. python scripts/paper/tab_panel.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from defenses.decision import EASY_ATTACKS, HARD_ATTACKS  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_BAR_POINTS,
    HEADLINE_SUCCESS,
    PANEL_DATASETS,
    SECOND_BAR_POINTS,
    SECOND_SUCCESS,
    attack_label,
    clearing_cells,
    implanted_cells,
    build_parser,
    dataset_label,
    load_coverage,
    word_list,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_panel.py"
DATASET_ORDER = PANEL_DATASETS
RATE_ORDER = (0.01, 0.05, 0.1)
RATE_HEADERS = ("1%", "5%", "10%")


def attack_order(cells: list[dict]) -> list[str]:
    """Attacks present in the panel, easy first then hard, each ordered as decision.py fixes it.

    An attack the panel declares but never runs (excluded folder tokens, a
    future addition) falls back to alphabetical order after the 2 named groups,
    so a new attack cannot silently drop off the table.
    """
    present = {cell["attack"] for cell in cells}
    named = [attack for attack in EASY_ATTACKS + HARD_ATTACKS if attack in present]
    leftover = sorted(present - set(named))
    ordered = named + leftover
    return ordered


def cell_text(cell: dict | None) -> str:
    """ASR / clean accuracy for a clearing cell, ASR alone with a dagger below the bar."""
    if cell is None or cell.get("asr") is None:
        return "--"
    asr = cell["asr"]
    if cell["asr_class"] == "source_mapped":
        return f"{asr:.3f}$^\\ddagger$"
    if cell["asr_class"] != "clears":
        return f"{asr:.3f}$^\\dagger$"
    clean_accuracy = cell.get("clean_accuracy")
    if clean_accuracy is None:
        return f"{asr:.3f} / --"
    mark = "" if cell[HEADLINE_SUCCESS] else "$^\\S$"
    return f"{asr:.3f} / {clean_accuracy:.3f}{mark}"


def panel_rows(
    cells_by_key: dict[tuple[str, str, float], dict],
    datasets: tuple[str, ...],
    attacks: list[str],
) -> list[list[str]]:
    """1 row per dataset and attack, 1 column per poison rate."""
    rows = []
    for dataset in datasets:
        for attack in attacks:
            row = [dataset_label(dataset), attack_label(attack)]
            row += [
                cell_text(cells_by_key.get((dataset, attack, rate)))
                for rate in RATE_ORDER
            ]
            rows.append(row)
    return rows


def benign_rows(
    benign_reference_accuracy: dict[str, float], datasets: tuple[str, ...]
) -> list[list[str]]:
    rows = [
        [dataset_label(dataset), f"{benign_reference_accuracy[dataset]:.3f}"]
        for dataset in datasets
        if dataset in benign_reference_accuracy
    ]
    return rows


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    coverage = load_coverage(args.results_dir)
    cells = coverage["cells"]
    asr_bar = coverage["asr_bar"]
    clearing = implanted_cells(coverage)
    successful = clearing_cells(coverage)
    second = clearing_cells(coverage, SECOND_SUCCESS)
    failing = [cell for cell in clearing if not cell[HEADLINE_SUCCESS]]

    datasets = DATASET_ORDER
    attacks = attack_order(cells)
    # A retrained variant shares its (dataset, attack, rate) slot with the run it
    # replaces, so the clearing one takes the slot whatever the folder order.
    cells_by_key = {}
    for cell in sorted(cells, key=lambda cell: cell["asr_class"] == "clears"):
        cells_by_key[(cell["dataset"], cell["attack"], cell["poison_rate"])] = cell
    source_mapped = [cell for cell in cells if cell["asr_class"] == "source_mapped"]
    rows = panel_rows(cells_by_key, datasets, attacks)

    write_table(
        path=os.path.join(args.paper_dir, "tables", "panel.tex"),
        generator=GENERATOR,
        inputs=[coverage_path],
        caption=(
            f"The ViT panel, from results/coverage/coverage.json: {len(clearing)} of "
            f"{len(cells)} declared models clear the attack success bar {asr_bar:.2f}, "
            f"and the {len(successful)} of them whose clean accuracy is within "
            f"{HEADLINE_BAR_POINTS} points of the benign model carry every table in "
            "this paper. A model that clears prints attack success rate over clean "
            "accuracy. A model below the bar prints only its attack success rate, "
            r"marked $^\dagger$, so its exclusion stays visible. $^\S$ marks a model "
            f"that clears but loses more than {HEADLINE_BAR_POINTS} points of clean "
            r"accuracy. $^\ddagger$ marks a TaCT model that maps its whole source "
            "class to the target without the trigger, excluded because it is not a "
            "trigger backdoor."
        ),
        label="tab:panel",
        header=["dataset", "attack", *RATE_HEADERS],
        rows=rows,
        align="ll" + "r" * len(RATE_ORDER),
    )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "panel_benign.tex"),
        generator=GENERATOR,
        inputs=[coverage_path],
        caption=(
            "Benign-model clean accuracy per dataset, the reference every "
            "clean-accuracy-drop figure in this paper subtracts against. Same "
            "architecture, optimizer and 15 epochs as every backdoored model in the "
            "panel."
        ),
        label="tab:panel-benign",
        header=["dataset", "benign clean accuracy"],
        rows=benign_rows(coverage["benign_reference_accuracy"], datasets),
        align="lr",
    )

    macros = {
        "panel_cells_clearing": (
            str(len(clearing)),
            "cells in the ViT panel clearing the attack success bar",
        ),
        "panel_cells_successful": (
            str(len(successful)),
            "cells in the ViT panel that are successful backdoors, clearing the "
            f"attack success bar with clean accuracy within {HEADLINE_BAR_POINTS} "
            "points of the benign model, the population every result spans",
        ),
        "panel_cells_successful_five_point": (
            str(len(second)),
            "cells in the ViT panel clearing the attack success bar with clean "
            f"accuracy within {SECOND_BAR_POINTS} points of the benign model",
        ),
        "panel_cells_failing_clean_bar": (
            str(len(failing)),
            "cells clearing the attack success bar that lose more than "
            f"{HEADLINE_BAR_POINTS} points of clean accuracy, excluded from every "
            "result",
        ),
        "panel_cells_failing_clean_bar_five_point": (
            str(len(clearing) - len(second)),
            "cells clearing the attack success bar that lose more than "
            f"{SECOND_BAR_POINTS} points of clean accuracy",
        ),
        "panel_cells_failing_clean_bar_names": (
            word_list(
                [
                    f"{attack_label(cell['attack'])} at "
                    f"{cell['poison_rate'] * 100:g}\\% on {dataset_label(cell['dataset'])}"
                    for cell in sorted(failing, key=lambda cell: cell["folder_name"])
                ]
            ),
            "the cells clearing the attack success bar that lose more than "
            f"{HEADLINE_BAR_POINTS} points of clean accuracy, as a phrase",
        ),
        "panel_clean_bar_points": (
            str(HEADLINE_BAR_POINTS),
            "the headline clean-accuracy bar, in points below the benign model",
        ),
        "panel_clean_bar_points_second": (
            str(SECOND_BAR_POINTS),
            "the second clean-accuracy bar every number is also given at, in points",
        ),
        "panel_attacks_successful": (
            str(len({cell["attack"] for cell in successful})),
            "attacks with at least 1 successful cell",
        ),
        "panel_cells_total": (
            str(len(cells)),
            "cells the ViT panel declares, clearing and below the bar together",
        ),
        "panel_datasets": (str(len(datasets)), "datasets in the ViT panel"),
        "panel_cells_source_mapped": (
            str(len(source_mapped)),
            "TaCT cells excluded because clean source-class images lose their label",
        ),
        "panel_attacks_clearing": (
            str(len({cell["attack"] for cell in clearing})),
            "attacks with at least 1 cell clearing the attack success bar",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "panel.macros.json"),
        GENERATOR,
        [coverage_path],
        macros,
    )
    print(
        f"panel: {len(clearing)}/{len(cells)} cells clear, {len(successful)} "
        f"successful, {len(datasets)} datasets"
    )


if __name__ == "__main__":
    main()

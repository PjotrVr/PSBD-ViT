"""Which predictions survive each placement: the triggered and the clean shift ratio.

PSBD flags an input whose prediction barely moves under the perturbation, so a
placement detects exactly when it moves clean predictions and leaves triggered
ones in place. Every psbd_metrics.json already records, per rate, the share of
clean and of triggered predictions that change (the shift ratio of each split).
This reads both at each placement's adaptive rate, where by construction about
0.8 of clean validation predictions change, and asks how many triggered
predictions change at the same rate. It needs no new forward pass.

Attacks are grouped by where the trigger sits. A patch trigger (BadNets, TaCT,
Label-Consistent) covers a few tokens and a global trigger (Blend, SIG, WaNet,
LF, BPP) covers every token, which is the split the mechanism section turns on.

    PYTHONPATH=. python scripts/paper/mech_survival.py \
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._style import TEXT_WIDTH, attack_color, legend_above  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from defenses.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    attack_label,
    build_parser,
    clearing_cells,
    figure_sidecar,
    fmt,
    load_coverage,
    load_psbd_metrics,
    mean_or_none,
    rate_row,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_survival.py"
PLACEMENTS = {"tm": RECOMMENDED_PLACEMENT, "rd": PUBLISHED_PLACEMENT}
PLACEMENT_NAMES = {"tm": "PSBD-TM", "rd": "PSBD-RD"}
PATCH_ATTACKS = ("badnet_a2o", "tact", "lc")
GLOBAL_ATTACKS = ("blend", "sig", "wanet", "lf", "bpp")
ATTACK_ORDER = PATCH_ATTACKS + GLOBAL_ATTACKS
# The placements that separate what lets a patch trigger through: whole-token
# masking against partial perturbation at the same site, and the same token mask
# on a sublayer input, on a branch output and on the residual stream itself.
# key, placement id, label.
SITE_ROWS = (
    (
        "mask_attention_input",
        "before_attention_norm_token_mask",
        "token mask, attention input",
    ),
    (
        "mask_attention_output",
        "before_attention_residual_token_mask",
        "token mask, attention branch output",
    ),
    (
        "mask_stream",
        "after_attention_residual_token_mask",
        "token mask, stream after the attention add",
    ),
    ("dropout_attention_input", "before_attention_norm", "dropout, attention input"),
    (
        "channel_attention_input",
        "before_attention_norm_channel_mask",
        "channel mask, attention input",
    ),
    (
        "noise_attention_input",
        "before_attention_norm_gaussian",
        "Gaussian noise, attention input",
    ),
    ("dropout_stream", "post_residual", "dropout, stream after both adds (PSBD-RD)"),
)
SITE_ATTACK = "badnet_a2o"


def reading_at(block: dict, rate: float) -> dict | None:
    """Clean, validation and triggered shift ratio and AUROC at 1 rate of 1 placement."""
    row = rate_row(block, rate)
    if row is None:
        return None
    shift = row["shift_ratio"]
    reading = {
        "rate": rate,
        "validation": shift["validation"],
        "clean": shift["clean"],
        "triggered": shift["backdoor"],
        "auroc": row["detection_psu_ratio"].get(HEADLINE_KEY, {}).get("auroc"),
    }
    return reading


def cell_record(report: dict) -> dict:
    """1 model's adaptive-rule reading and full rate curve at both placements."""
    record = {"attack": report["attack"], "folder": report["folder_name"]}
    for key, placement in PLACEMENTS.items():
        block = report.get("placements", {}).get(placement)
        if block is None:
            record[key] = None
            continue
        rate = psbd_rate(block, "adaptive")
        curve = [reading_at(block, row["rate"]) for row in block.get("rates", [])]
        record[key] = {
            "adaptive": reading_at(block, rate) if rate is not None else None,
            "curve": [point for point in curve if point is not None],
        }
    return record


def collect_records(results_dir: str) -> list[dict]:
    """Every clearing panel model that carries both placements at the adaptive rule."""
    records = []
    for cell in clearing_cells(load_coverage(results_dir)):
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        if report is None:
            continue
        record = cell_record(report)
        if all(record[key] and record[key]["adaptive"] for key in PLACEMENTS):
            records.append(record)
    return records


def adaptive_mean(records: list[dict], key: str, field: str) -> float | None:
    values = [record[key]["adaptive"][field] for record in records]
    return mean_or_none([value for value in values if value is not None])


def by_attack(records: list[dict]) -> dict[str, list[dict]]:
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["attack"]].append(record)
    ordered = {attack: grouped[attack] for attack in ATTACK_ORDER if attack in grouped}
    return ordered


def mean_curve(records: list[dict], key: str) -> list[dict]:
    """The clean validation and triggered shift ratio averaged over models, per rate."""
    per_rate = collections.defaultdict(list)
    for record in records:
        for point in record[key]["curve"]:
            per_rate[point["rate"]].append(point)
    curve = [
        {
            "rate": rate,
            "validation": mean_or_none([point["validation"] for point in points]),
            "triggered": mean_or_none([point["triggered"] for point in points]),
            "models": len(points),
        }
        for rate, points in sorted(per_rate.items())
        # A rate only some models were swept at would bend the mean toward them.
        if len(points) == len(records)
    ]
    return curve


def write_survival_table(args, grouped: dict[str, list[dict]], inputs) -> None:
    rows = []
    for attack, records in grouped.items():
        rows.append(
            [
                attack_label(attack),
                str(len(records)),
                fmt(adaptive_mean(records, "tm", "triggered"), places=2),
                fmt(adaptive_mean(records, "tm", "auroc")),
                fmt(adaptive_mean(records, "rd", "triggered"), places=2),
                fmt(adaptive_mean(records, "rd", "auroc")),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "survival.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Share of triggered predictions that change under each placement at its "
            "adaptive rate, where about four in five clean predictions change, and the "
            "resulting mean AUROC. Patch triggers first, then global triggers. A "
            "detector needs the triggered share far below the clean one."
        ),
        label="tab:survival",
        header=[
            "attack",
            "n",
            "TM shift",
            "TM AUROC",
            "RD shift",
            "RD AUROC",
        ],
        rows=rows,
        align="lrrrrr",
    )


def site_readings(results_dir: str) -> dict[str, list[dict]]:
    """Each site row's adaptive-rule reading on every clearing BadNets model."""
    readings = collections.defaultdict(list)
    for cell in clearing_cells(load_coverage(results_dir)):
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        if report is None or report["attack"] != SITE_ATTACK:
            continue
        for key, placement, _ in SITE_ROWS:
            block = report.get("placements", {}).get(placement)
            rate = psbd_rate(block, "adaptive") if block else None
            reading = reading_at(block, rate) if rate is not None else None
            if reading is not None:
                readings[key].append(reading)
    return readings


def mean_field(readings: list[dict], field: str) -> float | None:
    return mean_or_none([r[field] for r in readings if r[field] is not None])


def write_site_table(args, readings: dict[str, list[dict]], inputs) -> dict:
    rows = []
    macros = {}
    for key, _, label in SITE_ROWS:
        group = readings.get(key, [])
        rows.append(
            [
                label,
                str(len(group)),
                fmt(mean_field(group, "clean"), places=2),
                fmt(mean_field(group, "triggered"), places=2),
                fmt(mean_field(group, "auroc")),
            ]
        )
        macros[f"survival_site_{key}_triggered"] = (
            fmt(mean_field(group, "triggered"), places=2),
            f"share of triggered BadNets predictions that change, {label}, adaptive rate",
        )
    macros["survival_site_models"] = (
        str(min(len(readings.get(key, [])) for key, _, _ in SITE_ROWS)),
        "BadNets models behind the site table",
    )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "survival_sites.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "BadNets under each perturbation at its adaptive rate. The clean column is "
            "the share of clean test predictions that change, the triggered column the "
            "share of triggered predictions that change, and AUROC the resulting "
            "detection."
        ),
        label="tab:survival-sites",
        header=["perturbation", "n", "clean", "triggered", "AUROC"],
        rows=rows,
        align="lrrrr",
    )
    return macros


def write_survival_figure(args, grouped: dict[str, list[dict]], inputs) -> dict:
    """Triggered against clean shift ratio over the swept rates, 1 panel per placement."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.3), sharey=True)
    plotted = {}
    for axis, key in zip(axes, PLACEMENTS):
        plotted[key] = {}
        for attack, records in grouped.items():
            curve = mean_curve(records, key)
            plotted[key][attack] = curve
            axis.plot(
                [point["validation"] for point in curve],
                [point["triggered"] for point in curve],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                linestyle="-" if attack in PATCH_ATTACKS else "--",
                color=attack_color(attack),
                label=attack_label(attack),
            )
        axis.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, linestyle=":")
        axis.axvline(ADAPTIVE_SHIFT_TARGET, color="0.3", linewidth=0.8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(PLACEMENT_NAMES[key])
        axis.set_xlabel("share of clean predictions that change")
    axes[0].set_ylabel("share of triggered predictions that change")
    fig.tight_layout()
    legend_above(fig, list(axes), columns=4)

    path = os.path.join(args.paper_dir, "figures", "mech_survival.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    figure_sidecar(
        path=path.replace(".pdf", ".json"),
        generator=GENERATOR,
        inputs=inputs,
        plotted=plotted,
    )
    return plotted


def group_mean(grouped: dict, attacks: tuple, key: str, field: str) -> float | None:
    records = [record for attack in attacks for record in grouped.get(attack, [])]
    return adaptive_mean(records, key, field) if records else None


def survival_macros(records: list[dict], grouped: dict) -> dict:
    macros = {
        "survival_models": (str(len(records)), "models behind the survival table"),
    }
    for key, name in PLACEMENT_NAMES.items():
        macros[f"survival_{key}_clean"] = (
            fmt(adaptive_mean(records, key, "clean"), places=2),
            f"mean share of clean test predictions that change under {name} at its adaptive rate",
        )
        macros[f"survival_{key}_rate"] = (
            fmt(adaptive_mean(records, key, "rate"), places=2),
            f"mean rate the adaptive rule picks for {name}",
        )
        for group, attacks in (("patch", PATCH_ATTACKS), ("global", GLOBAL_ATTACKS)):
            macros[f"survival_{key}_{group}_triggered"] = (
                fmt(group_mean(grouped, attacks, key, "triggered"), places=2),
                f"mean share of triggered predictions that change under {name}, {group} triggers",
            )
            macros[f"survival_{key}_{group}_auroc"] = (
                fmt(group_mean(grouped, attacks, key, "auroc")),
                f"mean AUROC of {name} at the adaptive rule, {group} triggers",
            )
        # Every attack the order names gets its macros, so an attack with no
        # successful model left prints the empty-population dash instead of
        # leaving the paper an undefined macro.
        for attack in ATTACK_ORDER:
            attack_records = grouped.get(attack, [])
            macros[f"survival_{key}_{attack}_triggered"] = (
                fmt(adaptive_mean(attack_records, key, "triggered"), places=2),
                f"mean share of triggered {attack} predictions that change under "
                f"{name}, over {len(attack_records)} models",
            )
            macros[f"survival_{key}_{attack}_auroc"] = (
                fmt(adaptive_mean(attack_records, key, "auroc")),
                f"mean AUROC of {name} at the adaptive rule on {attack}, over "
                f"{len(attack_records)} models",
            )
    return macros


def main() -> None:
    args = build_parser(__doc__).parse_args()
    records = collect_records(args.results_dir)
    grouped = by_attack(records)
    inputs = [
        os.path.join(args.results_dir, "coverage", "coverage.json"),
        f"{args.results_dir}/<folder>/psbd_metrics.json ({len(records)} models)",
    ]

    write_survival_table(args, grouped, inputs)
    write_survival_figure(args, grouped, inputs)
    macros = survival_macros(records, grouped)
    macros.update(write_site_table(args, site_readings(args.results_dir), inputs))
    write_macros(
        os.path.join(args.paper_dir, "tables", "survival.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"survival: {len(records)} models, "
        + ", ".join(
            f"{key} patch {macros[f'survival_{key}_patch_triggered'][0]} "
            f"global {macros[f'survival_{key}_global_triggered'][0]}"
            for key in PLACEMENTS
        )
    )


if __name__ == "__main__":
    main()

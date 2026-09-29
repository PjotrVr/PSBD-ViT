"""T?: the H41 probe union, read on the ordinary (non-adaptive) backdoored ViT-B/16 models.

H41 built the min-rank probe union against an attacker trained to evade 1
probed operator. This table reads the same union rule where nobody trained
against a probe: the models the paper's headline reads. 6 probe sets, each
a row: PSBD-TM alone, PSBD-TM plus PSBD-RD, PSBD-TM plus token masking on the
attention branch output, the 3-probe pool from the adaptive-attacker section
(PSBD-TM, attention-input dropout, MLP-norm-out gain scaling), that pool plus
PSBD-RD, and every basis placement present on every one of them. Every number
comes from the per-model rows of
results/_experiments/probe_union/probe_union.json, written by
experiments/probe_union/measure.py, restricted here to the successful cells of
the panel (scripts.paper._common.clearing_cells). The record was written over the
ASR-only panel, so its own summaries span models the panel now leaves out and
are recomputed from its rows instead of read.

    PYTHONPATH=. .venv/bin/python scripts/paper/tab_probe_union.py \\
        --results-dir results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    bootstrap_ci,
    build_parser,
    ci_text,
    clearing_cells,
    fmt,
    load_coverage,
    load_json,
    mean_or_none,
    write_macros,
    write_wide_table,
)

GENERATOR = "scripts/paper/tab_probe_union.py"
SIG_INVERTED_FOLDER = "vit_cifar10_sig_0_1"
RECORD_PATH = os.path.join("results", "_experiments", "probe_union", "probe_union.json")

# Row order and the reader-facing name of each probe set's placements. The
# reference row carries no gain column entry, since it is what every other row
# is measured against.
# Probe short names, expanded in the table caption. The full descriptions ran a
# row to 69 characters, which pushed the 6-column float 198pt past the text block.
PROBE_LEGEND = (
    "TM is token masking at the attention input, RD dropout after both residual "
    "adds, TM-out token masking on the attention branch output, DI dropout at the "
    "attention input and GS gain scaling of the MLP LayerNorm output"
)
ROW_LABELS = (
    ("psbd_tm", "TM alone"),
    ("psbd_tm_rd", "TM + RD"),
    ("psbd_tm_attn_branch", "TM + TM-out"),
    ("adaptive_3probe", "TM + DI + GS"),
    ("adaptive_4probe", "TM + DI + GS + RD"),
    ("all_65_basis", "every basis placement on every model"),
)


def load_record(results_dir: str) -> dict:
    path = os.path.join(results_dir, "_experiments", "probe_union", "probe_union.json")
    record = load_json(path)
    if record is None:
        raise SystemExit(
            f"{path} does not exist, run experiments/probe_union/measure.py first"
        )
    return record


def restrict_to_panel(record: dict, panel: set[str], resamples: int, seed: int) -> dict:
    """Every probe set's summary and paired gain over PSBD-TM, recomputed on the panel models."""
    reference = {
        row["folder"]: row["auroc"]
        for row in record["probe_sets"]["psbd_tm"]["per_model"]
        if row["folder"] in panel
    }
    restricted = {}
    for name, block in record["probe_sets"].items():
        rows = [row for row in block["per_model"] if row["folder"] in panel]
        summary = {
            "n_models": len(rows),
            "auroc_mean": mean_or_none([row["auroc"] for row in rows]),
            "tpr_at_0.10_mean": mean_or_none([row["tpr_at_0.10"] for row in rows]),
            "tpr_at_0.20_mean": mean_or_none([row["tpr_at_0.20"] for row in rows]),
        }
        gain = None
        if "gain_over_psbd_tm" in block:
            deltas = [
                row["auroc"] - reference[row["folder"]]
                for row in rows
                if row["folder"] in reference
            ]
            low, high = bootstrap_ci(deltas, resamples, seed)
            gain = {
                "n": len(deltas),
                "mean_gain": mean_or_none(deltas),
                "ci_low": low,
                "ci_high": high,
            }
        restricted[name] = {
            "summary": summary,
            "per_model": rows,
            "gain_over_psbd_tm": gain,
        }
    return restricted


def table_row(name: str, label: str, probe_sets: dict) -> list[str]:
    block = probe_sets[name]
    summary = block["summary"]
    gain = block.get("gain_over_psbd_tm")
    if gain is None:
        gain_text = "--"
    else:
        gain_text = f"{fmt(gain['mean_gain'], signed=True)} {ci_text(gain['ci_low'], gain['ci_high'])}"
    row = [
        label,
        str(summary["n_models"]),
        fmt(summary["auroc_mean"]),
        fmt(summary["tpr_at_0.10_mean"]),
        fmt(summary["tpr_at_0.20_mean"]),
        gain_text,
    ]
    return row


def main() -> None:
    args = build_parser(__doc__).parse_args()
    record = load_record(args.results_dir)
    panel = {
        cell["folder_name"] for cell in clearing_cells(load_coverage(args.results_dir))
    }
    probe_sets = restrict_to_panel(record, panel, args.bootstrap, args.seed)
    n_models = probe_sets["psbd_tm"]["summary"]["n_models"]

    inputs = [
        RECORD_PATH,
        f"{args.results_dir}/coverage/coverage.json ({len(panel)} successful cells)",
    ]

    rows = [table_row(name, label, probe_sets) for name, label in ROW_LABELS]
    write_wide_table(
        path=os.path.join(args.paper_dir, "tables", "probe_union.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            f"The min-rank union of probes on the {n_models} backdoored ViT-B/16 models, with "
            "the paired AUROC gain over token masking at the attention input alone. "
            f"{PROBE_LEGEND}."
        ),
        label="tab:probe-union",
        header=[
            "Probes",
            "n",
            "AUROC",
            "TPR@FPR 10%",
            "TPR@FPR 20%",
            "Gain over TM [95% CI]",
        ],
        rows=rows,
        align="lrrrrl",
    )

    branch = probe_sets["psbd_tm_attn_branch"]
    tm = probe_sets["psbd_tm"]["summary"]
    tm_rd = probe_sets["psbd_tm_rd"]
    wanet = {
        name: row
        for name, row in record["wanet_cifar10"].items()
        if row and row["folder"] in panel
    }
    # The SIG model is out of the panel when it fails the clean-accuracy bar, and
    # its macro then prints the empty-population dash rather than a number read
    # off a model no other table carries.
    sig = next(
        (row for row in tm_rd["per_model"] if row["folder"] == SIG_INVERTED_FOLDER),
        None,
    )
    every_basis = probe_sets["all_65_basis"]["gain_over_psbd_tm"]
    macros = {
        "probe_union_all_basis_gain": (
            fmt(every_basis["mean_gain"], signed=True),
            "paired AUROC gain of the union of every fully swept placement over PSBD-TM",
        ),
        "probe_union_sig_cifar10_tm_rd": (
            fmt(sig["auroc"] if sig else None),
            "PSBD-TM + PSBD-RD union on SIG at 10% on CIFAR-10, the other inverted "
            "cell, a dash when that model is not in the panel",
        ),
        "probe_union_three_probe_auroc": (
            fmt(probe_sets["adaptive_3probe"]["summary"]["auroc_mean"]),
            "mean AUROC of the 3-probe union of the adaptive section on the ordinary models",
        ),
        "probe_union_three_probe_n": (
            str(probe_sets["adaptive_3probe"]["summary"]["n_models"]),
            "models holding all 3 probes of the adaptive section",
        ),
        "probe_union_tm_branch_gain": (
            fmt(branch["gain_over_psbd_tm"]["mean_gain"], signed=True),
            "paired AUROC gain of the PSBD-TM + branch-output union over PSBD-TM alone",
        ),
        "probe_union_tm_branch_auroc": (
            fmt(branch["summary"]["auroc_mean"]),
            "mean AUROC of the PSBD-TM + branch-output union",
        ),
        "probe_union_n_models": (
            str(n_models),
            "backdoored ViT-B/16 models the probe-union check reads, "
            "the models with both headline placements",
        ),
        "probe_union_psbd_tm_auroc": (
            fmt(tm["auroc_mean"]),
            "mean AUROC of PSBD-TM alone over the models the probe-union check reads",
        ),
        "probe_union_tm_rd_gain": (
            fmt(tm_rd["gain_over_psbd_tm"]["mean_gain"], signed=True),
            "paired AUROC gain of the PSBD-TM + PSBD-RD union over PSBD-TM alone",
        ),
        "probe_union_tm_rd_gain_ci": (
            ci_text(
                tm_rd["gain_over_psbd_tm"]["ci_low"],
                tm_rd["gain_over_psbd_tm"]["ci_high"],
            ),
            "95% bootstrap interval of that gain",
        ),
        "probe_union_wanet_cifar10_tm": (
            fmt(wanet["psbd_tm"]["auroc"]) if "psbd_tm" in wanet else "--",
            "PSBD-TM alone on WaNet at 10% on CIFAR-10, the inverted cell the "
            "headline names",
        ),
        "probe_union_wanet_cifar10_tm_rd": (
            fmt(wanet["psbd_tm_rd"]["auroc"]) if "psbd_tm_rd" in wanet else "--",
            "PSBD-TM + PSBD-RD union on the same cell",
        ),
    }
    write_macros(
        sidecar_path=os.path.join(args.paper_dir, "tables", "probe_union.macros.json"),
        generator=GENERATOR,
        inputs=inputs,
        macros=macros,
    )

    print(f"wrote {args.paper_dir}/tables/probe_union.tex")
    print(f"wrote {args.paper_dir}/tables/probe_union.macros.json")


if __name__ == "__main__":
    main()

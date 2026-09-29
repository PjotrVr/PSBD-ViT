"""Why a patch trigger survives PSBD-TM and not PSBD-RD, and how legible each trigger is.

Reads the per-model records experiments/why_token_masking_works/measure.py
writes, keeps the models the panel keeps (scripts.paper._common.excluded_folders
drops the diverged, source-mapped and clean-accuracy failures), summarizes them
with the experiment's own summarize and turns that into macros and 1 appendix
table. The experiment's summary.json spans every model it ran on, so it is not
read.

- Masking only the trigger's tokens at the attention input, in chosen blocks,
  against the same number of random tokens.
- The real PSBD-TM operator at the adaptive rate, with each pass split by how
  many of blocks 9 to 12 masked every trigger token.
- PSBD-RD's dropout restricted to the trigger's positions, to every other
  position, or to as many random positions.
- A fixed share of tokens kept visible in every block, per attack.

    PYTHONPATH=. python scripts/paper/mech_causal.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from experiments.why_token_masking_works.measure import (  # noqa: E402
    read_model_records,
    summarize,
)
from scripts.paper._common import (  # noqa: E402
    attack_label,
    build_parser,
    excluded_folders,
    fmt,
    word_list,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_causal.py"
PATCH_GROUP = "badnet_a2o"
FOUR_TOKEN_GROUP = "badnet_a2o_4_tokens"
CONDITIONAL_TACT = "tact_trigger_conditional"
GLOBAL_ATTACKS = ("blend", "bpp", "lf", "sig", "wanet")
VISIBLE_SHARE = "0.3"
# Rows of the appendix table: label, section of the record, key.
MASK_ROWS = (
    ("trigger tokens, all 12 blocks", "all_12"),
    ("trigger tokens, blocks 1 to 4", "blocks_1_4"),
    ("trigger tokens, blocks 5 to 8", "blocks_5_8"),
    ("trigger tokens, blocks 9 to 12", "blocks_9_12"),
    ("random tokens, all 12 blocks", "random_all_12"),
)
DROPOUT_ROWS = (
    ("every position (PSBD-RD)", "all_tokens"),
    ("trigger positions only", "trigger_only"),
    ("every position but the trigger", "all_but_trigger"),
    ("as many random positions", "random_same_count"),
)


def load_record(results_dir: str) -> dict:
    """The experiment's summary over the per-model records of panel models only."""
    directory = os.path.join(results_dir, "_experiments", "why_token_masking_works")
    if not os.path.isdir(directory):
        raise SystemExit(
            f"{directory} does not exist, run experiments/why_token_masking_works/measure.py"
        )
    excluded = excluded_folders(results_dir)
    records = [
        record
        for record in read_model_records(directory)
        if record["folder"] not in excluded
    ]
    record = summarize(records)
    return record


def masking_macros(record: dict) -> dict:
    masks = record["deterministic_masking"][PATCH_GROUP]
    macros = {
        f"causal_mask_{key}": (
            fmt(masks[key]["triggered_kept"]),
            f"share of triggered BadNets predictions kept, {label}",
        )
        for label, key in MASK_ROWS
    }
    macros["causal_mask_clean_kept_min"] = (
        fmt(min(masks[key]["clean_kept"] for _, key in MASK_ROWS)),
        "lowest share of clean BadNets predictions kept across the masking rows",
    )
    macros["causal_models"] = (
        str(masks["all_12"]["models"]),
        "BadNets models behind the causal tests",
    )
    conditional = record["deterministic_masking"][CONDITIONAL_TACT]["all_12"]
    macros["causal_tact_mask_all"] = (
        fmt(conditional["triggered_kept"]),
        "share of triggered TaCT predictions kept with the trigger masked in every block",
    )
    macros["causal_tact_models"] = (
        str(conditional["models"]),
        "TaCT models whose clean source class keeps its label",
    )
    return macros


def stochastic_macros(record: dict) -> dict:
    four = record["stochastic_token_mask_by_trigger_tokens"][FOUR_TOKEN_GROUP]
    late = four["triggered"]["late_all"]
    overall = record["stochastic_token_mask"][PATCH_GROUP]["overall"]
    macros = {
        "causal_tm_triggered_kept": (
            fmt(overall["triggered_kept"]),
            "share of triggered BadNets predictions kept per PSBD-TM pass",
        ),
        "causal_tm_clean_kept": (
            fmt(overall["clean_kept"]),
            "share of clean BadNets-model predictions kept per PSBD-TM pass",
        ),
        "causal_tm_late_none": (
            fmt(late["0"]["kept"]),
            "triggered kept when no block of 9 to 12 masked every trigger token",
        ),
        "causal_tm_late_all": (
            fmt(late["4"]["kept"]),
            "triggered kept when all of blocks 9 to 12 masked every trigger token",
        ),
        "causal_tm_late_all_share": (
            fmt(late["4"]["n"] / sum(entry["n"] for entry in late.values())),
            "share of passes in which all of blocks 9 to 12 masked every trigger token",
        ),
    }
    return macros


def dropout_macros(record: dict) -> dict:
    dropout = record["residual_dropout"][PATCH_GROUP]
    macros = {
        f"causal_rd_{key}": (
            fmt(dropout[key]["triggered_kept"]),
            f"share of triggered BadNets predictions kept, dropout on {label}",
        )
        for label, key in DROPOUT_ROWS
    }
    macros["causal_rd_clean_kept"] = (
        fmt(dropout["all_tokens"]["clean_kept"]),
        "share of clean BadNets-model predictions kept under PSBD-RD",
    )
    return macros


def visibility_macros(record: dict) -> dict:
    subsets = record["visible_subsets"]
    macros = {}
    # An attack with no panel model left prints the empty-population dash.
    for attack in GLOBAL_ATTACKS:
        reading = subsets.get(attack, {}).get(VISIBLE_SHARE, {})
        models = reading.get("models", 0)
        macros[f"causal_visible_{attack}_trigger"] = (
            fmt(reading.get("excess_retention"), places=2),
            f"{attack} trigger effect retained with {VISIBLE_SHARE} of tokens "
            f"visible, over {models} models",
        )
        macros[f"causal_visible_{attack}_clean"] = (
            fmt(reading.get("clean_accuracy_retention"), places=2),
            f"{attack} clean accuracy retained with {VISIBLE_SHARE} of tokens "
            f"visible, over {models} models",
        )
    legible = [
        attack for attack in GLOBAL_ATTACKS if attack != "wanet" and attack in subsets
    ]
    legible_words = word_list([attack_label(attack) for attack in legible])
    per_trigger = [
        subsets[attack][VISIBLE_SHARE]["excess_retention"] for attack in legible
    ]
    per_clean = [
        subsets[attack][VISIBLE_SHARE]["clean_accuracy_retention"] for attack in legible
    ]
    macros["causal_visible_global_attacks"] = (
        legible_words,
        "the global-trigger attacks other than WaNet with a panel model in the "
        "legibility test",
    )
    macros["causal_visible_global_trigger_min"] = (
        fmt(min(per_trigger), places=2),
        f"lowest trigger effect retained by {legible_words}",
    )
    macros["causal_visible_global_trigger_max"] = (
        fmt(max(per_trigger), places=2),
        f"highest trigger effect retained by {legible_words}",
    )
    macros["causal_visible_global_clean_min"] = (
        fmt(min(per_clean), places=2),
        "lowest clean accuracy retained on those models",
    )
    macros["causal_visible_global_clean_max"] = (
        fmt(max(per_clean), places=2),
        "highest clean accuracy retained on those models",
    )
    macros["causal_visible_share"] = (
        f"{float(VISIBLE_SHARE):.0%}".replace("%", "\\%"),
        "share of tokens kept visible in the legibility test",
    )
    return macros


def write_causal_table(args, record: dict, inputs: list[str]) -> None:
    masks = record["deterministic_masking"][PATCH_GROUP]
    dropout = record["residual_dropout"][PATCH_GROUP]
    rows = [["\\multicolumn{3}{l}{\\emph{masked at the attention input}}"]]
    rows += [
        [label, fmt(masks[key]["triggered_kept"]), fmt(masks[key]["clean_kept"])]
        for label, key in MASK_ROWS
    ]
    rows.append(["\\multicolumn{3}{l}{\\emph{PSBD-RD dropout applied to}}"])
    rows += [
        [label, fmt(dropout[key]["triggered_kept"]), fmt(dropout[key]["clean_kept"])]
        for label, key in DROPOUT_ROWS
    ]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "causal_patch.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Share of predictions kept on the "
            f"{masks['all_12']['models']} BadNets models, triggered and clean, when "
            "only chosen tokens are perturbed. The masks are deterministic. The "
            "dropout rows use PSBD-RD's adaptive rate with ten passes."
        ),
        label="tab:causal-patch",
        header=["perturbed", "triggered kept", "clean kept"],
        rows=rows,
        align="lrr",
    )


def main() -> None:
    args = build_parser(__doc__).parse_args()
    record = load_record(args.results_dir)
    inputs = [
        "results/_experiments/why_token_masking_works/<folder>.json "
        f"({record['models']} panel models)"
    ]
    macros = {}
    for builder in (
        masking_macros,
        stochastic_macros,
        dropout_macros,
        visibility_macros,
    ):
        macros.update(builder(record))
    write_causal_table(args, record, inputs)
    write_macros(
        os.path.join(args.paper_dir, "tables", "causal.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"causal: mask all {macros['causal_mask_all_12'][0]}, random "
        f"{macros['causal_mask_random_all_12'][0]}, late none "
        f"{macros['causal_tm_late_none'][0]} all {macros['causal_tm_late_all'][0]}, "
        f"rd trigger only {macros['causal_rd_trigger_only'][0]}"
    )


if __name__ == "__main__":
    main()

"""PSBD-TM against the best residual placement, not only against PSBD-RD.

PSBD-RD is our adaptation of the original site, so it is the named baseline. A
reader can still ask whether some other residual placement would have done as
well as PSBD-TM. This reads every declared placement that perturbs the residual
stream or a branch output with an operator other than whole-token masking, keeps
those present on every model the headline reads, picks the one with the highest
mean AUROC at the adaptive rule and reports PSBD-TM's paired margin over it.
The pick is made on the same models it is measured on, which flatters the pick
and so makes the margin conservative.

    PYTHONPATH=. python scripts/paper/tab_best_residual.py \
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from cli.compare_detectors import psbd_rate  # noqa: E402
from defenses.decision import RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    bootstrap_ci,
    build_parser,
    clearing_cells,
    fmt,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    mean_or_none,
    placement_words,
    rate_row,
    write_macros,
)

GENERATOR = "scripts/paper/tab_best_residual.py"
RESIDUAL_POSITIONS = {
    "pre_residual",
    "post_residual",
    "before_attention_residual",
    "after_attention_residual",
    "before_mlp_residual",
    "after_mlp_residual",
}
REPORTED_ATTACKS = ("badnet_a2o", "tact", "wanet", "sig")


def residual_candidates(declaration: dict) -> list[str]:
    """Declared placements on a residual site whose operator is not a token mask."""
    candidates = [
        entry["id"]
        for entry in declaration["basis"]
        if entry["position"] in RESIDUAL_POSITIONS and entry["operator"] != "token_mask"
    ]
    return candidates


def adaptive_auroc(report: dict, placement: str) -> float | None:
    block = report.get("placements", {}).get(placement)
    if block is None:
        return None
    rate = psbd_rate(block, "adaptive")
    row = rate_row(block, rate) if rate is not None else None
    if row is None:
        return None
    return row["detection_psu_ratio"].get(HEADLINE_KEY, {}).get("auroc")


def read_models(results_dir: str, placements: list[str]) -> list[dict]:
    """Every clearing model's adaptive AUROC at PSBD-TM and each candidate."""
    models = []
    for cell in clearing_cells(load_coverage(results_dir)):
        report = load_psbd_metrics(results_dir, cell["folder_name"])
        if report is None:
            continue
        aurocs = {
            placement: adaptive_auroc(report, placement)
            for placement in [RECOMMENDED_PLACEMENT, *placements]
        }
        if aurocs[RECOMMENDED_PLACEMENT] is not None:
            models.append({"attack": report["attack"], "auroc": aurocs})
    return models


def main() -> None:
    args = build_parser(__doc__).parse_args()
    candidates = residual_candidates(load_declaration(args.declaration))
    models = read_models(args.results_dir, candidates)
    complete = [
        placement
        for placement in candidates
        if all(model["auroc"][placement] is not None for model in models)
    ]
    means = {
        placement: mean_or_none([model["auroc"][placement] for model in models])
        for placement in complete
    }
    best = max(means, key=means.get)
    deltas = [
        model["auroc"][RECOMMENDED_PLACEMENT] - model["auroc"][best] for model in models
    ]
    low, high = bootstrap_ci(deltas, args.bootstrap, args.seed)

    macros = {
        "best_residual_candidates": (
            str(len(complete)),
            "residual placements, token masks excluded, present on every headline model",
        ),
        "best_residual_models": (str(len(models)), "models behind the comparison"),
        "best_residual_name": (placement_words(best), "the best residual placement"),
        "best_residual_auroc": (
            fmt(means[best]),
            "its mean AUROC at the adaptive rule",
        ),
        "best_residual_gain": (
            fmt(mean_or_none(deltas), signed=True),
            "mean paired AUROC gain of PSBD-TM over the best residual placement",
        ),
        "best_residual_gain_low": (
            fmt(low, signed=True),
            "lower bound of the 95% bootstrap interval on that gain",
        ),
        "best_residual_gain_high": (
            fmt(high, signed=True),
            "upper bound of the 95% bootstrap interval on that gain",
        ),
    }
    for attack in REPORTED_ATTACKS:
        group = [model for model in models if model["attack"] == attack]
        macros[f"best_residual_{attack}_auroc"] = (
            fmt(mean_or_none([model["auroc"][best] for model in group])),
            f"mean AUROC of the best residual placement on {attack}",
        )
    inputs = [
        args.declaration,
        f"{args.results_dir}/<folder>/psbd_metrics.json ({len(models)} models)",
    ]
    write_macros(
        os.path.join(args.paper_dir, "tables", "best_residual.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        f"best residual: {best} {fmt(means[best])} of {len(complete)}, "
        f"TM gain {macros['best_residual_gain'][0]} [{macros['best_residual_gain_low'][0]}, "
        f"{macros['best_residual_gain_high'][0]}] over {len(models)} models"
    )


if __name__ == "__main__":
    main()

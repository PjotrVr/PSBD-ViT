"""Breaking-point curves: the clean and triggered shift ratio along every rate ladder.

The shift ratio at rate p is the share of (image, pass) predictions that differ from
the unperturbed answer, so over a ladder it traces the distribution of the
breaking point p* (critical_rate.py): the curve at p is the share of passes whose
image has already broken. This script writes those curves per model rather than
as a median, for every placement with a ladder, on the ViT and the Swin panel
(`scripts.paper._common.clearing_cells` at the 2 point bar, Swin through
`swin_coverage`) and on the benign references of both architectures, whose clean
and triggered curves should coincide.

The curves come from each model's psbd_metrics.json: `shift_ratio.clean` is the
whole clean analysis split and `shift_ratio.backdoor` every triggered row, which
includes triggered rows the backdoor did not capture. For each model and
placement the rate at which a curve crosses 0.5 is interpolated on the ladder,
None when it never does.

The output, results/_experiments/why_psbd_works/shift_curves.json, is also the
sidecar of the notebook figure drawn from it. CPU only, seconds.

    PYTHONPATH=. python experiments/why_psbd_works/shift_curves.py
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from experiments._paths import experiment_result_path  # noqa: E402
from experiments.why_psbd_works.measure import SLUG  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    clearing_cells,
    load_coverage,
    load_psbd_metrics,
    swin_coverage,
)

DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
MIN_RATES = 4
CROSSING = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output-root", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    models = panel_models(args)
    curves = []
    for architecture, folder, attack, dataset in models:
        report = load_psbd_metrics(args.results_dir, folder)
        if report is None:
            continue
        for placement, block in report["placements"].items():
            curve = placement_curve(block)
            if curve is not None:
                curves.append(
                    {
                        "architecture": architecture,
                        "folder": folder,
                        "attack": attack,
                        "dataset": dataset,
                        "placement": placement,
                        **curve,
                    }
                )
    out_path = experiment_result_path(SLUG, "shift_curves.json", args.output_root)
    with open(out_path, "w") as handle:
        json.dump({"crossing": CROSSING, "curves": curves}, handle)
    print(f"{len(curves)} curves over {len(models)} models written to {out_path}")


def panel_models(args):
    models = [
        ("vit", cell["folder_name"], cell["attack"], cell["dataset"])
        for cell in clearing_cells(load_coverage(args.results_dir))
    ]
    swin = swin_coverage(args.results_dir, args.checkpoints_dir)
    models += [
        ("swin", cell["folder_name"], cell["attack"], cell["dataset"])
        for cell in clearing_cells(swin)
    ]
    for architecture in ("vit", "swin"):
        models += [
            (architecture, f"{architecture}_{dataset}_benign", "benign", dataset)
            for dataset in DATASETS
        ]
    return models


def placement_curve(block):
    rows = [r for r in block.get("rates", []) if r.get("shift_ratio")]
    rows = sorted(rows, key=lambda r: r["rate"])
    if len(rows) < MIN_RATES:
        return None
    rates = [r["rate"] for r in rows]
    clean = [r["shift_ratio"].get("clean") for r in rows]
    triggered = [r["shift_ratio"].get("backdoor") for r in rows]
    if None in clean or None in triggered:
        return None
    curve = {
        "rates": rates,
        "clean": clean,
        "triggered": triggered,
        "clean_crossing": crossing_rate(rates, clean),
        "triggered_crossing": crossing_rate(rates, triggered),
        "triggered_peak": max(triggered),
        "adaptive_rate": block.get("adaptive_rate"),
    }
    return curve


# The rate at which a rising curve first reaches the crossing level, by linear
# interpolation between the 2 ladder rates around it.
def crossing_rate(rates, values):
    if values[0] >= CROSSING:
        return rates[0]
    for low, high, below, above in zip(rates, rates[1:], values, values[1:]):
        if below < CROSSING <= above:
            fraction = (CROSSING - below) / (above - below)
            rate = low + fraction * (high - low)
            return rate
    return None


if __name__ == "__main__":
    main()

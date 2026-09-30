"""Prediction depth against PSBD-TM on the current panel, with the paper generator's reader.

`scripts.paper.mech_prediction_depth.measure_cell` pairs `depth_soft` against PSBD-TM at the
adaptive 0.8 rule within cell: AUROC at q0.25 and TPR at q0.01. It reads the headline panel
(`experiments.probe_union.measure.select_models`) and lists the panel models that have no
`prediction_depth.json` record, which need a GPU pass.

    PYTHONPATH=. .venv/bin/python experiments/prediction_depth/panel.py
"""

import collections
import statistics

from defenses.decision import HARD_ATTACKS
from experiments.probe_union.measure import select_models
from scripts.paper._common import bootstrap_ci
from scripts.paper.mech_prediction_depth import measure_cell


def row(label: str, cells: list[dict]) -> None:
    if not cells:
        print(f"{label}: none")
        return
    auroc = [c["depth_auroc"] - c["psbd_auroc"] for c in cells]
    tpr = [c["depth_tpr"] - c["psbd_tpr"] for c in cells]
    print(
        f"{label:28s} n {len(cells):2d}"
        f" depth auroc {statistics.mean(c['depth_auroc'] for c in cells):.3f}"
        f" psbd auroc {statistics.mean(c['psbd_auroc'] for c in cells):.3f}"
        f" delta {statistics.mean(auroc):+.3f} {[round(x, 3) for x in bootstrap_ci(auroc, 5000, 0)]}"
        f" | tpr1 depth {statistics.mean(c['depth_tpr'] for c in cells):.3f}"
        f" psbd {statistics.mean(c['psbd_tpr'] for c in cells):.3f}"
        f" delta {statistics.mean(tpr):+.3f} {[round(x, 3) for x in bootstrap_ci(tpr, 5000, 0)]}"
        f" wins {sum(t > 0 for t in tpr)}"
    )


def main() -> None:
    cells = []
    missing = []
    for model in select_models("results"):
        measured = measure_cell("results", model)
        if measured is None:
            missing.append(model["folder_name"])
            continue
        measured.update(attack=model["attack"], rate=model["poison_rate"])
        cells.append(measured)
    print(f"hard attacks {HARD_ATTACKS}")
    print(f"panel models without a depth record: {missing}")
    row("all", cells)
    hard = [c for c in cells if c["attack"] in HARD_ATTACKS]
    row("hard", hard)
    row("hard at 1% and 5%", [c for c in hard if c["rate"] in (0.01, 0.05)])
    by_attack = collections.defaultdict(list)
    for c in cells:
        by_attack[c["attack"]].append(c)
    for attack, members in sorted(by_attack.items()):
        row(f"  {attack}", members)


if __name__ == "__main__":
    main()

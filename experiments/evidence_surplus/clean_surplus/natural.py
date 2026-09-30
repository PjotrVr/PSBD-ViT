"""C, the natural version: do the clean images that never flip carry the most surplus?

CPU only. Prediction 4 of PREDICTIONS.md: over PSBD-TM's 10 passes in
experiments/why_psbd_works, the clean images whose answer never changes have a
higher margin retention (the share of the own-class logit margin a pass keeps,
the surplus proxy) than the clean images that flip, on every model measured. The
reading per model is the median retention of each group and the AUROC with which
retention separates the never-flipping images from the rest.

    PYTHONPATH=. python experiments/evidence_surplus/clean_surplus/natural.py
"""

import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

from experiments._paths import experiment_result_path  # noqa: E402
from experiments.evidence_surplus.common import SLUG  # noqa: E402
from experiments.why_psbd_works.measure import auroc_low_is_positive  # noqa: E402

SOURCE = os.path.join("results", "_experiments", "why_psbd_works")


def main():
    rows = []
    for name in sorted(os.listdir(SOURCE)):
        if not name.startswith(("vit_", "swin_")) or not name.endswith(".json"):
            continue
        with open(os.path.join(SOURCE, name)) as handle:
            record = json.load(handle)
        clean = record["operators"]["token_mask"]["clean"]
        pairs = [
            (kept, retention)
            for kept, retention in zip(clean["kept"], clean["margin_retention"])
            if kept is not None and retention is not None
        ]
        never = [r for kept, r in pairs if kept == 1.0]
        flipped = [r for kept, r in pairs if kept < 1.0]
        if not never or not flipped:
            continue
        rows.append(
            {
                "model": name[: -len(".json")],
                "never_flip_share": len(never) / len(pairs),
                "never_flip_median_retention": statistics.median(never),
                "flipped_median_retention": statistics.median(flipped),
                # A high retention flags the never-flipping image, so the negation
                # is the low-is-positive score the helper reads.
                "auroc": auroc_low_is_positive(
                    [-r for r in never], [-r for r in flipped]
                ),
            }
        )
    holds = sum(
        r["never_flip_median_retention"] > r["flipped_median_retention"] for r in rows
    )
    payload = {"rows": rows, "models": len(rows), "prediction_4_holds_on": holds}
    with open(
        experiment_result_path(SLUG, "clean_surplus_natural.json"), "w"
    ) as handle:
        json.dump(payload, handle, indent=2)
    print(f"never-flip retention above flipped on {holds} of {len(rows)} models")


if __name__ == "__main__":
    main()

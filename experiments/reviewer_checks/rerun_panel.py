"""Rerun reviewer checks 1 and 2 on the current paper panel and write a dated record.

The panel selection is the experiment's own (`measure.selected_cells`, which follows
`scripts.paper._common.clearing_cells`, the models successful at the 2-point bar). The
relaxed-coverage follow-up keeps its share rather than its count: the original bar was
60 of 69 models, so the bar becomes that share of the current panel, rounded up. Check 3
needs a GPU and is copied unchanged from the 2026-09-11 record.

    PYTHONPATH=. .venv/bin/python experiments/reviewer_checks/rerun_panel.py
"""

import json
import math
import sys
import time

import experiments.reviewer_checks.measure as measure

OLD = "results/_experiments/reviewer_checks/reviewer_checks.json"
NEW = (
    "results/_experiments/reviewer_checks/reviewer_checks_"
    + time.strftime("%Y-%m-%d")
    + ".json"
)
ORIGINAL_COVERAGE_SHARE = 60 / 69


def main():
    n_panel = len(measure.selected_cells("results"))
    measure.CHECK2B_MIN_COVERAGE = math.ceil(ORIGINAL_COVERAGE_SHARE * n_panel)
    sys.argv = ["measure.py", "--skip-check3", "--output", NEW]
    measure.main()
    sys.argv = ["measure.py", "--check2-relaxed-only", "--output", NEW]
    measure.main()
    with open(NEW) as handle:
        report = json.load(handle)
    with open(OLD) as handle:
        report["check3_mask_seed"] = json.load(handle)["check3_mask_seed"]
    report["check3_source"] = f"{OLD}, GPU run of 2026-09-11, copied unchanged"
    report["check2b_min_coverage"] = measure.CHECK2B_MIN_COVERAGE
    with open(NEW, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {NEW} on {n_panel} models")


if __name__ == "__main__":
    main()

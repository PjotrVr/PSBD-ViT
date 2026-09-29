import sys
import collections

sys.path.insert(0, ".")
from scipy.stats import spearmanr
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import select_rate_adaptively

cells = clearing_cells(load_coverage("results"))


def read(metrics, placement):
    block = metrics["placements"].get(placement)
    if not block:
        return None
    shift = {r["rate"]: r["shift_ratio"]["validation"] for r in block["rates"]}
    rate = select_rate_adaptively(shift)
    if rate is None:
        return None
    row = next(r for r in block["rates"] if r["rate"] == rate)
    histogram = row.get("shift_target_histogram", {}).get("clean")
    share = (
        histogram[metrics["target_label"]] / sum(histogram)
        if histogram and sum(histogram)
        else None
    )
    return share, row["detection_psu_ratio"]["q0.25"]["auroc"]


rows = []
for cell in cells:
    metrics = load_psbd_metrics("results", cell["folder_name"])
    if metrics is None:
        continue
    tm, rd = (
        read(metrics, "before_attention_norm_token_mask"),
        read(metrics, "post_residual"),
    )
    if tm and rd and tm[0] is not None and rd[0] is not None:
        rows.append((metrics["attack"], tm[0], tm[1], rd[0], rd[1]))
print("n", len(rows))
print("TM share vs TM auroc", spearmanr([r[1] for r in rows], [r[2] for r in rows]))
print("RD share vs RD auroc", spearmanr([r[3] for r in rows], [r[4] for r in rows]))
print(
    "RD share minus TM share vs RD auroc minus TM auroc",
    spearmanr([r[3] - r[1] for r in rows], [r[4] - r[2] for r in rows]),
)

print("scale_up by rate")
by = collections.defaultdict(list)
for cell in cells:
    metrics = load_psbd_metrics("results", cell["folder_name"])
    if metrics is None:
        continue
    block = metrics["placements"].get("input_pixels_scale_up")
    if not block:
        continue
    best = max(r["detection_psu_ratio"]["q0.25"]["auroc"] for r in block["rates"])
    shift = {r["rate"]: r["shift_ratio"]["validation"] for r in block["rates"]}
    rate = select_rate_adaptively(shift)
    adaptive = next(
        (
            r["detection_psu_ratio"]["q0.25"]["auroc"]
            for r in block["rates"]
            if r["rate"] == rate
        ),
        None,
    )
    by[(metrics["attack"], metrics["poison_rate"])].append((best, adaptive))
for key, values in sorted(by.items()):
    adaptive = [v[1] for v in values if v[1] is not None]
    print(
        key,
        len(values),
        f"oracle={sum(v[0] for v in values) / len(values):.3f}",
        f"adaptive={sum(adaptive) / len(adaptive):.3f}"
        if adaptive
        else "adaptive=none",
    )

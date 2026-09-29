import sys
import collections

sys.path.insert(0, ".")
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import select_rate_adaptively

cells = clearing_cells(load_coverage("results"))
placements = [
    "before_attention_norm_token_mask",
    "post_residual",
    "after_embedding_token_mask",
]
table = collections.defaultdict(lambda: collections.defaultdict(list))
rates = collections.defaultdict(list)
for cell in cells:
    metrics = load_psbd_metrics("results", cell["folder_name"])
    if metrics is None:
        continue
    attack, target = metrics["attack"], metrics["target_label"]
    for placement in placements:
        block = metrics["placements"].get(placement)
        if not block:
            continue
        shift = {
            row["rate"]: row["shift_ratio"]["validation"] for row in block["rates"]
        }
        rate = select_rate_adaptively(shift)
        if rate is None:
            continue
        rates[(placement, attack)].append(rate)
        row = next(r for r in block["rates"] if r["rate"] == rate)
        histogram = row.get("shift_target_histogram", {}).get("clean")
        if not histogram or sum(histogram) == 0 or target is None:
            continue
        share = histogram[target] / sum(histogram)
        top = max(histogram) / sum(histogram)
        table[placement][attack].append((share, top, 1 / len(histogram)))
for placement in placements:
    print(placement)
    for attack, rows in sorted(table[placement].items()):
        n = len(rows)
        print(
            f"  {attack:12s} n={n:2d} target_share={sum(r[0] for r in rows) / n:.3f} top_class_share={sum(r[1] for r in rows) / n:.3f} uniform={sum(r[2] for r in rows) / n:.3f} mean_rate={sum(rates[(placement, attack)]) / len(rates[(placement, attack)]):.2f}"
        )

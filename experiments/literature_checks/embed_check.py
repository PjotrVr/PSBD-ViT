import sys
import collections

sys.path.insert(0, ".")
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import select_rate_adaptively, select_rate_at_matched_shift

coverage = load_coverage("results")
cells = clearing_cells(coverage)
print("clearing cells", len(cells), "keys", list(cells[0].keys())[:12])
placements = [
    "after_embedding_token_mask",
    "before_attention_norm_token_mask",
    "post_residual",
]
by_attack = collections.defaultdict(lambda: collections.defaultdict(list))
for cell in cells:
    folder = cell.get("folder") or cell.get("folder_name")
    metrics = load_psbd_metrics("results", folder)
    if metrics is None:
        continue
    attack = metrics["attack"]
    for placement in placements:
        block = metrics["placements"].get(placement)
        if not block:
            continue
        shift = {
            row["rate"]: row["shift_ratio"]["validation"] for row in block["rates"]
        }
        for rule_name, rate in (
            ("adaptive", select_rate_adaptively(shift)),
            ("matched", select_rate_at_matched_shift(shift, 0.6)),
        ):
            if rate is None:
                continue
            row = next(r for r in block["rates"] if r["rate"] == rate)
            auroc = row["detection_psu_ratio"]["q0.25"]["auroc"]
            by_attack[(placement, rule_name)][attack].append(
                (auroc, row["shift_ratio"]["clean"], row["shift_ratio"]["backdoor"])
            )
for key in sorted(by_attack):
    print(key)
    for attack, rows in sorted(by_attack[key].items()):
        n = len(rows)

        def mean(i):
            return sum(r[i] for r in rows) / n

        print(
            f"  {attack:16s} n={n:2d} auroc={mean(0):.3f} shift_clean={mean(1):.3f} shift_backdoor={mean(2):.3f}"
        )

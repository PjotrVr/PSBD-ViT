import sys
import json
import os
import collections

sys.path.insert(0, ".")
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import select_rate_adaptively

cells = clearing_cells(load_coverage("results"))
detectors = [
    "confidence",
    "strip",
    "scale_up",
    "ibd_psc",
    "ibd_psc_calibrated",
    "teco",
    "cd_l",
    "ted",
    "beatrix",
    "sentinet",
]
table = collections.defaultdict(lambda: collections.defaultdict(list))
for cell in cells:
    folder = cell["folder_name"]
    metrics = load_psbd_metrics("results", folder)
    if metrics is None:
        continue
    attack = metrics["attack"]
    for name in detectors:
        path = f"results/{folder}/detectors/{name}_metrics.json"
        if os.path.exists(path):
            record = json.load(open(path))
            if record.get("status") == "scored":
                table[name][attack].append(record["detection"]["q0.25"]["auroc"])
    for placement, label in (
        ("before_attention_norm_token_mask", "PSBD-TM"),
        ("post_residual", "PSBD-RD"),
        ("mlp_neurons_channel_mask", "mlp_neurons_channel_mask"),
        ("before_attention_norm_gaussian", "gaussian_attn_input"),
    ):
        block = metrics["placements"].get(placement)
        if not block:
            continue
        shift = {r["rate"]: r["shift_ratio"]["validation"] for r in block["rates"]}
        rate = select_rate_adaptively(shift)
        if rate is None:
            continue
        row = next(r for r in block["rates"] if r["rate"] == rate)
        table[label][attack].append(row["detection_psu_ratio"]["q0.25"]["auroc"])
attacks = ["badnet_a2o", "tact", "blend", "lf", "sig", "wanet", "bpp"]
print(
    "detector".ljust(26)
    + "".join(a[:8].rjust(10) for a in attacks)
    + "      mean     n"
)
for name, per in table.items():
    values = [v for a in attacks for v in per.get(a, [])]
    print(
        name.ljust(26)
        + "".join(
            (
                f"{sum(per[a]) / len(per[a]):.3f}({len(per[a])})" if per.get(a) else "-"
            ).rjust(10)
            for a in attacks
        )
        + f"   {sum(values) / len(values):.3f}  {len(values)}"
    )

print("gain_scale target share at the largest rate")
share = collections.defaultdict(list)
for cell in cells:
    metrics = load_psbd_metrics("results", cell["folder_name"])
    if metrics is None:
        continue
    block = metrics["placements"].get("mlp_norm_out_gain_scale")
    if not block:
        continue
    row = max(block["rates"], key=lambda r: r["rate"])
    histogram = row.get("shift_target_histogram", {}).get("clean")
    if histogram and sum(histogram):
        share[metrics["attack"]].append(
            (
                row["rate"],
                histogram[metrics["target_label"]] / sum(histogram),
                row["shift_ratio"]["clean"],
            )
        )
for attack, rows in sorted(share.items()):
    print(
        attack,
        len(rows),
        "rate",
        rows[0][0],
        f"target_share={sum(r[1] for r in rows) / len(rows):.3f}",
        f"clean_shift={sum(r[2] for r in rows) / len(rows):.3f}",
    )

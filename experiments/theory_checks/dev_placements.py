import sys
import collections
import numpy as np

sys.path.insert(0, ".")
from scripts.paper._common import load_psbd_metrics
from defenses.decision import select_rate_adaptively, select_rate_at_matched_shift

DEV = [
    "vit_cifar10_wanet_0_1",
    "vit_tiny_wanet_0_05",
    "vit_cifar10_bpp_0_05",
    "vit_gtsrb_bpp_0_01",
    "vit_cifar10_blend_0_1",
    "vit_cifar10_badnet_a2o_0_01",
    "vit_gtsrb_tact_0_05",
    "vit_cifar10_tact_0_01",
    "vit_cifar100_bpp_0_01",
    "vit_tiny_lf_0_01",
]
cnt = collections.Counter()
tab = collections.defaultdict(dict)
for f in DEV:
    m = load_psbd_metrics("results", f)
    for k, b in m["placements"].items():
        sh = {
            r["rate"]: r["shift_ratio"]["validation"]
            for r in b["rates"]
            if r.get("shift_ratio")
        }
        ra = select_rate_adaptively(sh)
        rm = select_rate_at_matched_shift(sh, 0.6) if sh else None
        if ra is None:
            continue
        row = next(r for r in b["rates"] if r["rate"] == ra)
        rowm = next(r for r in b["rates"] if r["rate"] == rm)
        tab[k][f] = (
            row["detection_psu_ratio"]["q0.25"]["auroc"],
            row["shift_ratio"]["backdoor"],
            rowm["detection_psu_ratio"]["q0.25"]["auroc"],
        )
for k, v in sorted(tab.items(), key=lambda kv: -len(kv[1])):
    if len(v) < 1 or not any(
        s in k
        for s in (
            "droppath",
            "head_mask",
            "residual_channel_mask",
            "before_attention_token_mask",
            "after_mlp_residual",
            "blocks_9_12",
            "blocks_5_8_token",
            "after_embedding_token",
        )
    ):
        continue
    print(
        f"{k:55s} n={len(v)} adaptive {np.mean([x[0] for x in v.values()]):.3f} matched {np.mean([x[2] for x in v.values()]):.3f} | "
        + " ".join(f"{v[f][0]:.2f}" if f in v else " -- " for f in DEV)
    )
print("cols", [f.replace("vit_", "") for f in DEV])

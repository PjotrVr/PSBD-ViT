import sys
import json

sys.path.insert(0, ".")
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics

cells = clearing_cells(load_coverage("results"))
print("clearing cells", len(cells))
out = {}
for c in cells:
    f = c["folder_name"]
    m = load_psbd_metrics("results", f)
    if m is None:
        continue
    out[f] = {"attack": m["attack"], "dataset": m["dataset"]}
    for pl in (
        "before_attention_norm_token_mask",
        "post_residual",
        "before_attention_norm_gaussian",
        "before_mlp_norm_token_mask",
        "before_mlp_gaussian",
    ):
        b = m["placements"].get(pl)
        if not b:
            continue
        lad = [
            (
                r["rate"],
                r["shift_ratio"]["validation"],
                r["shift_ratio"]["backdoor"],
                r["detection_psu_ratio"]["q0.25"]["auroc"],
                r["detection"]["q0.25"]["auroc"] if "detection" in r else None,
            )
            for r in b["rates"]
        ]
        out[f][pl] = lad
json.dump(out, open(sys.argv[1], "w"))
print(len(out))

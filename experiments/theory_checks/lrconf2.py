import statistics as st
import sys
import os
import numpy as np
import collections

sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import pair_clean_to_backdoor, select_rate_adaptively
from defenses.cache import baseline_path, load_baseline, read_split_manifest

cells = clearing_cells(load_coverage("results"))
rng = np.random.default_rng(0)
rows = []
for c in cells:
    f = c["folder_name"]
    m = load_psbd_metrics("results", f)
    if m is None:
        continue
    d = os.path.join("results", f, "psbd")
    man = read_split_manifest(d)
    pc = {}
    for s in ("clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(d, s))
        pc[s] = probs.gather(1, labels.view(-1, 1).long()).squeeze(1).double()
    cl = pair_clean_to_backdoor(pc["clean"], man).numpy()
    bd = pc["backdoor"].numpy()
    n = len(bd)
    # pair-level folds so a clean image and its triggered twin share a fold
    fold = rng.integers(0, 2, n)
    x = np.r_[cl, bd]
    y = np.r_[np.zeros(n), np.ones(n)]
    fo = np.r_[fold, fold]
    z = -np.log1p(-np.minimum(x, 1 - 1e-12))
    s = np.zeros_like(z)
    for k in (0, 1):
        tr = fo != k
        te = fo == k
        edges = np.unique(np.quantile(z[tr], np.linspace(0, 1, 41)))
        bt = np.clip(np.searchsorted(edges, z[tr], side="right") - 1, 0, len(edges) - 2)
        be = np.clip(np.searchsorted(edges, z[te], side="right") - 1, 0, len(edges) - 2)
        nb = len(edges) - 1
        lr = (np.bincount(bt[y[tr] == 1], minlength=nb) + 0.5) / (
            np.bincount(bt[y[tr] == 0], minlength=nb) + 0.5
        )
        s[te] = lr[be]
    astar = roc_auc_score(y, s)
    out = {"f": f, "attack": m["attack"], "astar": astar, "conf": roc_auc_score(y, z)}
    for pl in ("before_attention_norm_token_mask", "post_residual"):
        b = m["placements"][pl]
        r = select_rate_adaptively(
            {x["rate"]: x["shift_ratio"]["validation"] for x in b["rates"]}
        )
        out[pl] = next(x for x in b["rates"] if x["rate"] == r)["detection_psu_ratio"][
            "q0.25"
        ]["auroc"]
    rows.append(out)

print(
    "cross-fitted LR-optimal confidence mean",
    st.mean(r["astar"] for r in rows),
    " raw conf",
    st.mean(r["conf"] for r in rows),
)
for pl in ("before_attention_norm_token_mask", "post_residual"):
    print(
        pl,
        "mean",
        st.mean(r[pl] for r in rows),
        "beats A* in",
        sum(r[pl] > r["astar"] for r in rows),
        "of",
        len(rows),
        "mean margin",
        st.mean(r[pl] - r["astar"] for r in rows),
    )
by = collections.defaultdict(list)
for r in rows:
    by[r["attack"]].append(r)
for a, v in by.items():
    print(
        f"  {a:12s} A* {st.mean(x['astar'] for x in v):.3f} TM {st.mean(x['before_attention_norm_token_mask'] for x in v):.3f} RD {st.mean(x['post_residual'] for x in v):.3f}"
    )

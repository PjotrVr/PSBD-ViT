import sys
import os
import collections
import numpy as np

sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import select_rate_adaptively, pair_clean_to_backdoor
from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)

cells = clearing_cells(load_coverage("results"))
out = collections.defaultdict(lambda: collections.defaultdict(list))
tpr = collections.defaultdict(list)
for c in cells:
    f = c["folder_name"]
    m = load_psbd_metrics("results", f)
    if m is None:
        continue
    t = m["target_label"]
    d = os.path.join("results", f, "psbd")
    man = read_split_manifest(d)
    for pl in ("before_attention_norm_token_mask", "post_residual"):
        b = m["placements"][pl]
        rate = select_rate_adaptively(
            {r["rate"]: r["shift_ratio"]["validation"] for r in b["rates"]}
        )
        row = next(r for r in b["rates"] if r["rate"] == rate)
        tpr[pl].append(row["detection_psu_ratio"]["q0.25"]["tpr"])
        sc = {}
        for s in ("clean", "backdoor"):
            probs, labels, _ = load_baseline(baseline_path(d, s))
            pp, am = load_dropout_pass_probs(dropout_pass_path(d, pl, rate, s))
            pc = (
                probs.gather(1, labels.view(-1, 1).long())
                .squeeze(1)
                .float()
                .clamp_min(1e-6)
            )
            frac = 1 - pp.float().mean(0) / pc
            if s == "clean" and t is not None:
                pp2 = pp.float().clone()
                hit = (am.long() == t) & (labels.view(1, -1).long() != t)
                pp2[hit] = pc.expand_as(pp2)[hit]
                frac2 = 1 - pp2.mean(0) / pc
            else:
                frac2 = frac
            sc[s] = (frac, frac2)
        cl = pair_clean_to_backdoor(sc["clean"][0], man)
        cl2 = pair_clean_to_backdoor(sc["clean"][1], man)
        bd = sc["backdoor"][0]
        y = np.r_[np.zeros(len(cl)), np.ones(len(bd))]
        a0 = roc_auc_score(y, np.r_[-cl.numpy(), -bd.numpy()])
        a1 = roc_auc_score(y, np.r_[-cl2.numpy(), -bd.numpy()])
        out[pl][m["attack"]].append((a0, a1))
for pl, dd in out.items():
    print(pl, "mean TPR@q0.25", np.mean(tpr[pl]))
    for a, v in dd.items():
        v = np.array(v)
        print(
            f"  {a:12s} n={len(v)} auroc {v[:, 0].mean():.3f} -> target flips removed {v[:, 1].mean():.3f}  (delta {np.mean(v[:, 1] - v[:, 0]):+.3f}, max |delta| {np.abs(v[:, 1] - v[:, 0]).max():.3f})"
        )
    allv = np.array([x for v in dd.values() for x in v])
    print("  all", allv[:, 0].mean(), allv[:, 1].mean())

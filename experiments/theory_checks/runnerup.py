import sys
import os
import collections
import numpy as np

sys.path.insert(0, ".")
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import select_rate_adaptively
from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)

cells = clearing_cells(load_coverage("results"))
out = collections.defaultdict(lambda: collections.defaultdict(list))
for c in cells:
    f = c["folder_name"]
    m = load_psbd_metrics("results", f)
    if m is None:
        continue
    t = m["target_label"]
    if t is None:
        continue
    d = os.path.join("results", f, "psbd")
    probs, labels, _ = load_baseline(baseline_path(d, "validation"))
    top2 = probs.topk(2, dim=1).indices[:, 1]
    nontarget = labels.long() != t
    for pl in ("before_attention_norm_token_mask", "post_residual"):
        b = m["placements"][pl]
        rate = select_rate_adaptively(
            {r["rate"]: r["shift_ratio"]["validation"] for r in b["rates"]}
        )
        pp, am = load_dropout_pass_probs(dropout_pass_path(d, pl, rate, "validation"))
        shifted = (am.long() != labels.view(1, -1).long()) & nontarget.view(1, -1)
        ru_t = (top2 == t).view(1, -1).expand_as(shifted)
        land = am.long() == t
        # P(land target | shifted, runner-up is target) and (| runner-up not target)
        a = (land & shifted & ru_t).sum().item() / max((shifted & ru_t).sum().item(), 1)
        bb = (land & shifted & ~ru_t).sum().item() / max(
            (shifted & ~ru_t).sum().item(), 1
        )
        # P(land on runner-up | shifted)
        ru_land = (am.long() == top2.view(1, -1)).logical_and(
            shifted
        ).sum().item() / max(shifted.sum().item(), 1)
        share_ru_t = float((top2[nontarget] == t).float().mean())
        out[pl][m["attack"]].append((a, bb, ru_land, share_ru_t))
for pl, dd in out.items():
    print(pl)
    for a, v in dd.items():
        v = np.array(v)
        print(
            f"  {a:12s} n={len(v)} P(target|shift,runnerup=t) {v[:, 0].mean():.3f}  P(target|shift,runnerup!=t) {v[:, 1].mean():.3f}  P(lands on runner-up|shift) {v[:, 2].mean():.3f}  share runner-up=t {v[:, 3].mean():.3f}"
        )

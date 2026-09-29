import sys
import os
import json
import collections
import numpy as np
import torch

sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, norm
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import pair_clean_to_backdoor
from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.scores import to_rank

A, B = "before_attention_norm_token_mask", "pre_residual_blocks_9_12"
cells = clearing_cells(load_coverage("results"))
rng = np.random.default_rng(0)
rows = []


def auc(cl, bd):
    y = np.r_[np.zeros(len(cl)), np.ones(len(bd))]
    return roc_auc_score(y, np.r_[-cl, -bd])


for c in cells:
    f = c["folder_name"]
    m = load_psbd_metrics("results", f)
    if (
        m is None
        or B not in m["placements"]
        or m["placements"][B].get("adaptive_rate") is None
    ):
        continue
    d = os.path.join("results", f, "psbd")
    man = read_split_manifest(d)
    base = {
        s: load_baseline(baseline_path(d, s))
        for s in ("validation", "clean", "backdoor")
    }
    sc = {}
    for pl in (A, B):
        rate = m["placements"][pl]["adaptive_rate"]
        sc[pl] = {}
        for s in ("validation", "clean", "backdoor"):
            probs, labels, _ = base[s]
            pp, _ = load_dropout_pass_probs(dropout_pass_path(d, pl, rate, s))
            pc = (
                probs.gather(1, labels.view(-1, 1).long())
                .squeeze(1)
                .float()
                .clamp_min(1e-6)
            )
            sc[pl][s] = 1 - pp.float().mean(0) / pc
        sc[pl]["clean"] = pair_clean_to_backdoor(sc[pl]["clean"], man)
    out = {"f": f, "attack": m["attack"]}
    for pl, key in ((A, "tm"), (B, "late")):
        out[key] = auc(sc[pl]["clean"].numpy(), sc[pl]["backdoor"].numpy())
    mix = {s: 0.5 * (sc[A][s] + sc[B][s]) for s in ("validation", "clean", "backdoor")}
    out["mix"] = auc(mix["clean"].numpy(), mix["backdoor"].numpy())
    rk = {
        pl: {
            s: to_rank(sc[pl][s], sc[pl]["validation"])
            for s in ("validation", "clean", "backdoor")
        }
        for pl in (A, B)
    }
    un = {s: torch.minimum(rk[A][s], rk[B][s]) for s in ("clean", "backdoor")}
    out["union"] = auc(un["clean"].numpy(), un["backdoor"].numpy())
    mr = {s: 0.5 * (rk[A][s] + rk[B][s]) for s in ("clean", "backdoor")}
    out["meanrank"] = auc(mr["clean"].numpy(), mr["backdoor"].numpy())
    out["r_val"] = float(spearmanr(sc[A]["validation"], sc[B]["validation"])[0])
    out["r_bd"] = float(spearmanr(sc[A]["backdoor"], sc[B]["backdoor"])[0])
    # per-triggered-sample: which probe catches it at 10% clean-val quantile
    ta = float(np.quantile(sc[A]["validation"], 0.1))
    tb = float(np.quantile(sc[B]["validation"], 0.1))
    ca = (sc[A]["backdoor"] < ta).numpy()
    cb = (sc[B]["backdoor"] < tb).numpy()
    out["both"] = float((ca & cb).mean())
    out["onlyA"] = float((ca & ~cb).mean())
    out["onlyB"] = float((~ca & cb).mean())
    out["neither"] = float((~ca & ~cb).mean())
    # binormal prediction from marginal AUROCs and clean correlation (normal scores)
    zA = norm.ppf(np.clip(rk[A]["validation"].numpy(), 1e-4, 1 - 1e-4))
    zB = norm.ppf(np.clip(rk[B]["validation"].numpy(), 1e-4, 1 - 1e-4))
    r = float(np.corrcoef(zA, zB)[0, 1])
    out["rz"] = r
    muA = np.sqrt(2) * norm.ppf(min(out["tm"], 0.9999))
    muB = np.sqrt(2) * norm.ppf(min(max(out["late"], 1e-4), 0.9999))
    out["pred_sum"] = float(norm.cdf((muA + muB) / (2 * np.sqrt(1 + r))))
    cov = [[1, r], [r, 1]]
    n = 200000
    Zc = rng.multivariate_normal([0, 0], cov, n)
    Zb = rng.multivariate_normal([-muA, -muB], cov, n)
    mc = np.minimum(Zc[:, 0], Zc[:, 1])
    mb = np.minimum(Zb[:, 0], Zb[:, 1])
    out["pred_union"] = float(
        roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[-mc, -mb])
    )
    rows.append(out)
json.dump(rows, open(sys.argv[1], "w"))
k = (
    "tm",
    "late",
    "mix",
    "union",
    "meanrank",
    "pred_sum",
    "pred_union",
    "r_val",
    "r_bd",
    "rz",
    "both",
    "onlyA",
    "onlyB",
    "neither",
)
print("n", len(rows))
print({x: round(float(np.mean([r[x] for r in rows])), 4) for x in k})
by = collections.defaultdict(list)
for r in rows:
    by[r["attack"]].append(r)
for a, v in by.items():
    print(a, len(v), {x: round(float(np.mean([r[x] for r in v])), 3) for x in k})
print(
    "union>mix in",
    sum(r["union"] > r["mix"] for r in rows),
    "; pred union>pred sum in",
    sum(r["pred_union"] > r["pred_sum"] for r in rows),
    "; agree on sign",
    sum((r["union"] > r["mix"]) == (r["pred_union"] > r["pred_sum"]) for r in rows),
)

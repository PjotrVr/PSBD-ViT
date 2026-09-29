import sys
import json
import os

sys.path.insert(0, ".")
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
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
res = []
for c in cells:
    f = c["folder_name"]
    m = load_psbd_metrics("results", f)
    if m is None:
        continue
    d = os.path.join("results", f, "psbd")
    man = read_split_manifest(d)
    base = {
        s: load_baseline(baseline_path(d, s))
        for s in ("validation", "clean", "backdoor")
    }
    for pl in ("before_attention_norm_token_mask", "post_residual"):
        b = m["placements"].get(pl)
        if not b:
            continue
        rate = select_rate_adaptively(
            {r["rate"]: r["shift_ratio"]["validation"] for r in b["rates"]}
        )
        if rate is None:
            continue
        out = {
            "folder": f,
            "attack": m["attack"],
            "dataset": m["dataset"],
            "placement": pl,
            "rate": rate,
        }
        sc = {}
        for s in ("validation", "clean", "backdoor"):
            probs, labels, _ = base[s]
            pp, am = load_dropout_pass_probs(dropout_pass_path(d, pl, rate, s))
            k = pp.shape[0]
            pc = probs.gather(1, labels.view(-1, 1).long()).squeeze(1).float()
            frac = 1 - pp.float().mean(0) / pc.clamp_min(1e-6)
            ab = pc - pp.float().mean(0)
            cnt = (am.long() != labels.view(1, -1).long()).sum(0).float()
            sc[s] = dict(pc=pc, frac=frac, ab=ab, cnt=cnt)
        out["k"] = k
        cl = {
            key: pair_clean_to_backdoor(sc["clean"][key], man)
            for key in ("pc", "frac", "ab", "cnt")
        }
        bd = sc["backdoor"]
        y = np.r_[np.zeros(len(cl["frac"])), np.ones(len(bd["frac"]))]

        def auc(key, sign=-1):
            s = np.r_[sign * cl[key].numpy(), sign * bd[key].numpy()]
            return float(roc_auc_score(y, s))

        out["auc_frac"] = auc("frac")
        out["auc_abs"] = auc("ab")
        out["auc_cnt"] = auc("cnt")
        out["auc_conf"] = auc("pc", +1)
        hc = np.bincount(cl["cnt"].long().numpy(), minlength=k + 1) / len(cl["cnt"])
        hb = np.bincount(bd["cnt"].long().numpy(), minlength=k + 1) / len(bd["cnt"])
        hv = np.bincount(sc["validation"]["cnt"].long().numpy(), minlength=k + 1) / len(
            sc["validation"]["cnt"]
        )
        out["hist_clean"] = hc.tolist()
        out["hist_bd"] = hb.tolist()
        out["hist_val"] = hv.tolist()
        out["ptie_cnt"] = float((hc * hb).sum())
        # ties in frac
        out["frac_zero_val"] = float(
            (sc["validation"]["frac"].abs() < 1e-7).float().mean()
        )
        out["frac_zero_bd"] = float((bd["frac"].abs() < 1e-7).float().mean())
        lp_c = torch.log(cl["pc"].clamp_min(1e-6))
        lp_b = torch.log(bd["pc"].clamp_min(1e-6))
        out["dlogP"] = float(lp_b.mean() - lp_c.mean())
        out["meanP_clean"] = float(cl["pc"].mean())
        out["meanP_bd"] = float(bd["pc"].mean())
        allf = torch.cat([cl["frac"], bd["frac"]]).numpy()
        allp = torch.cat([cl["pc"], bd["pc"]]).numpy()
        out["r2_frac_conf"] = float(np.corrcoef(allf, allp)[0, 1] ** 2)
        # within clean: spearman frac vs confidence
        out["sp_clean_frac_conf"] = float(
            spearmanr(cl["frac"].numpy(), cl["pc"].numpy())[0]
        )
        # validation mean flip prob and overdispersion
        q = float(sc["validation"]["cnt"].mean() / k)
        from math import comb

        out["q_val"] = q
        out["binom_val"] = [
            comb(k, j) * q**j * (1 - q) ** (k - j) for j in range(k + 1)
        ]
        res.append(out)
json.dump(res, open(sys.argv[1], "w"))
print(len(res))

import json
import numpy as np
import collections

R = json.load(open("results/_experiments/theory_checks/cache_read.json"))
for pl in ("before_attention_norm_token_mask", "post_residual"):
    rs = [r for r in R if r["placement"] == pl]
    print("==", pl, len(rs), "k", set(r["k"] for r in rs))
    for key in (
        "auc_frac",
        "auc_abs",
        "auc_cnt",
        "auc_conf",
        "ptie_cnt",
        "frac_zero_val",
        "frac_zero_bd",
        "dlogP",
        "r2_frac_conf",
        "sp_clean_frac_conf",
        "q_val",
        "meanP_clean",
        "meanP_bd",
    ):
        v = [r[key] for r in rs]
        print(
            f"  {key:20s} mean {np.mean(v):.4f} median {np.median(v):.4f} min {min(v):.4f} max {max(v):.4f}"
        )
    # bound check: auc_frac - auc_cnt vs 0.5*ptie
    d = [(r["auc_frac"] - r["auc_cnt"], 0.5 * r["ptie_cnt"]) for r in rs]
    print(
        "  frac-cnt mean",
        np.mean([x for x, _ in d]),
        " half-tie mean",
        np.mean([y for _, y in d]),
        " violations |diff|>halftie",
        sum(abs(x) > y + 1e-9 for x, y in d),
    )
    # upper bound on count AUROC: 1-0.5 ptie
    print(
        "  cnt auc <= 1-ptie/2 ?",
        all(r["auc_cnt"] <= 1 - 0.5 * r["ptie_cnt"] + 1e-9 for r in rs),
    )
    # frac vs abs wins
    print(
        "  frac>abs in",
        sum(r["auc_frac"] > r["auc_abs"] for r in rs),
        "of",
        len(rs),
        " mean diff",
        np.mean([r["auc_frac"] - r["auc_abs"] for r in rs]),
    )
    hv = np.mean([r["hist_val"] for r in rs], 0)
    hb = np.mean([r["binom_val"] for r in rs], 0)
    print(
        "  validation count hist mean",
        np.round(hv, 3),
        " binomial at same q",
        np.round(hb, 3),
    )
    print(
        "  clean(paired) hist",
        np.round(np.mean([r["hist_clean"] for r in rs], 0), 3),
        " backdoor hist",
        np.round(np.mean([r["hist_bd"] for r in rs], 0), 3),
    )
    by = collections.defaultdict(list)
    for r in rs:
        by[r["attack"]].append(r)
    for a, v in by.items():
        print(
            f"   {a:12s} n={len(v)} frac {np.mean([x['auc_frac'] for x in v]):.3f} abs {np.mean([x['auc_abs'] for x in v]):.3f} cnt {np.mean([x['auc_cnt'] for x in v]):.3f} conf {np.mean([x['auc_conf'] for x in v]):.3f} ptie {np.mean([x['ptie_cnt'] for x in v]):.3f} dlogP {np.mean([x['dlogP'] for x in v]):.3f} bdhist {np.round(np.mean([x['hist_bd'] for x in v], 0), 3)}"
        )

import json
import glob
from math import comb

R = "results/_experiments/why_token_masking_works"
agg = {"obs": [0] * 5, "pred": [0.0] * 5, "kc": [0] * 5}
agg1 = {"obs": [0] * 5, "pred": [0.0] * 5}
chg = []
for f in sorted(glob.glob(R + "/vit_*.json")):
    d = json.load(open(f))
    if d["attack"] != "badnet_a2o":
        continue
    stm = d["stochastic_token_mask"]
    p = stm["rate"]
    m = len(d["trigger_positions"])
    la = stm["triggered"]["late_all"]
    obs = [la.get(str(j), [0, 0])[1] for j in range(5)]
    kc = [la.get(str(j), [0, 0])[0] for j in range(5)]
    n = sum(obs)
    q = p**m
    pred = [n * comb(4, j) * q**j * (1 - q) ** (4 - j) for j in range(5)]
    changed = 1 - sum(kc) / n
    det = d["deterministic_masking"]
    s1 = det["last_1"]["triggered_kept"]
    s2 = det["last_2"]["triggered_kept"]
    s4 = det["last_4"]["triggered_kept"]
    sdet = [1, s1, s2, (s2 + s4) / 2, s4]
    pred_det = sum(pred[j] / n * (1 - sdet[j]) for j in range(5))
    allnone = q**4  # all-or-nothing prediction of changed
    print(
        f"{d['folder']:32s} m={m} p={p} n={n} obs={obs} pred={[round(x, 1) for x in pred]} P4obs={obs[4] / n:.4f} P4pred={q**4:.5f} changed={changed:.4f} allornone={allnone:.5f} det_model={pred_det:.4f} s_j={[round(kc[j] / obs[j], 3) if obs[j] else None for j in range(5)]} sdet={[round(x, 3) for x in sdet]}"
    )
    A = agg if m == 4 else agg1
    for j in range(5):
        A["obs"][j] += obs[j]
        A["pred"][j] += pred[j]
        if m == 4:
            agg["kc"][j] += kc[j]
    chg.append((m, p, n, changed, allnone, pred_det))
for name, A in (("4-token", agg), ("1-token", agg1)):
    N = sum(A["obs"])
    print(
        name,
        "obs",
        A["obs"],
        "pred",
        [round(x, 1) for x in A["pred"]],
        "P4 obs",
        round(A["obs"][4] / N, 5),
        "pred",
        round(A["pred"][4] / N, 5),
    )
# chi-square like per-row ratio
N = sum(agg["obs"])
print("pooled 4-token changed", 1 - sum(agg["kc"]) / N)
w = [c for c in chg if c[0] == 4]
print(
    "4-token: mean changed",
    sum(c[3] * c[2] for c in w) / sum(c[2] for c in w),
    "all-or-none pred",
    sum(c[4] * c[2] for c in w) / sum(c[2] for c in w),
    "det model",
    sum(c[5] * c[2] for c in w) / sum(c[2] for c in w),
)
w = chg
print(
    "all 12: mean changed (unweighted)",
    sum(c[3] for c in w) / len(w),
    "all-or-none",
    sum(c[4] for c in w) / len(w),
    "det",
    sum(c[5] for c in w) / len(w),
)

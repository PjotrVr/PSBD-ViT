import collections
import json
import math
import statistics as st
import numpy as np
from scipy.stats import binom

L = json.load(open("results/_experiments/theory_checks/ladders.json"))
pl = "before_attention_norm_token_mask"
ex = list(L)[0]
print(ex, [(r[0], round(r[1], 3)) for r in L[ex][pl]])
nadap = []
rates = []
sig = []
neff_all = []


def kofn(p, n, r):
    return binom.sf(r - 1, n, 1 - p)  # P(Bin(n,1-p)>=r)


fits = []
for f, d in L.items():
    lad = d.get(pl)
    if not lad:
        continue
    shift = {r[0]: r[1] for r in lad}
    rate = min([r for r, s in shift.items() if s >= 0.8], default=None)
    if rate is None:
        continue
    s = shift[rate]
    rates.append(rate)
    sig.append(s)
    nadap.append(math.log(1 - s) / math.log(1 - rate))
    # per-rate AND n
    ns = [
        (r, math.log(1 - sv) / math.log(1 - r))
        for r, sv in sorted(shift.items())
        if 0 < sv < 1
    ]
    neff_all.append(
        (f, d["attack"], rate, round(s, 3), [(r, round(n, 2)) for r, n in ns])
    )
    # fit k-of-n with continuous n, r=rho*n via beta-ish: use grid over integer n,r
    ps = np.array(sorted(shift))
    ks = np.array([1 - shift[r] for r in ps])
    best = None
    for n in range(1, 200):
        for r in range(1, n + 1):
            pred = kofn(ps, n, r)
            e = float(((pred - ks) ** 2).sum())
            if best is None or e < best[0]:
                best = (e, n, r)
    fits.append((f, d["attack"], rate, best))
print(
    "adaptive rates",
    sorted(set(rates)),
)

print(collections.Counter(rates))
print(
    "achieved shift at adaptive: mean", st.mean(sig), "min", min(sig), "max", max(sig)
)
print(
    "AND-model n_eff at adaptive: median",
    st.median(nadap),
    "range",
    min(nadap),
    max(nadap),
)
for x in neff_all[:8]:
    print(x)
rs = [b[3][2] / b[3][1] for b in fits]
ns = [b[3][1] for b in fits]
es = [b[3][0] for b in fits]
print(
    "k-of-n fits: median n",
    st.median(ns),
    "median r/n",
    st.median(rs),
    "median sse",
    st.median(es),
)
for b in fits[:12]:
    print(b)

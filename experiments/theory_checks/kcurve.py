import json
import numpy as np
from scipy.optimize import curve_fit

d = json.load(open("paper/figures/fig_forward_passes.json"))["plotted"]
agg = d["pilot_group_aggregate"]
ks = np.array(sorted(int(k) for k in agg))
A = np.array([agg[str(k)]["auroc"]["mean"] for k in ks])
print(np.round(A, 4))


def f1(k, a, c):
    return a - c / k


p1, _ = curve_fit(f1, ks, A)
print("A_inf - C/k fit", p1, "rmse", np.sqrt(np.mean((f1(ks, *p1) - A) ** 2)))


def f2(k, a, c, e):
    return a - c / k**e


p2, _ = curve_fit(f2, ks, A, p0=[0.975, 0.01, 1])
print("power fit", p2, "rmse", np.sqrt(np.mean((f2(ks, *p2) - A) ** 2)))
# using only k>=3
m = ks >= 3
p3, _ = curve_fit(f1, ks[m], A[m])
print("k>=3 1/k fit", p3, "pred k=1", f1(1, *p3), "obs", A[0])
cells = d["pilot_cells_used"]
print(len(cells))
for f, c in cells.items():
    a = [c["by_k"][str(k)]["auroc"] for k in (1, 2, 3, 5, 10, 20)]
    print(f"{f:34s} rate {c['rate']} " + " ".join(f"{x:.4f}" for x in a))
print(d["pilot_cells_used"][list(cells)[0]]["by_k"]["1"])

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(1)
n = 400000


def aucs(muA, muB, r):
    cov = [[1, r], [r, 1]]
    Zc = rng.multivariate_normal([0, 0], cov, n)
    Zb = rng.multivariate_normal([-muA, -muB], cov, n)
    y = np.r_[np.zeros(n), np.ones(n)]
    u = roc_auc_score(y, -np.r_[Zc.min(1), Zb.min(1)])
    s = norm.cdf((muA + muB) / (2 * np.sqrt(1 + r)))
    opt = norm.cdf(
        np.sqrt((muA**2 + muB**2 - 2 * r * muA * muB) / (1 - r * r)) / np.sqrt(2)
    )
    return u, s, opt


for aucA in (0.966, 0.99):
    muA = np.sqrt(2) * norm.ppf(aucA)
    for r in (0.0, 0.31, 0.6):
        row = []
        for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            u, s, o = aucs(muA, lam * muA, r)
            row.append(f"lam {lam}: union {u:.4f} sum {s:.4f} opt {o:.4f}")
        print("AUC_A", aucA, "r", r)
        print("   " + "\n   ".join(row))
# inverted member
muA = np.sqrt(2) * norm.ppf(0.966)
for aucB in (0.4, 0.5):
    u, s, o = aucs(muA, np.sqrt(2) * norm.ppf(aucB), 0.31)
    print("inverted/null B", aucB, "union", round(u, 4), "sum", round(s, 4))

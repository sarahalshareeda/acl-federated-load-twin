# dm_test_perclient.py — per-client Diebold-Mariano (panel-correct).
# Runs DM WITHIN each building, then aggregates: median DM + % of clients significant.
# Reuses `sweep` from v4 cell 7 (its lists preserve per-client boundaries).

import numpy as np
from scipy.stats import t as tdist

LEVEL = 1.0
H = 24

def pinball(y, lo, md, up):
    L = np.zeros_like(y, dtype=float)
    for q, p in {0.1: lo, 0.5: md, 0.9: up}.items():
        e = y - p; L += np.maximum(q * e, (q - 1) * e)
    return L / 3.0

def dm(lA, lB, h=H):
    d = lA - lB; m = np.isfinite(d); d = d[m]; n = len(d)
    if n < 3 * h: return np.nan, np.nan
    var = np.var(d)
    for k in range(1, h):
        if k < n: var += 2 * np.cov(d[:-k], d[k:])[0, 1]
    stat = d.mean() / np.sqrt(max(var, 1e-12) / n)
    stat *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    return stat, 2 * tdist.cdf(-abs(stat), df=n - 1)

def perclient(A, B):
    yA = sweep[(LEVEL, A)]["y"]; loA = sweep[(LEVEL, A)]["lo"]; mdA = sweep[(LEVEL, A)]["md"]; upA = sweep[(LEVEL, A)]["up"]
    loB = sweep[(LEVEL, B)]["lo"]; mdB = sweep[(LEVEL, B)]["md"]; upB = sweep[(LEVEL, B)]["up"]
    stats = []; sig = 0; nfav = 0
    for k in range(len(yA)):
        y = yA[k]
        LA = pinball(y, loA[k], mdA[k], upA[k])
        LB = pinball(y, loB[k], mdB[k], upB[k])
        s, p = dm(LA, LB)
        if np.isfinite(s):
            stats.append(s)
            if np.isfinite(p) and p < 0.05:
                sig += 1
                if s < 0: nfav += 1
    stats = np.array(stats)
    return np.median(stats), 100 * nfav / len(stats), 100 * sig / len(stats)

pairs = [("modular_local", "integrated"),
         ("cqr_local", "integrated"),
         ("modular_local", "modular_fed"),
         ("cqr_local", "cqr_fed")]

print(f"Per-client DM at level={LEVEL} (negative favors first-named)\n")
print(f"{'comparison':38s} {'medianDM':>9s} {'%sig&fav':>9s} {'%sig':>6s}")
for A, B in pairs:
    md_, favp, sigp = perclient(A, B)
    print(f"{A+' vs '+B:38s} {md_:9.2f} {favp:8.0f}% {sigp:5.0f}%")

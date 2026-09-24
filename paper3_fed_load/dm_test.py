# dm_test.py — Diebold-Mariano significance tests between schemes.
# Run in the SAME v4 session (reuses `sweep` from cell 7: per (level,scheme) y/lo/md/up).
# Computes per-hour pinball loss for each scheme at LEVEL, then DM (HLN-corrected).

import numpy as np
from scipy.stats import t as tdist

LEVEL = 1.0          # degradation level to test at
H     = 24           # forecast horizon for HLN correction (day-ahead)
QS_LEVELS = [0.1, 0.5, 0.9]

def pinball_series(y, lo, md, up):
    preds = {0.1: lo, 0.5: md, 0.9: up}
    L = np.zeros_like(y, dtype=float)
    for q, p in preds.items():
        e = y - p
        L += np.maximum(q * e, (q - 1) * e)
    return L / len(preds)

def series(scheme):
    s = sweep[(LEVEL, scheme)]
    y = np.concatenate(s["y"]); lo = np.concatenate(s["lo"])
    md = np.concatenate(s["md"]); up = np.concatenate(s["up"])
    return y, pinball_series(y, lo, md, up)

def dm(lA, lB, h=H):
    d = lA - lB
    m = np.isfinite(d); d = d[m]; n = len(d)
    dbar = d.mean()
    var = np.var(d)
    for k in range(1, h):
        if k < n:
            g = np.cov(d[:-k], d[k:])[0, 1]; var += 2 * g
    stat = dbar / np.sqrt(max(var, 1e-12) / n)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    stat *= hln
    p = 2 * tdist.cdf(-abs(stat), df=n - 1)
    return stat, p

pairs = [
    ("modular_local", "integrated"),
    ("cqr_local", "integrated"),
    ("modular_local", "modular_fed"),
    ("cqr_local", "cqr_fed"),
]
print(f"Diebold-Mariano at level={LEVEL} (negative favors first-named):\n")
print(f"{'comparison':38s} {'DM':>8s} {'p-value':>12s}")
common_y = None
loss = {}
for sch in set([a for a, _ in pairs] + [b for _, b in pairs]):
    y, L = series(sch); loss[sch] = L
    common_y = y if common_y is None else common_y
for a, b in pairs:
    stat, p = dm(loss[a], loss[b])
    print(f"{a+' vs '+b:38s} {stat:8.2f} {p:12.2e}")

# For adaptive vs fixed@0, run adaptive_conformal.py first (needs its per-point arrays),
# then compare using the same dm() on their pinball-loss series.
print("\nNote: adaptive-vs-fixed@0 DM requires the adaptive run's predictions;\n"
      "compute pinball series from that store and call dm() the same way.")

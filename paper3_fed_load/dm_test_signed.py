# dm_test_signed.py -- per-client Diebold-Mariano for the score comparison,
# so the paper's main number carries a significance test rather than a mean.
#
#   exec(open(f"{PROJECT_DIR}/dm_test_signed.py").read())
#
# Run AFTER baselines.py in the same session: it reuses calibrate(), the
# cached rollouts, and the tuned step sizes recorded in results_baselines.csv.
# Uses the same pinball loss, the same HLN correction and the same h=24 as
# dm_test_perclient.py, so the new rows are comparable with tab:dm.
#
# The pair that matters is acl_asym vs acl. If the signed score does not beat
# the symmetric one significantly within buildings, the 50% cost figure is a
# fleet aggregate driven by a handful of large clients and must be reported
# as such.

import os
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

LEVEL = 1.0
H = 24


# ---- loss and test, lifted verbatim from dm_test_perclient.py -------------
def pinball(y, lo, md, up):
    L = np.zeros_like(y, dtype=float)
    for q, p in {0.1: lo, 0.5: md, 0.9: up}.items():
        e = y - p
        L += np.maximum(q * e, (q - 1) * e)
    return L / 3.0


def dm(lA, lB, h=H):
    d = lA - lB
    m = np.isfinite(d)
    d = d[m]
    n = len(d)
    if n < 3 * h:
        return np.nan, np.nan
    var = np.var(d)
    for k in range(1, h):
        if k < n:
            var += 2 * np.cov(d[:-k], d[k:])[0, 1]
    stat = d.mean() / np.sqrt(max(var, 1e-12) / n)
    stat *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    return stat, 2 * tdist.cdf(-abs(stat), df=n - 1)


# ---- the tuned step sizes, read back rather than retyped ------------------
bl = pd.read_csv(os.path.join(OUT_DIR, "results_baselines.csv"))
bl = bl[(bl.base == "MLP-point") & (bl.level == LEVEL)].set_index("scheme")


def tuned(method):
    kw = {}
    if method in bl.index:
        r = bl.loc[method]
        if "gamma" in bl.columns and np.isfinite(r.get("gamma", np.nan)):
            kw["gamma"] = float(r["gamma"])
        if "eta_frac" in bl.columns and np.isfinite(r.get("eta_frac", np.nan)):
            kw["eta_frac"] = float(r["eta_frac"])
    return kw


# ---- per-client panels ----------------------------------------------------
def panels_calibrated(method):
    """(y, lo, md, up) per client from the shared calibrate()."""
    kw = tuned(method)
    out = []
    for nm in clients:
        y, pt = test_pred[LEVEL][nm]
        blo = bup = np.asarray(pt, dtype=float)
        mid = np.asarray(pt, dtype=float)
        seed = calib_scores[0.0][nm]
        run_method = method
        if method == "acl_debias":
            e = calib_signed[0.0][nm]
            b = float(np.mean(e))
            blo = bup = blo + b
            mid = mid + b
            seed = np.abs(e - b)
            run_method = "acl"
        lo, up = calibrate(y, blo, bup, seed, run_method, **kw)
        out.append((np.asarray(y).ravel(), lo, mid, up))
    return out


def panels_sweep(scheme):
    """(y, lo, md, up) per client from the stored sweep."""
    s = sweep[(LEVEL, scheme)]
    return [(np.asarray(s["y"][k]).ravel(), np.asarray(s["lo"][k]).ravel(),
             np.asarray(s["md"][k]).ravel(), np.asarray(s["up"][k]).ravel())
            for k in range(len(s["y"]))]


print("building per-client panels ...")
P = {}
for m in ("acl", "acl_asym", "acl_debias", "pid"):
    P[m] = panels_calibrated(m)
for s in ("cqr_local", "modular_local", "modular_fed"):
    try:
        P[s] = panels_sweep(s)
    except (KeyError, NameError):
        print(f"  [skip] {s} not in sweep")


def compare(A, B):
    stats, sig, nfav = [], 0, 0
    n = min(len(P[A]), len(P[B]))
    for k in range(n):
        yA, loA, mdA, upA = P[A][k]
        yB, loB, mdB, upB = P[B][k]
        L = min(len(yA), len(yB))
        LA = pinball(yA[:L], loA[:L], mdA[:L], upA[:L])
        LB = pinball(yA[:L], loB[:L], mdB[:L], upB[:L])
        s, p = dm(LA, LB)
        if np.isfinite(s):
            stats.append(s)
            if np.isfinite(p) and p < 0.05:
                sig += 1
                if s < 0:
                    nfav += 1
    stats = np.array(stats)
    if stats.size == 0:
        return np.nan, np.nan, np.nan
    return np.median(stats), 100 * nfav / len(stats), 100 * sig / len(stats)


NICE = {"acl_asym": "adaptive, signed",
        "acl": "adaptive, symmetric",
        "acl_debias": "adaptive + intercept",
        "pid": "conformal PID",
        "cqr_local": "CQR$_\\mathrm{local}$",
        "modular_local": "modular$_\\mathrm{local}$",
        "modular_fed": "modular$_\\mathrm{fed}$"}

PAIRS = [("acl_asym", "acl"),            # the paper's main claim
         ("acl_asym", "acl_debias"),     # signed score vs static intercept
         ("acl_asym", "cqr_local"),      # vs the sharpest static local
         ("acl", "pid")]                 # expected NOT significant

print(f"\nPer-client DM at level={LEVEL} "
      f"(negative favors the first-named scheme)\n")
print(f"{'comparison':44s} {'medianDM':>9s} {'%sig&fav':>9s} {'%sig':>6s}")
rows = []
for A, B in PAIRS:
    if A not in P or B not in P:
        continue
    md_, favp, sigp = compare(A, B)
    rows.append((A, B, md_, favp, sigp))
    print(f"{NICE[A] + ' vs ' + NICE[B]:44s} {md_:9.2f} {favp:8.0f}% {sigp:5.0f}%")

print("\n% ---- rows for tab:dm ----")
for A, B, md_, favp, sigp in rows:
    print(f"{NICE[A]} vs.\\ {NICE[B]:<34} & {favp:.0f}\\% & {sigp:.0f}\\% \\\\")

print("\nreading: the last pair is a NEGATIVE control. If the two controllers")
print("are genuinely comparable, %sig&fav should sit far from 100, and the")
print("paper can say so instead of claiming a win it did not earn.")

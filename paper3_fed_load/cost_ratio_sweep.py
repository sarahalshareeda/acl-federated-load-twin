# cost_ratio_sweep.py -- the empirical optimum at many cost ratios, so
# Proposition 1 can be shown verified rather than asserted in a caption.
#
#   exec(open(f"{PROJECT_DIR}/cost_ratio_sweep.py").read())
#
# Needs in session: OUT_DIR, np, and either `sweep` in memory or
# rollout_resid_L1.npy on disk. Runs in a couple of seconds: cost_curve() is
# only quantiles over a residual array.
#
# Writes cost_optima_sweep.csv, which make_figs.py plots as the right panel of
# fig_costopt.png.
#
# NOTE the construction, because it is the reason the empirical and closed-form
# optima agree: the dispatched bound is placed at the (1 - alpha/2) quantile of
# the SIGNED residual, which is what Proposition 1 requires. A symmetric
# +/- Q_{1-alpha}(|e|) band would not land on the critical fractile unless the
# residual distribution were symmetric about zero.

import os
import csv
import numpy as np

C_H = 0.12
RATIOS = np.unique(np.round(np.geomspace(2.0, 30.0, 14), 2))
ALPHAS = np.linspace(0.005, 0.80, 1600)   # fine grid: the claim is about
                                          # agreement, so resolution matters
HOURS = 24


def _residuals():
    g = globals()
    if "sweep" in g:
        lv = max(g["LEVELS"]) if "LEVELS" in g else 1.0
        s = g["sweep"][(lv, "integrated")]
        y = np.concatenate([np.asarray(a).ravel() for a in s["y"]])
        md = np.concatenate([np.asarray(a).ravel() for a in s["md"]])
        e = y - md
        return e[np.isfinite(e)], f"sweep(level={lv})"
    p = os.path.join(OUT_DIR, "rollout_resid_L1.npy")
    if os.path.exists(p):
        e = np.load(p)
        return e[np.isfinite(e)], "rollout_resid_L1.npy"
    raise FileNotFoundError("no sweep in memory and no rollout_resid_L1.npy")


def optimum(e, c_h, c_p, alphas=ALPHAS):
    J = np.empty_like(alphas)
    for i, a in enumerate(alphas):
        q = np.quantile(e, 1.0 - a / 2.0)          # signed upper quantile
        J[i] = HOURS * (c_h * np.mean(np.maximum(q - e, 0.0))
                        + c_p * np.mean(np.maximum(e - q, 0.0)))
    k = int(np.argmin(J))
    return float(alphas[k]), float(J[k])


e, src = _residuals()
print(f"residuals from {src}, n={e.size}")

rows = []
for r in RATIOS:
    a_emp, j_min = optimum(e, C_H, C_H * r)
    a_th = 2.0 / (r + 1.0)                          # 2 c_h / (c_p + c_h)
    rows.append(dict(cp_ch_ratio=float(r),
                     alpha_star_emp=a_emp,
                     alpha_star_theory=a_th,
                     coverage_emp=1.0 - a_emp,
                     coverage_theory=1.0 - a_th,
                     abs_err=abs(a_emp - a_th),
                     J_min=j_min))

path = os.path.join(OUT_DIR, "cost_optima_sweep.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r_ in rows:
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                    for k, v in r_.items()})
print("saved cost_optima_sweep.csv")

print(f"\n{'c_p/c_h':>8}{'alpha* emp':>12}{'alpha* theory':>15}{'|err|':>9}")
for r_ in rows:
    print(f"{r_['cp_ch_ratio']:>8.2f}{r_['alpha_star_emp']:>12.4f}"
          f"{r_['alpha_star_theory']:>15.4f}{r_['abs_err']:>9.4f}")

err = np.array([r_["abs_err"] for r_ in rows])
grid = float(ALPHAS[1] - ALPHAS[0])
print(f"\nmax |empirical - closed form| = {err.max():.4f}")
print(f"alpha grid resolution         = {grid:.4f}")
if err.max() <= 2 * grid:
    print("Agreement is at the resolution of the grid, which is the honest")
    print("way to state it in the caption. Claiming more than the grid can")
    print("resolve is a claim a referee can check in one line.")

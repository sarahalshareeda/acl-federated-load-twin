#!/usr/bin/env python3
"""
Cost-optimal operating coverage for the reserve band (Proposition 3).

Given deployment (recursive-rollout) residuals e_t = P_t - Phat_t^med, this computes
the expected daily reserve cost (newsvendor form)

    J(alpha) = sum_t [ c_h * (Uhat_t - P_t)_+  +  c_p * (P_t - Uhat_t)_+ ]

where Uhat_t(alpha) = Phat_t^med + Q_tau(e) is the reserved upper bound at the
tau-quantile of the residuals, tau = 1 - alpha/2. c_h charges over-provisioned (unused)
reserve, c_p charges shortfall. It returns the coverage grid, J, the empirical
minimizer alpha*, and the closed-form critical fractile
tau* = c_p / (c_p + c_h)  ->  alpha* = 2(1 - tau*).

USAGE (Colab, on your real residuals)
--------------------------------------
    import numpy as np, pandas as pd
    from cost_optimize import cost_curve, plot_cost_curves
    # e: 1-D array of rollout residuals pooled over test hours (and buildings)
    e = np.load("rollout_residuals.npy")          # or build from your model_out
    out = cost_curve(e, c_h=0.12, c_p=1.08)        # ratio 9 -> 80% band
    print("empirical alpha*  =", out["alpha_star_emp"])
    print("closed-form alpha* =", out["alpha_star_theory"])
    # paper figure across three ratios:
    plot_cost_curves(e, c_h=0.12, ratios=(4,9,19), out="fig_costopt.png")

Building e from per-building rollout outputs: for each test day and building, take
e = actual_load - median_forecast over the 24-step rollout at lambda=1, then
concatenate. Use the SAME rollout used for coverage so the tails match.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED = "#C00000"


def cost_curve(e, c_h=0.12, c_p=1.08, alphas=None, hours=24):
    """Expected daily reserve cost vs miscoverage alpha (two-sided band).

    e      : 1-D residual sample (actual - median forecast), any sign.
    c_h    : holding cost per unit reserve.
    c_p    : shortfall (unserved-energy) cost per unit, c_p > c_h.
    returns dict with alphas, coverage, J (per day), and both minimizers.
    """
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if alphas is None:
        alphas = np.linspace(0.02, 0.60, 120)
    J = np.empty_like(alphas)
    for i, a in enumerate(alphas):
        tau = 1.0 - a / 2.0                      # upper quantile the reserve covers
        q = np.quantile(e, tau)                  # reserve offset (upside), Uhat - Phat
        overage = np.mean(np.maximum(q - e, 0.0))   # unused reserve  (Uhat - P)_+
        shortfall = np.mean(np.maximum(e - q, 0.0))  # unmet load     (P - Uhat)_+
        J[i] = hours * (c_h * overage + c_p * shortfall)
    k = int(np.argmin(J))
    tau_star = c_p / (c_p + c_h)
    return {
        "alphas": alphas,
        "coverage": 1.0 - alphas,
        "J": J,
        "alpha_star_emp": float(alphas[k]),
        "J_min": float(J[k]),
        "tau_star": float(tau_star),
        "alpha_star_theory": float(2.0 * (1.0 - tau_star)),
    }


def plot_cost_curves(e, c_h=0.12, ratios=(4, 9, 19), out="fig_costopt.png", hours=24):
    """Plot J(alpha) vs coverage for several c_p/c_h ratios; mark the optima."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    cols = ["#1f77b4", RED, "#2ca02c"]
    for r, col in zip(ratios, cols):
        out_r = cost_curve(e, c_h=c_h, c_p=c_h * r, hours=hours)
        cov = out_r["coverage"]
        Jn = out_r["J"] / out_r["J_min"]          # normalize each curve to its own min
        ax.plot(cov, Jn, color=col, lw=1.8, label=fr"$c_p/c_h={r}$")
        k = int(np.argmin(out_r["J"]))
        ax.plot(cov[k], Jn[k], "o", color=col, ms=6, zorder=5)
    ax.axvline(0.80, color="0.4", ls=(0, (4, 3)), lw=1.0)
    ax.text(0.805, ax.get_ylim()[1] * 0.98, "80% band", fontsize=8,
            color="0.35", va="top", ha="left")
    ax.set_xlabel("operating coverage  $1-\\alpha$")
    ax.set_ylabel("reserve cost $J(\\alpha)$ (norm.)")
    ax.set_xlim(0.40, 0.98)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved", out)
    for r in ratios:
        o = cost_curve(e, c_h=c_h, c_p=c_h * r, hours=hours)
        print(f"  ratio {r:>2}:  emp alpha*={o['alpha_star_emp']:.3f}"
              f"  theory alpha*={o['alpha_star_theory']:.3f}"
              f"  (coverage {1-o['alpha_star_theory']:.2f})")


def save_cost_csv(e, c_h=0.12, ratios=(4, 9, 19), outdir=".", hours=24, alphas=None):
    """Write the cost analysis to two CSVs so you can plot it yourself.

    cost_curve.csv  : long form, one row per (ratio, alpha) point
        columns: cp_ch_ratio, c_h, c_p, alpha, coverage, J_per_day, J_norm, is_optimum
    cost_optima.csv : one row per ratio with the empirical and closed-form optimum
        columns: cp_ch_ratio, c_h, c_p, tau_star, alpha_star_theory,
                 coverage_star_theory, alpha_star_empirical, coverage_star_empirical,
                 J_min_per_day
    Returns (curve_path, optima_path).
    """
    curve_path = os.path.join(outdir, "cost_curve.csv")
    optima_path = os.path.join(outdir, "cost_optima.csv")
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cp_ch_ratio", "c_h", "c_p", "alpha", "coverage",
                    "J_per_day", "J_norm", "is_optimum"])
        for r in ratios:
            o = cost_curve(e, c_h=c_h, c_p=c_h * r, alphas=alphas, hours=hours)
            k = int(np.argmin(o["J"]))
            for i, a in enumerate(o["alphas"]):
                w.writerow([r, c_h, round(c_h * r, 6), f"{a:.4f}",
                            f"{o['coverage'][i]:.4f}", f"{o['J'][i]:.6f}",
                            f"{o['J'][i] / o['J_min']:.6f}", int(i == k)])
    with open(optima_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cp_ch_ratio", "c_h", "c_p", "tau_star", "alpha_star_theory",
                    "coverage_star_theory", "alpha_star_empirical",
                    "coverage_star_empirical", "J_min_per_day"])
        for r in ratios:
            o = cost_curve(e, c_h=c_h, c_p=c_h * r, alphas=alphas, hours=hours)
            w.writerow([r, c_h, round(c_h * r, 6), f"{o['tau_star']:.4f}",
                        f"{o['alpha_star_theory']:.4f}",
                        f"{1 - o['alpha_star_theory']:.4f}",
                        f"{o['alpha_star_emp']:.4f}",
                        f"{1 - o['alpha_star_emp']:.4f}",
                        f"{o['J_min']:.6f}"])
    print("saved", curve_path)
    print("saved", optima_path)
    return curve_path, optima_path


def residuals_from_sweep(sweep, level, scheme="integrated"):
    """Deployment residuals e_t = actual - median forecast, pooled over the fleet,
    read straight from the dose-response `sweep` dict at a given degradation `level`.
    No new rollouts: reuses what the sweep already computed."""
    s = sweep[(level, scheme)]
    y = np.concatenate([np.asarray(a).ravel() for a in s["y"]])
    md = np.concatenate([np.asarray(a).ravel() for a in s["md"]])
    e = y - md
    return e[np.isfinite(e)]


def _representative_residuals(n=200_000, seed=0):
    """Right-skewed load residuals (spikes cost more) for an illustrative figure
    when no real sweep is available."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 8, n)
    spike = rng.lognormal(mean=2.6, sigma=0.7, size=n) - np.exp(2.6 + 0.7**2 / 2)
    return base + spike


# ---- integration: runs when exec()'d inside the Colab notebook ----
# Set your economics here. c_h = holding cost per kWh of over-provisioned reserve,
# c_p = shortfall cost per kWh. Ratio c_p/c_h fixes the optimal band (Prop. 3):
#   ratio 9 -> 80%,  ratio 4 -> 60%,  ratio 19 -> 90%.
C_H, C_P = 0.12, 1.08          # <-- edit c_p to your shortfall cost

# where Section 7 persists residuals (survives a Colab disconnect)
RESID_PATH = "/content/drive/MyDrive/Colab Notebooks/paper3_fed_load/model_out/rollout_resid_L1.npy"

_g = globals()
_e, _outdir, _src = None, ".", "synthetic"
if "sweep" in _g and "OUT_DIR" in _g:                       # best: live sweep in memory
    _lv = max(_g["LEVELS"]) if "LEVELS" in _g else 1.0
    _e = residuals_from_sweep(_g["sweep"], _lv)
    _outdir, _src = _g["OUT_DIR"], f"sweep(level={_lv})"
elif os.path.exists(RESID_PATH):                            # fallback: residuals saved to Drive
    _e = np.load(RESID_PATH)
    _e = _e[np.isfinite(_e)]
    _outdir, _src = os.path.dirname(RESID_PATH), "saved rollout_resid_L1.npy"

if _e is not None:
    # data tables to plot from yourself
    save_cost_csv(_e, c_h=C_H, ratios=(4, 9, 19), outdir=_outdir)
    # convenience PNG (optional; the CSVs above are the source of truth)
    plot_cost_curves(_e, c_h=C_H, ratios=(4, 9, 19),
                     out=os.path.join(_outdir, "fig_costopt.png"))
    _r = cost_curve(_e, c_h=C_H, c_p=C_P)
    print(f"[cost_optimize] source={_src}  residuals n={_e.size}  c_p/c_h={C_P/C_H:.0f}")
    print(f"[cost_optimize] cost-optimal alpha* (empirical) = {_r['alpha_star_emp']:.3f}"
          f"  -> operating coverage {1-_r['alpha_star_emp']:.2f}")
    print(f"[cost_optimize] closed-form (Prop. 3) alpha*     = {_r['alpha_star_theory']:.3f}")
else:                                                       # nothing available: illustrative curve
    print("[cost_optimize] no sweep in memory and no saved residuals -> using REPRESENTATIVE curve.")
    print("[cost_optimize] run Section 7 (the sweep) first, or restore rollout_resid_L1.npy.")
    _e = _representative_residuals()
    save_cost_csv(_e, c_h=C_H, ratios=(4, 9, 19), outdir=".")
    plot_cost_curves(_e, c_h=C_H, ratios=(4, 9, 19), out="fig_costopt.png")

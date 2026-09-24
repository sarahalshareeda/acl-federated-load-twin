# cost_table_cp.py -- prices both terms of the daily loss (9) so tab:cost stops
# looking as though `integrated' were a bargain, and adds fixed@0: the only
# static calibrator a deployed system could actually build.
#
# Needs in session: clients, ALPHA, OUT_DIR, rollout(), np, pd.
# Uses sweep (or sweep_L1.pkl / sweep.pkl) for the static schemes. For the
# adaptive schemes it reuses cached rollout dicts if present, else rolls out
# only the two splits it needs (calib@0 for the clean seed, test@1).
#
#   exec(open(f"{PROJECT_DIR}/cost_table_cp.py").read())
#
# Writes cost_table_cp.csv, which make_figs.py reads.

import os, pickle, time
import numpy as np
import pandas as pd

C_H, C_P = 0.12, 1.08     # $/kWh; ratio 9 -> alpha* = 0.20
HOURS    = 24
LV       = 1.0            # degradation level the table reports
GAMMA, WMAX = 0.02, 360   # the paper's ACL constants, not adaptive_conformal.py's


# ---- 0) sweep, from memory or disk ---------------------------------------
if "sweep" not in globals():
    for _f in ("sweep_L1.pkl", "sweep.pkl"):
        _p = os.path.join(OUT_DIR, _f)
        if os.path.exists(_p) and os.path.getsize(_p) > 0:
            with open(_p, "rb") as f:
                sweep = pickle.load(f)
            print(f"loaded {_f}, {len(sweep)} keys")
            break
    else:
        raise FileNotFoundError("no sweep in memory and no usable sweep pickle")


def from_sweep(scheme, lv=LV):
    s = sweep[(lv, scheme)]
    cat = lambda k: np.concatenate([np.asarray(a).ravel() for a in s[k]])
    return cat("y"), cat("lo"), cat("up")


# ---- 1) rollouts the adaptive and fixed@0 panels need ---------------------
def ensure_point_rollouts():
    """calib_scores[0.0] and test_pred[LV], for adaptive_modular and fixed@0"""
    have = ("calib_scores" in globals() and "test_pred" in globals()
            and 0.0 in calib_scores and LV in test_pred)
    if have:
        return calib_scores, test_pred
    cs, tp = {0.0: {}}, {LV: {}}
    t0 = time.time()
    for name in clients:
        yc, ptc = rollout(name, "pt", "reconstructed",
                          split="calib", seed_from="val", level=0.0)
        s = np.abs(yc - ptc); cs[0.0][name] = s[np.isfinite(s)]
        tp[LV][name] = rollout(name, "pt", "reconstructed",
                               split="test", level=LV)
    print(f"  point rollouts ({time.time()-t0:.0f}s)")
    return cs, tp


def ensure_int_rollouts():
    """cqr_calib[0.0] and test_int[LV], for adaptive_CQR"""
    have = ("cqr_calib" in globals() and "test_int" in globals()
            and 0.0 in cqr_calib and LV in test_int)
    if have:
        return cqr_calib, test_int
    cc, ti = {0.0: {}}, {LV: {}}
    t0 = time.time()
    for name in clients:
        yc, lo_c, md_c, up_c = rollout(name, "int", "reconstructed",
                                       split="calib", seed_from="val", level=0.0)
        E = np.maximum(lo_c - yc, yc - up_c)
        cc[0.0][name] = E[np.isfinite(E)]
        ti[LV][name] = rollout(name, "int", "reconstructed",
                               split="test", level=LV)
    print(f"  quantile rollouts ({time.time()-t0:.0f}s)")
    return cc, ti


def acl(y, base_lo, base_up, seed_scores, score_fn):
    """one building's ACL rollout at the paper's gamma and W"""
    W = list(seed_scores); a = ALPHA
    N = len(y); LO = np.empty(N); UP = np.empty(N)
    for b0 in range(0, N, 24):
        b1 = min(b0 + 24, N)
        r = np.quantile(np.asarray(W), min(0.999, max(0.001, 1 - a)))
        LO[b0:b1] = base_lo[b0:b1] - r
        UP[b0:b1] = base_up[b0:b1] + r
        day = []
        for i in range(b0, b1):
            if np.isfinite(y[i]):
                err = 0 if (LO[i] <= y[i] <= UP[i]) else 1
                a = min(0.999, max(0.001, a + GAMMA * (ALPHA - err)))
                day.append(score_fn(i))
        W.extend(day); W = W[-WMAX:]
    return LO, UP


def adaptive_modular_panel():
    cs, tp = ensure_point_rollouts()
    Y, LO, UP = [], [], []
    for name in clients:
        y, pt = tp[LV][name]
        lo, up = acl(y, pt, pt, cs[0.0][name],
                     lambda i, y=y, pt=pt: abs(y[i] - pt[i]))
        Y.append(y); LO.append(lo); UP.append(up)
    return map(np.concatenate, (Y, LO, UP))


def adaptive_cqr_panel():
    cc, ti = ensure_int_rollouts()
    Y, LO, UP = [], [], []
    for name in clients:
        y, lo0, md0, up0 = ti[LV][name]
        lo, up = acl(y, lo0, up0, cc[0.0][name],
                     lambda i, y=y, lo0=lo0, up0=up0:
                         max(lo0[i] - y[i], y[i] - up0[i]))
        Y.append(y); LO.append(lo); UP.append(up)
    return map(np.concatenate, (Y, LO, UP))


def fixed0_panel():
    """the only static calibrator a deployed system could build: the margin is
    frozen on clean scores and then meets degraded deployment unchanged"""
    cs, tp = ensure_point_rollouts()
    Y, LO, UP = [], [], []
    for name in clients:
        y, pt = tp[LV][name]
        s = cs[0.0][name]
        r = np.quantile(s, min(0.999, (1 - ALPHA) * (1 + 1 / max(len(s), 1))))
        Y.append(y); LO.append(pt - r); UP.append(pt + r)
    return map(np.concatenate, (Y, LO, UP))


panels = {}
for name in ["integrated", "modular_fed", "modular_local", "cqr_local"]:
    try:
        panels[name] = from_sweep(name)
    except KeyError:
        print(f"  [skip] {name!r} not in sweep")

print("building fixed@0 and the adaptive panels ...")
panels["fixed@0"] = tuple(fixed0_panel())
panels["adaptive_modular"] = tuple(adaptive_modular_panel())
panels["adaptive_cqr"] = tuple(adaptive_cqr_panel())


# ---- 2) price each panel -------------------------------------------------
def price(y, lo, up):
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(up)
    y, lo, up = y[m], lo[m], up[m]
    picp   = float(np.mean((y >= lo) & (y <= up)))     # two-sided, as reported
    exceed = float(np.mean(y > up))                    # what c_p actually charges
    width  = float(np.mean(up - lo))                   # MPIW80, kW
    over   = float(np.mean(np.maximum(up - y, 0.0)))   # (Uhat - P)^+, kW
    short  = float(np.mean(np.maximum(y - up, 0.0)))   # (P - Uhat)^+, kW
    hold_A = C_H * HOURS * width      # convention A: 24 * MPIW
    hold_B = C_H * HOURS * over       # convention B: matches eq (10)
    pen    = C_P * HOURS * short
    return dict(PICP80=picp, exceed=exceed, MPIW80=width,
                width_kWh=HOURS*width, over_kWh=HOURS*over,
                short_kWh=HOURS*short, hold_A=hold_A, hold_B=hold_B,
                penalty=pen, J_A=hold_A+pen, J_B=hold_B+pen)


df = pd.DataFrame([dict(scheme=n, **price(*p)) for n, p in panels.items()]
                  ).set_index("scheme")

s = sweep[(LV, "modular_local")]
yy = np.concatenate([np.asarray(a).ravel() for a in s["y"]])
md = np.concatenate([np.asarray(a).ravel() for a in s["md"]])
m = np.isfinite(yy) & np.isfinite(md)
SCHED = C_H * HOURS * float(np.mean(np.abs(yy[m] - md[m])))

df["total_A"] = SCHED + df.J_A
df["total_B"] = SCHED + df.J_B

ORDER = [o for o in ["integrated", "fixed@0", "modular_fed", "modular_local",
                     "cqr_local", "adaptive_modular", "adaptive_cqr"]
         if o in df.index]
df = df.loc[ORDER]

print(f"\nscheduling ${SCHED:.0f}/day/building | c_h={C_H} c_p={C_P} | lambda={LV}\n")
print(df[["PICP80", "exceed", "MPIW80", "over_kWh", "short_kWh",
          "hold_B", "penalty", "total_B"]]
      .round(dict(PICP80=3, exceed=3, MPIW80=1, over_kWh=0,
                  short_kWh=0, hold_B=0, penalty=0, total_B=0)).to_string())
df.round(4).to_csv(os.path.join(OUT_DIR, "cost_table_cp.csv"))
print("\nsaved cost_table_cp.csv")


# ---- 3) the comparisons the paper reports --------------------------------
def saving(base, prop="adaptive_cqr", col="total_B"):
    if base in df.index and prop in df.index:
        b, p_ = df.loc[base, col], df.loc[prop, col]
        return 100.0 * (b - p_) / b
    return float("nan")


print("\nadaptive_cqr vs ... (total operating cost, scheduling included)")
for base, note in (("fixed@0", "the only deployable static calibrator"),
                   ("modular_fed", "fleet-pooled, oracle-calibrated"),
                   ("cqr_local", "sharpest static local, oracle-calibrated")):
    if base in df.index:
        print(f"   vs {base:<18} {saving(base):>5.1f}% lower   ({note})")

print("\ncalibration-attributable only (scheduling excluded)")
for base in ("fixed@0", "modular_fed", "cqr_local"):
    if base in df.index:
        b, p_ = df.loc[base, "J_B"], df.loc["adaptive_cqr", "J_B"]
        print(f"   vs {base:<18} {100*(b-p_)/b:>5.1f}% lower")


# ---- 4) LaTeX body for tab:cost -----------------------------------------
PRETTY = {"integrated": "integrated", "fixed@0": "fixed@0",
          "modular_fed": r"modular$_\mathrm{fed}$",
          "modular_local": r"modular$_\mathrm{local}$",
          "cqr_local": r"CQR$_\mathrm{local}$",
          "adaptive_modular": r"adaptive$_\mathrm{modular}$",
          "adaptive_cqr": r"adaptive$_\mathrm{CQR}$"}
print("\n% ---- paste into tab:cost (components sum to total) ----")
for nm in ORDER:
    r = df.loc[nm]
    hold = round(r.hold_B); pen = round(r.penalty)
    sched = round(r.total_B - r.hold_B - r.penalty)
    print(f"{PRETTY[nm]:<32} & {r.PICP80:.2f} & {r.exceed:.2f} & "
          f"{r.over_kWh:.0f} & {hold:.0f} & {pen:.0f} & "
          f"{hold+pen+sched:.0f} \\\\")

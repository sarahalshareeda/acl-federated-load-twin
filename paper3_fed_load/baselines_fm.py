# baselines_fm.py -- symmetric vs signed ACL on the Chronos backbone, so
# Section V-G stops reporting only one of the two scores Section II-D defines.
#
#   exec(open(f"{PROJECT_DIR}/baselines_fm.py").read())
#
# Needs in session: clients, ALPHA, OUT_DIR, imetrics(), np, pd, and
# fm_ckpt.pkl on disk. No rollouts: everything comes from the cache, so this
# runs in seconds.
#
# The symmetric rows reproduce results_fm_adaptive_cqr.csv exactly. If they do
# not, the shared code below has changed the semantics of your original
# fm_adaptive_cqr and nothing else here can be trusted.

import os, pickle
import numpy as np
import pandas as pd

GAMMA, WMAX = 0.02, 360
ASTAR = ALPHA
FM_LEVELS = [0.0, 0.5, 1.0]
C_H, C_P, HOURS = 0.12, 1.08, 24


def _key(lv, name, split):
    return (round(float(lv), 3), name, split)


with open(os.path.join(OUT_DIR, "fm_ckpt.pkl"), "rb") as f:
    ckpt = pickle.load(f)
print(f"loaded {len(ckpt)} cached FM rollouts")

fm_test = {lv: {n: ckpt[_key(lv, n, "test")] for n in clients} for lv in FM_LEVELS}
fm_calib = {lv: {n: ckpt[_key(lv, n, "calib")] for n in clients} for lv in FM_LEVELS}

# clean seeds. The symmetric score is the max of the two signed ones, which is
# how the two formulations relate in eq. (16) of the paper.
seed_up, seed_lo, seed_sym = {}, {}, {}
for name in clients:
    y, lo, md, up = fm_calib[0.0][name]
    s_up = y - up
    s_lo = lo - y
    m = np.isfinite(s_up) & np.isfinite(s_lo)
    seed_up[name], seed_lo[name] = s_up[m], s_lo[m]
    seed_sym[name] = np.maximum(s_lo[m], s_up[m])


def _q(w, p):
    return float(np.quantile(np.asarray(w), min(0.999, max(0.001, p))))


def acl_fm(name, lv, signed):
    """One building, one level. Day-blocked radius, hourly state update."""
    y, lo, md, up = fm_test[lv][name]
    N = len(y)
    LO = np.empty(N); UP = np.empty(N)
    a = ASTAR
    if signed:
        Wlo, Wup = list(seed_lo[name]), list(seed_up[name])
    else:
        W = list(seed_sym[name])

    for b0 in range(0, N, HOURS):
        b1 = min(b0 + HOURS, N)
        if signed:
            r_lo, r_up = _q(Wlo, 1 - a / 2), _q(Wup, 1 - a / 2)
        else:
            r_lo = r_up = _q(W, 1 - a)
        LO[b0:b1] = lo[b0:b1] - r_lo
        UP[b0:b1] = up[b0:b1] + r_up

        d_lo, d_up, d_sym = [], [], []
        for i in range(b0, b1):
            if not np.isfinite(y[i]):
                continue
            err = 0 if (LO[i] <= y[i] <= UP[i]) else 1
            a = min(0.999, max(0.001, a + GAMMA * (ASTAR - err)))
            if signed:
                d_lo.append(lo[i] - y[i]); d_up.append(y[i] - up[i])
            else:
                d_sym.append(max(lo[i] - y[i], y[i] - up[i]))
        if signed:
            Wlo.extend(d_lo); Wlo = Wlo[-WMAX:]
            Wup.extend(d_up); Wup = Wup[-WMAX:]
        else:
            W.extend(d_sym); W = W[-WMAX:]
    return y, LO, md, UP


def cost(y, lo, up):
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(up)
    y, lo, up = y[m], lo[m], up[m]
    over = float(np.mean(np.maximum(up - y, 0.0)))
    short = float(np.mean(np.maximum(y - up, 0.0)))
    return dict(EXC=float(np.mean(y > up)),
                hold_B=C_H * HOURS * over,
                penalty=C_P * HOURS * short,
                J_B=C_H * HOURS * over + C_P * HOURS * short)


rows = []
for signed in (False, True):
    tag = "fm_adaptive_cqr" + ("_signed" if signed else "")
    for lv in FM_LEVELS:
        Y, L, M, U = [], [], [], []
        for name in clients:
            y, l, m_, u = acl_fm(name, lv, signed)
            Y.append(y); L.append(l); M.append(m_); U.append(u)
        Y, L, M, U = (np.concatenate(v) for v in (Y, L, M, U))
        r = imetrics(Y, L, M, U)
        r.update(cost(Y, L, U))
        r.update(level=lv, scheme=tag)
        rows.append(r)

df = pd.DataFrame(rows)[["level", "scheme", "MAE", "WAPE", "PICP80", "MPIW80",
                         "MIS80", "QS", "EXC", "hold_B", "penalty", "J_B"]]
df.to_csv(os.path.join(OUT_DIR, "results_fm_signed.csv"), index=False)
print("\nsaved results_fm_signed.csv\n")
print(df.round(2).to_string(index=False))

print("\nwidth and cost at each level (symmetric -> signed)")
for lv in FM_LEVELS:
    s = df[(df.scheme == "fm_adaptive_cqr") & (df.level == lv)].iloc[0]
    g = df[(df.scheme == "fm_adaptive_cqr_signed") & (df.level == lv)].iloc[0]
    print(f"  lambda={lv:.1f}  MPIW {s.MPIW80:6.1f} -> {g.MPIW80:6.1f} kW "
          f"({s.MPIW80/g.MPIW80:.2f}x)   J ${s.J_B:5.0f} -> ${g.J_B:5.0f}   "
          f"PICP {s.PICP80:.2f} -> {g.PICP80:.2f}")

# ---- gate: the symmetric rows must reproduce the published FM numbers ------
print("\nverification against results_fm_adaptive_cqr.csv")
try:
    ref = pd.read_csv(os.path.join(OUT_DIR, "results_fm_adaptive_cqr.csv"))
    ok = True
    for lv in FM_LEVELS:
        a = df[(df.scheme == "fm_adaptive_cqr") & (df.level == lv)].iloc[0]
        b = ref[ref.level == lv]
        if b.empty:
            continue
        b = b.iloc[0]
        dp = abs(a.PICP80 - b.PICP80)
        dw = abs(a.MPIW80 - b.MPIW80) / max(b.MPIW80, 1e-9)
        good = dp <= 0.01 and dw <= 0.02
        ok &= good
        print(f"   lambda={lv:.1f}  PICP {a.PICP80:.2f} vs {b.PICP80:.2f}, "
              f"MPIW {a.MPIW80:.1f} vs {b.MPIW80:.1f}   "
              f"[{'ok' if good else 'MISMATCH'}]")
    if not ok:
        print("\n   A MISMATCH means the shared acl_fm() does not behave like your")
        print("   original adaptive_cqr. Fix that before using the signed rows.")
except FileNotFoundError:
    print("   results_fm_adaptive_cqr.csv not found, skipping the gate")

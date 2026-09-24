# baselines.py -- every competing and ablated calibrator on one interface, so
# the paper can answer "where is the comparison against the works in Table I".
#
#   exec(open(f"{PROJECT_DIR}/baselines.py").read())
#
# Needs in session: clients, ALPHA, OUT_DIR, rollout(), np, pd.
# Reuses calib_scores / test_pred / cqr_calib / test_int if they are already
# built (rebuild_min.py or cost_table_cp.py leaves them behind); otherwise it
# rolls out only the splits it needs, once, and caches them.
#
# METHODS
#   split      static quantile frozen on clean scores, never updates.
#              This is fixed@0 and also plain split conformal [Lei et al.].
#   windowed   rolling-window quantile, alpha held at the target. Ablation:
#              the score window WITHOUT the alpha controller.
#   aci        alpha controller on a FROZEN calibration set.
#              READ THIS BEFORE CITING IT AS "ACI". Gibbs & Candes compute the
#              quantile over a RECENT lookback, not a frozen calibration set,
#              so this is an ABLATION of the paper's own layer, not faithful
#              ACI. Faithful ACI is, to within the window length, the same
#              algorithm as `acl' below. Presenting this row as ACI would be a
#              strawman and a referee who knows that paper will say so.
#   acl        alpha controller + rolling score window. The paper's layer, and
#              honestly an ACI-style controller. Label it that way. The
#              paper's contribution is the cost-optimal alpha* that this layer
#              is pointed at, not the controller itself.
#   pid        conformal PID control [Angelopoulos, Candes & Tibshirani 2023],
#              proportional + bounded integral. The derivative term in that
#              paper is a learned "scorecaster" over future scores; it is
#              omitted here and the omission is stated in the caption, since
#              training an auxiliary score model is a separate contribution.
#   acl_asym   ACL with SIGNED scores and independent upper/lower radii. The
#              daily loss charges only upper breaches, so tying the upper
#              radius to a symmetric absolute score spends width that the
#              cost never pays for.
#   acl_tod    ACL with a separate controller and window per time-of-day
#              group. The bandit already treats peak and off-peak separately;
#              this makes the calibration layer consistent with it.
#   acl_debias THE REFEREE'S QUESTION. Each building's mean clean-calibration
#              residual is added back to its forecast, and then the ORDINARY
#              symmetric layer runs on the corrected forecast. If this recovers
#              most of what acl_asym wins, the finding is "remove the offset",
#              which any practitioner would do with an intercept, and the
#              signed-score result is not a contribution about calibration.
#              If acl_asym still wins, the gap is the part a mean correction
#              cannot reach: the offset drifts, and signed scores track it
#              while a fixed intercept does not.
#
# FAIRNESS
#   Every method sees the identical backbone, split, seed scores and
#   degradation stream, and dispatches ONE interval per day held for 24 h.
#   gamma (for aci/acl) and eta (for pid) are swept over the same grid size
#   and each method is reported at its own best setting, so no method is
#   handicapped by a hyperparameter chosen for another.

import os, time, itertools
import numpy as np
import pandas as pd

C_H, C_P = 0.12, 1.08
HOURS = 24
ALPHA_STAR = ALPHA if "ALPHA" in globals() else 0.20

# Override either of these in a cell BEFORE the exec line, e.g.
#   BASELINE_LEVELS = [1.0]        # fast pass, ~8 min instead of ~30
#   BASELINE_PEAK   = set(range(9, 22))
LEVELS = globals().get("BASELINE_LEVELS", [0.0, 0.25, 0.5, 0.75, 1.0])
PEAK = globals().get("BASELINE_PEAK", set(range(8, 21)))

WMAX = 360                              # the paper's score-window length
# If a method's best setting lands on the EDGE of its grid, the grid was too
# narrow and the comparison is not fair to that method. Override and re-run.
GAMMA_GRID = globals().get("BASELINE_GAMMA_GRID",
                           [0.005, 0.01, 0.02, 0.05, 0.10])
ETA_FRAC_GRID = globals().get("BASELINE_ETA_GRID",
                              [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
K_I = 0.30          # integral gain, as a multiple of eta
C_SAT = 12.0        # hours of one-sided error before integral action saturates

BASE_GAMMA = 0.02   # what the paper currently reports


# ======================================================================
# 0) rollouts, cached
# ======================================================================
def _ensure(levels):
    """calib_scores[0.0], cqr_calib[0.0], test_pred[lv], test_int[lv]"""
    g = globals()
    cs = g.setdefault("calib_scores", {})
    cc = g.setdefault("cqr_calib", {})
    tp = g.setdefault("test_pred", {})
    ti = g.setdefault("test_int", {})

    # A level counts as cached only when EVERY current client is present.
    # Testing the level key alone lets a stale or half-built dict through, and
    # the gap then surfaces much later as a KeyError on one building name,
    # after the rollouts have already been paid for.
    def complete(d, lv):
        return lv in d and all(n in d[lv] for n in clients)

    # signed clean residuals, needed for the de-biased baseline. Kept separate
    # because calib_scores stores |y - pt| and the sign cannot be recovered.
    sg = g.setdefault("calib_signed", {})
    if not complete(sg, 0.0):
        t0 = time.time(); sg[0.0] = {}
        for nm in clients:
            yc, ptc = rollout(nm, "pt", "reconstructed",
                              split="calib", seed_from="val", level=0.0)
            e = np.asarray(yc) - np.asarray(ptc)
            sg[0.0][nm] = e[np.isfinite(e)]
        print(f"  signed clean residuals ({time.time()-t0:.0f}s)")

    if not (complete(cs, 0.0) and complete(cc, 0.0)):
        t0 = time.time(); cs[0.0], cc[0.0] = {}, {}  # noqa: E702
        for nm in clients:
            yc, ptc = rollout(nm, "pt", "reconstructed",
                              split="calib", seed_from="val", level=0.0)
            s = np.abs(yc - ptc); cs[0.0][nm] = s[np.isfinite(s)]
            yq, lo, md, up = rollout(nm, "int", "reconstructed",
                                     split="calib", seed_from="val", level=0.0)
            E = np.maximum(lo - yq, yq - up); cc[0.0][nm] = E[np.isfinite(E)]
        print(f"  clean calib rollouts ({time.time()-t0:.0f}s)")

    for lv in levels:
        if complete(tp, lv) and complete(ti, lv):
            continue
        t0 = time.time(); tp[lv], ti[lv] = {}, {}
        for nm in clients:
            tp[lv][nm] = rollout(nm, "pt", "reconstructed",
                                 split="test", level=lv)
            ti[lv][nm] = rollout(nm, "int", "reconstructed",
                                 split="test", level=lv)
        print(f"  test rollouts, level {lv:.2f} ({time.time()-t0:.0f}s)")
    return cs, cc, tp, ti


# ======================================================================
# 1) the calibrators, one signature
# ======================================================================
def _q(w, p):
    return float(np.quantile(np.asarray(w), min(0.999, max(0.001, p))))


def calibrate(y, base_lo, base_up, seed, method,
              alpha=None, gamma=BASE_GAMMA, eta_frac=0.10, wmax=WMAX):
    """One building. Returns (LO, UP), both length len(y).

    The interval is recomputed once per 24 h block and held, matching
    day-ahead dispatch. State updates hourly on realized load.
    """
    a0 = ALPHA_STAR if alpha is None else alpha
    N = len(y)
    LO = np.empty(N); UP = np.empty(N)
    seed = np.asarray(seed, dtype=float)
    seed = seed[np.isfinite(seed)]

    asym = method == "acl_asym"
    tod = method == "acl_tod"
    groups = ("peak", "off") if tod else ("all",)

    def grp(i):
        return ("peak" if (i % 24) in PEAK else "off") if tod else "all"

    # per-group state
    a = {g: a0 for g in groups}
    W = {g: list(seed) for g in groups}
    if asym:
        # signed scores: seed both tails from the symmetric clean scores, which
        # is the only information a deployed system has before it sees a miss
        Wlo = list(seed); Wup = list(seed)

    # PID state, in score units
    eta = eta_frac * float(np.quantile(seed, 0.95) - np.quantile(seed, 0.50) + 1e-9)
    q_pid = _q(seed, 1 - a0)
    integ = 0.0
    step = 0

    for b0 in range(0, N, HOURS):
        b1 = min(b0 + HOURS, N)

        # ---- form the radius for this day ----
        if method == "split":
            r = _q(seed, (1 - a0) * (1 + 1 / max(len(seed), 1)))
            LO[b0:b1] = base_lo[b0:b1] - r; UP[b0:b1] = base_up[b0:b1] + r
        elif method == "pid":
            r = max(q_pid, 0.0)
            LO[b0:b1] = base_lo[b0:b1] - r; UP[b0:b1] = base_up[b0:b1] + r
        elif asym:
            r_up = _q(Wup, 1 - a["all"] / 2)
            r_lo = _q(Wlo, 1 - a["all"] / 2)
            LO[b0:b1] = base_lo[b0:b1] - r_lo; UP[b0:b1] = base_up[b0:b1] + r_up
        else:
            # ONE radius per group per day, formed from the state at the start
            # of the block and then held. Recomputing it hourly as alpha drifts
            # would let the dispatched interval move during the day, which the
            # day-ahead commitment does not allow, and would not reproduce the
            # paper's published widths.
            r_g = {g: _q(seed if method == "aci" else W[g], 1 - a[g])
                   for g in groups}
            for i in range(b0, b1):
                r = r_g[grp(i)]
                LO[i] = base_lo[i] - r; UP[i] = base_up[i] + r

        # ---- observe the day, update state hourly ----
        fresh = {g: [] for g in groups}
        fresh_lo, fresh_up = [], []
        for i in range(b0, b1):
            if not np.isfinite(y[i]):
                continue
            step += 1
            covered = LO[i] <= y[i] <= UP[i]
            err = 0.0 if covered else 1.0
            g = grp(i)

            if method in ("aci", "acl", "acl_asym", "acl_tod"):
                a[g] = min(0.999, max(0.001, a[g] + gamma * (a0 - err)))
            elif method == "pid":
                integ += (err - a0)
                q_pid = q_pid + eta * (err - a0) \
                    + K_I * eta * np.tanh(integ / C_SAT)

            if method in ("windowed", "acl", "acl_tod"):
                fresh[g].append(max(base_lo[i] - y[i], y[i] - base_up[i]))
            elif asym:
                fresh_lo.append(base_lo[i] - y[i])
                fresh_up.append(y[i] - base_up[i])

        for g in groups:
            if fresh[g]:
                W[g].extend(fresh[g]); W[g] = W[g][-wmax:]
        if asym and fresh_up:
            Wlo.extend(fresh_lo); Wlo = Wlo[-wmax:]
            Wup.extend(fresh_up); Wup = Wup[-wmax:]

    return LO, UP


# ======================================================================
# 2) pricing
# ======================================================================
def _pinball(y, q, tau):
    u = y - q
    return float(np.mean(np.maximum(tau * u, (tau - 1.0) * u)))


def price(y, lo, up, md=None):
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(up)
    if md is not None:
        m &= np.isfinite(md)
    y, lo, up = y[m], lo[m], up[m]
    over = float(np.mean(np.maximum(up - y, 0.0)))
    short = float(np.mean(np.maximum(y - up, 0.0)))

    # QS over the dispatched triple, matching eq. (21) of the paper. The
    # nominal levels are alpha*/2 and 1-alpha*/2, not 0.1 and 0.9, whenever
    # alpha* differs from 0.2; both coincide at the reported setting.
    lo_tau, up_tau = ALPHA_STAR / 2.0, 1.0 - ALPHA_STAR / 2.0
    qs = [_pinball(y, lo, lo_tau), _pinball(y, up, up_tau)]
    if md is not None:
        qs.append(_pinball(y, md[m], 0.5))
    QS = float(np.mean(qs))

    # MIS, eq. (22)
    MIS = float(np.mean((up - lo)
                        + (2.0 / ALPHA_STAR) * np.maximum(lo - y, 0.0)
                        + (2.0 / ALPHA_STAR) * np.maximum(y - up, 0.0)))

    return dict(PICP80=float(np.mean((y >= lo) & (y <= up))),
                EXC=float(np.mean(y > up)),
                MPIW80=float(np.mean(up - lo)),
                QS=QS, MIS80=MIS,
                hold_B=C_H * HOURS * over,
                penalty=C_P * HOURS * short,
                J_B=C_H * HOURS * over + C_P * HOURS * short)


def run(method, level, base="pt", alpha=None, gamma=BASE_GAMMA,
        eta_frac=0.10, cache={}):
    """Fleet-wide evaluation of one method at one degradation level."""
    cs, cc, tp, ti = cache["r"]
    Y, LO, UP, MD = [], [], [], []
    for nm in clients:
        if base == "pt":
            y, pt = tp[level][nm]; blo = bup = np.asarray(pt, dtype=float)
            seed = cs[0.0][nm]; mid = np.asarray(pt, dtype=float)
            if method == "acl_debias":
                # shift the forecast by this building's mean clean residual,
                # then calibrate symmetrically around the corrected forecast
                e = calib_signed[0.0][nm]
                b = float(np.mean(e))
                blo = bup = blo + b
                mid = mid + b
                seed = np.abs(e - b)
        else:
            y, lo0, md0, up0 = ti[level][nm]
            blo, bup = lo0, up0; seed = cc[0.0][nm]
            mid = np.asarray(md0, dtype=float)
        run_method = "acl" if method == "acl_debias" else method
        l, u = calibrate(y, blo, bup, seed, run_method,
                         alpha=alpha, gamma=gamma, eta_frac=eta_frac)
        Y.append(y); LO.append(l); UP.append(u); MD.append(mid)
    Y, LO, UP, MD = (np.concatenate(v) for v in (Y, LO, UP, MD))
    return price(Y, LO, UP, MD)


# ======================================================================
# 3) hyperparameter sweep -- each method at its own best setting
# ======================================================================
def tune(method, base="pt", cache=None):
    """Pick gamma (or eta) at lambda=1 by lowest operating cost among the
    settings that hold coverage within 2 points of nominal. Selecting on cost
    rather than width is what the paper's objective actually asks for."""
    if method in ("split", "windowed"):
        return {}
    grid = ([{"gamma": g} for g in GAMMA_GRID] if method != "pid"
            else [{"eta_frac": e} for e in ETA_FRAC_GRID])
    rows = []
    for kw in grid:
        m = run(method, 1.0, base=base, cache=cache, **kw)
        rows.append((kw, m))
        tag = list(kw.values())[0]
        print(f"     {method:<10} {list(kw)[0]}={tag:<6} "
              f"PICP {m['PICP80']:.3f}  MPIW {m['MPIW80']:6.1f}  "
              f"J ${m['J_B']:.0f}")
    ok = [(k, m) for k, m in rows if abs(m["PICP80"] - (1 - ALPHA_STAR)) <= 0.02]
    pool = ok or rows
    best = min(pool, key=lambda km: km[1]["J_B"])
    print(f"     -> {method}: {best[0]}")
    return best[0]


# ======================================================================
# 4) drive everything
# ======================================================================
cache = {"r": _ensure(LEVELS)}

MLP_METHODS = ["split", "windowed", "aci", "acl", "pid",
               "acl_debias", "acl_asym", "acl_tod"]
CQR_METHODS = ["acl", "pid", "acl_asym"]

print("\ntuning at lambda=1 (each method at its own best setting)")
best_pt = {m: tune(m, "pt", cache) for m in MLP_METHODS}
best_int = {m: tune(m, "int", cache) for m in CQR_METHODS}

rows = []
print("\nsweeping degradation levels")
for lv in LEVELS:
    for m in MLP_METHODS:
        r = run(m, lv, base="pt", cache=cache, **best_pt[m])
        rows.append(dict(scheme=m, base="MLP-point", level=lv, **r,
                         **best_pt[m]))
    for m in CQR_METHODS:
        r = run(m, lv, base="int", cache=cache, **best_int[m])
        rows.append(dict(scheme=m + "_cqr", base="MLP-quantile", level=lv, **r,
                         **best_int[m]))
    print(f"  level {lv:.2f} done")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "results_baselines.csv"), index=False)
print("\nsaved results_baselines.csv")

CITE = {"split": "split conformal [Lei et al.]",
        "windowed": "window only, alpha fixed (ablation)",
        "aci": "alpha only, scores frozen (ablation)",
        "acl": "ACL = ACI-style [Gibbs&Candes] + window",
        "pid": "conformal PID [Angelopoulos 2023]",
        "acl_debias": "per-building intercept + symmetric ACL",
        "acl_asym": "ACL, signed scores (new)",
        "acl_tod": "ACL, time-of-day (new)"}

print(f"\nlambda = 1, point backbone\n{'method':<34}{'PICP':>6}{'EXC':>6}"
      f"{'MPIW':>8}{'hold':>7}{'imb':>6}{'J':>7}")
for m in MLP_METHODS:
    r = df[(df.scheme == m) & (df.level == 1.0)].iloc[0]
    print(f"{CITE[m]:<34}{r.PICP80:>6.2f}{r.EXC:>6.2f}{r.MPIW80:>8.1f}"
          f"{r.hold_B:>7.0f}{r.penalty:>6.0f}{r.J_B:>7.0f}")


# ======================================================================
# 5) the composition experiment: does cost-optimal alpha* help EVERY updater?
# ======================================================================
print("\ncost-optimal alpha* on top of each updater (lambda=1, point backbone)")
print(f"{'updater':<34}{'a=0.10 (convention)':>22}{'a=0.20 (newsvendor)':>22}"
      f"{'change':>9}")
comp = []
for m in ["aci", "pid", "acl", "acl_asym"]:
    a_conv = run(m, 1.0, base="pt", cache=cache, alpha=0.10, **best_pt[m])
    a_star = run(m, 1.0, base="pt", cache=cache, alpha=0.20, **best_pt[m])
    d = 100 * (a_conv["J_B"] - a_star["J_B"]) / a_conv["J_B"]
    comp.append(dict(scheme=m, J_conv=a_conv["J_B"], J_star=a_star["J_B"],
                     PICP_conv=a_conv["PICP80"], PICP_star=a_star["PICP80"],
                     pct=d))
    print(f"{CITE[m]:<34}{'$' + format(a_conv['J_B'], '.0f'):>22}"
          f"{'$' + format(a_star['J_B'], '.0f'):>22}{d:>8.1f}%")
pd.DataFrame(comp).to_csv(os.path.join(OUT_DIR, "results_alpha_composition.csv"),
                          index=False)
print("saved results_alpha_composition.csv")


# ======================================================================
# 6) verification against rows the paper already reports
# ======================================================================
# ======================================================================
# 5b) LaTeX rows for tab:cost, generated rather than transcribed
# ======================================================================
SCHED = 143.0   # scheduling term, common to every scheme (Section V-F)
TCITE = {"split": "split conformal, frozen at $\\lambda{=}0$",
         "acl": "adaptive, symmetric score",
         "acl_debias": "\\quad + per-building intercept",
         "acl_asym": "adaptive, signed score",
         "pid": "conformal PID~\\cite{angelopoulos2023pid}",
         "windowed": "score window only~\\cite{pfmcp2026}",
         "aci": "$\\alpha$ only, window frozen",
         "acl_tod": "adaptive, time-of-day"}
print("\n% ---- rows for tab:cost, lambda=1, point backbone ----")
print("% idle capacity is hold/(c_h) in kWh/day; total includes "
      f"${SCHED:.0f}/day scheduling")
for m in MLP_METHODS:
    r = df[(df.scheme == m) & (df.level == 1.0)]
    if r.empty:
        continue
    r = r.iloc[0]
    idle = r.hold_B / C_H              # kWh/day
    hold, pen = round(r.hold_B), round(r.penalty)
    print(f"{TCITE.get(m, m):<44} & {r.PICP80:.2f} & {r.EXC:.2f} & "
          f"{idle:.0f} & {hold:.0f} & {pen:.0f} & "
          f"{hold + pen + SCHED:.0f} \\\\")


print("\nverification (these should match the published numbers)")
chk = [("split @1 vs fixed@0 row", ("split", 1.0, "pt"), (0.444, 72.4)),
       ("acl @1 vs adaptive row",  ("acl", 1.0, "pt"),   (0.792, 132.9)),
       ("acl_cqr @1 vs adaptive_CQR row", ("acl", 1.0, "int"), (0.791, 131.7))]
for lab, (m, lv, b), (picp0, mpiw0) in chk:
    key = m + ("_cqr" if b == "int" else "")
    r = df[(df.scheme == key) & (df.level == lv)]
    if r.empty:
        print(f"   {lab:<34} [absent]")
        continue
    r = r.iloc[0]
    dp, dw = abs(r.PICP80 - picp0), abs(r.MPIW80 - mpiw0) / mpiw0
    flag = "ok" if (dp <= 0.01 and dw <= 0.02) else "MISMATCH"
    print(f"   {lab:<34} PICP {r.PICP80:.3f} vs {picp0:.3f}, "
          f"MPIW {r.MPIW80:.1f} vs {mpiw0:.1f}   [{flag}]")
print("\nA MISMATCH means the shared interface changed the semantics of the "
      "original layer.\nFix that before any number here goes into the paper.")

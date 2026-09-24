# twin_replay_signed.py -- the mid-horizon degradation replay, run under both
# nonconformity scores so Section IV-G matches Section II-D.
#
# Run it through run_twin_demo.py --figures, which supplies OUT_DIR.
# Needs resid_structured.npz on disk. Runs in seconds; no rollouts.
#
# The earlier symmetric version took np.abs() of both residual arrays. The npz
# already stores signed residuals (y - forecast), so the signed layer costs
# nothing extra to evaluate.
#
# Rendering matches make_figs.py:
#   1. PDF alongside the PNG. dt_live is a line plot, so vector output is what
#      keeps the strokes and the tick labels sharp.
#   2. Drawn at COL = 3.5 in and placed at \columnwidth, so it is neither
#      magnified nor shrunk. Point sizes are absolute and only mean what they
#      say at the placement width, so place it at \columnwidth exactly.
#   3. Point sizes match the capacity-economics panels (cell 46 of the research
#      notebook, which writes fig_3b and fig_3c), not make_figs.py. The two
#      figures sit within two pages of each other, so they must read alike.
#
# Saves dt_live.pdf and dt_live.png, and leaves RESULTS in the namespace.

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

assert "OUT_DIR" in globals(), "OUT_DIR not set; run through run_twin_demo.py"
NPZ = os.path.join(OUT_DIR, "resid_structured.npz")
assert os.path.exists(NPZ), f"missing {NPZ}; section 8 of the research notebook writes it"

GAMMA, WINDOW, ASTAR = 0.02, 360, 0.2
A_LO, A_HI = 0.001, 0.999        # the clip on alpha_t, eq. (aci_update)
SEED_H = 360                      # clean hours reserved as the calibration seed
RECOVER_AT = 0.78                 # nominal 0.80 within the 0.02 band of Table II

z = np.load(NPZ)
K = len([k for k in z.files if k.startswith("e1_")])
fleet = [(z[f"e0_{k}"], z[f"e1_{k}"]) for k in range(K)]   # SIGNED, sign kept
n_hours = min(min(len(a), len(b)) for a, b in fleet)
n_days = (n_hours - SEED_H) // 24
event_day = n_days // 2
print(f"{K} buildings, {n_days} stream days, event at day {event_day}")


def qcorr(s, p):
    s = np.asarray(s); s = s[np.isfinite(s)]
    return float(np.quantile(s, min(0.999, p * (1 + 1 / max(len(s), 1)))))


def replay(signed):
    """Returns (cov, wid, nmiss, nstep).

    cov, wid: frozen and adaptive calibrators, K x n_days.
    nmiss, nstep: per building, the ACL's miss count and number of updates over
    the whole stream, outage included. Their ratio is the time-averaged
    miscoverage that Proposition 3 bounds.
    clipsum: per building, sum_t |c_t|, the total correction the clip applied.
    """
    cov = {s: np.full((K, n_days), np.nan) for s in ("frozen", "adaptive")}
    wid = {s: np.full((K, n_days), np.nan) for s in ("frozen", "adaptive")}
    nmiss, nstep, clipsum = np.zeros(K), np.zeros(K), np.zeros(K)

    for k, (e0, e1) in enumerate(fleet):
        seed = e0[np.isfinite(e0)][:WINDOW]

        # frozen: set once on clean scores, never updated
        if signed:
            f_up = qcorr(seed, 1 - ASTAR / 2)
            f_lo = qcorr(-seed, 1 - ASTAR / 2)
        else:
            f_up = f_lo = qcorr(np.abs(seed), 1 - ASTAR)

        # adaptive: the ACL state
        a = ASTAR
        if signed:
            Wup, Wlo = list(seed), list(-seed)
        else:
            W = list(np.abs(seed))

        for d in range(n_days):
            h0 = SEED_H + d * 24
            src = e0 if d < event_day else e1          # the degradation event
            day = src[h0:h0 + 24]
            day = day[np.isfinite(day)]
            if day.size == 0:
                continue

            if signed:
                a_up, a_lo = _q(Wup, 1 - a / 2), _q(Wlo, 1 - a / 2)
            else:
                a_up = a_lo = _q(W, 1 - a)

            cov["frozen"][k, d] = np.mean((day <= f_up) & (-day <= f_lo))
            wid["frozen"][k, d] = f_up + f_lo
            cov["adaptive"][k, d] = np.mean((day <= a_up) & (-day <= a_lo))
            wid["adaptive"][k, d] = a_up + a_lo

            for e in day:                              # evening: grade, update
                err = 0.0 if (-a_lo <= e <= a_up) else 1.0
                nmiss[k] += err; nstep[k] += 1
                raw = a + GAMMA * (ASTAR - err)
                a = float(np.clip(raw, A_LO, A_HI))
                clipsum[k] += abs(a - raw)             # |c_t|, zero unless it binds
                if signed:
                    Wup.append(e); Wlo.append(-e)
                else:
                    W.append(abs(e))
            if signed:
                Wup = Wup[-WINDOW:]; Wlo = Wlo[-WINDOW:]
            else:
                W = W[-WINDOW:]
    return cov, wid, nmiss, nstep, clipsum


def _q(w, p):
    return float(np.quantile(np.asarray(w), min(0.999, max(0.001, p))))


def roll(x, w=7):
    return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


out, prop3 = {}, {}
for signed in (False, True):
    cov, wid, nmiss, nstep, clipsum = replay(signed)
    tag = "signed" if signed else "symmetric"
    prop3[tag] = (nmiss, nstep, clipsum)
    out[tag] = dict(
        pf=roll(np.nanmean(cov["frozen"], axis=0)),
        pa=roll(np.nanmean(cov["adaptive"], axis=0)),
        wf=roll(np.nanmean(wid["frozen"], axis=0)),
        wa=roll(np.nanmean(wid["adaptive"], axis=0)),
    )

# ---- figure ---------------------------------------------------------------
# Arial first, Liberation Sans as the metric-compatible fallback. All text
# black: coloured labels read as low contrast at print size, so colour is left
# to carry scheme identity in the lines, where it distinguishes something.
# Point sizes, axis weights and tick geometry are copied from the block that
# writes fig_3b and fig_3c, so the twin figure and the economics panels read as
# one set on the page.
import matplotlib.font_manager as fm

_have = {f.name for f in fm.fontManager.ttflist}
FAMILY = next((n for n in ("Arial", "Helvetica", "Liberation Sans",
                           "Nimbus Sans", "DejaVu Sans") if n in _have),
              "DejaVu Sans")
print(f"font resolved to: {FAMILY}")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FAMILY],
    "mathtext.fontset": "custom",
    "mathtext.rm": f"{FAMILY}:regular",
    "mathtext.it": f"{FAMILY}:italic",
    "mathtext.bf": f"{FAMILY}:bold",
    "font.size": 7.0, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.0,
    "text.color": "black", "axes.labelcolor": "black",
    "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.grid": True, "grid.color": "0.78", "grid.alpha": 1.0,
    "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.linewidth": 0.9, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.borderpad": 0.2,
    "legend.labelspacing": 0.22, "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4, "legend.columnspacing": 0.9,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})
INK = "black"
# Exactly the role colours cost_regimes assigns in the capacity-economics
# figure, so a scheme keeps one colour across the paper: Fixed@0 amber, ACL
# symmetric green, ACL signed purple. Red stays reserved for the reference the
# reader measures against, as it is for the empirical minima in fig_3b and the
# known-tariff line in fig_3c.
C_FROZEN, C_SYM, C_SIG = "#D97706", "#009E73", "#800080"
C_REF = "#DC2626"

# Stroke weights and dash patterns taken straight from fig_3b, which draws its
# three curves as one solid 1.9 for the case in focus and two broken 1.2 for
# the cases it is compared against, with a 1.4 red dash for the reference (the
# same weight fig_3c gives its known-tariff line). Applied here, the signed
# layer is the argument, the symmetric layer is the comparison, and the frozen
# calibrator is the failure the figure exists to show.
LW_SIG, LW_CTX, LW_REF = 1.9, 1.2, 1.4
DS_SYM, DS_FRZ = "--", ":"

# Equal panels. Drawn at COL = 3.5 in and placed at \columnwidth, so nothing
# is magnified or shrunk and the point sizes above land as stated. Place it at
# \columnwidth exactly, not at a fraction of it.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1]})
S, G = out["symmetric"], out["signed"]

ax1.plot(S["pf"], color=C_FROZEN, lw=LW_CTX, ls=DS_FRZ, label="frozen")
ax1.plot(S["pa"], color=C_SYM, lw=LW_CTX, ls=DS_SYM,
         label="adaptive, symmetric")
ax1.plot(G["pa"], color=C_SIG, lw=LW_SIG, label="adaptive, signed")
ax1.axhline(0.8, ls="--", c=C_REF, lw=LW_REF)
ax1.axvline(event_day, c="0.35", lw=0.9)
ax1.set_ylabel(r"fleet PICP$_{80}$")
ax1.legend(loc="lower left")
ax1.annotate("degradation event", (event_day, ax1.get_ylim()[0]),
             xytext=(event_day + 2, 0.50), fontsize=6.0, color=INK)

ax2.plot(S["wf"], color=C_FROZEN, lw=LW_CTX, ls=DS_FRZ)
ax2.plot(S["wa"], color=C_SYM, lw=LW_CTX, ls=DS_SYM)
ax2.plot(G["wa"], color=C_SIG, lw=LW_SIG)
ax2.axvline(event_day, c="0.35", lw=0.9)
ax2.set_ylabel("width (kW)"); ax2.set_xlabel("stream day", labelpad=1.5)

fig.tight_layout(pad=0.25)
for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
    fig.savefig(os.path.join(OUT_DIR, "dt_live" + ext),
                bbox_inches="tight", pad_inches=0.02, **kw)
plt.close(fig)
print("saved dt_live.pdf  +  dt_live.png")

# ---- the numbers Section IV-G quotes --------------------------------------
print(f"\n{'':22}{'pre-event':>11}{'post-event':>12}{'recovery':>11}")
for tag in ("symmetric", "signed"):
    d = out[tag]
    for who, p in (("frozen", d["pf"]), ("adaptive", d["pa"])):
        pre = float(np.nanmean(p[:event_day]))
        post = float(np.nanmean(p[event_day + 7:]))
        rec = next((i - event_day for i in range(event_day, n_days)
                    if p[i] >= RECOVER_AT), None)
        rec_s = f"{rec} days" if rec is not None else "never"
        print(f"{tag + ', ' + who:<22}{pre:>11.2f}{post:>12.2f}{rec_s:>11}")

print(f"\nsteady-state width after the event (mean of last 30 days)")
for tag in ("symmetric", "signed"):
    w = out[tag]["wa"]
    print(f"   adaptive, {tag:<10} {float(np.nanmean(w[-30:])):.1f} kW")

# ---- Proposition 3: the long-run bound, per building ------------------------
# Claim (ii) of Section IV-G. With the clip written as a correction
#     c_t = alpha_{t+1} - alpha_t - gamma (alpha* - err_t),
# summing the update gives an exact identity, whatever the sizing schedule:
#     (1/T) sum_t err_t - alpha* = (alpha_1 - alpha_{T+1} + sum_t c_t) / (gamma T)
# and hence the bound of eq. (aci_bound):
#     | (1/T) sum_t err_t - alpha* |
#         <= ( max(alpha* - A_LO, A_HI - alpha*) + sum_t |c_t| ) / (gamma T).
# It is exact for this code, so a violation can only mean a bug. The clip term
# is what the layer gives up to keep every interval finite.
print("\nProposition 3, long-run miscoverage per building (outage included)")
C0 = max(ASTAR - A_LO, A_HI - ASTAR)
RESULTS_P3 = {}
for tag in ("symmetric", "signed"):
    nmiss, nstep, clipsum = prop3[tag]
    ok = nstep > 0
    T = nstep[ok]
    dev = np.abs(nmiss[ok] / T - ASTAR)
    base = C0 / (GAMMA * T)                       # the bound if the clip never binds
    bnd = (C0 + clipsum[ok]) / (GAMMA * T)        # eq. (aci_bound), exact
    viol = int((dev > bnd + 1e-12).sum())
    n_clip = int((clipsum[ok] > 0).sum())
    print(f"   {tag:<10} |avg err - {ASTAR}|: median {np.median(dev):.4f}, max {dev.max():.4f}")
    print(f"   {'':<10} bound: clip-free part {np.median(base):.4f}, "
          f"with clip term median {np.median(bnd):.4f}, max {bnd.max():.4f}")
    print(f"   {'':<10} clip binds in {n_clip} / {int(ok.sum())} buildings, "
          f"sum|c_t| median {np.median(clipsum[ok]):.3f}, max {clipsum[ok].max():.3f}")
    print(f"   {'':<10} inside the clip-free part: {int((dev <= base).sum())} / {int(ok.sum())}"
          f"   violations of the full bound: {viol}   (T = {int(np.median(T))})")
    assert viol == 0, "Proposition 3 is an identity for this code; a violation is a bug"
    RESULTS_P3[tag] = {"median_dev": float(np.median(dev)), "max_dev": float(dev.max()),
                       "base": float(np.median(base)), "bound_median": float(np.median(bnd)),
                       "n_clip": n_clip, "clip_median": float(np.median(clipsum[ok])),
                       "inside_base": int((dev <= base).sum()), "T": int(np.median(T))}

# Read by run_twin_demo.py for the comparison table. Section IV-G quotes the
# frozen post-event coverage, the recovery times and the steady-state widths.
RESULTS = {}
for tag in ("symmetric", "signed"):
    d = out[tag]
    RESULTS[tag] = {
        "frozen_post": float(np.nanmean(d["pf"][event_day + 7:])),
        "adaptive_post": float(np.nanmean(d["pa"][event_day + 7:])),
        "recovery_days": next((i - event_day for i in range(event_day, n_days)
                               if d["pa"][i] >= RECOVER_AT), None),
        "width": float(np.nanmean(d["wa"][-30:])),
    }

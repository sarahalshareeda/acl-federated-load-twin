# twin_live_signed.py -- the twin running, hour by hour, under the signed
# score. Replaces twin_live.py, whose inner loop takes np.abs(y - p) and so
# animates the symmetric layer the paper no longer leads with.
#
# Run it through run_twin_demo.py --figures --gif, which supplies OUT_DIR.
# Needs twin_stream.npz on disk. Saves twin_running.gif.
#
# The visible difference from the symmetric version: the dispatched band sits
# OFF-CENTRE on the forecast, because the two radii are estimated separately.
# That offset is the per-building bias federated averaging leaves behind, and
# it is what the symmetric layer pays for twice in width.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

assert "OUT_DIR" in globals(), "OUT_DIR not set; run through run_twin_demo.py"
NPZ = os.path.join(OUT_DIR, "twin_stream.npz")
assert os.path.exists(NPZ), f"missing {NPZ}; section 8 of the research notebook writes it"
print("rendering one frame per stream day; this takes a few minutes")

GAMMA, WINDOW, ASTAR, SEED_H = 0.02, 360, 0.2, 360
SHOW_K = 0                       # building rendered in the top panel
SIGNED = True                    # set False to reproduce the old symmetric run

# Colours match the paper's other figures: the signed layer is magenta in
# regimes.png, symmetry.png and cost_regimes.png, so a reader moving from the
# figures to the video sees the same scheme in the same colour.
C_BAND = "#B5179E"               # dispatched interval
C_FCST = "0.45"                  # point forecast
C_LOAD = "#111111"               # realized load
C_EVENT = "#C1121F"              # degradation event marker

z = np.load(NPZ)
K = len([k for k in z.files if k.startswith("y0_")])
data = [(z[f"y0_{k}"], z[f"p0_{k}"], z[f"y1_{k}"], z[f"p1_{k}"]) for k in range(K)]
n_hours = min(len(a) for a, _, _, _ in data)
n_days = (n_hours - SEED_H) // 24
event_day = n_days // 2
print(f"{K} buildings, {n_days} stream days, event at day {event_day}")


def _q(w, p):
    return float(np.quantile(np.asarray(w), min(0.999, max(0.001, p))))


# per-building ACL state, seeded on clean signed residuals
state = []
for y0, p0, _, _ in data:
    e = y0 - p0
    e = e[np.isfinite(e)][:WINDOW]
    if SIGNED:
        state.append({"a": ASTAR, "up": list(e), "lo": list(-e)})
    else:
        state.append({"a": ASTAR, "w": list(np.abs(e))})

hist = {"picp": [], "width": [], "day": []}
show = {"t": [], "y": [], "p": [], "lo": [], "up": []}


def step_day(d):
    covs, wids = [], []
    for k, (y0, p0, y1, p1) in enumerate(data):
        y, p = (y0, p0) if d < event_day else (y1, p1)
        h0 = SEED_H + d * 24
        st = state[k]

        if SIGNED:
            r_up = _q(st["up"], 1 - st["a"] / 2)
            r_lo = _q(st["lo"], 1 - st["a"] / 2)
        else:
            r_up = r_lo = _q(st["w"], 1 - st["a"])

        yy, pp = y[h0:h0 + 24], p[h0:h0 + 24]
        m = np.isfinite(yy) & np.isfinite(pp)
        if m.sum():
            e = yy[m] - pp[m]
            covs.append(np.mean((-r_lo <= e) & (e <= r_up)))
            wids.append(r_up + r_lo)

        if k == SHOW_K:
            for i in range(24):
                show["t"].append(d * 24 + i)
                show["y"].append(yy[i] if m[i] else np.nan)
                show["p"].append(pp[i] if m[i] else np.nan)
                show["lo"].append(pp[i] - r_lo if m[i] else np.nan)
                show["up"].append(pp[i] + r_up if m[i] else np.nan)

        for e in (yy[m] - pp[m]):                    # evening: grade, update
            err = 0.0 if (-r_lo <= e <= r_up) else 1.0
            st["a"] = float(np.clip(st["a"] + GAMMA * (ASTAR - err), .001, .999))
            if SIGNED:
                st["up"].append(e); st["lo"].append(-e)
            else:
                st["w"].append(abs(e))
        if SIGNED:
            st["up"] = st["up"][-WINDOW:]; st["lo"] = st["lo"][-WINDOW:]
        else:
            st["w"] = st["w"][-WINDOW:]

    hist["picp"].append(float(np.mean(covs)))
    hist["width"].append(float(np.mean(wids)))
    hist["day"].append(d)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(7, 6),
                                    gridspec_kw={"height_ratios": [1.6, 1, 1]})
fig.tight_layout(pad=2.6)
TAG = "signed" if SIGNED else "symmetric"


def draw(d):
    step_day(d)
    for ax in (axA, axB, axC):
        ax.clear()

    W = 7 * 24
    t = np.array(show["t"][-W:])
    axA.fill_between(t, np.array(show["lo"][-W:]), np.array(show["up"][-W:]),
                     color=C_BAND, alpha=0.16)
    axA.plot(t, np.array(show["up"][-W:]), color=C_BAND, lw=0.9)
    axA.plot(t, np.array(show["lo"][-W:]), color=C_BAND, lw=0.9)
    # the forecast itself, so the band's offset from it is visible. Without
    # this line a viewer sees a band and cannot tell it is off-centre.
    axA.plot(t, np.array(show["p"][-W:]), color=C_FCST, lw=0.9,
             ls=(0, (3, 2)), label="forecast")
    axA.plot(t, show["y"][-W:], color=C_LOAD, lw=1.1, label="realized load")
    axA.legend(fontsize=7, loc="upper right", ncol=2)
    axA.set_title(f"building {SHOW_K}: realized load within the dispatched "
                  f"interval ({TAG} score)   day {d+1}/{n_days}"
                  + ("    TELEMETRY DEGRADED" if d >= event_day else ""),
                  fontsize=9.5)
    axA.set_ylabel("load (kW)")

    r7 = [np.mean(hist["picp"][max(0, i - 6):i + 1])
          for i in range(len(hist["picp"]))]
    axB.plot(hist["day"], r7, "k-", lw=1.3)
    axB.axhline(.8, ls=":", c="k", lw=0.9)
    if d >= event_day:
        axB.axvline(event_day, c=C_EVENT, lw=1.0)
    axB.set_xlim(0, n_days); axB.set_ylim(0.3, 1.0)
    axB.set_ylabel(r"fleet PICP$_{80}$")

    axC.plot(hist["day"], hist["width"], "k-", lw=1.3)
    if d >= event_day:
        axC.axvline(event_day, c=C_EVENT, lw=1.0)
    axC.set_xlim(0, n_days)
    # fixed floor and a headroom-based ceiling, so the axis does not rescale
    # every frame and the early frames are not a degenerate one-point range
    axC.set_ylim(0, max(120.0, 1.15 * max(hist["width"])))
    axC.set_ylabel("width (kW)"); axC.set_xlabel("stream day")


anim = animation.FuncAnimation(fig, draw, frames=n_days, interval=120)
anim.save(os.path.join(OUT_DIR, "twin_running.gif"), writer="pillow", fps=8)
plt.close(fig)
print(f"saved twin_running.gif ({TAG} score)")

print(f"\nfinal fleet PICP (last 7 days): "
      f"{np.mean(hist['picp'][-7:]):.3f}")
print(f"steady-state width (last 30 days): "
      f"{np.mean(hist['width'][-30:]):.1f} kW")

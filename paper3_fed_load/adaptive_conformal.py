# adaptive_conformal.py — fifth contribution: online/adaptive conformal.
# Seeds calibration ONLY on clean (level-0) data, then self-corrects on revealed day-ahead
# actuals. Shows it recovers ~0.80 coverage across ALL deployment degradation levels, where a
# frozen clean calibrator (fixed@0) collapsed. RUN IN THE SAME SESSION as model_federated_core_v4
# (reuses: rollout, clients, model_pt, LEVELS, ALPHA, imetrics, qs_score, OUT_DIR, np, pd).

import os, time
import numpy as np, pandas as pd

GAMMA = 0.02      # ACI learning rate
WMAX  = 360       # rolling residual-window size (~30 days)

def qlvl(n):
    return min(0.999, (1 - ALPHA) * (1 + 1 / max(n, 1)))

# --- 1) per-client calib residual scores at each level; per-client test point preds -------
print("rolling out calib + test (point) per level...")
calib_scores = {lv: {} for lv in LEVELS}      # |residual| on calib, per client
test_pred    = {lv: {} for lv in LEVELS}      # (yact, point) on test, per client
for lv in LEVELS:
    t0 = time.time()
    for name in clients:
        yc, ptc = rollout(name, "pt", "reconstructed", split="calib", seed_from="val", level=lv)
        s = np.abs(yc - ptc); calib_scores[lv][name] = s[np.isfinite(s)]
        yt, ptt = rollout(name, "pt", "reconstructed", split="test", level=lv)
        test_pred[lv][name] = (yt, ptt)
    print(f"  level {lv:.2f}  ({time.time()-t0:.0f}s)")

# --- 2) online adaptive conformal (seeded on CLEAN level-0 scores) ------------------------
def run_adaptive(name, lv):
    y, pt = test_pred[lv][name]
    W = list(calib_scores[0.0][name])          # seed with CLEAN scores only
    a = ALPHA; N = len(y); LO = np.empty(N); UP = np.empty(N)
    for b0 in range(0, N, 24):
        b1 = min(b0 + 24, N)
        r = np.quantile(np.array(W), min(0.999, max(0.001, 1 - a)))   # radius for the day
        for i in range(b0, b1):
            LO[i] = pt[i] - r; UP[i] = pt[i] + r
        day = []                                # reveal actuals, update alpha + window
        for i in range(b0, b1):
            if np.isfinite(y[i]):
                err = 0 if (LO[i] <= y[i] <= UP[i]) else 1
                a = min(0.999, max(0.001, a + GAMMA * (ALPHA - err)))
                day.append(abs(y[i] - pt[i]))
        W.extend(day); W = W[-WMAX:]
    return y, LO, pt, UP

# --- 3) compare adaptive vs fixed@0 vs matched (point + local conformal) ------------------
def eval_scheme(scheme, lv):
    Y, LO, MD, UP = [], [], [], []
    for name in clients:
        y, pt = test_pred[lv][name]
        if scheme == "adaptive":
            y, lo, md_, up = run_adaptive(name, lv)
        elif scheme == "fixed@0":
            r = np.quantile(calib_scores[0.0][name], qlvl(len(calib_scores[0.0][name])))
            lo, md_, up = pt - r, pt, pt + r
        else:  # matched
            r = np.quantile(calib_scores[lv][name], qlvl(len(calib_scores[lv][name])))
            lo, md_, up = pt - r, pt, pt + r
        Y.append(y); LO.append(lo); MD.append(md_); UP.append(up)
    return imetrics(np.concatenate(Y), np.concatenate(LO), np.concatenate(MD), np.concatenate(UP))

rows = []
for scheme in ["fixed@0", "matched", "adaptive"]:
    for lv in LEVELS:
        m = eval_scheme(scheme, lv); m.update(scheme=scheme, test_level=lv); rows.append(m)
adf = pd.DataFrame(rows)[["scheme", "test_level", "MAE", "WAPE", "PICP80", "MPIW80", "MIS80", "QS"]].round(2)
adf.to_csv(os.path.join(OUT_DIR, "results_adaptive.csv"), index=False)
print("saved results_adaptive.csv")

# --- 4) plot + coverage matrix -----------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for scheme in ["fixed@0", "matched", "adaptive"]:
        s = adf[adf.scheme == scheme].sort_values("test_level")
        ax[0].plot(s.test_level, s.PICP80, marker="o", label=scheme)
        ax[1].plot(s.test_level, s.MPIW80, marker="o", label=scheme)
    ax[0].axhline(0.8, ls="--", c="k", lw=1); ax[0].set_title("Coverage vs deployment degradation")
    ax[0].set_xlabel("test degradation level"); ax[0].set_ylabel("PICP80"); ax[0].legend(fontsize=9)
    ax[1].set_title("Interval width vs degradation"); ax[1].set_xlabel("test degradation level"); ax[1].set_ylabel("MPIW80 (W)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "adaptive.png"), dpi=130); plt.show()
except Exception as e:
    print("plot skipped:", e)

piv = adf.pivot(index="scheme", columns="test_level", values="PICP80")
print("\nPICP80 (rows = scheme, cols = deployment degradation):")
try: display(piv)
except NameError: print(piv.to_string())

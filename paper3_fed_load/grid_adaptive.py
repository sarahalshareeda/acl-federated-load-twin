# grid_adaptive.py — grid-search GAMMA x WMAX for adaptive conformal (cheap: rollouts done once).
# Run in the SAME session as model_federated_core_v4.ipynb. If adaptive_conformal.py already ran,
# it reuses calib_scores/test_pred; otherwise it computes them once (~10-12 min), then the grid is fast.
# Reuses: rollout, clients, LEVELS, ALPHA, OUT_DIR, np, pd.

import os, time
import numpy as np, pandas as pd

# ---- grid ----
GAMMAS = [0.02, 0.05, 0.08, 0.10, 0.15]
WMAXS  = [180, 360, 480, 720, 1440]
TARGET = ALPHA          # ACI target miscoverage (0.20 -> aim 80%); try 0.16 to inflate coverage

# ---- ensure rollout predictions exist (compute once if needed) ----
try:
    calib_scores; test_pred
    print("reusing calib_scores / test_pred from memory")
except NameError:
    print("rolling out calib + test (point) per level (one-time)...")
    calib_scores = {lv: {} for lv in LEVELS}; test_pred = {lv: {} for lv in LEVELS}
    for lv in LEVELS:
        t0 = time.time()
        for name in clients:
            yc, ptc = rollout(name, "pt", "reconstructed", split="calib", seed_from="val", level=lv)
            s = np.abs(yc - ptc); calib_scores[lv][name] = s[np.isfinite(s)]
            yt, ptt = rollout(name, "pt", "reconstructed", split="test", level=lv)
            test_pred[lv][name] = (yt, ptt)
        print(f"  level {lv:.2f} ({time.time()-t0:.0f}s)")

def run_cfg(name, lv, gamma, wmax, target):
    y, pt = test_pred[lv][name]
    W = list(calib_scores[0.0][name]); a = ALPHA; N = len(y); LO = np.empty(N); UP = np.empty(N)
    for b0 in range(0, N, 24):
        b1 = min(b0 + 24, N)
        r = np.quantile(np.asarray(W), min(0.999, max(0.001, 1 - a)))
        LO[b0:b1] = pt[b0:b1] - r; UP[b0:b1] = pt[b0:b1] + r
        day = []
        for i in range(b0, b1):
            if np.isfinite(y[i]):
                err = 0 if (LO[i] <= y[i] <= UP[i]) else 1
                a = min(0.999, max(0.001, a + gamma * (target - err))); day.append(abs(y[i] - pt[i]))
        W.extend(day); W = W[-wmax:]
    return y, LO, pt, UP

# ---- sweep ----
print(f"grid search: {len(GAMMAS)}x{len(WMAXS)} configs over {len(LEVELS)} levels...")
rows = []
for g in GAMMAS:
    for w in WMAXS:
        picps, qss = [], []
        for lv in LEVELS:
            Y, LO, PT, UP = [], [], [], []
            for name in clients:
                y, lo, pt, up = run_cfg(name, lv, g, w, TARGET)
                Y.append(y); LO.append(lo); PT.append(pt); UP.append(up)
            Y = np.concatenate(Y); LO = np.concatenate(LO); PT = np.concatenate(PT); UP = np.concatenate(UP)
            m = np.isfinite(Y) & np.isfinite(LO) & np.isfinite(UP); Y, LO, PT, UP = Y[m], LO[m], PT[m], UP[m]
            picps.append(np.mean((Y >= LO) & (Y <= UP)))
            qss.append(np.mean([np.mean(np.maximum(q*(Y-p), (q-1)*(Y-p))) for q, p in {0.1:LO,0.5:PT,0.9:UP}.items()]))
        rows.append(dict(gamma=g, wmax=w, mean_PICP=np.mean(picps),
                         calib_err=np.mean(np.abs(np.array(picps) - (1 - ALPHA))), mean_QS=np.mean(qss)))
    print(f"  gamma={g} done")

grid = pd.DataFrame(rows).round(3).sort_values("calib_err").reset_index(drop=True)
grid.to_csv(os.path.join(OUT_DIR, "results_grid_adaptive.csv"), index=False)
print("\nbest configs (lowest mean |PICP-0.80|):")
try: display(grid.head(10))
except NameError: print(grid.head(10).to_string(index=False))

# heatmap of calibration error
try:
    import matplotlib.pyplot as plt
    piv = grid.pivot(index="gamma", columns="wmax", values="calib_err")
    plt.figure(figsize=(6, 4)); plt.imshow(piv.values, aspect="auto", cmap="viridis_r")
    plt.colorbar(label="mean |PICP-0.80|")
    plt.xticks(range(len(piv.columns)), piv.columns); plt.yticks(range(len(piv.index)), piv.index)
    plt.xlabel("WMAX"); plt.ylabel("GAMMA"); plt.title("Adaptive-conformal calibration error")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            plt.text(j, i, f"{piv.values[i,j]:.03f}", ha="center", va="center", color="w", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "grid_adaptive.png"), dpi=130); plt.show()
except Exception as e:
    print("plot skipped:", e)

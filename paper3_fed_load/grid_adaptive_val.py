# grid_adaptive_val.py — same GAMMA x WMAX grid as grid_adaptive.py, but tuned on the
# VALIDATION split, so the test stream is never used to choose gamma or W.
# Run in the SAME session as model_federated_core_v5_clean.ipynb (needs rollout, clients,
# LEVELS, ALPHA, OUT_DIR). Writes results_grid_adaptive_val.csv.

import os, time
import numpy as np, pandas as pd

GAMMAS = [0.02, 0.05, 0.08, 0.10, 0.15]
WMAXS  = [180, 360, 480, 720, 1440]
TARGET = ALPHA

# calibration scores seed the window (as in the paper); the tuning stream is the val split,
# seeded for lags from the end of the train split. The test split is never touched.
calib_sc, val_pred = {lv: {} for lv in LEVELS}, {lv: {} for lv in LEVELS}
for lv in LEVELS:
    t0 = time.time()
    for name in clients:
        yc, ptc = rollout(name, "pt", "reconstructed", split="calib", seed_from="val", level=lv)
        s = np.abs(yc - ptc); calib_sc[lv][name] = s[np.isfinite(s)]
        yv, ptv = rollout(name, "pt", "reconstructed", split="val", seed_from="train", level=lv)
        val_pred[lv][name] = (yv, ptv)
    print(f"  level {lv:.2f} ({time.time()-t0:.0f}s)")


def run_cfg(name, lv, gamma, wmax, target):
    y, pt = val_pred[lv][name]
    W = list(calib_sc[0.0][name]); a = ALPHA; N = len(y); LO = np.empty(N); UP = np.empty(N)
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


print(f"grid search on VALIDATION: {len(GAMMAS)}x{len(WMAXS)} configs over {len(LEVELS)} levels...")
rows = []
for g in GAMMAS:
    for w in WMAXS:
        picps = []
        for lv in LEVELS:
            Y, LO, UP = [], [], []
            for name in clients:
                y, lo, pt, up = run_cfg(name, lv, g, w, TARGET)
                Y.append(y); LO.append(lo); UP.append(up)
            Y = np.concatenate(Y); LO = np.concatenate(LO); UP = np.concatenate(UP)
            m = np.isfinite(Y) & np.isfinite(LO) & np.isfinite(UP)
            picps.append(np.mean((Y[m] >= LO[m]) & (Y[m] <= UP[m])))
        rows.append(dict(gamma=g, wmax=w, mean_PICP=np.mean(picps),
                         calib_err=np.mean(np.abs(np.array(picps) - (1 - ALPHA)))))
    print(f"  gamma={g} done")

grid_val = pd.DataFrame(rows).round(4).sort_values("calib_err").reset_index(drop=True)
grid_val.to_csv(os.path.join(OUT_DIR, "results_grid_adaptive_val.csv"), index=False)
print("\nbest configs on VALIDATION (lowest mean |PICP-0.80|):")
print(grid_val.head(10).to_string(index=False))
print("\nrow for the paper's choice (gamma=0.02, W=360):")
print(grid_val[(grid_val.gamma == 0.02) & (grid_val.wmax == 360)].to_string(index=False))

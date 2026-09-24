# fm_eval.py — evaluate the Chronos backbone under the SAME calibration layer, WITH CHECKPOINTING.
# Run AFTER chronos_backbone.py in the same session (needs fm_forecast, clients, ALPHA, imetrics,
# qs_score, OUT_DIR, np, pd). Chronos rollouts are cached to Drive per (level,client,split), so a
# disconnect resumes instead of restarting. To resume: re-run chronos_backbone.py, then this file.

import os, time, pickle
import numpy as np, pandas as pd

FM_LEVELS = [0.0, 0.5, 1.0]        # add 0.25/0.75 for full dose-response (slower)
GAMMA, WMAX, ASTAR = 0.02, 360, ALPHA
CKPT = os.path.join(OUT_DIR, "fm_ckpt.pkl")
SAVE_EVERY = 5                     # write checkpoint to Drive every N buildings

def qlvl(n): return min(0.999, (1 - ALPHA) * (1 + 1 / max(n, 1)))
def key(lv, name, split): return (round(float(lv), 3), name, split)

# ---- 1) checkpointed Chronos rollouts (test + calib per level) ---------------------------
ckpt = {}
if os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        ckpt = pickle.load(f)
    print(f"resumed checkpoint: {len(ckpt)} cached (level,client,split) entries")

todo = [(lv, name) for lv in FM_LEVELS for name in clients]
n_new = 0
t0 = time.time()
for j, (lv, name) in enumerate(todo):
    kt, kc = key(lv, name, "test"), key(lv, name, "calib")
    fresh = False
    if kt not in ckpt:
        ckpt[kt] = fm_forecast(name, "test", lv, "calib"); fresh = True
    if kc not in ckpt:
        ckpt[kc] = fm_forecast(name, "calib", lv, "val"); fresh = True
    if fresh:
        n_new += 1
        if n_new % SAVE_EVERY == 0:
            with open(CKPT, "wb") as f:
                pickle.dump(ckpt, f)
            print(f"  [{j+1}/{len(todo)}] cached; saved checkpoint ({len(ckpt)} entries, {time.time()-t0:.0f}s)")
with open(CKPT, "wb") as f:      # final save
    pickle.dump(ckpt, f)
print(f"rollouts complete: {len(ckpt)} entries cached at {CKPT}")

# assemble from checkpoint
fm_test = {lv: {} for lv in FM_LEVELS}; fm_calib = {lv: {} for lv in FM_LEVELS}; fm_clean = {}
for lv in FM_LEVELS:
    for name in clients:
        fm_test[lv][name] = ckpt[key(lv, name, "test")]
        fm_calib[lv][name] = ckpt[key(lv, name, "calib")]
for name in clients:
    y, lo, md, up = fm_calib[0.0][name]; s = np.abs(y - md); fm_clean[name] = s[np.isfinite(s)]

# ---- 2) online adaptive (seeded clean) ---------------------------------------------------
def adaptive(name, lv):
    y, lo, md, up = fm_test[lv][name]
    W = list(fm_clean[name]); a = ALPHA; N = len(y); LO = np.empty(N); UP = np.empty(N)
    for b0 in range(0, N, 24):
        b1 = min(b0 + 24, N)
        r = np.quantile(np.asarray(W), min(0.999, max(0.001, 1 - a)))
        LO[b0:b1] = md[b0:b1] - r; UP[b0:b1] = md[b0:b1] + r
        day = []
        for i in range(b0, b1):
            if np.isfinite(y[i]):
                err = 0 if (LO[i] <= y[i] <= UP[i]) else 1
                a = min(0.999, max(0.001, a + GAMMA * (ASTAR - err))); day.append(abs(y[i] - md[i]))
        W.extend(day); W = W[-WMAX:]
    return y, LO, md, UP

# ---- 3) schemes and scoring --------------------------------------------------------------
rows = []
for lv in FM_LEVELS:
    loc_mod, loc_cqr = {}, {}
    for name in clients:
        y, lo, md, up = fm_calib[lv][name]
        s = np.abs(y - md); s = s[np.isfinite(s)]; loc_mod[name] = np.quantile(s, qlvl(len(s)))
        E = np.maximum(lo - y, y - up); E = E[np.isfinite(E)]; loc_cqr[name] = np.quantile(E, qlvl(len(E)))
    for scheme in ["fm_native", "fm_modular_local", "fm_cqr_local", "fm_adaptive"]:
        Y, LO, MD, UP = [], [], [], []
        for name in clients:
            y, lo, md, up = fm_test[lv][name]
            if scheme == "fm_native":          L, M, U = lo, md, up
            elif scheme == "fm_modular_local": L, M, U = md - loc_mod[name], md, md + loc_mod[name]
            elif scheme == "fm_cqr_local":     L, M, U = lo - loc_cqr[name], md, up + loc_cqr[name]
            else:                              y, L, M, U = adaptive(name, lv)
            Y.append(y); LO.append(L); MD.append(M); UP.append(U)
        r = imetrics(np.concatenate(Y), np.concatenate(LO), np.concatenate(MD), np.concatenate(UP))
        r.update(level=lv, scheme=scheme); rows.append(r)

fm_df = pd.DataFrame(rows)[["level", "scheme", "MAE", "WAPE", "PICP80", "MPIW80", "MIS80", "QS"]].round(2)
fm_df.to_csv(os.path.join(OUT_DIR, "results_fm.csv"), index=False)
print("\nsaved results_fm.csv")
try: display(fm_df.sort_values(["scheme", "level"]))
except NameError: print(fm_df.sort_values(["scheme", "level"]).to_string(index=False))

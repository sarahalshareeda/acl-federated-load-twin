# sweep7.py -- section 7 of the research notebook, as a file.
#
#   exec(open(f"{PROJECT_DIR}/sweep7.py").read())
#
# Needs in session, i.e. after cells 1, 2, 3, "#4 load trained models", 5 and 6:
#   LEVELS, clients, calibrate, rollout, ALPHA, OUT_DIR, np, pd, time, defaultdict
#
# WHY THIS IS A FILE. Pasting this cell into Colab kept losing its leading
# whitespace, which silently dedented `for name in clients:` out of
# `for lv in LEVELS:`. That runs the inner loop once, after the level loop has
# finished, with lv left at its final value of 1.0: five levels calibrated, one
# level swept. That is exactly how sweep.pkl ended up holding only level 1.0,
# which is what truncated results_baselines.csv to two rows and flattened the
# left panel of symmetry.png. Reading the file off disk cannot mangle it.
#
# Does all of section 7 in one call:
#   1. the five-level sweep
#   2. an assert that 25 keys came back, so a broken loop fails loudly here
#      rather than quietly downstream
#   3. results_doseresponse.csv
#   4. rollout_resid_L1.npy
#   5. sweep_all5.pkl, written to a NEW name so the existing level-1 sweep.pkl
#      is left alone and cannot be confused with this one
#
# Takes 15 to 25 minutes. Prints one line per level as it goes.

import os
import time
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd


def qs_score(y, levels):
    return np.mean([np.mean(np.maximum(q * (y - p), (q - 1) * (y - p)))
                    for q, p in levels.items()])


def imetrics(y, lo, md_, up):
    y = y.ravel(); lo = lo.ravel(); md_ = md_.ravel(); up = up.ravel()
    m = np.isfinite(y) & np.isfinite(md_) & np.isfinite(lo) & np.isfinite(up)
    y, lo, md_, up = y[m], lo[m], md_[m], up[m]
    return dict(
        MAE=np.mean(np.abs(y - md_)),
        WAPE=100 * np.sum(np.abs(y - md_)) / (np.sum(np.abs(y)) + 1e-9),
        PICP80=np.mean((y >= lo) & (y <= up)),
        MPIW80=np.mean(up - lo),
        MIS80=np.mean((up - lo)
                      + (2 / ALPHA) * (lo - y) * (y < lo)
                      + (2 / ALPHA) * (y - up) * (y > up)),
        QS=qs_score(y, {0.1: lo, 0.5: md_, 0.9: up}),
    )


sweep = defaultdict(lambda: {"y": [], "lo": [], "md": [], "up": []})


def put(key, y, lo, md_, up):
    s = sweep[key]
    s["y"].append(y); s["lo"].append(lo)
    s["md"].append(md_); s["up"].append(up)


print(f"sweeping {len(LEVELS)} levels x {len(clients)} clients")
for lv in LEVELS:
    t0 = time.time()
    fq, Qcqr, loc_mod, loc_cqr = calibrate(lv)
    for name in clients:
        yi, lo, md_, up = rollout(name, "int", "reconstructed", level=lv)
        yp, pt = rollout(name, "pt", "reconstructed", level=lv)
        put((lv, "integrated"),    yi, lo,                md_, up)
        put((lv, "cqr_fed"),       yi, lo - Qcqr,         md_, up + Qcqr)
        put((lv, "cqr_local"),     yi, lo - loc_cqr[name], md_, up + loc_cqr[name])
        put((lv, "modular_fed"),   yi, pt - fq,           pt,  pt + fq)
        put((lv, "modular_local"), yi, pt - loc_mod[name], pt, pt + loc_mod[name])
    print(f"level {lv:.2f} done in {time.time() - t0:.0f}s")

# fail loudly here, not silently three scripts later
_levels = sorted({k[0] for k in sweep})
if len(sweep) != 5 * len(LEVELS):
    raise SystemExit(
        f"expected {5 * len(LEVELS)} keys, got {len(sweep)} at levels {_levels}.\n"
        "The inner loop did not run once per level.")
print(f"sweep OK: {len(sweep)} keys, levels {_levels}")

# ---- results_doseresponse.csv -------------------------------------------
rows = []
for (lv, scheme), s in sweep.items():
    y = np.concatenate(s["y"]); lo = np.concatenate(s["lo"])
    md_ = np.concatenate(s["md"]); up = np.concatenate(s["up"])
    r = imetrics(y, lo, md_, up)
    r.update(level=lv, scheme=scheme)
    rows.append(r)

sweep_df = (pd.DataFrame(rows)[["level", "scheme", "MAE", "WAPE", "PICP80",
                                "MPIW80", "MIS80", "QS"]]
            .sort_values(["scheme", "level"]).round(2))
sweep_df.to_csv(os.path.join(OUT_DIR, "results_doseresponse.csv"), index=False)
print("saved results_doseresponse.csv")

# ---- full-degradation residuals, so cost_optimize survives a disconnect --
_s1 = sweep[(max(LEVELS), "integrated")]
_y1 = np.concatenate([np.asarray(a).ravel() for a in _s1["y"]])
_m1 = np.concatenate([np.asarray(a).ravel() for a in _s1["md"]])
np.save(os.path.join(OUT_DIR, "rollout_resid_L1.npy"), _y1 - _m1)
print("saved rollout_resid_L1.npy")

# ---- the sweep itself, under a NEW name ---------------------------------
# sweep.pkl on Drive holds level 1.0 only. Writing there would either destroy
# a known-good file or, if the write failed part way, leave a truncated one,
# which has already happened once in this project.
_dst = os.path.join(OUT_DIR, "sweep_all5.pkl")
with open(_dst, "wb") as _f:
    pickle.dump({k: {kk: [np.asarray(a) for a in vv] for kk, vv in v.items()}
                 for k, v in sweep.items()}, _f)
print(f"saved sweep_all5.pkl, {len(sweep)} keys, "
      f"{os.path.getsize(_dst) / 1e6:.1f} MB")

print("\nnext, in a new cell:")
print("  BASELINE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]")
print('  exec(open(f"{PROJECT_DIR}/baselines.py").read())')
print('  exec(open(f"{PROJECT_DIR}/make_figs.py").read())')
print("watch for: width ratio: [1.35, 1.56, 1.72, 1.84, 1.96]")

sweep_df

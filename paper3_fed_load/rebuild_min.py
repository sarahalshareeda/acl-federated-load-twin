# rebuild_min.py -- rebuilds only what cost_table_cp.py needs after sweep.pkl
# was lost: the lambda=1 panels for the six schemes, plus the clean (lambda=0)
# calibration scores the ACL seeds from. Roughly a third of cell 7's work.
#
# Needs in session: clients, ALPHA, OUT_DIR, rollout() (cell 5), qhi() (cell 6).
#
#   exec(open(f"{PROJECT_DIR}/rebuild_min.py").read())

import os, time, pickle
import numpy as np
from collections import defaultdict

LV = 1.0

calib_scores = {0.0: {}, LV: {}}     # |y - pt|   point nonconformity
cqr_calib    = {0.0: {}, LV: {}}     # CQR nonconformity
test_pred    = {LV: {}}
test_int     = {LV: {}}

for lv in (0.0, LV):
    t0 = time.time()
    for name in clients:
        yc, ptc = rollout(name, "pt", "reconstructed",
                          split="calib", seed_from="val", level=lv)
        s = np.abs(yc - ptc)
        calib_scores[lv][name] = s[np.isfinite(s)]

        yi, lo, md_, up = rollout(name, "int", "reconstructed",
                                  split="calib", seed_from="val", level=lv)
        E = np.maximum(lo - yi, yi - up)
        cqr_calib[lv][name] = E[np.isfinite(E)]
    print(f"  calib rollouts, level {lv:.2f}  ({time.time()-t0:.0f}s)")

t0 = time.time()
for name in clients:
    test_pred[LV][name] = rollout(name, "pt", "reconstructed",
                                  split="test", level=LV)
    test_int[LV][name]  = rollout(name, "int", "reconstructed",
                                  split="test", level=LV)
print(f"  test rollouts, level {LV:.2f}  ({time.time()-t0:.0f}s)")

# conformal radii at the deployment level, exactly as cell 6 forms them
fq      = qhi(np.concatenate([calib_scores[LV][n] for n in clients]))
Qcqr    = qhi(np.concatenate([cqr_calib[LV][n]    for n in clients]))
loc_mod = {n: qhi(calib_scores[LV][n]) for n in clients}
loc_cqr = {n: qhi(cqr_calib[LV][n])    for n in clients}

sweep = defaultdict(lambda: {"y": [], "lo": [], "md": [], "up": []})


def put(key, y, lo, md_, up):
    s = sweep[key]
    s["y"].append(y); s["lo"].append(lo); s["md"].append(md_); s["up"].append(up)


for name in clients:
    yi, lo, md_, up = test_int[LV][name]
    yp, pt          = test_pred[LV][name]
    put((LV, "integrated"),    yi, lo,                  md_, up)
    put((LV, "cqr_fed"),       yi, lo - Qcqr,           md_, up + Qcqr)
    put((LV, "cqr_local"),     yi, lo - loc_cqr[name],  md_, up + loc_cqr[name])
    put((LV, "modular_fed"),   yi, pt - fq,             pt,  pt + fq)
    put((LV, "modular_local"), yi, pt - loc_mod[name],  pt,  pt + loc_mod[name])

print(f"rebuilt {len(sweep)} panels at lambda={LV}")

# atomic save: a future interruption leaves the old file intact, not a stub
tmp = os.path.join(OUT_DIR, "sweep_L1.pkl.tmp")
with open(tmp, "wb") as f:
    pickle.dump({k: {kk: [np.asarray(a) for a in vv] for kk, vv in v.items()}
                 for k, v in sweep.items()}, f, protocol=4)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, os.path.join(OUT_DIR, "sweep_L1.pkl"))
print("saved sweep_L1.pkl,", os.path.getsize(os.path.join(OUT_DIR, "sweep_L1.pkl")), "bytes")

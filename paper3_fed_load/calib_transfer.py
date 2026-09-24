# calib_transfer.py — calibration-transfer robustness.
# Calibrate conformal ONCE at a fixed degradation level, then apply that frozen calibrator
# across ALL test degradation levels, and compare to matched (per-level) calibration.
# RUN THIS IN THE SAME SESSION as model_federated_core_v4.ipynb (reuses: calibrate, rollout,
# clients, model_int, model_pt, LEVELS, imetrics, qs_score, ALPHA, OUT_DIR, np, pd).

import os, time
import numpy as np, pandas as pd

CALIB_AT = [0.0, 0.5, 1.0]       # frozen-calibration levels to test
FOCUS = "cqr_local"              # scheme shown in the plot (change to modular_local etc.)

# 1) calibrators at each level
print("computing calibrators per level...")
cal = {}
for lv in LEVELS:
    t0 = time.time(); cal[lv] = calibrate(lv)
    print(f"  calibrated @ {lv:.2f}  ({time.time()-t0:.0f}s)")

# 2) raw test predictions per level (rolled out once, reused for every calibrator)
print("test rollouts per level...")
raw = {}
for lv in LEVELS:
    t0 = time.time(); pc = {}
    for name in clients:
        yi, lo, md_, up = rollout(name, "int", "reconstructed", level=lv)
        _, pt = rollout(name, "pt", "reconstructed", level=lv)
        pc[name] = (yi, lo, md_, up, pt)
    raw[lv] = pc
    print(f"  test rollout @ {lv:.2f}  ({time.time()-t0:.0f}s)")

# 3) assemble intervals for (calibrated-at c, tested-at t, scheme)
def assemble(c, t, scheme):
    fq, Qcqr, loc_mod, loc_cqr = cal[c]
    Y, LO, MD, UP = [], [], [], []
    for name, (yi, lo, md_, up, pt) in raw[t].items():
        Y.append(yi)
        if scheme == "cqr_fed":       L, M, U = lo - Qcqr, md_, up + Qcqr
        elif scheme == "cqr_local":   L, M, U = lo - loc_cqr[name], md_, up + loc_cqr[name]
        elif scheme == "modular_fed": L, M, U = pt - fq, pt, pt + fq
        elif scheme == "modular_local": L, M, U = pt - loc_mod[name], pt, pt + loc_mod[name]
        else:                          L, M, U = lo, md_, up
        LO.append(L); MD.append(M); UP.append(U)
    return imetrics(np.concatenate(Y), np.concatenate(LO), np.concatenate(MD), np.concatenate(UP))

# 4) build the transfer table across schemes
schemes = ["cqr_fed", "cqr_local", "modular_fed", "modular_local"]
rows = []
for scheme in schemes:
    for setting in [f"fixed@{c}" for c in CALIB_AT] + ["matched"]:
        for t in LEVELS:
            c = t if setting == "matched" else float(setting.split("@")[1])
            m = assemble(c, t, scheme)
            m.update(scheme=scheme, calib=setting, test_level=t)
            rows.append(m)
tf = pd.DataFrame(rows)[["scheme", "calib", "test_level", "MAE", "WAPE", "PICP80", "MPIW80", "MIS80", "QS"]].round(2)
tf.to_csv(os.path.join(OUT_DIR, "results_calibtransfer.csv"), index=False)
print("saved results_calibtransfer.csv")

# 5) focus plot: coverage & width vs test level, per calibration setting (one scheme)
try:
    import matplotlib.pyplot as plt
    sub = tf[tf.scheme == FOCUS]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for setting in sorted(sub.calib.unique()):
        s = sub[sub.calib == setting].sort_values("test_level")
        ax[0].plot(s.test_level, s.PICP80, marker="o", label=setting)
        ax[1].plot(s.test_level, s.MPIW80, marker="o", label=setting)
    ax[0].axhline(0.8, ls="--", c="k", lw=1); ax[0].set_title(f"{FOCUS}: coverage vs deployment degradation")
    ax[0].set_xlabel("test degradation level"); ax[0].set_ylabel("PICP80"); ax[0].legend(fontsize=8)
    ax[1].set_title(f"{FOCUS}: interval width vs degradation"); ax[1].set_xlabel("test degradation level"); ax[1].set_ylabel("MPIW80 (W)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "calibtransfer.png"), dpi=130); plt.show()
except Exception as e:
    print("plot skipped:", e)

# compact readout: coverage matrix for the focus scheme
piv = tf[tf.scheme == FOCUS].pivot(index="calib", columns="test_level", values="PICP80")
print(f"\nPICP80 for {FOCUS} (rows = calibrated at, cols = tested at):")
try: display(piv)
except NameError: print(piv.to_string())

# bandit_seasons.py -- multi-season replay: does bandit feedback converge
# given time? Reproduces the notebook cell verbatim, then saves the per-season
# costs the cell computed and discarded.
#
#   exec(open(f"{PROJECT_DIR}/bandit_seasons.py").read())
#
# Needs in session: OUT_DIR, and resid_structured.npz on disk.
# Writes bandit_hist_seasons.npy (as before) and bandit_seasons.npz (new,
# for the right panel of bandit_arms.png).
#
# NOTE the ACI streams score np.abs(e) throughout, so this study uses the
# SYMMETRIC nonconformity score. That is deliberate and stated in the paper:
# the level the bandit converges to follows from the cost ratio alone
# (Proposition 1) and is unchanged by the score, while the absolute bill
# levels would fall under the signed construction.

import os
import numpy as np

NPZ = os.path.join(OUT_DIR, "resid_structured.npz")
GAMMA_ACI, WINDOW = 0.02, 360
C_H = 0.12
PEAK = lambda h: (8 <= h) & (h < 20)          # noqa: E731
RATIOS = {True: 19.0, False: 4.0}             # unknown to the bandit
ARMS = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
LIN_ALPHA = 0.3
N_REP = 6                                     # simulated seasons

z = np.load(NPZ)
K = len([k for k in z.files if k.startswith("e1_")])
base = [(z[f"e0_{k}"], z[f"e1_{k}"], z[f"hod_{k}"]) for k in range(K)]
n1 = min(len(e1) for _, e1, _ in base)
d1 = n1 // 24                                 # days per season
flt = [(e0, np.tile(e1[:n1], N_REP), np.tile(hod[:n1], N_REP))
       for e0, e1, hod in base]
n_days = d1 * N_REP


class Stream:
    def __init__(self, e0):
        s = np.abs(e0[np.isfinite(e0)])
        self.win = list(s[-WINDOW:])
        self.alpha = 0.2

    def step(self, e, a_star):
        r = np.quantile(np.asarray(self.win),
                        1 - np.clip(self.alpha, 0.01, 0.99))
        err = 1.0 if abs(e) > r else 0.0
        self.alpha = float(np.clip(self.alpha + GAMMA_ACI * (a_star - err),
                                   0.001, 0.999))
        self.win.append(abs(e))
        del self.win[:max(0, len(self.win) - WINDOW)]
        return r, err


def run(policy, period_fn, days, fleet, on_day=None):
    periods = sorted({period_fn(h) for h in range(24)})
    streams = {(k, p): Stream(fleet[k][0]) for k in range(K) for p in periods}
    day_costs = []
    hist = []
    for d in range(days):
        tgt = {p: policy(d, p) for p in periods}
        dc = {p: 0.0 for p in periods}
        for k in range(K):
            _, e1, hod = fleet[k]
            for t in range(d * 24, (d + 1) * 24):
                if not np.isfinite(e1[t]):
                    continue
                p = period_fn(hod[t])
                r, _ = streams[(k, p)].step(e1[t], tgt[p])
                cp = C_H * RATIOS[bool(PEAK(hod[t]))]
                dc[p] += C_H * max(r - e1[t], 0) + cp * max(e1[t] - r, 0)
        day_costs.append(sum(dc.values()))
        hist.append(dict(tgt))
        if on_day:
            on_day(d, dc)
    return np.array(day_costs), hist


two_p = lambda h: bool(PEAK(h))               # noqa: E731
one_p = lambda h: "all"                       # noqa: E731

# season-static policies: one season, then scale
dc_fix, _ = run(lambda d, p: 0.20, one_p, d1, base)
dc_orc, _ = run(lambda d, p: 2 / (RATIOS[p] + 1), two_p, d1, base)
fix_season, orc_season = dc_fix.sum(), dc_orc.sum()


class Bandit:
    def __init__(self):
        self.A = {(p, a): 1.0 for p in (True, False)
                  for a in range(len(ARMS))}
        self.b = {(p, a): 0.0 for p in (True, False)
                  for a in range(len(ARMS))}
        self.pending = {}

    def __call__(self, d, p):
        best, arg = -1e18, 0
        for a in range(len(ARMS)):
            u = (self.b[(p, a)] / self.A[(p, a)]
                 + LIN_ALPHA * np.sqrt(np.log(max(d, 2)) / self.A[(p, a)]))
            if u > best:
                best, arg = u, a
        self.pending[p] = arg
        return ARMS[arg]

    def on_day(self, d, dc):
        for p, a in self.pending.items():
            self.A[(p, a)] += 1.0
            self.b[(p, a)] += -dc[p] / (K * 30.0)


B = Bandit()
dc_bnd, hist = run(B, two_p, n_days, flt, on_day=B.on_day)

print(f"oracle vs fixed saving (per season): "
      f"{100*(1-orc_season/fix_season):.1f}%")
for s in range(N_REP):
    c = dc_bnd[s*d1:(s+1)*d1].sum()
    h = np.array([[hh[True], hh[False]] for hh in hist[s*d1:(s+1)*d1]])
    print(f"season {s+1}: bandit saving {100*(1-c/fix_season):5.1f}%   "
          f"median targets peak={np.median(h[:, 0]):.2f} "
          f"off={np.median(h[:, 1]):.2f}")

np.save(os.path.join(OUT_DIR, "bandit_hist_seasons.npy"),
        np.array([[hh[True], hh[False]] for hh in hist]))
print("saved bandit_hist_seasons.npy")


# ---- what the original cell computed and threw away ----------------------
seasons = np.arange(1, N_REP + 1)
bnd_cost = np.array([dc_bnd[s*d1:(s+1)*d1].sum() for s in range(N_REP)])
saving = 100.0 * (1.0 - bnd_cost / fix_season)
oracle_saving = 100.0 * (1.0 - orc_season / fix_season)

np.savez(os.path.join(OUT_DIR, "bandit_seasons.npz"),
         season=seasons,
         bandit_cost=bnd_cost,
         fixed_cost=np.full(N_REP, fix_season),
         oracle_cost=np.full(N_REP, orc_season),
         saving_pct=saving,
         oracle_saving_pct=oracle_saving,
         days_per_season=d1,
         arms=ARMS)
print("saved bandit_seasons.npz")

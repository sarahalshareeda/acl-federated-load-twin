# bandit_seasons_save.py -- two lines bandit_seasons.py was missing.
#
# Run this IMMEDIATELY AFTER bandit_seasons.py, in the same session, while
# dc_bnd, hist, d1, N_REP, fix_season and orc_season are still in memory.
#
#   exec(open(f"{PROJECT_DIR}/bandit_seasons.py").read())
#   exec(open(f"{PROJECT_DIR}/bandit_seasons_save.py").read())
#
# bandit_seasons.py computes the per-season saving and prints it, then throws
# it away. The figure needs it on disk.

import os
import numpy as np

need = ["dc_bnd", "hist", "d1", "N_REP", "fix_season", "orc_season"]
missing = [n for n in need if n not in globals()]
if missing:
    raise NameError(f"run bandit_seasons.py first; missing {missing}")

seasons = np.arange(1, N_REP + 1)
bnd_cost = np.array([dc_bnd[s * d1:(s + 1) * d1].sum() for s in range(N_REP)])
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
         arms=np.array([0.05, 0.10, 0.20, 0.30, 0.40]))

print("saved bandit_seasons.npz")
print(f"\noracle saving per season: {oracle_saving:.1f}%")
for s in range(N_REP):
    flag = "  <- reaches oracle" if saving[s] >= oracle_saving else ""
    print(f"  season {s+1}: {saving[s]:5.1f}%{flag}")

# cluster_table.py -- rebuilds tab:cluster, the per-cluster split that is the
# last table in the paper without a file behind it.
#
#   exec(open(f"{PROJECT_DIR}/cluster_table.py").read())
#
# Needs in session: sweep (levels 0.0 AND 1.0, so cell 7 in full, not the
# level-1 trim), clients, PROJECT_DIR, OUT_DIR.
#
# Pools each cluster's clients the same way the dose-response cell pools the
# fleet, so the WAPE and PICP here are computed exactly as tab:dose's are and
# the two tables are comparable.

import os
import numpy as np
import pandas as pd

NEED = [(lv, sc) for lv in (0.0, 1.0)
        for sc in ("integrated", "modular_local")]
_missing = [k for k in NEED if k not in sweep]
if _missing:
    raise SystemExit(
        f"sweep is missing {_missing}\n"
        "Run cell 7 in full (all five levels) before this, and do not "
        "reassign sweep from the level-1 trim afterwards.")

# BDG2 names carry the use in the middle token, site_use_name, so the cluster
# is read off the client id itself. candidate_clients.csv is consulted only if
# it happens to be alongside, as a cross-check on the parse.
def use_from_name(nm):
    parts = nm.split("_")
    return parts[1].capitalize() if len(parts) >= 2 else "unknown"


names = list(clients.keys())          # the order put() appended in
use = {nm: use_from_name(nm) for nm in names}

for p in (os.path.join(PROJECT_DIR, "candidate_clients.csv"),
          os.path.join(OUT_DIR, "candidate_clients.csv")):
    if os.path.exists(p):
        meta = pd.read_csv(p).set_index("building_id")["primary_use"]
        bad = [nm for nm in names
               if nm in meta.index and meta[nm] != use[nm]]
        print(f"cross-checked against {os.path.basename(p)}: "
              f"{len(bad)} mismatches")
        if bad:
            print("  overriding from the file for:", bad[:5])
            for nm in bad:
                use[nm] = meta[nm]
        break

groups = {}
for i, nm in enumerate(names):
    groups.setdefault(use[nm], []).append(i)

print(f"clients by primary use: "
      f"{ {k: len(v) for k, v in sorted(groups.items())} }")


def pooled(level, scheme, idx):
    """WAPE and PICP over one cluster, pooled across clients and hours"""
    s = sweep[(level, scheme)]
    y = np.concatenate([np.asarray(s["y"][i]).ravel() for i in idx])
    lo = np.concatenate([np.asarray(s["lo"][i]).ravel() for i in idx])
    md = np.concatenate([np.asarray(s["md"][i]).ravel() for i in idx])
    up = np.concatenate([np.asarray(s["up"][i]).ravel() for i in idx])
    m = np.isfinite(y) & np.isfinite(md) & np.isfinite(lo) & np.isfinite(up)
    y, lo, md, up = y[m], lo[m], md[m], up[m]
    wape = 100 * np.sum(np.abs(y - md)) / (np.sum(np.abs(y)) + 1e-9)
    picp = np.mean((y >= lo) & (y <= up))
    return wape, picp


rows = []
for cl in ("Education", "Office"):
    if cl not in groups:
        print(f"  [skip] no clients tagged {cl}")
        continue
    idx = groups[cl]
    for scheme, label in (("integrated", "integrated"),
                          ("modular_local", "modular$_\\mathrm{local}$")):
        w0, p0 = pooled(0.0, scheme, idx)
        w1, p1 = pooled(1.0, scheme, idx)
        rows.append((cl, label, w0, w1, p0, p1))

print(f"\n{'cluster':<11}{'scheme':<26}"
      f"{'WAPE l=0':>9}{'WAPE l=1':>9}{'PICP l=0':>9}{'PICP l=1':>9}")
for cl, lab, w0, w1, p0, p1 in rows:
    plain = lab.replace("$_\\mathrm{local}$", "_local")
    print(f"{cl:<11}{plain:<26}{w0:9.1f}{w1:9.1f}{p0:9.2f}{p1:9.2f}")

print("\n% ---- rows for tab:cluster ----")
for cl, lab, w0, w1, p0, p1 in rows:
    print(f" & {lab:<26} & {w0:.1f} & {w1:.1f} & {p0:.2f} & {p1:.2f} \\\\")

pd.DataFrame(rows, columns=["cluster", "scheme", "WAPE_0", "WAPE_1",
                            "PICP_0", "PICP_1"]).to_csv(
    os.path.join(OUT_DIR, "results_cluster.csv"), index=False)
print("\nsaved results_cluster.csv")

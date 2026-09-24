# chronos_backbone.py — SCAFFOLD: swap a time-series foundation model (Chronos-Bolt)
# in as the forecaster, feeding the SAME conformal / adaptive calibration layer.
#
# Rationale: a foundation model forecasts a horizon directly from a load-history CONTEXT,
# bypassing the engineered per-step features. So the FM replaces the MLP + recursive rollout;
# feature-asymmetry becomes degradation of the CONTEXT window (blend with climatology), and your
# calibration layer (deployment-regime conformal + adaptive) wraps the FM's quantile outputs
# unchanged. This keeps every calibration finding backbone-agnostic.
#
# STATUS: integration path with TODOs. Needs the chronos package + GPU; test on a few clients first.
#
# In Colab (same project):  !pip -q install chronos-forecasting
# Reuses from v4 session: clients, CLI_DIR, LEVELS, ALPHA, imetrics, qs_score, OUT_DIR, np, pd, torch.

import os, numpy as np, pandas as pd, torch

CONTEXT = 168          # hours of history fed to the FM (7 days)
HORIZON = 24
QLO, QMID, QHI = 0.1, 0.5, 0.9

# ---- load the foundation model (Chronos-Bolt small is a good start) ---------------------
# In Colab first run:  !pip -q install chronos-forecasting
from chronos import BaseChronosPipeline
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map=_DEV,
    torch_dtype=(torch.bfloat16 if _DEV == "cuda" else torch.float32),
)
QLEVELS = [QLO, QMID, QHI]

def load_series(name):
    d = pd.read_parquet(os.path.join(CLI_DIR, name + ".parquet"))
    return d

def climatology(d):
    trl = d[d.split == "train"].dropna(subset=["load"])
    clim = trl.groupby([trl.index.hour, trl.index.dayofweek])["load"].mean().to_dict()
    g = trl["load"].mean()
    return clim, g

def degrade_context(ctx_vals, ctx_idx, clim, g, level):
    if level <= 0:
        return ctx_vals
    cl = np.array([clim.get((t.hour, t.dayofweek), g) for t in ctx_idx])
    return (1 - level) * ctx_vals + level * cl

def fm_forecast(name, split, level, seed_from):
    """Rolling day-ahead FM quantile forecast over `split`; returns y, lo, md, up."""
    d = load_series(name); clim, g = climatology(d)
    seg = d[d.split == split]
    hist = d[d.split.isin([seed_from, split])]["load"]         # continuous history up to each block
    y = seg["load"].values; N = len(y)
    lo = np.empty(N); md = np.empty(N); up = np.empty(N)
    hist_vals = hist.values; hist_idx = hist.index
    start = len(hist_vals) - N                                  # position where `split` begins
    for b0 in range(0, N, HORIZON):
        b1 = min(b0 + HORIZON, N)
        c_end = start + b0
        c_beg = max(0, c_end - CONTEXT)
        ctx = hist_vals[c_beg:c_end].astype(float)
        cidx = hist_idx[c_beg:c_end]
        ctx = degrade_context(ctx, cidx, clim, g, level)        # feature-asymmetry on the context
        ctx = np.nan_to_num(ctx, nan=g)
        # ---- Chronos-Bolt zero-shot quantile forecast ----
        ctx_t = torch.tensor(ctx, dtype=torch.float32)
        # predict_quantiles returns (quantiles, mean):
        #   quantiles shape [num_series, prediction_length, num_quantiles]
        q, _mean = pipe.predict_quantiles(
            ctx_t,                       # first positional arg (named 'inputs'/'context' by version)
            prediction_length=b1 - b0,
            quantile_levels=QLEVELS,
        )
        q = q[0].float().cpu().numpy()               # (h, 3) -> columns are QLO, QMID, QHI
        lo[b0:b1], md[b0:b1], up[b0:b1] = q[:, 0], q[:, 1], q[:, 2]
    return y, lo, md, up

# ---- calibration reuse -------------------------------------------------------------------
# Once fm_forecast works, calibrate exactly as in the MLP pipeline:
#   calib scores  = |y_calib - md_calib|  from fm_forecast(name,'calib',level,'val')
#   deployment-regime conformal + local/federated radii  (same code as model_federated_core_v4)
#   adaptive ACI  seeded on clean fm scores  (same adaptive_conformal loop)
# The point is: NOTHING in the calibration layer changes — only the forecaster does.
#
# Suggested first experiment: zero-shot Chronos-Bolt vs the trained MLP at level 0 and level 1,
# reporting WAPE / PICP80 / MPIW80 / QS with modular_local + adaptive calibration on top.
# ---- quick self-test: forecast ONE client, clean vs full degradation --------------------
if __name__ == "__main__" or True:
    name = list(clients)[0]
    for lv in (0.0, 1.0):
        y, lo, md, up = fm_forecast(name, "test", lv, "calib")
        m = np.isfinite(y) & np.isfinite(md)
        wape = 100 * np.sum(np.abs(y[m] - md[m])) / (np.sum(np.abs(y[m])) + 1e-9)
        picp = np.mean((y[m] >= lo[m]) & (y[m] <= up[m]))
        print(f"[{name}] level={lv:.1f}  WAPE={wape:5.1f}%  raw-PICP80={picp:.2f}  "
              f"shapes ok: {y.shape==md.shape}")
    print("\nFM forecaster works. Next: build calib scores from fm_forecast(...,'calib','val'),"
          " then reuse the deployment-regime conformal + adaptive layer verbatim.")

# model_out/

This is `OUT_DIR`. Both notebooks compute it as
`os.path.join(PROJECT_DIR, "model_out")`, so the folder name matters: rename it
and every script fails to find its inputs.

Everything here is **generated**. Run the notebooks and the files below appear.
Nothing needs to be created by hand.

## What writes what

| section of `model_federated_core_v5_clean.ipynb` | writes |
|---|---|
| 4. Train the two backbones | `model_int.pt`, `model_pt.pt` |
| 7. Dose-response sweep | `results_doseresponse.csv`, `rollout_resid_L1.npy`, `sweep.pkl` |
| 8. Derived artifacts | `resid_structured.npz`, `twin_stream.npz` |
| 9. Transfer, adaptive variants, grid | `results_calibtransfer.csv`, `results_grid_adaptive.csv`, `results_adaptive_cqr.csv` |
| 10. Baselines | `results_baselines.csv` |
| 11. Capacity economics | `cost_curve.csv`, `cost_optima_sweep.csv`, `bandit_hist_seasons.npy`, `bandit_seasons.npz`, `sweep_all5.pkl`, `sweep_L1.pkl`, `cost_table_cp.csv` |
| 12. Foundation-model backbone | `fm_ckpt.pkl`, `results_fm_signed.csv`, `results_fm_adaptive_cqr.csv` |
| 13. Significance tests | the Diebold-Mariano CSVs |
| 14. Figures | every `.pdf` and `.png`: `doseresponse`, `regimes`, `symmetry`, `fm_backbone`, `fig_costopt`, `bandit_arms`, `cost_regimes`, `fig_3b`, `fig_3c` |

| section of `twin_v2.ipynb` | writes |
|---|---|
| 3. Mirror | `acl_state_ckpt.pkl`, checkpointed every ten stream days |
| 4. Frozen replay | `dashboard_replay.html`, `twin_series.json` |
| 5. Figures | `dt_live.pdf`, `dt_live.png`, `twin_running.gif` |

## Order matters

Section 8 reads what section 7 leaves in memory, section 11 reads section 10's
output, and section 14 reads the CSVs from 10 through 13. Run the notebook top
to bottom the first time.

Sections 4 and 7 together take about 25 minutes. Everything after them is
minutes or seconds.

## If the folder is empty

That is the expected state of a fresh clone. Run the notebooks and it fills.

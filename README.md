# Emotion Forecaster — Anonymized Results

Raw model outputs were compressed and *inverted* relative to the 1–7 label range.  
A simple linear calibration fixed this.

## Headline (micro)
- **Raw:** MAE 1.81, RMSE 2.28, ρ −0.35  
- **Post-hoc calibration:** MAE 1.42, RMSE 1.71, ρ +0.35  
- **Calibration mapping:**  ŷ = 12.893 − 1.403 × ŷ_raw  (then clipped to [1, 7])

## What’s in this repo
- `reports/*_anon.csv` — anonymized predictions & per-user summaries (IDs are `U01, U02, …`)
  - e.g., `gru_preds_calibrated_cv_anon.csv`, `gru_fold_summary_calibrated_cv_anon.csv`,
    `fold_summary_pretty.csv`, `baseline_louo_anon.csv`, etc.
- `src/` and top-level scripts (`train_gru.py`, `train_baseline.py`, `score_shifts.py`,
  `sanity_check.py`, `plot_trajectories.py`) for training/evaluation and plotting.

## Notes
- Headline numbers are **micro** because per-fold ρ is unstable with tiny test sets.
- User IDs are anonymized to **U01, U02, …**; the private mapping is not in the repo.

## Reproduce
Use `reports/*_anon.csv` directly. (Optional: `python plot_trajectories.py` for figures.)

# Emotion Forecaster — Anonymized Results

Raw model outputs were compressed and *inverted* relative to the 1–7 label range.  
A simple linear calibration fixed this.

## Headline (micro)
- **Raw:** MAE 1.81, RMSE 2.28, ρ −0.35  
- **Post-hoc calibration:** MAE 1.42, RMSE 1.71, ρ +0.35  
- **Cross-validated calibration (no leakage):** ← paste your `CVCAL-> ...` line here

**Calibration mapping:**  ŷ = 12.893 − 1.403 × ŷ_raw  (then clipped to [1, 7])

## What’s in this repo
- `reports/*_anon.csv` — anonymized predictions & per-user summaries (IDs are `U01, U02, …`)
  - e.g., `gru_preds_calibrated_cv_anon.csv`, `gru_fold_summary_calibrated_cv_anon.csv`,
    `fold_summary_pretty.csv`, `baseline_louo_anon.csv`, etc.
- `src/` and top-level scripts (`train_gru.py`, `train_baseline.py`, `score_shifts.py`,
  `sanity_check.py`, `plot_trajectories.py`) for training/evaluation and plotting.

## Notes
- Per-fold correlations can be unstable when test sets are tiny, so we emphasize **global (micro)**
  metrics for the headline numbers.
- User identifiers are anonymized. The private mapping (`user_id_mapping.csv`, `user_id_legend.txt`)
  is **not** included in the repo.

## Reproduce (from predictions)
If you have local raw predictions, you can run the metric and calibration scripts
to regenerate the CSVs; otherwise, use the anonymized CSVs here directly for analysis/plots.

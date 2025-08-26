Raw model outputs were compressed and inverted; a simple linear calibration corrected this.

**Headline (micro):**
- Raw: MAE 1.81, RMSE 2.28, ρ −0.35
- Post-hoc calibration: MAE 1.42, RMSE 1.71, ρ +0.35
- CV calibration (train-only fit, test-fold apply): see anonymized CSVs in `reports/`.

Per-fold ρ can be unstable when test sets are tiny; we emphasize global (micro) metrics.
User identifiers are anonymized to `U01, U02, …`.

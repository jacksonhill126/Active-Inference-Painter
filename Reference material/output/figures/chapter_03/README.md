# `output/figures/chapter_03/` — Chapter 3 figures

PNG and GIF outputs from the Chapter 3 orchestrators
(`chapters/chapter_03/`). **Ephemeral and gitignored** — regenerate with:

```bash
./scripts/run_all_chapter_03.sh
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_03/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_3_1_mle.png` | `example_3_1_linear_regression_mle.py` |
| `example_3_2_gd.png` | `example_3_2_linear_regression_gd.py` |
| `example_3_3_multiple_regression.png` | `example_3_3_multiple_regression.py` |
| `example_3_4_mvn.png` | `example_3_4_multivariate_gaussian.py` |
| `example_3_5_blr_posteriors.png` / `example_3_5_blr_predictive.png` | `example_3_5_bayesian_linear_regression.py` |
| `example_3_6_lgs.png` | `example_3_6_lgs_food_localization.py` |
| `example_3_7_factor_analysis_em.png` | `example_3_7_factor_analysis_em.py` |
| `animation_blr_tightening.gif` | `animation_blr_tightening.py` |
| `animation_blr_predictive_band.gif` | `animation_blr_predictive_band.py` |
| `animation_em_convergence.gif` | `animation_em_convergence.py` |
| `animation_em_steps.gif` | `animation_em_steps.py` |
| `animation_lgs_online.gif` | `animation_lgs_online.py` |
| `animation_precision_sweep.gif` | `animation_precision_sweep.py` |
| `animation_bimodal_emergence.gif` | `animation_bimodal_emergence.py` |
| `animation_sufficient_statistics.gif` | `animation_sufficient_statistics.py` |
| `diagnostic_calibration.png` | `visualize_calibration.py` |
| `diagnostic_coverage.png` | `visualize_coverage.py` |
| `diagnostic_ppc_mean.png` / `diagnostic_ppc_range.png` / `diagnostic_ppc_std.png` | `visualize_posterior_predictive.py` |
